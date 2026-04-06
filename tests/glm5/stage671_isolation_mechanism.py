#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage671: P25 隔离机制破解 v3 — 纯Hidden State分析（不通过margin）

核心改变：不再通过logit/margin来测量干扰，而是直接分析hidden state的变化
这样可以避免hook注入导致模型forward异常的问题

实验设计：
  A. 隔离测量：注入能力X的方向到hidden state，看各层如何传播
  B. Norm压缩效应：直接比较norm层前后的hidden state变化
  C. 残差流衰减：注入信号在残差流中如何衰减
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

PROMPTS = {
    "disamb_a": "The river bank was muddy.",
    "disamb_b": "The bank approved the loan.",
    "syntax_a": "The key to the cabinet",
    "syntax_b": "The keys to the cabinet",
    "relation_a": "Water freezes at zero.",
    "relation_b": "Water boils at 100 degrees.",
    "coref_a": "Alice thanked Mary because Alice had won.",
    "coref_b": "Alice thanked Mary because Mary had won.",
    "style_a": "Choose the formal: The ceremony will",
    "style_b": "Choose the casual: The game will",
}

CAPABILITIES = ["disamb", "syntax", "relation", "coref", "style"]


def extract_all_hidden(model, tokenizer, prompt: str) -> Dict[int, torch.Tensor]:
    """提取每层最后一个token的hidden state"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden_states = {}
    hooks = []
    layers = discover_layers(model)

    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states[li] = output[0][:, -1, :].detach().cpu().float()
            else:
                hidden_states[li] = output[:, -1, :].detach().cpu().float()
        return hook_fn

    for li in range(len(layers)):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return hidden_states


def compute_directions(model, tokenizer) -> Dict[str, torch.Tensor]:
    """计算每种能力的编码方向"""
    directions = {}
    for cap in CAPABILITIES:
        h_a = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_a"])
        h_b = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_b"])
        
        # 用最后一层的差异
        last_layer = max(h_a.keys())
        diff = h_a[last_layer] - h_b[last_layer]
        if diff.norm() > 1e-6:
            directions[cap] = diff / (diff.norm() + 1e-10)
    return directions


def find_norm_modules(model) -> List[Tuple[str, object]]:
    norms = []
    for name, module in model.named_modules():
        mod_name = type(module).__name__.lower()
        if 'norm' in mod_name or 'rmsnorm' in mod_name or 'layernorm' in mod_name:
            norms.append((name, module))
    return norms


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage671: P25 隔离机制破解(v3) - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    norms = find_norm_modules(model)
    norm_types = set(type(m).__name__ for _, m in norms)

    print(f"  层数={num_layers}, Norm模块数={len(norms)}, 类型={norm_types}")

    # 计算编码方向
    print(f"  计算编码方向...")
    directions = compute_directions(model, tokenizer)
    print(f"  方向: {list(directions.keys())}")

    # ========== 实验A: 残差流信号衰减 ==========
    print(f"\n{'='*70}")
    print(f"实验A: 残差流信号衰减 — 注入方向在各层的对齐度")
    print(f"{'='*70}")

    # 对每种能力，看各层的增量差与最终方向的对齐
    print(f"\n  各层增量差与最终方向的对齐度(cos):")
    for cap in CAPABILITIES:
        h_a = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_a"])
        h_b = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_b"])
        
        last = max(h_a.keys())
        d_final = (h_a[last] - h_b[last]).flatten()
        d_unit = d_final / (d_final.norm() + 1e-10)

        # 计算逐层增量差
        sorted_keys = sorted(h_a.keys())
        prev_a, prev_b = None, None
        layer_alignments = []
        layer_norms = []

        for li in sorted_keys:
            ha = h_a[li].flatten()
            hb = h_b[li].flatten()
            h_diff = ha - hb
            layer_norms.append(float(h_diff.norm()))

            if prev_a is not None:
                inc_diff = (ha - prev_a) - (hb - prev_b)
                cos_val = float(torch.dot(inc_diff, d_unit) / (inc_diff.norm() * d_unit.norm() + 1e-10))
                cos_val = max(-1.0, min(1.0, cos_val))
                layer_alignments.append(cos_val)
            prev_a = ha
            prev_b = hb

        # 打印：每5层一个数据点
        print(f"\n  {cap}:")
        step = max(1, len(layer_alignments) // 8)
        for i in range(0, len(layer_alignments), step):
            li = sorted_keys[i + 1]  # +1 because first layer has no increment
            cos_v = layer_alignments[i]
            norm_v = layer_norms[i + 1]
            bar = "+" * int(cos_v * 20) if cos_v > 0 else "-" * int(abs(cos_v) * 20)
            print(f"    L{li:>2}: cos={cos_v:>6.3f} |{bar:<20}| ||diff||={norm_v:.1f}")

        # 统计
        pos_count = sum(1 for c in layer_alignments if c > 0)
        neg_count = sum(1 for c in layer_alignments if c < 0)
        avg_abs_cos = statistics.mean(abs(c) for c in layer_alignments)
        print(f"    正对齐:{pos_count} 负对齐:{neg_count} 平均|cos|:{avg_abs_cos:.3f}")

    # ========== 实验B: 层间旋转与抵消 ==========
    print(f"\n{'='*70}")
    print(f"实验B: 相邻层增量差之间的旋转角")
    print(f"{'='*70}")

    for cap in CAPABILITIES[:3]:
        h_a = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_a"])
        h_b = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_b"])

        sorted_keys = sorted(h_a.keys())
        prev_a, prev_b = None, None
        inc_diffs = []

        for li in sorted_keys:
            if prev_a is not None:
                ha = h_a[li].flatten()
                hb = h_b[li].flatten()
                inc = (ha - prev_a) - (hb - prev_b)
                inc_diffs.append(inc)
            prev_a = h_a[li].flatten()
            prev_b = h_b[li].flatten()

        # 相邻层旋转角
        angles = []
        for i in range(len(inc_diffs) - 1):
            cos_v = float(torch.dot(inc_diffs[i], inc_diffs[i+1]) / 
                         (inc_diffs[i].norm() * inc_diffs[i+1].norm() + 1e-10))
            cos_v = max(-1.0, min(1.0, cos_v))
            angle = float(torch.acos(torch.tensor(cos_v))) * 180 / 3.14159
            angles.append(angle)

        avg_angle = statistics.mean(angles)
        std_angle = statistics.stdev(angles) if len(angles) > 1 else 0
        
        print(f"  {cap}: 平均旋转角={avg_angle:.1f}° (std={std_angle:.1f}°), "
              f"范围=[{min(angles):.0f}°, {max(angles):.0f}°]")

    # ========== 实验C: 注入信号传播分析 ==========
    print(f"\n{'='*70}")
    print(f"实验C: 归一化对信号的影响 (纯数学分析)")
    print(f"{'='*70}")

    # 不需要真正注入，直接分析norm的效果
    # RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * gamma
    # LayerNorm: x_norm = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    # 分析：如果我们在hidden state h上添加bias b
    # h' = h + b
    # h'_norm = norm(h + b)
    # 信号残存 = cos(h'_norm, h_norm) * ||h'_norm|| / ||h_norm||

    for cap in CAPABILITIES[:3]:
        h_a = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_a"])
        h_b = extract_all_hidden(model, tokenizer, PROMPTS[f"{cap}_b"])
        last = max(h_a.keys())
        h = h_a[last].flatten()  # 原始hidden state

        # 注入不同能力的方向
        print(f"\n  原始向量(来自{cap}_a): ||h||={h.norm():.2f}")
        
        for inject_cap in CAPABILITIES:
            if inject_cap not in directions:
                continue
            
            inject_dir = directions[inject_cap]
            
            for amp in [1.0, 10.0, 50.0]:
                bias = inject_dir.flatten() * amp * (h.norm() / 100)  # 幅度相对于h
                h_injected = h + bias
                
                # 模拟RMSNorm: 归一化到单位球然后乘以gamma
                # 简化版：只看方向变化
                h_unit = h / (h.norm() + 1e-10)
                h_inj_unit = h_injected / (h_injected.norm() + 1e-10)
                
                cos_after = float(torch.dot(h_unit, h_inj_unit))
                cos_after = max(-1.0, min(1.0, cos_after))
                angle_change = float(torch.acos(torch.tensor(cos_after))) * 180 / 3.14159
                
                # 信号残存率（RMSNorm后，如果gamma不变，幅度不变，只有方向变化）
                # 真正的残存 = 投影到原始方向的分量
                retention = cos_after
                
                print(f"    注入{inject_cap:>8} amp={amp:>4.0f}: "
                      f"方向偏移={angle_change:>6.2f}°, "
                      f"信号残存={retention:.4f}")

    # ========== 实验D: 跨能力方向的几何关系 ==========
    print(f"\n{'='*70}")
    print(f"实验D: 跨能力方向的几何关系")
    print(f"{'='*70}")

    # 计算所有能力方向的cos similarity矩阵
    dir_list = [(cap, directions[cap].flatten()) for cap in CAPABILITIES if cap in directions]
    
    print(f"\n  Cos Similarity矩阵:")
    header = f"  {'':>10}"
    for cap, _ in dir_list:
        header += f"  {cap:>10}"
    print(header)
    print("  " + "-" * (12 + 12 * len(dir_list)))
    
    cos_matrix = {}
    for cap1, d1 in dir_list:
        row = f"  {cap1:>10}"
        for cap2, d2 in dir_list:
            cos_v = float(torch.dot(d1, d2))
            cos_matrix[(cap1, cap2)] = cos_v
            marker = "★" if cap1 == cap2 else " "
            row += f"  {marker}{cos_v:>8.3f}"
        print(row)
    
    # 分析
    off_diag = [abs(cos_matrix[(c1, c2)]) for c1, _ in dir_list for c2, _ in dir_list if c1 != c2]
    avg_off = statistics.mean(off_diag)
    print(f"\n  非对角线|cos|均值: {avg_off:.4f}")
    print(f"  隔离几何指标: {'强隔离(<0.1)' if avg_off < 0.1 else '中等隔离(0.1-0.3)' if avg_off < 0.3 else '弱隔离(>0.3)'}")

    # ========== 实验E: 归一化类型与架构分析 ==========
    print(f"\n{'='*70}")
    print(f"实验E: 归一化架构分析")
    print(f"{'='*70}")

    # 统计norm的位置
    input_norms = [n for n, _ in norms if 'input_layernorm' in n or 'input_norm' in n]
    post_attn_norms = [n for n, _ in norms if 'post_attention' in n or 'post_attn' in n]
    qk_norms = [n for n, _ in norms if 'q_norm' in n or 'k_norm' in n]
    final_norms = [n for n, _ in norms if 'final' in n or 'norm' in n.split('.')[-1]]
    
    print(f"  Input norms: {len(input_norms)}")
    print(f"  Post-attention norms: {len(post_attn_norms)}")
    print(f"  Q/K norms: {len(qk_norms)}")
    print(f"  Final norms: {len(final_norms)}")
    print(f"  Total norms: {len(norms)}")
    print(f"  Norm per layer: {len(norms) / (num_layers + 1e-10):.1f}")
    print(f"  Norm type: {list(norm_types)}")

    # 分析归一化对隔离的贡献
    # 如果每个block有2个norm(input + post_attn)，每层信号被归一化2次
    # 每次归一化都会"重置"幅度的差异
    norms_per_block = len(input_norms) / (num_layers + 1e-10)
    print(f"\n  每层归一化次数: {norms_per_block:.1f}")
    print(f"  全链路总归一化次数: {norms_per_block * num_layers:.0f}")
    print(f"  隔离放大效应估算: 每次归一化压缩偏差~{(1-avg_off):.3f}, "
          f"经过{norms_per_block * num_layers:.0f}次后 → "
          f"{(1-avg_off) ** (norms_per_block * num_layers):.6f}")

    # 保存
    result_data = {
        "model": model_key,
        "num_layers": num_layers,
        "norm_types": list(norm_types),
        "norm_count": len(norms),
        "norms_per_layer": norms_per_block,
        "avg_offdiag_cos": avg_off,
        "cos_matrix": {f"{k[0]}_{k[1]}": v for k, v in cos_matrix.items()},
        "isolation_estimate": float((1 - avg_off) ** (norms_per_block * num_layers)),
    }

    out_path = OUTPUT_DIR / f"stage671_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")

    free_model(model)
    return result_data


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
