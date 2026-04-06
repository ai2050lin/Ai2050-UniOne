#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage667: P20 跨能力因果干扰实验

目标：验证不同能力的编码方向之间是否存在因果干扰
- 对每个能力类型，提取其"编码方向"(d_final = h_A_last - h_B_last)
- 在模型前向传播时，注入其他能力的编码方向
- 测量注入后当前能力的margin变化（干扰度）
- 测试跨能力的正交性是否提供"天然隔离"

实验设计：
1. 对25个样本(5能力×5样本)，提取各自的d_final方向
2. 交叉注入：用能力X的d_final方向注入到能力Y的测试中
3. 测量margin变化率 = (margin_intervened - margin_baseline) / |margin_baseline|
4. 对角线（自注入）应该有强效应，非对角线（跨能力）应该弱

额外测试：
- 注入放大系数（0.1, 0.5, 1.0）vs 干扰度关系
- 每对能力的正交性(cos) vs 干扰度的相关性
- 哪些能力对之间干扰最强

判伪标准：
INV-294: "跨能力干扰度 < 30% → 编码隔离有效"
INV-295: "自注入(对角线)干扰度 > 跨注入(非对角线) → 编码是能力特异的"
INV-296: "cos(方向)越低 → 干扰度越低 → 正交性=隔离机制"
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[TestCase] = [
    # 消歧 (3)
    TestCase(capability="disamb", pair_id="bank",
             prompt_a="The river bank was muddy.",
             positive_a=" shore", negative_a=" finance",
             prompt_b="The bank approved the loan.",
             positive_b=" finance", negative_b=" shore"),
    TestCase(capability="disamb", pair_id="plant",
             prompt_a="The plant was green.",
             positive_a=" leaf", negative_a=" factory",
             prompt_b="The plant closed down.",
             positive_b=" factory", negative_b=" leaf"),
    TestCase(capability="disamb", pair_id="bat",
             prompt_a="The bat flew through the cave.",
             positive_a=" animal", negative_a=" sports",
             prompt_b="He swung the bat at the ball.",
             positive_b=" sports", negative_b=" animal"),
    # 语法 (3)
    TestCase(capability="syntax", pair_id="subj_verb",
             prompt_a="The key to the cabinet",
             positive_a=" is", negative_a=" are",
             prompt_b="The keys to the cabinet",
             positive_b=" are", negative_b=" is"),
    TestCase(capability="syntax", pair_id="clause_agree",
             prompt_a="The report that was written by the interns",
             positive_a=" was", negative_a=" were",
             prompt_b="The reports that were written by the intern",
             positive_b=" were", negative_b=" was"),
    TestCase(capability="syntax", pair_id="plural_pp",
             prompt_a="The label on the bottles",
             positive_a=" is", negative_a=" are",
             prompt_b="The labels on the bottle",
             positive_b=" are", negative_b=" is"),
    # 关系 (3)
    TestCase(capability="relation", pair_id="capital",
             prompt_a="Paris is the capital of France. The capital of France is",
             positive_a=" Paris", negative_a=" Berlin",
             prompt_b="Berlin is the capital of Germany. The capital of Germany is",
             positive_b=" Berlin", negative_b=" Paris"),
    TestCase(capability="relation", pair_id="author",
             prompt_a="Shakespeare wrote Hamlet. The author of Hamlet is",
             positive_a=" Shakespeare", negative_a=" Dickens",
             prompt_b="Tolstoy wrote War and Peace. The author of War and Peace is",
             positive_b=" Tolstoy", negative_b=" Shakespeare"),
    TestCase(capability="relation", pair_id="chemical",
             prompt_a="The chemical symbol for water is",
             positive_a=" H2O", negative_a=" CO2",
             prompt_b="The chemical symbol for carbon dioxide is",
             positive_b=" CO2", negative_b=" H2O"),
    # 指代 (3)
    TestCase(capability="coref", pair_id="winner",
             prompt_a="Alice thanked Mary because Alice had won the prize. The person who won was",
             positive_a=" Alice", negative_a=" Mary",
             prompt_b="Alice thanked Mary because Mary had won the prize. The person who won was",
             positive_b=" Mary", negative_b=" Alice"),
    TestCase(capability="coref", pair_id="apology",
             prompt_a="John apologized to David because John was late. The person who was late was",
             positive_a=" John", negative_a=" David",
             prompt_b="John apologized to David because David was late. The person who was late was",
             positive_b=" David", negative_b=" John"),
    TestCase(capability="coref", pair_id="help",
             prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
             positive_a=" Emma", negative_a=" Sara",
             prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
             positive_b=" Sara", negative_b=" Emma"),
    # 风格 (3)
    TestCase(capability="style", pair_id="formal_rewrite",
             prompt_a="Choose the more formal rewrite: I need your help with this request.",
             positive_a=" assistance", negative_a=" help",
             prompt_b="Choose the more casual rewrite: I require your assistance with this request.",
             positive_b=" help", negative_b=" assistance"),
    TestCase(capability="style", pair_id="formal_word",
             prompt_a="Choose the more formal next word: The ceremony will",
             positive_a=" commence", negative_a=" start",
             prompt_b="Choose the more casual next word: The game will",
             positive_b=" start", negative_b=" commence"),
    TestCase(capability="style", pair_id="formal_request",
             prompt_a="Choose the more formal request ending: Please review the attached file and",
             positive_a=" advise", negative_a=" tell",
             prompt_b="Choose the more casual request ending: Please look at the file and",
             positive_b=" tell", negative_b=" advise"),
]

CAPABILITIES = ["disamb", "syntax", "relation", "coref", "style"]


def extract_d_final(model, tokenizer, prompt_a: str, prompt_b: str, num_layers: int) -> torch.Tensor:
    """提取最后一层的d_final = h_A[-1] - h_B[-1]"""
    device = next(model.parameters()).device
    
    def get_last_hidden(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = {}
        hooks = []
        layers = discover_layers(model)
        
        def make_hook(li):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden[li] = output[0][:, -1, :].detach().cpu().float()
                else:
                    hidden[li] = output[:, -1, :].detach().cpu().float()
            return hook_fn
        
        for li in range(len(layers)):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))
        
        with torch.no_grad():
            model(**inputs)
        for h in hooks:
            h.remove()
        return hidden
    
    hidden_a = get_last_hidden(prompt_a)
    hidden_b = get_last_hidden(prompt_b)
    
    return hidden_a[num_layers - 1].flatten() - hidden_b[num_layers - 1].flatten()


def inject_direction_and_measure(model, tokenizer, case: TestCase,
                                   inject_direction: torch.Tensor,
                                   inject_strength: float,
                                   inject_layer: int) -> Dict:
    """
    注入一个方向向量到指定层，测量margin变化
    
    inject_direction: 要注入的方向向量(1D)
    inject_strength: 注入强度
    inject_layer: 注入到哪一层(-1=最后一层)
    """
    layers = discover_layers(model)
    num_layers = len(layers)
    target_layer = inject_layer if inject_layer >= 0 else num_layers - 1
    
    # 归一化方向
    dir_norm = inject_direction.norm()
    if dir_norm < 1e-10:
        return {"error": "zero direction"}
    dir_unit = inject_direction / dir_norm
    
    # bias = strength * dir_unit * d_final_typical_scale
    # 用一个合理的幅度：目标层increment_diff的典型范数
    # bias = strength * dir_unit * 典型增量差范数的比例
    # 注入幅度应该与模型hidden state同量级
    bias = inject_strength * dir_unit * 5.0  # 更大的幅度
    
    # 测量baseline margin
    import gc
    torch.cuda.empty_cache()
    margin_a_base = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                    score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    
    # 注入并测量
    def make_bias_hook(bias_tensor):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                b = bias_tensor.to(hidden.device, hidden.dtype)
                hidden = hidden + b.unsqueeze(0).unsqueeze(0)
                return (hidden,) + output[1:]
            else:
                b = bias_tensor.to(output.device, output.dtype)
                return output + b.unsqueeze(0).unsqueeze(0)
        return hook_fn
    
    hooks = []
    if target_layer < len(layers):
        hooks.append(layers[target_layer].register_forward_hook(make_bias_hook(bias)))
    
    try:
        torch.cuda.empty_cache()
        margin_a_int = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                       score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    finally:
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()
    
    delta = margin_a_int - margin_a_base
    relative_change = delta / (abs(margin_a_base) + 1e-10)
    
    return {
        "margin_base": margin_a_base,
        "margin_intervened": margin_a_int,
        "delta": delta,
        "relative_change": relative_change,
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage667: P20 跨能力因果干扰实验 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    
    # ========== 阶段1：提取所有样本的编码方向 ==========
    print(f"\n阶段1：提取编码方向...")
    
    case_directions = {}  # (capability, pair_id) -> d_final
    cap_avg_directions = {}  # capability -> avg d_final (归一化)
    
    for case in CASES:
        print(f"  [{case.capability}/{case.pair_id}] ", end="", flush=True)
        d_final = extract_d_final(model, tokenizer, case.prompt_a, case.prompt_b, num_layers)
        case_directions[(case.capability, case.pair_id)] = d_final
        print(f"||d||={d_final.norm():.4f}")
    
    # 计算每个能力的平均方向
    for cap in CAPABILITIES:
        directions = [case_directions[(cap, c.pair_id)] for c in CASES if c.capability == cap
                      and (cap, c.pair_id) in case_directions]
        if directions:
            avg_dir = sum(directions)
            avg_dir = avg_dir / (avg_dir.norm() + 1e-10)
            cap_avg_directions[cap] = avg_dir
    
    # ========== 阶段2：跨能力正交性矩阵 ==========
    print(f"\n阶段2：跨能力正交性矩阵(cos similarity)...")
    
    cap_cos_matrix = {}
    for cap_x in CAPABILITIES:
        for cap_y in CAPABILITIES:
            if cap_x in cap_avg_directions and cap_y in cap_avg_directions:
                cos_val = float(torch.dot(cap_avg_directions[cap_x], cap_avg_directions[cap_y]))
                cap_cos_matrix[(cap_x, cap_y)] = cos_val
    
    print(f"  {'':>10}", end="")
    for cap in CAPABILITIES:
        print(f"  {cap:>10}", end="")
    print()
    for cap_x in CAPABILITIES:
        print(f"  {cap_x:>10}", end="")
        for cap_y in CAPABILITIES:
            cos_val = cap_cos_matrix.get((cap_x, cap_y), 0)
            print(f"  {cos_val:>10.4f}", end="")
        print()
    
    # ========== 阶段3：交叉注入实验 ==========
    print(f"\n阶段3：交叉注入实验(strength=0.5, 注入最后一层)...")
    
    # 注入策略：用能力X的平均方向注入到能力Y的样本中
    # 测量margin的相对变化
    interference_matrix = {}  # (inject_cap, target_cap) -> avg relative change
    n_samples_per_pair = {}
    
    strengths_to_test = [0.3, 1.0]
    
    for inject_cap in CAPABILITIES:
        if inject_cap not in cap_avg_directions:
            continue
        inject_dir = cap_avg_directions[inject_cap]
        
        for target_cap in CAPABILITIES:
            target_cases = [c for c in CASES if c.capability == target_cap]
            relative_changes = []
            
            for case in target_cases[:3]:  # 限制每个target只用3个样本
                for strength in strengths_to_test:
                    result = inject_direction_and_measure(
                        model, tokenizer, case, inject_dir, strength, -1)
                    if "error" not in result:
                        relative_changes.append({
                            "strength": strength,
                            "rel_change": result["relative_change"],
                            "delta": result["delta"],
                            "margin_base": result["margin_base"],
                        })
            
            key = (inject_cap, target_cap)
            interference_matrix[key] = relative_changes
            n_samples_per_pair[key] = len(target_cases)
    
    # 打印干扰矩阵（strength=1.0时）
    print(f"\n  干扰矩阵(相对margin变化, strength=1.0):")
    print(f"  {'注入→目标':>15}", end="")
    for cap in CAPABILITIES:
        print(f"  {cap:>10}", end="")
    print()
    for inject_cap in CAPABILITIES:
        print(f"  {inject_cap:>15}", end="")
        for target_cap in CAPABILITIES:
            key = (inject_cap, target_cap)
            if key in interference_matrix:
                vals_s1 = [v["rel_change"] for v in interference_matrix[key] if v["strength"] == 1.0]
                avg_rc = statistics.mean(vals_s1) * 100 if vals_s1 else 0
                marker = "*" if inject_cap == target_cap else " "
                print(f" {marker}{avg_rc:>9.1f}%", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()
    print("  (* = 自注入/对角线)")
    
    # ========== 阶段4：详细统计 ==========
    print(f"\n阶段4：详细统计...")
    
    diagonal_interference = []  # 自注入
    off_diagonal_interference = []  # 跨能力注入
    
    for key, vals in interference_matrix.items():
        inject_cap, target_cap = key
        for v in vals:
            if v["strength"] == 1.0:
                if inject_cap == target_cap:
                    diagonal_interference.append(abs(v["rel_change"]))
                else:
                    off_diagonal_interference.append(abs(v["rel_change"]))
    
    avg_diag = statistics.mean(diagonal_interference) * 100 if diagonal_interference else 0
    avg_off = statistics.mean(off_diagonal_interference) * 100 if off_diagonal_interference else 0
    
    print(f"\n  自注入(对角线)平均干扰度: {avg_diag:.1f}%")
    print(f"  跨注入(非对角线)平均干扰度: {avg_off:.1f}%")
    print(f"  隔离比(跨/自): {avg_off / (avg_diag + 1e-10):.3f}")
    
    # ========== 阶段5：正交性 vs 干扰度相关性 ==========
    print(f"\n阶段5：正交性 vs 干扰度相关性...")
    
    cos_interference_pairs = []
    for inject_cap in CAPABILITIES:
        for target_cap in CAPABILITIES:
            if inject_cap == target_cap:
                continue
            key = (inject_cap, target_cap)
            if key in interference_matrix and key in cap_cos_matrix:
                vals_s1 = [v["rel_change"] for v in interference_matrix[key] if v["strength"] == 1.0]
                if vals_s1:
                    cos_val = cap_cos_matrix[key]
                    avg_interference = statistics.mean([abs(v) for v in vals_s1])
                    cos_interference_pairs.append((cos_val, avg_interference))
    
    if cos_interference_pairs:
        cos_vals = [p[0] for p in cos_interference_pairs]
        int_vals = [p[1] for p in cos_interference_pairs]
        
        # Pearson相关
        n_pairs = len(cos_vals)
        mean_c = statistics.mean(cos_vals)
        mean_i = statistics.mean(int_vals)
        cov = sum((c - mean_c) * (i - mean_i) for c, i in cos_interference_pairs) / n_pairs
        std_c = statistics.stdev(cos_vals) if n_pairs > 1 else 1
        std_i = statistics.stdev(int_vals) if n_pairs > 1 else 1
        pearson = cov / (std_c * std_i + 1e-10)
        
        print(f"  cos vs 干扰度的Pearson相关: r={pearson:.3f}")
        if pearson > 0.5:
            print(f"  → 强正相关：cos越大(方向越相似)→干扰越大，正交性是隔离机制")
        elif pearson < -0.3:
            print(f"  → 负相关：方向相似但干扰反而小，需要新解释")
        else:
            print(f"  → 弱相关：干扰度不完全由方向正交性决定")
    
    # ========== 阶段6：按强度分析 ==========
    print(f"\n阶段6：注入强度 vs 干扰度...")
    for strength in strengths_to_test:
        diag_s = [abs(v["rel_change"]) for key, vals in interference_matrix.items()
                  for v in vals if v["strength"] == strength and key[0] == key[1]]
        off_s = [abs(v["rel_change"]) for key, vals in interference_matrix.items()
                 for v in vals if v["strength"] == strength and key[0] != key[1]]
        avg_d = statistics.mean(diag_s) * 100 if diag_s else 0
        avg_o = statistics.mean(off_s) * 100 if off_s else 0
        print(f"  strength={strength}: 对角线={avg_d:.1f}%, 非对角线={avg_o:.1f}%, 隔离比={avg_o / (avg_d + 1e-10):.3f}")
    
    # ========== 判伪 ==========
    print(f"\n{'='*70}")
    print(f"判伪结论")
    print(f"{'='*70}")
    
    # INV-294: 跨能力干扰度 < 30%
    print(f"  INV-294: 跨能力干扰度={avg_off:.1f}% → "
          f"{'<30% 确认(隔离有效)' if avg_off < 30 else '>=30% 推翻(隔离不足)'}")
    
    # INV-295: 自注入 > 跨注入
    print(f"  INV-295: 自注入={avg_diag:.1f}% vs 跨注入={avg_off:.1f}% → "
          f"{'自>跨 确认(编码能力特异)' if avg_diag > avg_off else '自<=跨 推翻'}")
    
    # INV-296: cos vs 干扰度
    if cos_interference_pairs:
        print(f"  INV-296: Pearson r={pearson:.3f} → "
              f"{'>0.3 确认(正交性=隔离)' if pearson > 0.3 else '<0.3 推翻(正交性非唯一隔离)'}")
    
    # ========== 保存结果 ==========
    out_path = OUTPUT_DIR / f"stage667_{model_key}_{TIMESTAMP}.json"
    save_data = {
        "cos_matrix": {f"{k[0]}_{k[1]}": v for k, v in cap_cos_matrix.items()},
        "interference_summary": {
            "diagonal_avg_pct": avg_diag,
            "off_diagonal_avg_pct": avg_off,
            "isolation_ratio": avg_off / (avg_diag + 1e-10),
        },
        "cos_interference_correlation": pearson if cos_interference_pairs else None,
    }
    
    # 保存完整干扰矩阵
    for key, vals in interference_matrix.items():
        save_data[f"interference_{key[0]}_{key[1]}"] = vals
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return save_data


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
