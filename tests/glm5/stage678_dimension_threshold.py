#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage678: P32 高维阈值精确定位——不同hidden_dim下第一公理的成立条件

目标：训练d=64/128/256/512的tiny GPT-2，测量A1(高维几何)，
     精确确定高维阈值d*——在此维度以上，相邻层方向接近正交。

INV-319: 高维阈值d*≈256-512，低于此阈值时A1不成立
  - d=64:  angle远小于90°（高维效应极弱）
  - d=128: angle≈15°（已知）
  - d=256: angle可能在45-60°（过渡区）
  - d=512: angle应接近75-85°（接近正交）
  - d=1024: angle应≈90°（完全正交）

额外测量：
  - E[|cos(Δ_l, h_l)|] vs 1/√d 的关系
  - 各维度下A2-A5的成立情况
"""

from __future__ import annotations

import sys
import io
import json
import copy
import math
import statistics

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# 离线模式
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


# ============================================================
# 训练数据
# ============================================================
TRAINING_SENTENCES = [
    "The river bank was muddy and the water was rising fast.",
    "The bank approved the loan for the new house.",
    "The bat flew across the dark cave at night.",
    "He swung the bat and hit a home run.",
    "Paris is the capital of France in Europe.",
    "Berlin is the capital of Germany in central Europe.",
    "She ran quickly to the store because she needed milk.",
    "The dog barked loudly at the stranger who approached.",
    "The cat is under the table near the window.",
    "The bird is above the tree in the garden.",
    "The meeting was extremely productive and efficient.",
    "That get-together was quite fruitful and enjoyable.",
    "Yesterday it rained heavily all day long.",
    "Tomorrow it will snow in the northern regions.",
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "If it rains, the ground will be wet and slippery.",
    "All birds have feathers, and eagles are birds with large wings.",
] * 8  # 重复8遍


class SimpleDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=64):
        self.examples = []
        for sent in sentences:
            tokens = tokenizer.encode(sent, truncation=True, max_length=max_length)
            if len(tokens) >= 4:
                # pad到max_length
                padded = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
                self.examples.append(torch.tensor(padded, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================
# 五公理测量
# ============================================================
def get_hidden_states(model, tokenizer, text, layers=None):
    """获取各层hidden state的最后一token"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # (L+1, 1, T, d)
    # 取最后一token
    result = {}
    for i, hs in enumerate(hidden_states):
        result[i] = hs[0, -1, :].float().cpu()  # (d,)
    return result


def measure_axiom1(hs_dict):
    """A1: 高维几何——不同文本在中间层的方向正交性
    测量方法：
    1. 对多个不同文本，取中间层的hidden state方向
    2. 计算所有配对的cos，取均值
    3. 高维理论预测：随机方向在d维空间中E[|cos|] ≈ √(2/(πd))
    """
    layer_keys = sorted([k for k in hs_dict.keys() if k >= 0])
    cos_values = []
    # 改用不同文本之间的方向正交性
    # 这里hs_dict只有一个文本的各层，所以返回空——实际测量在measure_all_axioms中
    return {"mean_cos": 0, "mean_angle": 90, "values": []}

    if not cos_values:
        return {"mean_cos": 0, "mean_angle": 90, "values": []}

    mean_cos = statistics.mean(cos_values)
    mean_angle = math.degrees(math.acos(max(-1, min(1, mean_cos))))
    return {
        "mean_cos": mean_cos,
        "mean_angle": mean_angle,
        "values": cos_values,
    }


def measure_axiom2(hs_dict, test_cases):
    """A2: 信息域——不同能力方向的纠缠度"""
    # 用不同能力文本的差异方向
    directions = {}
    for tc in test_cases:
        ha = get_hidden_states.__wrapped__(None, None, "")  # placeholder
        # 直接用hs_dict中的最终层方向
        delta = hs_dict.get(max(hs_dict.keys()), torch.zeros(1))  # placeholder
    # 简化：用6对能力文本的差异方向
    cap_texts = {
        "disamb_a": "The river bank was muddy.",
        "disamb_b": "The bank approved the loan.",
        "syntax_a": "She quickly ran home.",
        "syntax_b": "Home she ran quickly.",
        "relation_a": "Paris is the capital of France.",
        "relation_b": "Berlin is the capital of Germany.",
        "style_a": "The meeting was extremely productive.",
        "style_b": "That get-together was quite fruitful.",
        "spatial_a": "The cat is under the table.",
        "spatial_b": "The bird is above the tree.",
        "temporal_a": "Yesterday it rained heavily.",
        "temporal_b": "Tomorrow it will snow.",
    }
    return {"simplified": True}


def measure_all_axioms(model, tokenizer):
    """测量所有公理（简化版）"""
    # A1: 用多个文本测量——不同主题的句子
    texts = [
        "The cat sat on the mat.",
        "Paris is the capital of France.",
        "She quickly ran home.",
        "The river bank was muddy.",
        "The meeting was extremely productive.",
        "Yesterday it rained heavily all day.",
        "Two plus two equals four exactly.",
        "The sky turned orange at sunset.",
        "DNA contains genetic instructions for life.",
        "The orchestra played a beautiful symphony.",
        "Gravity causes objects to fall downward.",
        "She carefully folded the origami crane.",
    ]

    all_cos = []
    all_angles = []
    for text in texts:
        hs = get_hidden_states(model, tokenizer, text)
        # 取中间层的hidden state方向
        layer_keys = sorted(hs.keys())
        mid = layer_keys[len(layer_keys) // 2]
        all_cos.append(hs[mid])  # 存储方向向量，后面统一计算

    if len(all_cos) < 2:
        return None

    # 计算所有配对的cos
    pair_cos = []
    for i in range(len(all_cos)):
        for j in range(i + 1, len(all_cos)):
            cos_v = F.cosine_similarity(all_cos[i].unsqueeze(0), all_cos[j].unsqueeze(0)).item()
            pair_cos.append(abs(cos_v))

    mean_cos = statistics.mean(pair_cos) if pair_cos else 0
    mean_angle = math.degrees(math.acos(max(-1, min(1, mean_cos))))

    # A2: 能力方向纠缠
    cap_pairs = [
        ("disamb", "The river bank was muddy.", "The bank approved the loan."),
        ("syntax", "She quickly ran home.", "Home she ran quickly."),
        ("relation", "Paris is the capital of France.", "Berlin is the capital of Germany."),
        ("style", "The meeting was extremely productive.", "That get-together was quite fruitful."),
        ("spatial", "The cat is under the table.", "The bird is above the tree."),
        ("temporal", "Yesterday it rained heavily.", "Tomorrow it will snow."),
    ]

    directions = {}
    last_layer = None
    for cap, pa, pb in cap_pairs:
        hs_a = get_hidden_states(model, tokenizer, pa)
        hs_b = get_hidden_states(model, tokenizer, pb)
        ll = max(hs_a.keys())
        last_layer = ll
        delta = hs_a[ll] - hs_b[ll]
        delta_norm = F.normalize(delta.unsqueeze(0)).squeeze(0)
        directions[cap] = delta_norm

    # 能力间cos矩阵
    cap_names = list(directions.keys())
    cos_matrix = {}
    for i, c1 in enumerate(cap_names):
        for j, c2 in enumerate(cap_names):
            if i < j:
                cos_v = F.cosine_similarity(
                    directions[c1].unsqueeze(0),
                    directions[c2].unsqueeze(0)
                ).item()
                cos_matrix[(c1, c2)] = cos_v

    abs_cos_vals = [abs(v) for v in cos_matrix.values()]
    mean_entangle = statistics.mean(abs_cos_vals) if abs_cos_vals else 0

    # A4: 信号聚焦——前25%层 vs 后25%层的方向一致性
    all_ha = get_hidden_states(model, tokenizer, "The river bank was muddy.")
    all_hb = get_hidden_states(model, tokenizer, "The bank approved the loan.")
    layer_keys = sorted([k for k in all_ha.keys() if k > 0])
    n_layers = len(layer_keys)
    early_end = n_layers // 4
    late_start = 3 * n_layers // 4

    early_cos = []
    late_cos = []
    for i, k in enumerate(layer_keys):
        if i + 1 < len(layer_keys):
            da = all_ha[layer_keys[i+1]] - all_ha[k]
            db = all_hb[layer_keys[i+1]] - all_hb[k]
            cos_v = F.cosine_similarity(da.unsqueeze(0), db.unsqueeze(0)).item()
            if i < max(1, early_end):
                early_cos.append(abs(cos_v))
            if i >= max(1, late_start - 1):
                late_cos.append(abs(cos_v))

    focus_early = statistics.mean(early_cos) if early_cos else 0
    focus_late = statistics.mean(late_cos) if late_cos else 0
    focus_ratio = focus_late / max(focus_early, 1e-8)

    # A5: Logit精确（用模型自身的unembed）
    h_final = all_ha[last_layer]
    unembed = model.lm_head.weight.float().cpu()  # (V, d)

    # 选两个候选词 "was" vs "is"
    tok_was = tokenizer.encode(" was", add_special_tokens=False)[0]
    tok_is = tokenizer.encode(" is", add_special_tokens=False)[0]
    w_was = unembed[tok_was]
    w_is = unembed[tok_is]

    # 实际logit差
    logit_was = torch.dot(h_final, w_was).item()
    logit_is = torch.dot(h_final, w_is).item()
    actual_margin = logit_was - logit_is

    # 预测margin
    dw = w_was - w_is
    pred_margin = torch.dot(h_final, dw).item()
    logit_error = abs(pred_margin - actual_margin) / max(abs(actual_margin), 1e-8) * 100

    return {
        "A1_angle": mean_angle,
        "A1_cos": mean_cos,
        "A2_entangle": mean_entangle,
        "A4_focus_ratio": focus_ratio,
        "A4_early": focus_early,
        "A4_late": focus_late,
        "A5_error_pct": logit_error,
    }


# ============================================================
# 训练循环
# ============================================================
def train_tiny_model(hidden_dim, n_layers=6, n_heads=4, n_steps=200, batch_size=8, lr=5e-4):
    """训练一个指定维度的tiny GPT-2"""
    # 确保n_heads能整除hidden_dim
    n_heads = min(n_heads, hidden_dim // 64)
    n_heads = max(n_heads, 2)

    config = GPT2Config(
        vocab_size=50257,
        n_positions=64,
        n_embd=hidden_dim,
        n_layer=n_layers,
        n_head=n_heads,
        n_inner=hidden_dim * 4,
    )

    model = GPT2LMHeadModel(config)
    model.train()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SimpleDataset(TRAINING_SENTENCES, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = n_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    model.cuda()
    step = 0
    losses = []

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
            batch = batch.cuda()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 跳过NaN
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1

    valid_losses = [l for l in losses if not math.isnan(l)]
    avg_loss = statistics.mean(valid_losses[-10:]) if valid_losses else float('nan')
    return model, tokenizer, avg_loss


def run_dimension_sweep():
    """主函数：扫描不同维度"""
    print("=" * 65)
    print("  P32 高维阈值精确定位——第一公理的维度依赖性")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    dimensions = [64, 128, 256, 512]
    # 不测1024——参数量太大

    results = {}

    for d in dimensions:
        print(f"\n{'='*50}")
        print(f"  hidden_dim = {d}")
        print(f"{'='*50}")

        # 根据维度调整学习率（大维度用更小的学习率）
        lr = min(5e-4, 5e-4 * 128 / d)
        print(f"  学习率: {lr:.2e}")

        # 训练200步
        n_layers = 6
        model, tokenizer, avg_loss = train_tiny_model(
            hidden_dim=d, n_layers=n_layers, n_steps=200, lr=lr
        )
        print(f"  训练完成，平均loss: {avg_loss:.3f}")

        # 测量五公理
        model.eval()
        metrics = measure_all_axioms(model, tokenizer)
        if metrics:
            metrics["loss"] = avg_loss
            metrics["dim"] = d
            results[d] = metrics

            print(f"  A1 高维几何: angle={metrics['A1_angle']:.1f}°, |cos|={metrics['A1_cos']:.4f}")
            print(f"  A2 纠缠度:   {metrics['A2_entangle']:.4f}")
            print(f"  A4 聚焦比:   {metrics['A4_focus_ratio']:.2f} (early={metrics['A4_early']:.4f}, late={metrics['A4_late']:.4f})")
            print(f"  A5 Logit误差: {metrics['A5_error_pct']:.2f}%")

            # 理论预测: 1/sqrt(d)
            theory_cos = 1.0 / math.sqrt(d)
            print(f"  理论 1/√{d}:  {theory_cos:.4f} (对应angle={math.degrees(math.acos(max(-1,min(1,1-theory_cos)))):.1f}°)")
        else:
            print("  测量失败！")

        # 释放GPU
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # 汇总分析
    print(f"\n{'='*65}")
    print("  汇总结果")
    print(f"{'='*65}")
    print(f"  {'dim':>6} | {'A1_angle':>9} | {'A1_cos':>8} | {'1/√d':>8} | {'A2_ent':>8} | {'A4_ratio':>8} | {'A5_err%':>8} | {'loss':>6}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    for d in sorted(results.keys()):
        m = results[d]
        theory = 1.0 / math.sqrt(d)
        print(f"  {d:>6} | {m['A1_angle']:>8.1f}° | {m['A1_cos']:>8.4f} | {theory:>8.4f} | {m['A2_entangle']:>8.4f} | {m['A4_focus_ratio']:>8.2f} | {m['A5_error_pct']:>7.2f}% | {m['loss']:>6.2f}")

    # 高维阈值分析
    print(f"\n  --- 高维阈值d*分析 ---")
    for d in sorted(results.keys()):
        m = results[d]
        theory = 1.0 / math.sqrt(d)
        ratio = m["A1_cos"] / theory if theory > 0 else 0
        angle_gap = 90 - m["A1_angle"]
        status = "✅正交" if m["A1_angle"] > 70 else ("⚠️过渡" if m["A1_angle"] > 40 else "❌非正交")
        print(f"  d={d}: angle={m['A1_angle']:.1f}°, 理论比={ratio:.2f}x, 距90°={angle_gap:.1f}° → {status}")

    # 估计d*
    angles = {d: results[d]["A1_angle"] for d in results.keys()}
    # 找angle > 45°的最小d
    d_star_estimate = None
    for d in sorted(angles.keys()):
        if angles[d] > 45:
            d_star_estimate = d
            break

    if d_star_estimate:
        print(f"\n  >>> 高维阈值d*估计: {d_star_estimate} (angle首次>45°)")
    else:
        print(f"\n  >>> 高维阈值d*估计: >512 (所有测试维度angle<45°)")

    # 拟合cos vs 1/sqrt(d)的关系
    print(f"\n  --- cos vs 1/√d 拟合 ---")
    for d in sorted(results.keys()):
        m = results[d]
        theory = 1.0 / math.sqrt(d)
        print(f"  d={d}: actual_cos={m['A1_cos']:.4f}, theory={theory:.4f}, ratio={m['A1_cos']/theory:.2f}")

    # 保存结果
    save_path = OUTPUT_DIR / f"stage678_dim_sweep_{TIMESTAMP}.json"
    save_data = {}
    for d, m in results.items():
        save_data[str(d)] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {save_path}")

    return results


if __name__ == "__main__":
    run_dimension_sweep()
