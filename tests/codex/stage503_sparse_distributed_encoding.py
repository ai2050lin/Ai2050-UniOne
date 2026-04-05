#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage503: 激活稀疏度与分布式编码的正面数学描述

目标：找到"分布式编码"的正面数学描述，而不仅仅是"不是X"。

分析方法：
  A1: 层间稀疏度演化——计算每层的激活稀疏度(基于Top-K占比)
  A2: 维度利用率——每个维度被"使用"的比例(非零激活维度占比)
  A3: 信息瓶颈分析——层间互信息估计（通过秩和有效维度）
  A4: 概念编码的维度重叠——不同概念共享多少活跃维度
  A5: 全局vs局部编码比例——一个概念的信息分散在多少维度上
"""

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodel_language_shared import (
    discover_layers,
    evenly_spaced_layers,
    free_model,
    get_model_device,
    load_model_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = PROJECT_ROOT / "tests" / "codex_temp"


def get_hidden_at_layers(model, tokenizer, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """获取文本在指定层的最后一个token隐藏状态"""
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    result = {}
    for li in layer_indices:
        result[li] = outputs.hidden_states[li][0, -1].float().cpu()  # [hidden_dim]
    return result


# ============================================================
# A1: 层间稀疏度演化
# ============================================================
def run_a1_sparsity_evolution(model, tokenizer, layer_indices: List[int]) -> Dict:
    """计算每层的激活稀疏度——Top-1%, Top-5%, Top-10%贡献了多少能量"""
    test_texts = [
        "猫是一种动物",
        "天空是蓝色的",
        "学习使人进步",
        "数学是科学的基础",
        "音乐能治愈心灵",
        "历史记录了过去",
        "地球绕太阳旋转",
        "水是生命之源",
    ]

    sparsity_by_layer = {li: [] for li in layer_indices}

    for text in test_texts:
        hidden_map = get_hidden_at_layers(model, tokenizer, text, layer_indices)
        for li, h in hidden_map.items():
            abs_h = h.abs()
            total = abs_h.sum().item()
            if total < 1e-8:
                continue
            sorted_vals, _ = torch.sort(abs_h, descending=True)
            n = len(sorted_vals)
            top1_pct = sorted_vals[:max(1, n // 100)].sum().item() / total
            top5_pct = sorted_vals[:max(1, n // 20)].sum().item() / total
            top10_pct = sorted_vals[:max(1, n // 10)].sum().item() / total
            sparsity_by_layer[li].append({
                "top1pct": top1_pct,
                "top5pct": top5_pct,
                "top10pct": top10_pct,
            })

    avg_by_layer = {}
    for li in layer_indices:
        vals = sparsity_by_layer[li]
        if vals:
            avg_by_layer[f"L{li}"] = {
                "top1pct": sum(v["top1pct"] for v in vals) / len(vals),
                "top5pct": sum(v["top5pct"] for v in vals) / len(vals),
                "top10pct": sum(v["top10pct"] for v in vals) / len(vals),
            }

    return avg_by_layer


# ============================================================
# A2: 维度利用率
# ============================================================
def run_a2_dimension_utilization(model, tokenizer, layer_indices: List[int]) -> Dict:
    """非零激活维度占比（使用阈值=最大值*0.01）"""
    test_texts = [
        "猫是动物", "天空很蓝", "学习进步",
        "数学基础", "音乐治愈", "历史过去",
    ]

    utilization_by_layer = {li: [] for li in layer_indices}

    for text in test_texts:
        hidden_map = get_hidden_at_layers(model, tokenizer, text, layer_indices)
        for li, h in hidden_map.items():
            threshold = h.abs().max().item() * 0.01
            active = (h.abs() > threshold).sum().item()
            total = h.numel()
            utilization_by_layer[li].append(active / total)

    avg_by_layer = {}
    for li in layer_indices:
        vals = utilization_by_layer[li]
        if vals:
            avg_by_layer[f"L{li}"] = sum(vals) / len(vals)

    return avg_by_layer


# ============================================================
# A3: 信息瓶颈——有效秩
# ============================================================
def run_a3_effective_rank(model, tokenizer, layer_indices: List[int]) -> Dict:
    """用奇异值分布估计每层的有效秩"""
    test_texts = [
        "猫", "狗", "鸟", "鱼", "红", "蓝", "大", "小",
        "快乐", "悲伤", "城市", "乡村", "春天", "冬天",
    ]

    rank_by_layer = {li: [] for li in layer_indices}
    hidden_dim = None

    for text in test_texts:
        hidden_map = get_hidden_at_layers(model, tokenizer, text, layer_indices)
        for li, h in hidden_map.items():
            if hidden_dim is None:
                hidden_dim = h.numel()
            # SVD
            sv = torch.linalg.svdvals(h.unsqueeze(0))  # [1, hidden_dim]
            sv = sv.float()
            total_sv = sv.sum().item()
            if total_sv < 1e-8:
                continue
            # 有效秩 = (sum(sv))^2 / sum(sv^2)
            effective_rank = (sv.sum().item() ** 2) / (sv ** 2).sum().item()
            rank_by_layer[li].append(min(effective_rank, h.numel()))

    avg_by_layer = {}
    for li in layer_indices:
        vals = rank_by_layer[li]
        if vals:
            avg_by_layer[f"L{li}"] = {
                "effective_rank": sum(vals) / len(vals),
                "rank_ratio": (sum(vals) / len(vals)) / hidden_dim if hidden_dim else 0,
            }

    return avg_by_layer


# ============================================================
# A4: 概念编码的维度重叠
# ============================================================
def run_a4_concept_dimension_overlap(model, tokenizer, layer_indices: List[int]) -> Dict:
    """不同概念在同一层共享多少活跃维度"""
    concepts = {
        "animals": ["猫", "狗", "鸟", "鱼"],
        "colors": ["红", "蓝", "绿", "黄"],
        "actions": ["跑", "走", "飞", "游"],
    }

    overlap_results = {}

    for li in layer_indices:
        # 收集所有概念的活跃维度
        concept_active_dims = {}
        for cat, words in concepts.items():
            all_active = set()
            for w in words:
                hidden_map = get_hidden_at_layers(model, tokenizer, w, [li])
                h = hidden_map[li]
                threshold = h.abs().max().item() * 0.05
                active_idx = set((h.abs() > threshold).nonzero(as_tuple=True)[0].tolist())
                all_active.update(active_idx)
            concept_active_dims[cat] = all_active

        # 计算重叠
        categories = list(concepts.keys())
        overlaps = {}
        for i, c1 in enumerate(categories):
            for j, c2 in enumerate(categories):
                if i < j:
                    s1 = concept_active_dims[c1]
                    s2 = concept_active_dims[c2]
                    if s1 and s2:
                        jaccard = len(s1 & s2) / len(s1 | s2)
                        overlap_pct = len(s1 & s2) / max(len(s1), len(s2))
                    else:
                        jaccard = 0
                        overlap_pct = 0
                    overlaps[f"{c1}_vs_{c2}"] = {
                        "jaccard": jaccard,
                        "overlap_pct": overlap_pct,
                        "size_c1": len(s1),
                        "size_c2": len(s2),
                        "intersection": len(s1 & s2),
                    }

        overlap_results[f"L{li}"] = overlaps

    return overlap_results


# ============================================================
# A5: 全局vs局部编码比例
# ============================================================
def run_a5_global_local_ratio(model, tokenizer, layer_indices: List[int]) -> Dict:
    """一个概念的信息分散在多少维度上——集中度分析"""
    test_concepts = ["猫", "苹果", "快乐", "城市", "春天"]

    concentration_by_layer = {li: [] for li in layer_indices}

    for concept in test_concepts:
        hidden_map = get_hidden_at_layers(model, tokenizer, concept, layer_indices)
        for li, h in hidden_map.items():
            abs_h = h.abs()
            sorted_vals, _ = torch.sort(abs_h, descending=True)
            n = len(sorted_vals)
            # 信息集中度：前10%维度占总能量的比例
            top10_energy = sorted_vals[:max(1, n // 10)].sum().item()
            total_energy = sorted_vals.sum().item()
            concentration = top10_energy / total_energy if total_energy > 0 else 0
            # 也计算：需要多少维度才能包含90%的能量
            cumsum = torch.cumsum(sorted_vals, dim=0)
            threshold_90 = total_energy * 0.9
            dims_for_90 = (cumsum >= threshold_90).nonzero(as_tuple=True)[0]
            nd90 = int(dims_for_90[0].item()) + 1 if len(dims_for_90) > 0 else n
            dims_ratio = nd90 / n

            concentration_by_layer[li].append({
                "top10_concentration": concentration,
                "dims_for_90pct": nd90,
                "dims_ratio": dims_ratio,
            })

    avg_by_layer = {}
    for li in layer_indices:
        vals = concentration_by_layer[li]
        if vals:
            avg_by_layer[f"L{li}"] = {
                "top10_concentration": sum(v["top10_concentration"] for v in vals) / len(vals),
                "dims_for_90pct": int(sum(v["dims_for_90pct"] for v in vals) / len(vals)),
                "dims_ratio": sum(v["dims_ratio"] for v in vals) / len(vals),
            }

    return avg_by_layer


# ============================================================
# 主函数
# ============================================================
def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"stage503_sparse_distributed_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage503] 模型: {model_key}")
    print(f"[Stage503] 输出目录: {out_dir}")

    model, tokenizer = load_model_bundle(model_key)
    layers = discover_layers(model)
    n_layers = len(layers)
    layer_indices = evenly_spaced_layers(model, count=8)
    print(f"[Stage503] 层数: {n_layers}, 采样层: {layer_indices}")

    try:
        # A1
        print("\n=== A1: 层间稀疏度演化 ===")
        a1 = run_a1_sparsity_evolution(model, tokenizer, layer_indices)
        for li_key in sorted(a1.keys()):
            v = a1[li_key]
            print(f"  {li_key}: Top1%={v['top1pct']:.4f} Top5%={v['top5pct']:.4f} Top10%={v['top10pct']:.4f}")

        # A2
        print("\n=== A2: 维度利用率 ===")
        a2 = run_a2_dimension_utilization(model, tokenizer, layer_indices)
        for li_key in sorted(a2.keys()):
            print(f"  {li_key}: {a2[li_key]:.4f}")

        # A3
        print("\n=== A3: 有效秩 ===")
        a3 = run_a3_effective_rank(model, tokenizer, layer_indices)
        for li_key in sorted(a3.keys()):
            v = a3[li_key]
            print(f"  {li_key}: effective_rank={v['effective_rank']:.1f} ratio={v['rank_ratio']:.4f}")

        # A4
        print("\n=== A4: 概念维度重叠 ===")
        a4 = run_a4_concept_dimension_overlap(model, tokenizer, layer_indices)
        for li_key in sorted(a4.keys()):
            print(f"  {li_key}:")
            for pair, info in a4[li_key].items():
                print(f"    {pair}: jaccard={info['jaccard']:.4f} overlap={info['overlap_pct']:.4f}")

        # A5
        print("\n=== A5: 全局vs局部编码 ===")
        a5 = run_a5_global_local_ratio(model, tokenizer, layer_indices)
        for li_key in sorted(a5.keys()):
            v = a5[li_key]
            print(f"  {li_key}: top10_conc={v['top10_concentration']:.4f} dims_90pct={v['dims_for_90pct']} dims_ratio={v['dims_ratio']:.4f}")

        # 综合描述
        print("\n=== 编码机制的正面数学描述 ===")
        # 取中间层的指标作为代表
        mid_key = f"L{layer_indices[len(layer_indices) // 2]}"
        description = {
            "sparsity": a1.get(mid_key, {}),
            "dimension_utilization": a2.get(mid_key, 0),
            "effective_rank_ratio": a3.get(mid_key, {}).get("rank_ratio", 0),
            "global_encoding_ratio": a5.get(mid_key, {}).get("dims_ratio", 0),
        }
        encoding_type = "highly_local" if description["global_encoding_ratio"] < 0.2 else \
                        "moderately_distributed" if description["global_encoding_ratio"] < 0.6 else \
                        "highly_distributed"

        print(f"  中间层({mid_key})编码类型: {encoding_type}")
        print(f"  稀疏度(Top10%能量占比): {description['sparsity'].get('top10pct', 0):.4f}")
        print(f"  维度利用率: {description['dimension_utilization']:.4f}")
        print(f"  有效秩比例: {description['effective_rank_ratio']:.4f}")
        print(f"  全局编码比例(90%能量需要的维度比例): {description['global_encoding_ratio']:.4f}")

        summary = {
            "model": model_key,
            "n_layers": n_layers,
            "timestamp": timestamp,
            "a1_sparsity": a1,
            "a2_utilization": a2,
            "a3_effective_rank": a3,
            "a4_overlap": a4,
            "a5_global_local": a5,
            "encoding_description": description,
            "encoding_type": encoding_type,
        }
        (out_dir / f"summary_{model_key}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )
        print(f"\n[Stage503] 结果已保存到 {out_dir}")

    finally:
        free_model(model)


if __name__ == "__main__":
    main()
