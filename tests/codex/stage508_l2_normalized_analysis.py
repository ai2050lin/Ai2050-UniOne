#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage508: L2距离 + 归一化残差流分析
核心问题：cos=1.0是norm爆炸的数值假象，用L2距离和归一化后重测量真正概念差异

分析维度：
- D1: L2距离矩阵 vs 余弦相似度矩阵（单token + 句子上下文）
- D2: 归一化残差流后逐层余弦相似度（消除norm影响）
- D3: 残差流norm逐层追踪（诊断爆炸层）
- D4: 层间信息流——相邻层L2距离的传播/衰减
- D5: 归一化后的概念层级/翻译信号逐层演化
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_glm4_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device,
)


# ============================================================
# 数据定义
# ============================================================
CONCEPT_HIERARCHY = {
    "具体": ["苹果", "猫", "北京"],
    "中类": ["水果", "动物", "城市"],
    "大类": ["食物", "生物", "地点"],
}

TRANSLATION_PAIRS = [
    ("apple", "苹果", "食物"), ("cat", "猫", "动物"), ("city", "城市", "地点"),
    ("water", "水", "自然"), ("book", "书", "物品"), ("sun", "太阳", "自然"),
]

UNRELATED_PAIRS = [
    ("apple", "猫"), ("cat", "城市"), ("water", "书"),
]


def get_sample_layers(model) -> List[int]:
    """均匀采样12层"""
    n = len(discover_layers(model))
    if n <= 12:
        return list(range(n))
    return [mapped_idx(n, i) for i in range(12)]


def mapped_idx(n_layers: int, gpt2_idx: int) -> int:
    if n_layers <= 1:
        return 0
    return max(0, min(n_layers - 1, round((n_layers - 1) * gpt2_idx / 11)))


def get_hidden_at_last_token(model, tokenizer, text: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """获取最后一token在指定层的hidden states"""
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    result = {}
    for i, li in enumerate(layer_indices):
        h = outputs.hidden_states[li + 1][:, -1, :]  # +1: index 0 is embedding
        result[li] = h.float().cpu()
    return result


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b, p=2).item()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dim() == 1:
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    return F.cosine_similarity(a, b, dim=-1).item()


def normalize(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)


# ============================================================
# D1: L2距离矩阵 vs 余弦相似度矩阵
# ============================================================
def analyze_D1_distance_vs_cosine(model, tokenizer, layer_indices):
    """D1: 对比L2距离和余弦相似度，在单token和句子上下文下"""
    all_words = CONCEPT_HIERARCHY["具体"] + CONCEPT_HIERARCHY["中类"] + CONCEPT_HIERARCHY["大类"]
    templates = {
        "single": None,
        "sentence": "这是一个关于{w}的句子。",
    }

    results = {}
    for mode, tmpl in templates.items():
        # 收集hidden states
        hs_map = {}
        for w in all_words:
            text = w if tmpl is None else tmpl.format(w=w)
            hs_map[w] = get_hidden_at_last_token(model, tokenizer, text, layer_indices)

        for li in layer_indices:
            l2_matrix = {}
            cos_matrix = {}
            for w1 in all_words:
                for w2 in all_words:
                    key = f"{w1}-{w2}"
                    l2_matrix[key] = round(l2_distance(hs_map[w1][li], hs_map[w2][li]), 4)
                    cos_matrix[key] = round(cos_sim(hs_map[w1][li], hs_map[w2][li]), 4)

            # 计算层级分离度
            same_level_l2, diff_level_l2 = [], []
            same_level_cos, diff_level_cos = [], []
            for w1 in all_words:
                for w2 in all_words:
                    if w1 >= w2:
                        continue
                    # 判断是否同层级
                    l1 = None
                    l2 = None
                    for level, words in CONCEPT_HIERARCHY.items():
                        if w1 in words:
                            l1 = level
                        if w2 in words:
                            l2 = level
                    if l1 == l2:
                        same_level_l2.append(l2_matrix[f"{w1}-{w2}"])
                        same_level_cos.append(cos_matrix[f"{w1}-{w2}"])
                    else:
                        diff_level_l2.append(l2_matrix[f"{w1}-{w2}"])
                        diff_level_cos.append(cos_matrix[f"{w1}-{w2}"])

            avg_same_l2 = float(np.mean(same_level_l2)) if same_level_l2 else 0
            avg_diff_l2 = float(np.mean(diff_level_l2)) if diff_level_l2 else 0
            avg_same_cos = float(np.mean(same_level_cos)) if same_level_cos else 0
            avg_diff_cos = float(np.mean(diff_level_cos)) if diff_level_cos else 0

            results[f"{mode}_L{li}"] = {
                "avg_same_level_l2": round(avg_same_l2, 4),
                "avg_diff_level_l2": round(avg_diff_l2, 4),
                "l2_separation": round(avg_diff_l2 - avg_same_l2, 4),  # 正=好的层级分离
                "avg_same_level_cos": round(avg_same_cos, 4),
                "avg_diff_level_cos": round(avg_diff_cos, 4),
                "cos_separation": round(avg_diff_cos - avg_same_cos, 4),
            }

    # 找最佳层（用L2分离度）
    l2_keys = [k for k in results if k.startswith("single_L") and "separation" in str(results[k])]
    best_l2_key = max(l2_keys, key=lambda k: results[k]["l2_separation"]) if l2_keys else None
    if best_l2_key:
        results["_best_l2_layer"] = {best_l2_key: results[best_l2_key]}

    cos_keys = [k for k in results if k.startswith("single_L") and "separation" in str(results[k])]
    best_cos_key = max(cos_keys, key=lambda k: results[k]["cos_separation"]) if cos_keys else None
    if best_cos_key:
        results["_best_cos_layer"] = {best_cos_key: results[best_cos_key]}

    return {"D1_distance_vs_cosine": results}


# ============================================================
# D2: 归一化残差流后余弦相似度
# ============================================================
def analyze_D2_normalized_cosine(model, tokenizer, layer_indices):
    """D2: 对每层hidden state做L2归一化后再算余弦——消除norm影响"""
    all_words = CONCEPT_HIERARCHY["具体"] + CONCEPT_HIERARCHY["中类"] + CONCEPT_HIERARCHY["大类"]

    # 单token + 句子上下文
    modes = {
        "single": lambda w: w,
        "sentence": lambda w: f"这是一个关于{w}的句子。",
    }

    results = {}
    for mode_name, text_fn in modes.items():
        # 收集并归一化
        norm_hs_map = {}
        raw_norms = {}
        for w in all_words:
            hs = get_hidden_at_last_token(model, tokenizer, text_fn(w), layer_indices)
            norm_hs_map[w] = {}
            raw_norms[w] = {}
            for li in layer_indices:
                h = hs[li]
                raw_norms[w][li] = round(torch.norm(h, p=2).item(), 4)
                norm_hs_map[w][li] = normalize(h)

        for li in layer_indices:
            # 归一化后的层级分离
            same_cos, diff_cos = [], []
            for w1 in all_words:
                for w2 in all_words:
                    if w1 >= w2:
                        continue
                    l1 = next((lv for lv, ws in CONCEPT_HIERARCHY.items() if w1 in ws), None)
                    l2 = next((lv for lv, ws in CONCEPT_HIERARCHY.items() if w2 in ws), None)
                    c = cos_sim(norm_hs_map[w1][li], norm_hs_map[w2][li])
                    if l1 == l2:
                        same_cos.append(c)
                    else:
                        diff_cos.append(c)

            avg_norms = float(np.mean([raw_norms[w][li] for w in all_words]))

            results[f"{mode_name}_L{li}"] = {
                "norm_avg": round(avg_norms, 2),
                "same_level_cos": round(float(np.mean(same_cos)), 4) if same_cos else 0,
                "diff_level_cos": round(float(np.mean(diff_cos)), 4) if diff_cos else 0,
                "hierarchy_signal": round(
                    float(np.mean(diff_cos)) - float(np.mean(same_cos)), 4
                ) if same_cos and diff_cos else 0,
            }

    return {"D2_normalized_cosine": results}


# ============================================================
# D3: 残差流norm逐层追踪
# ============================================================
def analyze_D3_norm_trajectory(model, tokenizer, layer_indices):
    """D3: 残差流norm、方差、方向变化逐层追踪"""
    words = CONCEPT_HIERARCHY["具体"] + CONCEPT_HIERARCHY["中类"]

    results = {}
    for w in words:
        hs = get_hidden_at_last_token(model, tokenizer, w, layer_indices)
        results[w] = {}
        for li in layer_indices:
            h = hs[li]
            results[w][f"L{li}"] = {
                "norm": round(torch.norm(h, p=2).item(), 4),
                "std": round(h.std().item(), 4),
                "mean": round(h.mean().item(), 4),
                "max": round(h.max().item(), 4),
                "min": round(h.min().item(), 4),
            }

    # 跨词统计
    cross_word = {}
    for li in layer_indices:
        norms = [results[w][f"L{li}"]["norm"] for w in words]
        stds = [results[w][f"L{li}"]["std"] for w in words]
        cross_word[f"L{li}"] = {
            "norm_mean": round(float(np.mean(norms)), 2),
            "norm_std": round(float(np.std(norms)), 4),
            "norm_range": round(float(np.max(norms) - np.min(norms)), 4),
            "std_mean": round(float(np.mean(stds)), 4),
            "std_range": round(float(np.max(stds) - np.min(stds)), 4),
        }
    results["_cross_word"] = cross_word

    return {"D3_norm_trajectory": results}


# ============================================================
# D4: 层间信息流——L2距离传播
# ============================================================
def analyze_D4_inter_layer_flow(model, tokenizer, layer_indices):
    """D4: 追踪概念对之间的L2距离如何在层间传播/衰减"""
    pairs = [
        ("苹果", "水果", "同层级内"),
        ("苹果", "猫", "跨层级远"),
        ("apple", "苹果", "翻译对"),
        ("猫", "狗", "同层级(补充)"),
    ]

    results = {}
    for w1, w2, rel in pairs:
        hs1 = get_hidden_at_last_token(model, tokenizer, w1, layer_indices)
        hs2 = get_hidden_at_last_token(model, tokenizer, w2, layer_indices)

        l2_raw = []
        l2_norm = []
        cos_vals = []
        for li in layer_indices:
            l2_raw.append(l2_distance(hs1[li], hs2[li]))
            l2_norm.append(l2_distance(normalize(hs1[li]), normalize(hs2[li])))
            cos_vals.append(cos_sim(hs1[li], hs2[li]))

        # 计算变化趋势
        l2_change = l2_raw[-1] - l2_raw[0] if len(l2_raw) > 1 else 0
        cos_change = cos_vals[-1] - cos_vals[0] if len(cos_vals) > 1 else 0

        results[f"{w1}-{w2}"] = {
            "relation": rel,
            "l2_first": round(l2_raw[0], 4),
            "l2_last": round(l2_raw[-1], 4),
            "l2_change": round(l2_change, 4),
            "l2_normalized_first": round(l2_norm[0], 4),
            "l2_normalized_last": round(l2_norm[-1], 4),
            "cos_first": round(cos_vals[0], 4),
            "cos_last": round(cos_vals[-1], 4),
            "cos_change": round(cos_change, 4),
        }

    return {"D4_inter_layer_flow": results}


# ============================================================
# D5: 归一化后的翻译/层级信号逐层演化
# ============================================================
def analyze_D5_normalized_signal_evolution(model, tokenizer, layer_indices):
    """D5: 概念层级、翻译对齐、无关——三种关系在归一化后的逐层对比"""
    # 归一化后计算三类关系
    categories = {
        "translation": [("apple", "苹果"), ("cat", "猫"), ("water", "水")],
        "hierarchy": [("苹果", "水果"), ("猫", "动物"), ("城市", "地点")],
        "unrelated": [("apple", "猫"), ("water", "城市"), ("book", "猫")],
    }

    results = {}
    for li in layer_indices:
        cat_scores = {}
        for cat, pairs in categories.items():
            scores = []
            for w1, w2 in pairs:
                hs1 = get_hidden_at_last_token(model, tokenizer, w1, layer_indices)
                hs2 = get_hidden_at_last_token(model, tokenizer, w2, layer_indices)
                scores.append(cos_sim(normalize(hs1[li]), normalize(hs2[li])))
            cat_scores[cat] = round(float(np.mean(scores)), 4)

        cat_scores["translation_vs_unrelated"] = round(
            cat_scores["translation"] - cat_scores["unrelated"], 4
        )
        cat_scores["hierarchy_vs_unrelated"] = round(
            cat_scores["hierarchy"] - cat_scores["unrelated"], 4
        )
        results[f"L{li}"] = cat_scores

    return {"D5_normalized_signal_evolution": results}


# ============================================================
# 主函数
# ============================================================
def load_model(model_name: str):
    if model_name == "qwen3":
        return load_qwen3_model()
    elif model_name == "glm4":
        return load_glm4_model()
    elif model_name == "deepseek7b":
        return load_deepseek7b_model()
    elif model_name == "gemma4":
        return load_gemma4_model()
    else:
        raise ValueError(f"未知模型: {model_name}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"[Stage508] L2距离+归一化残差流分析 | 模型: {model_name}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t0 = time.time()

    # 加载模型
    model, tokenizer = load_model(model_name)
    layer_indices = get_sample_layers(model)
    print(f"  层数: {len(discover_layers(model))}, 采样层: {layer_indices}")

    summary = {"model": model_name, "timestamp": datetime.now().isoformat(), "layer_indices": layer_indices}

    # 运行5个分析
    analyses = [
        ("D1", lambda: analyze_D1_distance_vs_cosine(model, tokenizer, layer_indices)),
        ("D2", lambda: analyze_D2_normalized_cosine(model, tokenizer, layer_indices)),
        ("D3", lambda: analyze_D3_norm_trajectory(model, tokenizer, layer_indices)),
        ("D4", lambda: analyze_D4_inter_layer_flow(model, tokenizer, layer_indices)),
        ("D5", lambda: analyze_D5_normalized_signal_evolution(model, tokenizer, layer_indices)),
    ]

    for name, fn in analyses:
        print(f"  运行 {name}...", end=" ", flush=True)
        t1 = time.time()
        result = fn()
        summary.update(result)
        print(f"完成 ({time.time()-t1:.1f}s)")

    # 保存
    out_dir = Path("tests/codex_temp") / f"stage508_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{model_name}.json"

    # 移除D3中的raw向量数据
    d3 = summary.get("D3_norm_trajectory", {})
    for w in list(d3.keys()):
        if w.startswith("_"):
            continue
        for lk in d3[w]:
            pass  # 已经是标量，不需要清理

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n  完成! 耗时 {elapsed:.1f}s")
    print(f"  结果: {out_path}")

    # 打印关键摘要
    print("\n=== 关键发现 ===")

    d1 = summary.get("D1_distance_vs_cosine", {})
    for prefix in ["single", "sentence"]:
        best_l2 = None
        best_l2_val = -999
        for k, v in d1.items():
            if k.startswith(prefix) and "l2_separation" in v:
                if v["l2_separation"] > best_l2_val:
                    best_l2_val = v["l2_separation"]
                    best_l2 = k
        if best_l2:
            v = d1[best_l2]
            print(f"  D1 [{prefix}] 最佳L2分离: {best_l2} "
                  f"(same={v['avg_same_level_l2']:.2f}, diff={v['avg_diff_level_l2']:.2f}, "
                  f"sep={v['l2_separation']:.2f})")

    d2 = summary.get("D2_normalized_cosine", {})
    for prefix in ["single", "sentence"]:
        best_h = None
        best_h_val = -999
        for k, v in d2.items():
            if k.startswith(prefix) and "hierarchy_signal" in v:
                if v["hierarchy_signal"] > best_h_val:
                    best_h_val = v["hierarchy_signal"]
                    best_h = k
        if best_h:
            v = d2[best_h]
            print(f"  D2 [{prefix}] 最佳归一化层级信号: {best_h} "
                  f"(signal={v['hierarchy_signal']:.4f}, norm_avg={v['norm_avg']:.1f})")

    d3 = summary.get("D3_norm_trajectory", {}).get("_cross_word", {})
    if d3:
        # 找norm最大的层和最小的层
        norms = {k: v["norm_mean"] for k, v in d3.items()}
        max_norm_k = max(norms, key=norms.get)
        min_norm_k = min(norms, key=norms.get)
        print(f"  D3 norm范围: {min_norm_k}={norms[min_norm_k]:.1f} -> {max_norm_k}={norms[max_norm_k]:.1f} "
              f"({norms[max_norm_k]/max(norms[min_norm_k],0.001):.0f}x)")

    d5 = summary.get("D5_normalized_signal_evolution", {})
    if d5:
        layers = sorted(d5.keys())
        if layers:
            first = d5[layers[0]]
            last = d5[layers[-1]]
            print(f"  D5 归一化翻译信号: L0={first.get('translation_vs_unrelated','N/A'):.4f} -> "
                  f"深={last.get('translation_vs_unrelated','N/A'):.4f}")


if __name__ == "__main__":
    main()
