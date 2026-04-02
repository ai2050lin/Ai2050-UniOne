# -*- coding: utf-8 -*-
"""
Stage468: 3D空间概念编码分析 — 参考脉冲神经网络视角
=================================================================

核心问题：
  "苹果"这个概念在神经网络中是怎么形成的？
  运行时它的编码是如何被激活和使用的？
  参考大脑脉冲神经网络的3D空间组织方式。

类比映射（SNN → Transformer）：
  - 脉冲神经元的脉冲发放 → 神经元的高激活值（稀疏性）
  - 3D空间拓扑 → 层间连接形成的编码路径
  - 最小传送量原理 → ~18%神经元参与每个概念的编码
  - 层次化编码（特征→形状→类别→概念） → 低层→高层的逐层抽象
  - 局部片区编码 → SVD/PCA的主成分方向
  - 路径叠加编码 → 多个因子方向的非线性组合
  - 突触可塑性 → 权重矩阵的训练形成

实验模块：

  Exp1: 逐层激活演化 — "苹果"在36层中的编码如何演化
  Exp2: 3D坐标映射 — 将激活向量映射到3D空间，观察拓扑结构
  Exp3: 脉冲稀疏性分析 — 每层多少神经元参与编码"苹果"
  Exp4: 编码形成路径 — 从输入token到概念编码的完整路径
  Exp5: 苹果vs其他水果 — 编码相似度与属性的关联

用法：
  python stage468_3d_spike_concept_encoding.py qwen3
  python stage468_3d_spike_concept_encoding.py deepseek
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
    QWEN3_MODEL_PATH,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage468_3d_spike_concept_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集（聚焦水果类别，详细分析苹果） ====================
FRUIT_CONCEPTS = {
    "apple": {"color": "red", "size": 3, "taste": "sweet", "shape": "round"},
    "banana": {"color": "yellow", "size": 3, "taste": "sweet", "shape": "curved"},
    "orange": {"color": "orange", "size": 3, "taste": "sour_sweet", "shape": "round"},
    "grape": {"color": "purple", "size": 2, "taste": "sweet", "shape": "round_small"},
    "mango": {"color": "orange", "size": 3, "taste": "sweet", "shape": "oval"},
    "peach": {"color": "pink", "size": 3, "taste": "sweet", "shape": "round"},
    "lemon": {"color": "yellow", "size": 2, "taste": "sour", "shape": "oval"},
    "cherry": {"color": "red", "size": 1, "taste": "sweet", "shape": "round_small"},
    "watermelon": {"color": "green", "size": 5, "taste": "sweet", "shape": "round_large"},
    "strawberry": {"color": "red", "size": 1, "taste": "sweet", "shape": "heart"},
    "pear": {"color": "green", "size": 3, "taste": "sweet", "shape": "pear"},
    "pineapple": {"color": "yellow", "size": 4, "taste": "sour_sweet", "shape": "oval"},
    "coconut": {"color": "brown", "size": 4, "taste": "sweet", "shape": "round"},
    "kiwi": {"color": "green", "size": 2, "taste": "sour_sweet", "shape": "oval_small"},
    "blueberry": {"color": "blue", "size": 1, "taste": "sweet", "shape": "round_small"},
    "melon": {"color": "green", "size": 4, "taste": "sweet", "shape": "round_large"},
    "fig": {"color": "purple", "size": 2, "taste": "sweet", "shape": "pear"},
    "plum": {"color": "purple", "size": 2, "taste": "sweet", "shape": "round"},
    "lime": {"color": "green", "size": 2, "taste": "sour", "shape": "round_small"},
}

# 不同表述方式（测试编码收敛性）
APPLE_EXPRESSIONS = [
    "The apple",
    "A red apple",
    "An apple of red color",
    "Red is apple's color",
    "I like eating apples",
    "The fruit called apple",
    "Apples are red fruits",
]


def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    if isinstance(obj, (bool,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ==================== 模型加载 ====================
def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA: {want_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()

    layer_count = len(discover_layers(model))
    neuron_dim = discover_layers(model)[0].mlp.gate_proj.out_features
    print(f"  Layers: {layer_count}, NeuronDim: {neuron_dim}")
    return model, tokenizer, layer_count, neuron_dim


# ==================== 激活提取（支持多token位置） ====================
def extract_activation_per_layer(model, tokenizer, prompt, target_word, layer_count):
    """提取目标词在所有层中的激活向量"""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    token_ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # 找到目标词的token位置
    word_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    target_pos = None
    for i, tid in enumerate(token_ids):
        if tid in word_tokens:
            target_pos = i
            break

    layer_payload_map = {i: "neuron_in" for i in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)

    try:
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            model(**encoded)

        per_layer = {}
        for li in range(layer_count):
            buf = buffers[li]
            if buf is not None:
                pos = target_pos if target_pos is not None else -1
                if pos < 0:
                    pos = buf.shape[1] + pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                per_layer[li] = buf[0, pos].float().numpy()

        return per_layer, tokens, target_pos
    finally:
        remove_hooks(handles)


def extract_all_layers(model, tokenizer, word, layer_count):
    """提取单个词在所有层的激活"""
    acts, tokens, pos = extract_activation_per_layer(model, tokenizer, f"The {word}", word, layer_count)
    return acts, tokens


# ==================== Exp1: 逐层编码演化 ====================
def run_exp1_evolution(apple_acts, all_fruit_acts, layer_count):
    """分析'苹果'的编码在逐层中如何演化"""
    print("\n" + "=" * 60)
    print("  Exp1: Apple encoding evolution across layers")
    print("=" * 60)

    results = {}

    # 计算苹果与其他水果的相似度
    for li in range(layer_count):
        if li not in apple_acts:
            continue

        apple_vec = apple_acts[li]

        # 与其他水果的相似度
        similarities = {}
        for fruit, acts in all_fruit_acts.items():
            if fruit == "apple" or li not in acts:
                continue
            cos = np.dot(apple_vec, acts[li]) / max(
                np.linalg.norm(apple_vec) * np.linalg.norm(acts[li]), EPS
            )
            similarities[fruit] = round(float(cos), 4)

        # 排序：最相似的5个水果
        sorted_fruits = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

        # 激活统计
        activation_mean = float(np.mean(np.abs(apple_vec)))
        activation_max = float(np.max(np.abs(apple_vec)))
        sparsity = float(np.sum(np.abs(apple_vec) < activation_mean) / len(apple_vec))

        # 最近的邻居（苹果的"邻居"在哪一层变化）
        nearest = sorted_fruits[0] if sorted_fruits else ("none", 0)

        results[f"layer_{li}"] = {
            "nearest_neighbor": nearest[0],
            "nearest_cos": nearest[1],
            "top5_similar": sorted_fruits,
            "activation_mean": round(activation_mean, 4),
            "activation_max": round(activation_max, 4),
            "sparsity": round(sparsity, 4),
        }

        if li % 5 == 0 or li == layer_count - 1:
            print(f"  L{li}: nearest={nearest[0]}(cos={nearest[1]:.4f}), "
                  f"act_mean={activation_mean:.4f}, sparse={sparsity:.3f}")

    return results


# ==================== Exp2: 3D坐标映射 ====================
def run_exp2_3d_mapping(apple_acts, all_fruit_acts, layer_count):
    """将概念编码映射到3D空间，参考大脑3D拓扑"""
    print("\n" + "=" * 60)
    print("  Exp2: 3D coordinate mapping (PCA -> 3D)")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    results = {"layers": {}}

    # 选择关键层
    key_layers = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35]
    key_layers = [l for l in key_layers if l < layer_count]

    for li in key_layers:
        if li not in apple_acts:
            continue

        # 收集所有水果的激活
        fruit_names = []
        fruit_vecs = []
        for fruit, acts in all_fruit_acts.items():
            if li in acts:
                fruit_vecs.append(acts[li])
                fruit_names.append(fruit)

        if len(fruit_vecs) < 5:
            continue

        matrix = np.array(fruit_vecs)

        # PCA到3维
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        n_components = min(3, len(fruit_vecs) - 1)
        pca = PCA(n_components=n_components)
        coords_3d = pca.fit_transform(scaled)

        # 找到苹果的3D坐标
        apple_idx = fruit_names.index("apple") if "apple" in fruit_names else 0
        apple_3d = coords_3d[apple_idx].tolist()

        # 计算苹果到所有其他水果的3D距离
        distances = {}
        for j, name in enumerate(fruit_names):
            if name == "apple":
                continue
            dist = float(np.linalg.norm(np.array(apple_3d) - coords_3d[j]))
            distances[name] = round(dist, 4)

        # 找3D最近邻
        sorted_dist = sorted(distances.items(), key=lambda x: x[1])[:5]

        # PCA解释方差
        explained_var = pca.explained_variance_ratio_.tolist()

        results[f"layer_{li}"] = {
            "apple_3d": [round(c, 4) for c in apple_3d],
            "nearest_3d": sorted_dist,
            "pca_explained_var": [round(v, 4) for v in explained_var],
            "n_fruits": len(fruit_vecs),
            "all_coords": {name: [round(c, 4) for c in coords_3d[i].tolist()]
                          for i, name in enumerate(fruit_names)},
        }

        print(f"  L{li}: Apple@({apple_3d[0]:.3f},{apple_3d[1]:.3f},{apple_3d[2]:.3f}), "
              f"nearest={sorted_dist[0][0]}(d={sorted_dist[0][1]:.3f}), "
              f"PCA_var={sum(explained_var):.3f}")

    return results


# ==================== Exp3: 脉冲稀疏性分析 ====================
def run_exp3_sparsity(apple_acts, all_fruit_acts, layer_count):
    """分析编码苹果时每层的神经元参与度（参考SNN稀疏激活）"""
    print("\n" + "=" * 60)
    print("  Exp3: Spike-like sparsity analysis")
    print("=" * 60)

    results = {}

    # 对苹果在每一层计算稀疏性
    for li in range(layer_count):
        if li not in apple_acts:
            continue

        vec = apple_acts[li]
        abs_vec = np.abs(vec)

        # 多种稀疏性度量
        threshold_mean = np.mean(abs_vec)
        threshold_std = np.std(abs_vec)
        threshold_75 = np.percentile(abs_vec, 75)
        threshold_90 = np.percentile(abs_vec, 90)

        # 模拟"脉冲发放"：激活值超过阈值的神经元"发放"
        active_mean = float(np.sum(abs_vec > threshold_mean) / len(vec))
        active_75 = float(np.sum(abs_vec > threshold_75) / len(vec))
        active_90 = float(np.sum(abs_vec > threshold_90) / len(vec))

        # Top-k稀疏性（前10%的神经元）
        k10 = max(1, int(len(vec) * 0.10))
        k05 = max(1, int(len(vec) * 0.05))
        k01 = max(1, int(len(vec) * 0.01))
        top10_energy = float(np.sum(np.sort(abs_vec)[-k10:] ** 2) / np.sum(abs_vec ** 2))
        top05_energy = float(np.sum(np.sort(abs_vec)[-k05:] ** 2) / np.sum(abs_vec ** 2))
        top01_energy = float(np.sum(np.sort(abs_vec)[-k01:] ** 2) / np.sum(abs_vec ** 2))

        # Gini系数（衡量不均匀度）
        sorted_abs = np.sort(abs_vec)
        n = len(sorted_abs)
        gini = float(2 * np.sum((np.arange(1, n + 1) * sorted_abs)) / (n * np.sum(sorted_abs)) - (n + 1) / n)

        results[f"layer_{li}"] = {
            "active_ratio_mean": round(active_mean, 4),
            "active_ratio_75pct": round(active_75, 4),
            "active_ratio_90pct": round(active_90, 4),
            "top10pct_energy": round(top10_energy, 4),
            "top5pct_energy": round(top05_energy, 4),
            "top1pct_energy": round(top01_energy, 4),
            "gini_coefficient": round(gini, 4),
        }

        if li % 5 == 0 or li == layer_count - 1:
            print(f"  L{li}: active(>mean)={active_mean:.3f}, "
                  f"top10%energy={top10_energy:.3f}, "
                  f"top1%energy={top01_energy:.3f}, "
                  f"gini={gini:.3f}")

    # 全层统计
    gini_vals = [r["gini_coefficient"] for r in results.values()]
    active_vals = [r["active_ratio_mean"] for r in results.values()]
    energy_vals = [r["top10pct_energy"] for r in results.values()]

    results["summary"] = {
        "avg_active_ratio": round(float(np.mean(active_vals)), 4),
        "avg_gini": round(float(np.mean(gini_vals)), 4),
        "avg_top10_energy": round(float(np.mean(energy_vals)), 4),
        "min_active_ratio": round(float(np.min(active_vals)), 4),
        "max_active_ratio": round(float(np.max(active_vals)), 4),
    }

    print(f"\n  [Summary] avg_active={np.mean(active_vals):.3f}, "
          f"avg_gini={np.mean(gini_vals):.3f}, "
          f"avg_top10%energy={np.mean(energy_vals):.3f}")

    return results


# ==================== Exp4: 编码形成路径分析 ====================
def run_exp4_encoding_path(apple_acts, layer_count, neuron_dim):
    """分析从Layer0到最后一层，苹果编码的形成路径"""
    print("\n" + "=" * 60)
    print("  Exp4: Encoding formation path")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    results = {}

    # 收集所有层的苹果激活
    layer_vecs = []
    valid_layers = []
    for li in range(layer_count):
        if li in apple_acts:
            layer_vecs.append(apple_acts[li])
            valid_layers.append(li)

    if len(layer_vecs) < 3:
        return results

    matrix = np.array(layer_vecs)  # shape: (n_layers, neuron_dim)

    # 标准化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    # PCA到3维观察编码路径
    pca = PCA(n_components=3)
    path_3d = pca.fit_transform(scaled)

    # 逐层变化量（编码的"速度"）
    changes = []
    for i in range(1, len(valid_layers)):
        change = float(np.linalg.norm(scaled[i] - scaled[i - 1]))
        changes.append(change)

    # 关键转折点（编码变化最大的层）
    if changes:
        max_change_idx = int(np.argmax(changes))
        max_change_layer = valid_layers[max_change_idx + 1] if max_change_idx + 1 < len(valid_layers) else valid_layers[-1]

        # 前5个最大变化层
        sorted_changes = sorted(enumerate(changes), key=lambda x: x[1], reverse=True)[:5]
        top_changes = [(valid_layers[idx + 1], round(ch, 4)) for idx, ch in sorted_changes]

        results["path_dynamics"] = {
            "max_change_layer": int(max_change_layer),
            "max_change_value": round(float(np.max(changes)), 4),
            "avg_change": round(float(np.mean(changes)), 4),
            "total_path_length": round(float(np.sum(changes)), 4),
            "top5_change_layers": top_changes,
        }

        print(f"  Max encoding change at L{max_change_layer} (change={np.max(changes):.4f})")
        print(f"  Total path length: {np.sum(changes):.4f}")
        print(f"  Top5 change layers: {top_changes}")

    # 3D路径坐标
    results["path_3d"] = {
        layer: [round(c, 4) for c in path_3d[i].tolist()]
        for i, layer in enumerate(valid_layers)
    }
    results["pca_explained_var"] = [round(float(v), 4) for v in pca.explained_variance_ratio_.tolist()]

    # 编码收敛分析：前后半段编码的稳定性
    mid = len(valid_layers) // 2
    if mid > 0:
        early_stability = 1.0 - float(np.mean(changes[:mid])) / max(float(np.mean(changes)), EPS)
        late_stability = 1.0 - float(np.mean(changes[mid:])) / max(float(np.mean(changes)), EPS)
        results["convergence"] = {
            "early_stability": round(early_stability, 4),
            "late_stability": round(late_stability, 4),
            "converging": late_stability > early_stability,
        }
        print(f"  Early stability: {early_stability:.4f}, Late stability: {late_stability:.4f}")
        print(f"  Encoding is {'converging' if late_stability > early_stability else 'diverging'}")

    return results


# ==================== Exp5: 苹果vs其他水果 — 属性关联 ====================
def run_exp5_attribute_correlation(apple_acts, all_fruit_acts, layer_count):
    """分析苹果与其他水果的编码相似度如何随属性变化"""
    print("\n" + "=" * 60)
    print("  Exp5: Apple vs other fruits - attribute correlation")
    print("=" * 60)

    results = {}

    for li in range(layer_count):
        if li not in apple_acts:
            continue

        apple_vec = apple_acts[li]
        correlations = []

        for fruit, attrs in FRUIT_CONCEPTS.items():
            if fruit == "apple" or fruit not in all_fruit_acts or li not in all_fruit_acts[fruit]:
                continue

            fruit_vec = all_fruit_acts[fruit][li]
            cos = np.dot(apple_vec, fruit_vec) / max(
                np.linalg.norm(apple_vec) * np.linalg.norm(fruit_vec), EPS
            )

            # 计算属性相似度
            apple_attrs = FRUIT_CONCEPTS["apple"]
            attr_match = sum(1 for k, v in attrs.items() if apple_attrs.get(k) == v)
            attr_total = len(set(list(apple_attrs.keys()) + list(attrs.keys())))
            attr_similarity = attr_match / max(attr_total, 1)

            correlations.append({
                "fruit": fruit,
                "cos_sim": round(float(cos), 4),
                "attr_similarity": round(attr_similarity, 4),
                "shared_attrs": {k: v for k, v in attrs.items() if apple_attrs.get(k) == v},
            })

        # 相关性分析：编码相似度 vs 属性相似度
        cos_sims = [c["cos_sim"] for c in correlations]
        attr_sims = [c["attr_similarity"] for c in correlations]

        if len(cos_sims) >= 3:
            pearson_r = float(np.corrcoef(cos_sims, attr_sims)[0, 1]) if np.std(cos_sims) > EPS and np.std(attr_sims) > EPS else 0.0
        else:
            pearson_r = 0.0

        # 按属性匹配排序
        by_attr = sorted(correlations, key=lambda x: x["attr_similarity"], reverse=True)

        results[f"layer_{li}"] = {
            "pearson_r": round(float(np.nan_to_num(pearson_r)), 4),
            "top_by_attr": by_attr[:5],
            "avg_cos_sim": round(float(np.mean(cos_sims)), 4) if cos_sims else 0,
        }

        if li % 5 == 0 or li == layer_count - 1:
            print(f"  L{li}: pearson_r(cos vs attr)={np.nan_to_num(pearson_r):.4f}, "
                  f"avg_cos={np.mean(cos_sims):.4f}")
            if by_attr:
                top = by_attr[0]
                print(f"    Most similar by attr: {top['fruit']}(attr_sim={top['attr_similarity']:.2f}, cos={top['cos_sim']:.4f})")

    # 跨层总结
    pearsons = [r["pearson_r"] for r in results.values() if isinstance(r, dict)]
    if pearsons:
        results["cross_layer_summary"] = {
            "avg_pearson_r": round(float(np.mean(pearsons)), 4),
            "max_pearson_r": round(float(np.max(pearsons)), 4),
            "layer_best": int(list(results.keys())[int(np.argmax(pearsons))].replace("layer_", "")),
            "encoding_reflects_attributes": float(np.max(pearsons)) > 0.3,
        }

    return results


# ==================== Exp6: 不同表述的编码收敛性 ====================
def run_exp6_convergence(model, tokenizer, layer_count):
    """不同表述方式的苹果编码是否收敛到同一点"""
    print("\n" + "=" * 60)
    print("  Exp6: Encoding convergence across expressions")
    print("=" * 60)

    results = {}

    # 提取每种表述的逐层激活
    expression_acts = {}
    for expr in APPLE_EXPRESSIONS:
        acts, tokens, pos = extract_activation_per_layer(
            model, tokenizer, expr, "apple", layer_count
        )
        expression_acts[expr] = acts

    # 计算表述间的平均相似度（逐层）
    expr_names = list(expression_acts.keys())
    for li in range(layer_count):
        layer_sims = []
        for i in range(len(expr_names)):
            for j in range(i + 1, len(expr_names)):
                e1 = expression_acts[expr_names[i]]
                e2 = expression_acts[expr_names[j]]
                if li in e1 and li in e2:
                    cos = np.dot(e1[li], e2[li]) / max(
                        np.linalg.norm(e1[li]) * np.linalg.norm(e2[li]), EPS
                    )
                    layer_sims.append(float(cos))

        if layer_sims:
            results[f"layer_{li}"] = {
                "mean_similarity": round(float(np.mean(layer_sims)), 4),
                "std_similarity": round(float(np.std(layer_sims)), 4),
                "min_similarity": round(float(np.min(layer_sims)), 4),
                "max_similarity": round(float(np.max(layer_sims)), 4),
            }

        if li % 5 == 0 or li == layer_count - 1:
            if layer_sims:
                print(f"  L{li}: mean_cos={np.mean(layer_sims):.4f} "
                      f"(range=[{np.min(layer_sims):.4f}, {np.max(layer_sims):.4f}])")

    # 收敛趋势
    means = [r["mean_similarity"] for r in results.values() if isinstance(r, dict)]
    if len(means) >= 4:
        early = np.mean(means[:len(means) // 2])
        late = np.mean(means[len(means) // 2:])
        results["convergence_trend"] = {
            "early_layers_avg": round(float(early), 4),
            "late_layers_avg": round(float(late), 4),
            "converging": late > early,
            "convergence_delta": round(float(late - early), 4),
        }
        print(f"\n  Convergence: early={early:.4f}, late={late:.4f}, "
              f"delta={late - early:+.4f} ({'converging' if late > early else 'diverging'})")

    return results


# ==================== 主流程 ====================
def main():
    if len(sys.argv) < 2:
        print("Usage: python stage468_3d_spike_concept_encoding.py [qwen3|deepseek]")
        sys.exit(1)

    model_choice = sys.argv[1].lower()
    if model_choice == "qwen3":
        model_path = QWEN3_MODEL_PATH
        model_name = "qwen3_4b"
    elif model_choice == "deepseek":
        model_path = DEEPSEEK7B_MODEL_PATH
        model_name = "deepseek_7b"
    else:
        print(f"Unknown model: {model_choice}")
        sys.exit(1)

    print("=" * 70)
    print(f"Stage468: 3D Spatial Concept Encoding (SNN-inspired)")
    print(f"Model: {model_name}")
    print("=" * 70)

    t0 = time.time()

    # 1. 加载模型
    print("\n[1/3] Loading model...")
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)

    # 2. 提取所有水果的逐层激活
    print(f"\n[2/3] Extracting fruit activations ({len(FRUIT_CONCEPTS)} fruits, {layer_count} layers)...")
    all_fruit_acts = {}
    for fruit in FRUIT_CONCEPTS:
        acts, tokens = extract_all_layers(model, tokenizer, fruit, layer_count)
        all_fruit_acts[fruit] = acts
    print(f"  Extracted {len(all_fruit_acts)} fruits")

    apple_acts = all_fruit_acts.get("apple", {})

    # 3. 运行实验
    print(f"\n[3/3] Running experiments...")

    all_results = {
        "model": model_name,
        "layer_count": layer_count,
        "neuron_dim": neuron_dim,
    }

    # Exp1: 逐层演化
    e1 = run_exp1_evolution(apple_acts, all_fruit_acts, layer_count)
    all_results["exp1_evolution"] = e1

    # Exp2: 3D映射
    e2 = run_exp2_3d_mapping(apple_acts, all_fruit_acts, layer_count)
    all_results["exp2_3d_mapping"] = e2

    # Exp3: 脉冲稀疏性
    e3 = run_exp3_sparsity(apple_acts, all_fruit_acts, layer_count)
    all_results["exp3_sparsity"] = e3

    # Exp4: 编码路径
    e4 = run_exp4_encoding_path(apple_acts, layer_count, neuron_dim)
    all_results["exp4_encoding_path"] = e4

    # Exp5: 属性关联
    e5 = run_exp5_attribute_correlation(apple_acts, all_fruit_acts, layer_count)
    all_results["exp5_attribute_correlation"] = e5

    # Exp6: 编码收敛性（需要模型，在释放前运行）
    print("\n  [Running Exp6: encoding convergence - needs model]")
    e6 = run_exp6_convergence(model, tokenizer, layer_count)
    all_results["exp6_convergence"] = e6

    # 释放模型
    print("\n  Releasing model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 保存结果
    output_path = OUTPUT_DIR / f"{model_name}_full_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n  Results: {output_path}")

    # 生成报告
    generate_report(all_results, model_name)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Stage468 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")


def generate_report(all_data, model_name):
    lines = [
        f"# Stage468: 3D Space Concept Encoding Analysis (SNN-inspired) - {model_name}",
        "",
        f"**Time**: 2026-04-01 14:30",
        f"**Model**: {model_name}",
        f"**Layers**: {all_data['layer_count']}, NeuronDim: {all_data['neuron_dim']}",
        "",
        "---",
        "",
    ]

    # Exp3 Summary
    e3 = all_data.get("exp3_sparsity", {})
    e3s = e3.get("summary", {})
    lines.append("## Exp3: Spike-like Sparsity (Apple encoding)")
    lines.append("")
    lines.append(f"- Avg active ratio (>mean): {e3s.get('avg_active_ratio', 0):.4f}")
    lines.append(f"- Avg Gini coefficient: {e3s.get('avg_gini', 0):.4f}")
    lines.append(f"- Avg top-10% energy: {e3s.get('avg_top10_energy', 0):.4f}")
    lines.append("")

    # Exp4 Path
    e4 = all_data.get("exp4_encoding_path", {})
    e4p = e4.get("path_dynamics", {})
    e4c = e4.get("convergence", {})
    lines.append("## Exp4: Encoding Formation Path")
    lines.append("")
    lines.append(f"- Max change layer: L{e4p.get('max_change_layer', '?')}")
    lines.append(f"- Total path length: {e4p.get('total_path_length', 0):.4f}")
    lines.append(f"- Early stability: {e4c.get('early_stability', 0):.4f}")
    lines.append(f"- Late stability: {e4c.get('late_stability', 0):.4f}")
    lines.append(f"- Converging: {'Yes' if e4c.get('converging', False) else 'No'}")
    lines.append("")

    # Exp5 Attribute
    e5 = all_data.get("exp5_attribute_correlation", {})
    e5s = e5.get("cross_layer_summary", {})
    lines.append("## Exp5: Encoding-Attribute Correlation")
    lines.append("")
    lines.append(f"- Avg Pearson r (cos vs attr): {e5s.get('avg_pearson_r', 0):.4f}")
    lines.append(f"- Max Pearson r: {e5s.get('max_pearson_r', 0):.4f} (L{e5s.get('layer_best', '?')})")
    lines.append(f"- Encoding reflects attributes: {'Yes' if e5s.get('encoding_reflects_attributes', False) else 'No'}")
    lines.append("")

    # Exp6 Convergence
    e6 = all_data.get("exp6_convergence", {})
    e6t = e6.get("convergence_trend", {})
    lines.append("## Exp6: Encoding Convergence (different expressions)")
    lines.append("")
    lines.append(f"- Early layers avg similarity: {e6t.get('early_layers_avg', 0):.4f}")
    lines.append(f"- Late layers avg similarity: {e6t.get('late_layers_avg', 0):.4f}")
    lines.append(f"- Convergence delta: {e6t.get('convergence_delta', 0):+.4f}")
    lines.append(f"- Converging: {'Yes' if e6t.get('converging', False) else 'No'}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Apple Encoding Mechanism Summary")
    lines.append("See detailed analysis in the response.")
    lines.append("")

    report_path = OUTPUT_DIR / f"{model_name}_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
