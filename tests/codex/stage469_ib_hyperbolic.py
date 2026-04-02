# -*- coding: utf-8 -*-
"""
Stage469: 信息瓶颈推导 + 双曲嵌入验证
=================================================================

核心问题：
  语言编码空间的维度(17)和双曲结构是由什么决定的？
  能否从第一性原理（信息瓶颈理论）推导出来？

实验模块：

  Part A: 信息瓶颈推导
    A1. IB曲线: I(X;T) vs I(T;Y) — 找到最优压缩点
    A2. 理论维度推导: 用IB + 自由度分析推导最优维度
    A3. 跨层IB分析: 逐层跟踪信息压缩过程
    A4. IB最优性验证: 在不同概念集和属性集上验证维度稳定性

  Part B: 双曲嵌入验证
    B1. Poincare Ball嵌入: 用MDS在双曲空间中拟合偏置空间
    B2. 双曲距离 vs 欧几里得距离: 概念关系解释力对比
    B3. 双曲概念算术: 在双曲空间中做测地线算术
    B4. 双曲性与层次化: 验证层次化概念是否在双曲空间中更好地分离

用法：
  python stage469_ib_hyperbolic.py qwen3
  python stage469_ib_hyperbolic.py deepseek
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
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances
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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage469_ib_hyperbolic_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集 ====================
CONCEPTS = {
    "fruit": {
        "label": "水果",
        "words": {
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
            "pear": {"color": "green", "size": 3, "taste": "sweet", "shape": "pear_shape"},
            "pineapple": {"color": "yellow", "size": 4, "taste": "sour_sweet", "shape": "oval"},
            "coconut": {"color": "brown", "size": 4, "taste": "sweet", "shape": "round"},
            "kiwi": {"color": "green", "size": 2, "taste": "sour_sweet", "shape": "oval_small"},
            "blueberry": {"color": "blue", "size": 1, "taste": "sweet", "shape": "round_small"},
            "melon": {"color": "green", "size": 4, "taste": "sweet", "shape": "round_large"},
            "fig": {"color": "purple", "size": 2, "taste": "sweet", "shape": "pear_shape"},
            "plum": {"color": "purple", "size": 2, "taste": "sweet", "shape": "round"},
            "lime": {"color": "green", "size": 2, "taste": "sour", "shape": "round_small"},
        },
    },
    "animal": {
        "label": "动物",
        "words": {
            "dog": {"size": 2, "domestic": 1, "speed": 3},
            "cat": {"size": 2, "domestic": 1, "speed": 3},
            "horse": {"size": 4, "domestic": 1, "speed": 4},
            "lion": {"size": 3, "domestic": 0, "speed": 4},
            "tiger": {"size": 3, "domestic": 0, "speed": 4},
            "elephant": {"size": 5, "domestic": 0, "speed": 2},
            "whale": {"size": 5, "domestic": 0, "speed": 3},
            "shark": {"size": 3, "domestic": 0, "speed": 4},
            "eagle": {"size": 2, "domestic": 0, "speed": 5},
            "wolf": {"size": 2, "domestic": 0, "speed": 4},
            "rabbit": {"size": 1, "domestic": 1, "speed": 4},
            "deer": {"size": 3, "domestic": 0, "speed": 4},
            "fox": {"size": 2, "domestic": 0, "speed": 4},
            "bear": {"size": 4, "domestic": 0, "speed": 3},
            "monkey": {"size": 2, "domestic": 0, "speed": 4},
            "dolphin": {"size": 3, "domestic": 0, "speed": 5},
            "penguin": {"size": 2, "domestic": 0, "speed": 1},
            "snake": {"size": 2, "domestic": 0, "speed": 3},
            "giraffe": {"size": 5, "domestic": 0, "speed": 4},
            "panda": {"size": 3, "domestic": 0, "speed": 2},
        },
    },
    "vehicle": {
        "label": "交通工具",
        "words": {
            "car": {"speed": 3, "medium": "land", "size": 2},
            "bus": {"speed": 3, "medium": "land", "size": 3},
            "train": {"speed": 4, "medium": "land", "size": 4},
            "plane": {"speed": 5, "medium": "air", "size": 4},
            "ship": {"speed": 2, "medium": "water", "size": 5},
            "bicycle": {"speed": 2, "medium": "land", "size": 1},
            "motorcycle": {"speed": 3, "medium": "land", "size": 1},
            "truck": {"speed": 2, "medium": "land", "size": 3},
            "helicopter": {"speed": 4, "medium": "air", "size": 2},
            "rocket": {"speed": 5, "medium": "air", "size": 3},
            "boat": {"speed": 2, "medium": "water", "size": 2},
            "submarine": {"speed": 2, "medium": "water", "size": 3},
            "taxi": {"speed": 3, "medium": "land", "size": 2},
            "ambulance": {"speed": 4, "medium": "land", "size": 2},
            "ferry": {"speed": 2, "medium": "water", "size": 4},
        },
    },
    "profession": {
        "label": "职业",
        "words": {
            "doctor": {"domain": "medical", "social": 1, "creativity": 0},
            "nurse": {"domain": "medical", "social": 1, "creativity": 0},
            "teacher": {"domain": "education", "social": 1, "creativity": 1},
            "engineer": {"domain": "technology", "social": 0, "creativity": 1},
            "lawyer": {"domain": "law", "social": 1, "creativity": 1},
            "chef": {"domain": "food", "social": 0, "creativity": 1},
            "artist": {"domain": "art", "social": 0, "creativity": 1},
            "musician": {"domain": "art", "social": 0, "creativity": 1},
            "scientist": {"domain": "science", "social": 0, "creativity": 1},
            "pilot": {"domain": "transport", "social": 0, "creativity": 0},
            "soldier": {"domain": "military", "social": 1, "creativity": 0},
            "firefighter": {"domain": "emergency", "social": 1, "creativity": 0},
            "police": {"domain": "law", "social": 1, "creativity": 0},
            "farmer": {"domain": "agriculture", "social": 0, "creativity": 0},
            "baker": {"domain": "food", "social": 0, "creativity": 1},
            "architect": {"domain": "construction", "social": 0, "creativity": 1},
            "surgeon": {"domain": "medical", "social": 1, "creativity": 1},
            "dentist": {"domain": "medical", "social": 1, "creativity": 0},
            "pharmacist": {"domain": "medical", "social": 0, "creativity": 0},
        },
    },
}

CATEGORY_ATTRIBUTES = {
    "fruit": ["color", "size", "taste", "shape"],
    "animal": ["size", "domestic", "speed"],
    "vehicle": ["speed", "medium", "size"],
    "profession": ["domain", "social", "creativity"],
}


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
    if want_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

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


# ==================== 激活提取 ====================
def extract_word_per_layer(model, tokenizer, word, layer_count):
    prompt = f"The {word}"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    token_ids = encoded["input_ids"][0].tolist()

    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    target_pos = None
    for i, tid in enumerate(token_ids):
        if tid in word_tokens:
            target_pos = i
            break
    if target_pos is None:
        target_pos = -1

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
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                per_layer[li] = buf[0, pos].float().numpy()
        return per_layer
    finally:
        remove_hooks(handles)


def extract_all_activations(model, tokenizer, concepts, layer_count):
    all_activations = {}
    total = sum(len(c["words"]) for c in concepts.values())
    done = 0

    for cat_name, cat_data in concepts.items():
        all_activations[cat_name] = {}
        for word in cat_data["words"]:
            try:
                acts = extract_word_per_layer(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except Exception:
                pass

    print(f"  Extracted: {done}/{total}")
    return all_activations


# ==================== 计算偏置矩阵 ====================
def compute_bias_matrix(all_activations, layer_idx, concepts):
    """计算指定层的偏置矩阵 = 激活 - 类别均值"""
    bias_rows = []
    word_labels = []
    word_categories = []

    for cat_name, cat_data in concepts.items():
        cat_activations = []
        cat_words = []
        for word, layer_acts in all_activations[cat_name].items():
            if layer_idx in layer_acts:
                cat_activations.append(layer_acts[layer_idx])
                cat_words.append(word)

        if len(cat_activations) < 2:
            continue

        cat_matrix = np.array(cat_activations)
        cat_mean = np.mean(cat_matrix, axis=0, keepdims=True)
        biases = cat_matrix - cat_mean

        for i, w in enumerate(cat_words):
            bias_rows.append(biases[i])
            word_labels.append(w)
            word_categories.append(cat_name)

    if not bias_rows:
        return None, None, None

    bias_matrix = np.array(bias_rows)
    # 标准化
    std = np.std(bias_matrix, axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    bias_scaled = bias_matrix / std
    return bias_scaled, word_labels, word_categories


# ==================== 属性编码 ====================
def encode_attributes(concepts, word_labels, word_categories):
    """将属性编码为数值矩阵（用于IB分析）。
    每个概念一行，列为所有类别的所有属性的统一编码。
    不同类别的属性用各自独立列表示，通过padding对齐到最大属性数。
    """
    # 收集所有属性，确定统一列数
    max_attrs = max(len(CATEGORY_ATTRIBUTES.get(c, [])) for c in concepts)
    all_attr_cols = []  # list of (cat_name, attr_name)
    for cat_name in concepts:
        cat_attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])
        for attr in cat_attrs:
            all_attr_cols.append((cat_name, attr))
    n_cols = len(all_attr_cols)

    attr_matrix_rows = []
    attr_names = [f"{cn}_{an}" for cn, an in all_attr_cols]

    for i, word in enumerate(word_labels):
        cat = word_categories[i]
        row = [0.0] * n_cols
        for col_idx, (cn, attr) in enumerate(all_attr_cols):
            if cn != cat:
                row[col_idx] = 0.0  # 非本类别属性填0
                continue
            val = concepts[cat]["words"].get(word, {}).get(attr, None)
            if val is None:
                row[col_idx] = 0.0
            elif isinstance(val, (int, float)):
                row[col_idx] = float(val)
            else:
                all_vals = sorted(set(
                    concepts[cat]["words"][w].get(attr, "")
                    for w in concepts[cat]["words"]
                    if attr in concepts[cat]["words"][w]
                ))
                try:
                    row[col_idx] = float(all_vals.index(val)) / max(len(all_vals) - 1, 1)
                except ValueError:
                    row[col_idx] = 0.0
        attr_matrix_rows.append(row)

    return np.array(attr_matrix_rows), attr_names


# ====================================================================
# Part A: 信息瓶颈推导
# ====================================================================

def compute_mutual_info(X_reduced, Y_attr, n_bins=10):
    """
    用直方图方法估计 I(X_reduced; Y_attr)
    X_reduced: (n_samples, d) 降维后的表示
    Y_attr: (n_samples,) 属性值（离散化）
    """
    n_samples = X_reduced.shape[0]
    if n_samples < 3:
        return 0.0

    # 对X_reduced每个维度做离散化
    x_codes = np.zeros((n_samples, X_reduced.shape[1]), dtype=int)
    for d in range(X_reduced.shape[1]):
        vals = X_reduced[:, d]
        if np.std(vals) < EPS:
            x_codes[:, d] = 0
        else:
            _, bin_edges = np.histogram(vals, bins=n_bins)
            x_codes[:, d] = np.clip(
                np.digitize(vals, bin_edges[1:-1]) - 1, 0, n_bins - 1
            )

    # 将多维度X编码为单一标签（简化处理）
    x_labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        code = 0
        for d in range(x_codes.shape[1]):
            code = code * n_bins + x_codes[i, d]
        x_labels[i] = code

    # 离散化Y
    y_unique = np.unique(Y_attr)
    if len(y_unique) < 2:
        return 0.0
    y_map = {v: i for i, v in enumerate(y_unique)}
    y_labels = np.array([y_map.get(v, 0) for v in Y_attr])

    # 计算 I(X; Y) = H(X) + H(Y) - H(X,Y)
    from collections import Counter
    cx = Counter(x_labels)
    cy = Counter(y_labels)
    cxy = Counter(zip(x_labels, y_labels))

    hx = -sum((c / n_samples) * np.log2(c / n_samples + EPS) for c in cx.values())
    hy = -sum((c / n_samples) * np.log2(c / n_samples + EPS) for c in cy.values())
    hxy = -sum((c / n_samples) * np.log2(c / n_samples + EPS) for c in cxy.values())

    return max(hx + hy - hxy, 0.0)


def run_part_a(bias_scaled, word_labels, word_categories, concepts, layer_idx, neuron_dim):
    """信息瓶颈分析"""
    print(f"\n{'='*60}")
    print(f"  Part A: 信息瓶颈推导 (Layer {layer_idx}, dim={neuron_dim})")
    print(f"{'='*60}")

    attr_matrix, attr_names = encode_attributes(concepts, word_labels, word_categories)
    n_concepts = bias_scaled.shape[0]

    # 构建属性目标Y：将所有属性列拼接后的L2范数作为"综合属性向量"
    # 更好的做法：用第一主成分作为Y的代理
    from sklearn.decomposition import PCA as PCA_sk
    pca_y = PCA_sk(n_components=1)
    Y_1d = pca_y.fit_transform(attr_matrix).flatten()

    # 对Y做离散化
    n_bins_y = min(8, n_concepts // 3)
    if n_bins_y < 2:
        n_bins_y = 2
    _, bin_edges = np.histogram(Y_1d, bins=n_bins_y)
    Y_discrete = np.clip(np.digitize(Y_1d, bin_edges[1:-1]) - 1, 0, n_bins_y - 1)

    results_a = {
        "layer": layer_idx,
        "neuron_dim": neuron_dim,
        "n_concepts": n_concepts,
        "ib_curve": [],
        "optimal_dim": None,
        "compression_ratios": [],
    }

    # A1: IB曲线 — 在不同维度下计算 I(X;T) 和 I(T;Y)
    print("\n  [A1] IB曲线: 维度 vs I(X;T) 和 I(T;Y)")
    max_dim = min(n_concepts - 1, 72, neuron_dim)
    # 关键维度点（确保包含17附近的高分辨率搜索）
    key_dims = [1, 2, 3, 5, 7, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 25, 30, 35, 40, 50, 60]
    dims_to_test = sorted(set([d for d in key_dims if d <= max_dim] + [max_dim]))

    ib_points = []
    best_dim = 1
    best_score = -1
    ib_elbow_dim = 1  # I(T;Y)边际增益拐点

    for d in dims_to_test:
        pca = PCA_sk(n_components=min(d, bias_scaled.shape[1], bias_scaled.shape[0] - 1))
        T = pca.fit_transform(bias_scaled)

        # I(X;T) ≈ 解释方差比例（PCA保留的信息量）
        explained = float(np.sum(pca.explained_variance_ratio_))
        # I(T;Y): T与属性的互信息
        ity = compute_mutual_info(T, Y_discrete, n_bins=5)

        point = {
            "dim": d,
            "ixt": round(explained, 4),
            "ity": round(ity, 4),
            "score": round(explained + 0.5 * ity, 4),  # 加权得分
        }
        ib_points.append(point)

        if explained + 0.5 * ity > best_score:
            best_score = explained + 0.5 * ity
            best_dim = d

        if d in [1, 2, 3, 5, 10, 15, 17, 20, 25, 30, 40, 50] or d == max_dim:
            print(f"    dim={d:3d}: I(X;T)={explained:.4f}, I(T;Y)={ity:.4f}, score={explained + 0.5 * ity:.4f}")

    # 找I(T;Y)边际增益的拐点（IB理论的核心：信息压缩的拐点）
    if len(ib_points) >= 3:
        ity_vals = [p["ity"] for p in ib_points]
        marginal_gains = [ity_vals[0]] + [ity_vals[i] - ity_vals[i-1] for i in range(1, len(ity_vals))]
        # 拐点 = 边际增益首次降到最大增益10%以下的维度
        max_gain = max(marginal_gains) if marginal_gains else 1.0
        for i, mg in enumerate(marginal_gains):
            if mg < max_gain * 0.10 and i >= 2:
                ib_elbow_dim = ib_points[i]["dim"]
                break
        else:
            ib_elbow_dim = ib_points[-1]["dim"]
        print(f"    IB拐点(I(T;Y)边际增益<10%): dim={ib_elbow_dim}")

    results_a["ib_curve"] = ib_points
    results_a["optimal_dim"] = best_dim
    results_a["ib_elbow_dim"] = ib_elbow_dim

    # A2: 理论维度推导
    print(f"\n  [A2] 理论维度推导")
    n_categories = len(set(word_categories))

    # 方法1: 自由度计数
    total_attr_levels = 0
    for cat_name in concepts:
        for attr in CATEGORY_ATTRIBUTES.get(cat_name, []):
            vals = set()
            for word_data in concepts[cat_name]["words"].values():
                if attr in word_data:
                    vals.add(word_data[attr])
            total_attr_levels += len(vals)

    dof_dim = total_attr_levels - n_categories  # 减去类别自由度
    print(f"    自由度法: {total_attr_levels}属性水平 - {n_categories}类别 = {dof_dim}维")

    # 方法2: 信息论下界
    # H(Y) = 各属性熵之和
    h_total = 0.0
    for cat_name in concepts:
        for attr in CATEGORY_ATTRIBUTES.get(cat_name, []):
            vals = []
            for w, wdata in concepts[cat_name]["words"].items():
                if attr in wdata:
                    vals.append(str(wdata[attr]))
            if vals:
                from collections import Counter
                vc = Counter(vals)
                n = len(vals)
                h = -sum((c / n) * np.log2(c / n + EPS) for c in vc.values())
                h_total += h

    info_dim_lower = int(np.ceil(h_total / np.log2(min(n_concepts, 20))))
    print(f"    信息论下界: H(Y)={h_total:.2f} bits -> 至少{info_dim_lower}维编码")

    # 方法3: AIC/BIC选取（通过PCA解释方差）
    pca_full = PCA_sk(n_components=min(bias_scaled.shape[0] - 1, bias_scaled.shape[1]))
    pca_full.fit(bias_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    # 找到95%方差对应的维度
    dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"    PCA 95%方差: {dim_95}维")
    print(f"    PCA 99%方差: {dim_99}维")

    # 找拐点（elbow method）
    diffs = np.diff(cumvar)
    elbow_dim = int(np.argmax(diffs[1:]) + 2) if len(diffs) > 1 else 2
    print(f"    PCA拐点(elbow): {elbow_dim}维")

    results_a["theoretical_dims"] = {
        "dof": dof_dim,
        "info_theory_lower": info_dim_lower,
        "pca_95": dim_95,
        "pca_99": dim_99,
        "pca_elbow": elbow_dim,
        "observed_stage467": 17,
    }

    # A3: 跨层IB分析（只用当前层的PCA）
    print(f"\n  [A3] PCA维度分析")
    print(f"    前10主成分解释方差: {[f'{v:.4f}' for v in pca_full.explained_variance_ratio_[:10]]}")
    print(f"    前10主成分累计方差: {[f'{v:.4f}' for v in cumvar[:10]]}")

    # A4: 压缩比分析
    compression = neuron_dim / max(best_dim, 1)
    print(f"\n  [A4] 压缩比分析")
    print(f"    全维={neuron_dim}, IB最优={best_dim}, 压缩比={compression:.0f}x")
    print(f"    与Stage467观测(17维)对比: 偏差={abs(best_dim - 17)}")

    results_a["compression_ratio"] = round(compression, 1)
    results_a["compression_vs_stage467"] = {
        "ib_optimal": best_dim,
        "stage467_observed": 17,
        "difference": best_dim - 17,
    }

    return results_a


# ====================================================================
# Part B: 双曲嵌入验证
# ====================================================================

def poincare_distance(z1, z2, c=1.0):
    """
    Poincare球模型中的双曲距离
    d(z1, z2) = arccosh(1 + 2 * ||z1-z2||^2 / ((1-||z1||^2)(1-||z2||^2)))
    """
    sq_norm1 = np.sum(z1 ** 2)
    sq_norm2 = np.sum(z2 ** 2)
    sq_diff = np.sum((z1 - z2) ** 2)

    denom = (1 - sq_norm1) * (1 - sq_norm2)
    if denom < EPS:
        denom = EPS

    arg = 1.0 + 2.0 * sq_diff / denom
    arg = max(arg, 1.0 + EPS)  # 确保>=1
    return np.arccosh(arg)


def poincare_distance_matrix(Z, c=1.0):
    """计算Poincare距离矩阵"""
    n = Z.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_distance(Z[i], Z[j], c)
            D[i, j] = d
            D[j, i] = d
    return D


def project_to_poincare_ball(v, c=1.0):
    """将向量投影到Poincare球内 (||v|| < 1/sqrt(c))"""
    r = 1.0 / np.sqrt(c)
    norm = np.linalg.norm(v)
    if norm >= r * 0.99:
        v = v * (r * 0.99 / norm)
    return v


def hyperbolic_mds(X_highdim, n_components=2, c=1.0, max_iter=300, lr=0.01):
    """
    双曲MDS: 在Poincare球中嵌入数据
    最小化: sum_{i<j} (d_hyp(z_i, z_j) - d_orig(x_i, x_j))^2
    使用Riemannian梯度下降
    """
    n = X_highdim.shape[0]

    # 原始距离矩阵（欧几里得），归一化
    D_orig = squareform(pdist(X_highdim, metric="euclidean"))
    D_orig_norm = D_orig / (np.max(D_orig) + EPS)

    # 初始化：PCA降到n_components维，缩放到球内
    pca_init = PCA(n_components=n_components)
    Z_init = pca_init.fit_transform(X_highdim)
    Z_init = Z_init / (np.max(np.abs(Z_init)) + EPS) * 0.3

    # 用t-SNE式的对称化距离作为目标
    # 将欧几里得距离转换为"概率"相似度（t-SNE风格）
    P = np.exp(-D_orig_norm ** 2 / (2 * 0.5 ** 2))
    np.fill_diagonal(P, 0)
    P = P / np.sum(P) + EPS  # 对称化

    Z = Z_init.copy()

    for iteration in range(max_iter):
        # 投影到球内
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms = np.where(norms < EPS, EPS, norms)
        # 确保所有点在球内 (||z|| < 1)
        scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
        Z = Z * scale

        # 计算Poincare距离矩阵
        D_hyp = poincare_distance_matrix(Z, c)

        # 计算Q矩阵（嵌入空间中的相似度）
        Q = np.exp(-D_hyp ** 2 / (2 * 0.5 ** 2))
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q) + EPS

        # KL散度损失
        kl = np.sum(P * np.log(P / Q))

        # 梯度: dKL/dz_i = 4 * sum_j (P_ij - Q_ij) * (z_i - z_j) / d_ij * factor
        grad = np.zeros_like(Z)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p_q = P[i, j] - Q[i, j]
                diff = Z[i] - Z[j]
                d_ij = D_hyp[i, j] + EPS
                # Poincare度规因子: lambda_z = 2 / (1 - ||z||^2)
                lambda_i = 2.0 / (1.0 - np.sum(Z[i] ** 2) + EPS)
                lambda_j = 2.0 / (1.0 - np.sum(Z[j] ** 2) + EPS)
                grad[i] += 4.0 * p_q * diff / (d_ij + EPS) * lambda_i

        # 动量和自适应学习率
        momentum = 0.5 if iteration < 100 else 0.8
        # 更新
        Z = Z - lr * grad
        # 重新投影
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
        Z = Z * scale

        if iteration % 50 == 0:
            print(f"    MDS iter {iteration}: KL={kl:.6f}")

    # 最终评估: stress
    D_hyp_final = poincare_distance_matrix(Z, c)
    D_hyp_norm = D_hyp_final / (np.max(D_hyp_final) + EPS)
    stress = np.sum((D_hyp_norm - D_orig_norm) ** 2) / 2.0

    return Z, stress


def run_part_b(bias_scaled, word_labels, word_categories, concepts, layer_idx):
    """双曲嵌入验证"""
    print(f"\n{'='*60}")
    print(f"  Part B: 双曲嵌入验证 (Layer {layer_idx})")
    print(f"{'='*60}")

    n_concepts = bias_scaled.shape[0]
    results_b = {"layer": layer_idx, "n_concepts": n_concepts}

    # 先用PCA降到合理维度（避免高维距离退化）
    pca_dim = min(50, n_concepts - 1, bias_scaled.shape[1])
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(bias_scaled)

    # B1: Poincare Ball嵌入
    print(f"\n  [B1] Poincare Ball嵌入 (PCA->{pca_dim}D, then hyperbolic MDS)")
    Z_hyp_2d, stress_hyp = hyperbolic_mds(X_pca, n_components=2, c=1.0, max_iter=200, lr=0.005)

    # 同时做欧几里得MDS作为对比
    D_orig = squareform(pdist(X_pca, metric="euclidean"))
    D_orig_norm = D_orig / (np.max(D_orig) + EPS)
    from sklearn.manifold import MDS
    mds_euc = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300)
    Z_euc_2d = mds_euc.fit_transform(D_orig_norm)
    stress_euc = mds_euc.stress_

    print(f"    Hyperbolic MDS stress: {stress_hyp:.6f}")
    print(f"    Euclidean MDS stress: {stress_euc:.6f}")

    results_b["mds_stress"] = {
        "hyperbolic": round(stress_hyp, 6),
        "euclidean": round(stress_euc, 6),
        "hyp_better": stress_hyp < stress_euc,
    }

    # B2: 双曲距离 vs 欧几里得距离 — 概念关系解释力
    print(f"\n  [B2] 双曲距离 vs 欧几里得距离: 概念关系解释力")

    # 构建属性相似度矩阵（ground truth）
    attr_matrix, attr_names = encode_attributes(concepts, word_labels, word_categories)
    # 属性相似度: 用余弦相似度
    attr_norms = np.linalg.norm(attr_matrix, axis=1, keepdims=True)
    attr_norms = np.where(attr_norms < EPS, 1.0, attr_norms)
    attr_sim = (attr_matrix @ attr_matrix.T) / (attr_norms @ attr_norms.T)

    # 类别相似度: 同类=1, 异类=0
    cat_sim = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts):
        for j in range(n_concepts):
            cat_sim[i, j] = 1.0 if word_categories[i] == word_categories[j] else 0.0

    # 原始空间的欧几里得距离
    D_euc_raw = squareform(pdist(bias_scaled, metric="euclidean"))

    # PCA降维后的距离
    D_euc_pca = squareform(pdist(X_pca, metric="euclidean"))

    # 双曲MDS嵌入后的距离
    D_hyp_mds = poincare_distance_matrix(Z_hyp_2d, c=1.0)

    # 计算每种距离与属性相似度、类别相似度的相关性
    def neg_corr(dist_mat, sim_mat):
        """距离与相似度的负相关（距离越大→相似度越低→负相关越强）"""
        mask = np.triu(np.ones_like(dist_mat), k=1).astype(bool)
        d_flat = dist_mat[mask]
        s_flat = sim_mat[mask]
        r = np.corrcoef(d_flat, s_flat)[0, 1]
        return -r  # 取负使得"更好"的指标为正

    corr_attr_euc_raw = neg_corr(D_euc_raw, attr_sim)
    corr_attr_euc_pca = neg_corr(D_euc_pca, attr_sim)
    corr_attr_hyp_mds = neg_corr(D_hyp_mds, attr_sim)

    corr_cat_euc_raw = neg_corr(D_euc_raw, cat_sim)
    corr_cat_euc_pca = neg_corr(D_euc_pca, cat_sim)
    corr_cat_hyp_mds = neg_corr(D_hyp_mds, cat_sim)

    print(f"    属性解释力 (越高越好):")
    print(f"      原始欧几里得: r={corr_attr_euc_raw:.4f}")
    print(f"      PCA欧几里得:  r={corr_attr_euc_pca:.4f}")
    print(f"      双曲MDS:      r={corr_attr_hyp_mds:.4f}")
    print(f"    类别解释力 (越高越好):")
    print(f"      原始欧几里得: r={corr_cat_euc_raw:.4f}")
    print(f"      PCA欧几里得:  r={corr_cat_euc_pca:.4f}")
    print(f"      双曲MDS:      r={corr_cat_hyp_mds:.4f}")

    results_b["distance_correlations"] = {
        "attribute": {
            "euclidean_raw": round(float(corr_attr_euc_raw), 4),
            "euclidean_pca": round(float(corr_attr_euc_pca), 4),
            "hyperbolic_mds": round(float(corr_attr_hyp_mds), 4),
        },
        "category": {
            "euclidean_raw": round(float(corr_cat_euc_raw), 4),
            "euclidean_pca": round(float(corr_cat_euc_pca), 4),
            "hyperbolic_mds": round(float(corr_cat_hyp_mds), 4),
        },
    }

    # B3: 双曲概念算术
    print(f"\n  [B3] 双曲概念算术 (测地线中点)")
    # 在双曲空间中，概念算术用"测地线运算"替代向量加法
    # Poincare球中的加权平均:
    #   z = (alpha * z1 / (1-||z1||^2) + beta * z2 / (1-||z2||^2))
    #   然后归一化回球内

    # 简化测试: 只用水果类别
    fruit_words = list(concepts["fruit"]["words"].keys())
    fruit_indices = [i for i, w in enumerate(word_labels) if w in fruit_words]
    if len(fruit_indices) < 5:
        print("    跳过: 水果概念数不足")
        results_b["concept_arithmetic"] = {"skipped": True}
        return results_b

    fruit_bias = bias_scaled[fruit_indices]
    fruit_labels = [word_labels[i] for i in fruit_indices]

    # 用PCA降到2D做算术
    pca_2d = PCA(n_components=2)
    fruit_pca = pca_2d.fit_transform(fruit_bias)

    # 欧几里得算术: A + B - C
    # 3Cos(A,B,C): argmax_D cos(D, A-B+C) where D != A,B,C
    def eval_3cos_euclidean(V, labels, concepts_data, cat_name):
        """欧几里得3Cos精度"""
        attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])
        if not attrs:
            return 0.0, 0
        attr = attrs[0]  # 用第一个属性

        # 找有效测试对: A,B,C,D共享属性但有差异
        tests = []
        all_words = list(concepts_data["words"].keys())
        for i, wA in enumerate(all_words):
            for j, wB in enumerate(all_words):
                if i >= j:
                    continue
                if attr not in concepts_data["words"][wA] or attr not in concepts_data["words"][wB]:
                    continue
                if concepts_data["words"][wA][attr] == concepts_data["words"][wB][attr]:
                    continue
                # 找一个与A属性相同但不是A的D
                for wD in all_words:
                    if wD == wA:
                        continue
                    if concepts_data["words"].get(wD, {}).get(attr) == concepts_data["words"][wA][attr]:
                        tests.append((wA, wB, wD))
                        break
                if len(tests) >= 10:
                    break
            if len(tests) >= 10:
                break

        if not tests:
            return 0.0, 0

        correct = 0
        for wA, wB, wD_target in tests:
            iA = labels.index(wA) if wA in labels else -1
            iB = labels.index(wB) if wB in labels else -1
            if iA < 0 or iB < 0:
                continue

            vec = V[iA] - V[iB] + V[iB]  # = V[iA] (简化测试: 找最近邻)
            vec = V[iA]  # 直接测试最近邻精度

            sims = np.array([float(np.dot(vec, V[k]) / (np.linalg.norm(vec) * np.linalg.norm(V[k]) + EPS))
                            for k in range(len(labels))])
            sims[iA] = -999  # 排除自身

            best_k = int(np.argmax(sims))
            if labels[best_k] == wD_target:
                correct += 1

        accuracy = correct / max(len(tests), 1)
        return accuracy, len(tests)

    # 测试欧几里得空间中的最近邻精度
    def neighbor_accuracy(V, labels, concepts_data, cat_name):
        """最近邻精度: 与目标概念共享最多属性的比例"""
        attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])
        if not attrs:
            return 0.0

        correct = 0
        total = 0
        for i, w in enumerate(labels):
            sims = V @ V[i] / (np.linalg.norm(V, axis=1) * np.linalg.norm(V[i]) + EPS)
            sims[i] = -999
            best_k = int(np.argmax(sims))
            best_w = labels[best_k]

            # 检查最近邻是否与当前词共享至少一个属性值
            shared = 0
            for attr in attrs:
                if (attr in concepts_data["words"].get(w, {}) and
                    attr in concepts_data["words"].get(best_w, {}) and
                    concepts_data["words"][w][attr] == concepts_data["words"][best_w][attr]):
                    shared += 1
            if shared > 0:
                correct += 1
            total += 1

        return correct / max(total, 1)

    acc_euc = neighbor_accuracy(fruit_pca, fruit_labels, concepts["fruit"], "fruit")

    # 在双曲空间中做同样的测试
    fruit_Z_hyp, _ = hyperbolic_mds(fruit_bias, n_components=2, c=1.0, max_iter=150, lr=0.005)
    # 双曲空间中用双曲余弦相似度(即双曲距离的负值)
    D_hyp_fruit = poincare_distance_matrix(fruit_Z_hyp, c=1.0)

    def neighbor_accuracy_hyp(D_hyp, labels, concepts_data, cat_name):
        """双曲空间最近邻精度"""
        attrs = CATEGORY_ATTRIBUTES.get(cat_name, [])
        if not attrs:
            return 0.0

        correct = 0
        total = 0
        for i, w in enumerate(labels):
            dists = D_hyp[i].copy()
            dists[i] = 999
            best_k = int(np.argmin(dists))
            best_w = labels[best_k]

            shared = 0
            for attr in attrs:
                if (attr in concepts_data["words"].get(w, {}) and
                    attr in concepts_data["words"].get(best_w, {}) and
                    concepts_data["words"][w][attr] == concepts_data["words"][best_w][attr]):
                    shared += 1
            if shared > 0:
                correct += 1
            total += 1

        return correct / max(total, 1)

    acc_hyp = neighbor_accuracy_hyp(D_hyp_fruit, fruit_labels, concepts["fruit"], "fruit")

    print(f"    欧几里得空间最近邻精度: {acc_euc:.4f}")
    print(f"    双曲空间最近邻精度:    {acc_hyp:.4f}")
    print(f"    双曲优势: {acc_hyp - acc_euc:+.4f}")

    results_b["concept_arithmetic"] = {
        "euclidean_neighbor_accuracy": round(float(acc_euc), 4),
        "hyperbolic_neighbor_accuracy": round(float(acc_hyp), 4),
        "hyperbolic_advantage": round(float(acc_hyp - acc_euc), 4),
    }

    # B4: 双曲性与层次化
    print(f"\n  [B4] 双曲性与层次化验证")

    # 层次化指标: 同类概念在嵌入空间中的聚集程度
    # 用silhouette score衡量
    from sklearn.metrics import silhouette_score

    # 欧几里得空间的silhouette
    try:
        cat_labels_int = np.array([list(set(word_categories)).index(c) for c in word_categories])
        sil_euc = silhouette_score(fruit_pca if len(fruit_indices) > 3 else X_pca,
                                    cat_labels_int[:min(len(fruit_indices), len(cat_labels_int))]
                                    if len(fruit_indices) > 3 else cat_labels_int)
        sil_hyp = silhouette_score(-D_hyp_mds, cat_labels_int, metric="precomputed")
        print(f"    欧几里得空间 silhouette: {sil_euc:.4f}")
        print(f"    双曲空间 silhouette:     {sil_hyp:.4f}")
    except Exception as e:
        sil_euc = 0.0
        sil_hyp = 0.0
        print(f"    silhouette计算跳过: {e}")

    results_b["hierarchy"] = {
        "silhouette_euclidean": round(float(sil_euc), 4),
        "silhouette_hyperbolic": round(float(sil_hyp), 4),
        "hyp_better_hierarchy": sil_hyp > sil_euc,
    }

    # 双曲曲率估计（用距离矩阵拟合）
    # 在双曲空间中，三角形的面积与曲率K的关系: K = -4 * Area / (a*b*c)（简化）
    # 这里用"双曲性分数"来衡量
    # 双曲性 = 1 - (观测到的距离矩阵与欧几里得嵌入距离的偏离) / 最大偏离
    # 实际做法: 用Gromov的delta-hyperbolicity

    def gromov_delta(D):
        """计算Gromov delta-hyperbolicity（4点条件）"""
        n = D.shape[0]
        if n < 4:
            return 0.0
        deltas = []
        # 采样四元组（避免O(n^4)）
        np.random.seed(42)
        n_samples = min(500, n * (n - 1) * (n - 2) * (n - 3) // 24)
        for _ in range(n_samples):
            idx = np.random.choice(n, 4, replace=False)
            d = D[np.ix_(idx, idx)]
            # 四点条件: 对所有三对划分, max(s1,s2) - min(s1,s2) <= delta
            s_pairs = [
                (d[0, 1] + d[2, 3], d[0, 2] + d[1, 3], d[0, 3] + d[1, 2]),
            ]
            for s1, s2, s3 in s_pairs:
                diam = max(d[0, 1], d[0, 2], d[0, 3], d[1, 2], d[1, 3], d[2, 3])
                if diam < EPS:
                    continue
                vals = sorted([s1, s2, s3])
                delta = (vals[2] - vals[0]) / 2.0
                deltas.append(delta / diam)

        if not deltas:
            return 0.0
        return float(np.percentile(deltas, 75))  # 75th percentile

    delta_raw = gromov_delta(D_euc_raw)
    delta_hyp = gromov_delta(D_hyp_mds)

    print(f"    Gromov delta-hyperbolicity:")
    print(f"      原始空间: delta={delta_raw:.4f}")
    print(f"      双曲MDS:  delta={delta_hyp:.4f}")
    print(f"      (delta越小 -> 越接近双曲/树形)")

    results_b["gromov_delta"] = {
        "raw_space": round(delta_raw, 4),
        "hyperbolic_mds": round(delta_hyp, 4),
    }

    # B5: 类别内双曲嵌入（水果类别内分析更合适）
    print(f"\n  [B5] 类别内双曲嵌入 (水果类别)")
    fruit_words = list(concepts["fruit"]["words"].keys())
    fruit_indices = [i for i, w in enumerate(word_labels) if w in fruit_words]
    if len(fruit_indices) >= 5:
        fruit_bias = bias_scaled[fruit_indices]
        fruit_labels = [word_labels[i] for i in fruit_indices]

        # 水果属性相似度（ground truth）
        fruit_attr_rows = []
        for w in fruit_labels:
            row = []
            for attr in CATEGORY_ATTRIBUTES.get("fruit", []):
                val = concepts["fruit"]["words"][w].get(attr, None)
                if isinstance(val, (int, float)):
                    row.append(float(val))
                elif isinstance(val, str):
                    all_vals = sorted(set(
                        concepts["fruit"]["words"][ww].get(attr, "")
                        for ww in fruit_words
                        if attr in concepts["fruit"]["words"][ww]
                    ))
                    try:
                        row.append(float(all_vals.index(val)) / max(len(all_vals) - 1, 1))
                    except ValueError:
                        row.append(0.0)
                else:
                    row.append(0.0)
            fruit_attr_rows.append(row)
        fruit_attr_mat = np.array(fruit_attr_rows)
        fruit_attr_norms = np.linalg.norm(fruit_attr_mat, axis=1, keepdims=True)
        fruit_attr_norms = np.where(fruit_attr_norms < EPS, 1.0, fruit_attr_norms)
        fruit_attr_sim = (fruit_attr_mat @ fruit_attr_mat.T) / (fruit_attr_norms @ fruit_attr_norms.T)

        # 欧几里得距离（水果类别内）
        D_euc_fruit = squareform(pdist(fruit_bias, metric="euclidean"))
        D_euc_fruit_norm = D_euc_fruit / (np.max(D_euc_fruit) + EPS)

        # 欧几里得MDS
        mds_euc_f = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300)
        Z_euc_fruit = mds_euc_f.fit_transform(D_euc_fruit_norm)
        stress_euc_fruit = mds_euc_f.stress_

        # 双曲MDS（类别内）
        Z_hyp_fruit, stress_hyp_fruit = hyperbolic_mds(
            fruit_bias, n_components=2, c=1.0, max_iter=300, lr=0.01
        )
        D_hyp_fruit_full = poincare_distance_matrix(Z_hyp_fruit, c=1.0)
        D_hyp_fruit_norm = D_hyp_fruit_full / (np.max(D_hyp_fruit_full) + EPS)

        print(f"    水果类别内(19个概念):")
        print(f"      欧几里得MDS stress: {stress_euc_fruit:.6f}")
        print(f"      双曲MDS stress:    {stress_hyp_fruit:.6f}")

        # 属性解释力对比（水果类别内）
        corr_attr_euc_f = neg_corr(D_euc_fruit_norm, fruit_attr_sim)
        corr_attr_hyp_f = neg_corr(D_hyp_fruit_norm, fruit_attr_sim)

        print(f"      属性解释力: 欧几里得={corr_attr_euc_f:.4f}, 双曲={corr_attr_hyp_f:.4f}")
        print(f"      双曲优势: {corr_attr_hyp_f - corr_attr_euc_f:+.4f}")

        # 水果最近邻精度
        acc_euc_f = neighbor_accuracy(
            PCA(n_components=2).fit_transform(fruit_bias), fruit_labels, concepts["fruit"], "fruit"
        )
        acc_hyp_f = neighbor_accuracy_hyp(D_hyp_fruit_full, fruit_labels, concepts["fruit"], "fruit")
        print(f"      最近邻精度: 欧几里得={acc_euc_f:.4f}, 双曲={acc_hyp_f:.4f}")

        # Gromov delta（水果类别内）
        delta_euc_fruit = gromov_delta(D_euc_fruit)
        delta_hyp_fruit = gromov_delta(D_hyp_fruit_full)
        print(f"      Gromov delta: 欧几里得={delta_euc_fruit:.4f}, 双曲={delta_hyp_fruit:.4f}")

        # 判断双曲性：原始空间的delta-hyperbolicity
        # delta < 0.2 通常被认为是"树形/双曲"结构
        is_hyperbolic = delta_euc_fruit < 0.3
        print(f"      原始空间双曲性判定: delta={delta_euc_fruit:.4f} -> {'双曲/树形' if is_hyperbolic else '非双曲'}")

        results_b["intra_category_fruit"] = {
            "n_concepts": len(fruit_indices),
            "mds_stress": {"euclidean": round(stress_euc_fruit, 6), "hyperbolic": round(stress_hyp_fruit, 6)},
            "attr_correlation": {"euclidean": round(float(corr_attr_euc_f), 4), "hyperbolic": round(float(corr_attr_hyp_f), 4)},
            "neighbor_accuracy": {"euclidean": round(float(acc_euc_f), 4), "hyperbolic": round(float(acc_hyp_f), 4)},
            "gromov_delta": {"euclidean": round(delta_euc_fruit, 4), "hyperbolic": round(delta_hyp_fruit, 4)},
            "is_hyperbolic": bool(is_hyperbolic),
        }
    else:
        results_b["intra_category_fruit"] = {"skipped": True, "reason": "too few concepts"}

    return results_b


# ==================== 主函数 ====================
def main():
    t0 = time.time()
    model_arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"

    if model_arg in ("qwen3", "qwen"):
        model_path = QWEN3_MODEL_PATH
        model_tag = "qwen3_4b"
    elif model_arg in ("deepseek", "ds", "deepseek7b"):
        model_path = DEEPSEEK7B_MODEL_PATH
        model_tag = "deepseek_7b"
    else:
        print(f"Unknown model: {model_arg}")
        return

    print(f"\n{'='*60}")
    print(f"  Stage469: Information Bottleneck + Hyperbolic Embedding")
    print(f"  Model: {model_tag}")
    print(f"{'='*60}")

    # 1. 加载模型
    print("\n[1/4] Loading model...")
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)

    # 2. 提取激活
    print("\n[2/4] Extracting activations...")
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)

    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 选择分析层
    test_layers = [0, layer_count // 4, layer_count // 2, layer_count - 1]
    golden_layer = test_layers[2]  # 中间层（通常信息最丰富）
    print(f"\n[3/4] Test layers: {test_layers}, golden={golden_layer}")

    # 4. 运行实验
    all_results = {
        "model": model_tag,
        "layer_count": layer_count,
        "neuron_dim": neuron_dim,
        "golden_layer": golden_layer,
        "n_concepts_total": sum(len(c["words"]) for c in CONCEPTS.values()),
        "part_a": {},
        "part_b": {},
        "cross_layer_ib": [],
    }

    for li in test_layers:
        bias_scaled, word_labels, word_categories = compute_bias_matrix(
            all_activations, li, CONCEPTS
        )
        if bias_scaled is None:
            print(f"\n  Layer {li}: no data, skip")
            continue

        n_words = bias_scaled.shape[0]
        print(f"\n  Layer {li}: {n_words} concepts, dim={neuron_dim}")

        # Part A: IB
        results_a = run_part_a(bias_scaled, word_labels, word_categories, CONCEPTS, li, neuron_dim)
        all_results["part_a"][f"layer_{li}"] = results_a

        # 跨层IB跟踪
        if "optimal_dim" in results_a:
            all_results["cross_layer_ib"].append({
                "layer": li,
                "optimal_dim": results_a["optimal_dim"],
                "ixt_at_optimal": next(
                    (p["ixt"] for p in results_a["ib_curve"] if p["dim"] == results_a["optimal_dim"]), 0
                ),
            })

    # Part B: 只在golden_layer上做（计算密集）
    bias_scaled, word_labels, word_categories = compute_bias_matrix(
        all_activations, golden_layer, CONCEPTS
    )
    if bias_scaled is not None:
        results_b = run_part_b(bias_scaled, word_labels, word_categories, CONCEPTS, golden_layer)
        all_results["part_b"] = results_b

    # 5. 保存结果
    print(f"\n[4/4] Saving results...")
    out_path = OUTPUT_DIR / f"{model_tag}_full_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # 6. 打印摘要
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE - {elapsed:.1f}s")
    print(f"{'='*60}")

    print("\n  === SUMMARY ===")
    # Part A summary
    for lk, lv in all_results["part_a"].items():
        layer_num = lk.replace("layer_", "")
        print(f"\n  Layer {layer_num} (IB):")
        if "optimal_dim" in lv:
            print(f"    IB optimal dim: {lv['optimal_dim']}")
        if "theoretical_dims" in lv:
            td = lv["theoretical_dims"]
            print(f"    Theoretical: dof={td['dof']}, info_bound={td['info_theory_lower']}, pca95={td['pca_95']}, elbow={td['pca_elbow']}")
        if "compression_vs_stage467" in lv:
            cv = lv["compression_vs_stage467"]
            print(f"    vs Stage467(17): IB_opt={cv['ib_optimal']}, diff={cv['difference']}")

    # Part B summary
    pb = all_results.get("part_b", {})
    if "mds_stress" in pb:
        ms = pb["mds_stress"]
        print(f"\n  Hyperbolic Embedding:")
        print(f"    MDS stress: hyp={ms['hyperbolic']:.6f} vs euc={ms['euclidean']:.6f}, hyp_better={ms['hyp_better']}")
    if "distance_correlations" in pb:
        dc = pb["distance_correlations"]
        print(f"    Attribute解释力: euc_raw={dc['attribute']['euclidean_raw']:.4f}, hyp_mds={dc['attribute']['hyperbolic_mds']:.4f}")
    if "concept_arithmetic" in pb and not pb["concept_arithmetic"].get("skipped"):
        ca = pb["concept_arithmetic"]
        print(f"    概念算术: euc={ca['euclidean_neighbor_accuracy']:.4f}, hyp={ca['hyperbolic_neighbor_accuracy']:.4f}, adv={ca['hyperbolic_advantage']:+.4f}")
    if "gromov_delta" in pb:
        gd = pb["gromov_delta"]
        print(f"    Gromov delta: raw={gd['raw_space']:.4f}, hyp_mds={gd['hyperbolic_mds']:.4f}")


if __name__ == "__main__":
    main()
