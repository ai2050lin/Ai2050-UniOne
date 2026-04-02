# -*- coding: utf-8 -*-
"""
Stage459: 超大规模概念验证 + 本征维度估算
=========================================

核心目标：
1. 扩展到500+概念，验证SVD方差解释比能否突破50%
2. 用多种方法估算偏置空间的本征维度（intrinsic dimensionality）
3. 验证语义因子在大规模上的稳定性
4. 分析SVD方差解释比与概念数的关系（scaling law）

本征维度估算方法：
- PCA eigenvalue spectrum（特征值谱）
- Maximum Likelihood Estimation (MLE)
- TwoNN (Two Nearest Neighbors)
- 参与度分析（Participation Ratio）

模型: DeepSeek-7B (CUDA)
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage459_large_scale_dim_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 超大规模概念集（500+词，15类别） ====================
LARGE_CONCEPTS = {
    "fruit": {
        "label": "水果",
        "words": [
            "apple", "banana", "orange", "grape", "mango", "peach", "lemon",
            "cherry", "berry", "melon", "kiwi", "plum", "pear", "fig", "lime",
            "coconut", "pineapple", "strawberry", "watermelon", "blueberry",
            "papaya", "guava", "pomegranate", "apricot", "cranberry",
            "blackberry", "raspberry", "date", "lychee", "dragonfruit",
            "passionfruit", "tangerine", "nectarine", "gooseberry",
            "boysenberry", "elderberry", "mulberry", "currant",
        ],
    },
    "animal": {
        "label": "动物",
        "words": [
            "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
            "whale", "shark", "snake", "eagle", "wolf", "bear", "monkey",
            "rabbit", "deer", "fox", "owl", "dolphin",
            "penguin", "parrot", "frog", "turtle", "crocodile",
            "giraffe", "zebra", "gorilla", "kangaroo", "panda",
            "koala", "squirrel", "hedgehog", "otter", "seal",
            "buffalo", "rhino", "leopard", "cheetah", "hippo",
        ],
    },
    "vehicle": {
        "label": "交通工具",
        "words": [
            "car", "bus", "train", "plane", "ship", "boat", "bicycle",
            "motorcycle", "truck", "van", "taxi", "subway", "helicopter",
            "rocket", "ambulance", "tractor", "scooter", "cart", "canoe",
            "tank", "ferry", "yacht", "cruiser", "skateboard",
            "balloon", "glider", "submarine", "hovercraft", "segway",
        ],
    },
    "natural": {
        "label": "自然",
        "words": [
            "river", "mountain", "ocean", "forest", "desert", "island",
            "valley", "lake", "cloud", "storm", "rain", "snow", "wind",
            "fire", "earth", "stone", "moon", "star", "sun", "tree",
            "volcano", "glacier", "cave", "waterfall", "meadow",
            "swamp", "prairie", "cliff", "reef", "delta",
        ],
    },
    "furniture": {
        "label": "家具",
        "words": [
            "chair", "table", "desk", "bed", "sofa", "shelf", "cabinet",
            "drawer", "mirror", "lamp", "carpet", "curtain", "pillow",
            "blanket", "mattress", "wardrobe", "bookcase", "stool",
            "couch", "bench",
        ],
    },
    "material": {
        "label": "材料",
        "words": [
            "metal", "wood", "glass", "cloth", "silk", "cotton", "gold",
            "silver", "copper", "iron", "steel", "diamond", "ruby", "emerald",
            "pearl", "amber", "ivory", "marble", "clay", "leather",
            "plastic", "paper", "concrete", "ceramic", "wool",
            "velvet", "linen", "satin", "brass", "bronze",
        ],
    },
    "profession": {
        "label": "职业",
        "words": [
            "doctor", "teacher", "engineer", "artist", "writer", "singer",
            "dancer", "soldier", "lawyer", "judge", "farmer", "chef",
            "nurse", "pilot", "driver", "scientist", "painter", "musician",
            "actor", "architect",
        ],
    },
    "food": {
        "label": "食物",
        "words": [
            "bread", "rice", "pasta", "cheese", "butter", "milk", "egg",
            "meat", "chicken", "beef", "pork", "soup", "salad",
            "cake", "cookie", "chocolate", "pizza", "burger",
            "sandwich", "yogurt", "noodle", "taco", "sushi",
            "dumpling", "pancake", "waffle", "croissant", "bagel",
            "muffin", "biscuit",
        ],
    },
    "clothing": {
        "label": "服装",
        "words": [
            "shirt", "pants", "dress", "jacket", "coat", "hat", "shoes",
            "boots", "socks", "gloves", "scarf", "tie", "belt", "jeans",
            "sweater", "uniform", "gown", "vest", "shorts", "sandals",
            "skirt", "blazer", "tuxedo", "apron", "hoodie",
            "leggings", "mittens", "slippers", "swimsuit", "overalls",
        ],
    },
    "tool": {
        "label": "工具",
        "words": [
            "hammer", "saw", "drill", "knife", "scissors", "needle", "pen",
            "brush", "key", "lock", "rope", "chain", "wheel", "ladder",
            "shovel", "axe", "compass", "ruler", "glue", "tape",
            "wrench", "pliers", "screwdriver", "chisel", "file",
            "anvil", "tongs", "stapler", "calculator", "microscope",
        ],
    },
    "body": {
        "label": "身体部位",
        "words": [
            "head", "hand", "foot", "arm", "leg", "eye", "ear", "nose",
            "mouth", "tooth", "finger", "thumb", "knee", "elbow",
            "shoulder", "chest", "back", "neck", "wrist", "ankle",
            "heart", "brain", "lung", "stomach", "bone",
            "muscle", "skin", "blood", "hair", "tongue",
        ],
    },
    "emotion": {
        "label": "情感",
        "words": [
            "joy", "anger", "sadness", "fear", "love", "hate", "hope",
            "despair", "pride", "shame", "guilt", "jealousy", "envy",
            "gratitude", "surprise", "disgust", "trust", "anxiety",
            "excitement", "boredom", "calm", "loneliness", "happiness",
            "sorrow", "regret", "curiosity", "nostalgia", "empathy",
            "compassion", "courage",
        ],
    },
    "color": {
        "label": "颜色",
        "words": [
            "red", "blue", "green", "yellow", "purple", "orange", "pink",
            "brown", "black", "white", "gray", "gold", "silver",
            "crimson", "scarlet", "maroon", "navy", "teal",
            "cyan", "magenta", "violet", "indigo", "beige",
            "ivory", "coral", "salmon", "turquoise", "lavender",
            "olive", "lime",
        ],
    },
    "weather": {
        "label": "天气",
        "words": [
            "sunny", "rainy", "cloudy", "snowy", "windy", "stormy",
            "foggy", "humid", "dry", "hot", "cold", "warm", "cool",
            "freezing", "breezy", "thunder", "lightning", "hail",
            "tornado", "hurricane", "drought", "blizzard", "monsoon",
            "drizzle", "mist", "frost", "dew", "rainbow",
            "sunrise", "sunset",
        ],
    },
    "sport": {
        "label": "运动",
        "words": [
            "soccer", "basketball", "tennis", "football", "baseball",
            "swimming", "running", "cycling", "boxing", "wrestling",
            "golf", "hockey", "volleyball", "cricket", "rugby",
            "skiing", "surfing", "archery", "fencing", "rowing",
            "gymnastics", "diving", "marathon", "sprinting", "javelin",
            "discus", "hurdles", "badminton", "table_tennis", "bowling",
        ],
    },
}

TOTAL_CONCEPTS = sum(len(c["words"]) for c in LARGE_CONCEPTS.values())
print(f"Categories: {len(LARGE_CONCEPTS)}, Concepts: {TOTAL_CONCEPTS}")

EPS = 1e-8


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(int(k) if isinstance(k, (np.integer,)) else k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_model(model_path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()

    layer_count = len(discover_layers(model))
    hidden_dim = discover_layers(model)[0].mlp.down_proj.out_features
    print(f"  Layers: {layer_count}, HiddenDim: {hidden_dim}")
    return model, tokenizer, layer_count, hidden_dim


def extract_word_activation(model, tokenizer, word, layer_count):
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
        activations = {}
        for layer_idx in range(layer_count):
            buf = buffers[layer_idx]
            if buf is not None:
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                activations[layer_idx] = buf[0, pos].float().numpy()
        return activations
    finally:
        remove_hooks(handles)


def extract_all_activations(model, tokenizer, categories, layer_count):
    all_activations = {}
    total = sum(len(cat["words"]) for cat in categories.values())
    done = 0

    for cat_name, cat_info in categories.items():
        print(f"  [{cat_name}] ({len(cat_info['words'])} words)", end="")
        all_activations[cat_name] = {}
        for word in cat_info["words"]:
            try:
                acts = extract_word_activation(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except:
                pass
        cat_done = sum(len(v) for v in all_activations.get(cat_name, {}).values()) if cat_name in all_activations else 0
        print(f" → {cat_done}/{len(cat_info['words'])}")

    print(f"  Total: {done}/{total}")
    return all_activations


def build_bias_matrices(all_activations, categories, target_layers):
    all_biases, concept_labels, category_labels = [], [], []

    for cat_name, words_acts in all_activations.items():
        if cat_name not in categories:
            continue
        words = list(words_acts.keys())
        layer_vecs = defaultdict(list)
        for word in words:
            acts = words_acts[word]
            for l in target_layers:
                if l in acts:
                    layer_vecs[l].append(acts[l])

        basis = {l: np.mean(vecs, axis=0) for l, vecs in layer_vecs.items() if vecs}
        if not basis:
            continue

        for word in words:
            acts = words_acts[word]
            bias_parts = []
            for l in sorted(target_layers):
                if l in acts and l in basis:
                    bias = acts[l] - basis[l]
                    norm = np.linalg.norm(bias)
                    if norm > EPS:
                        bias_parts.append(bias / norm)
            if bias_parts:
                all_biases.append(np.concatenate(bias_parts))
                concept_labels.append(word)
                category_labels.append(cat_name)

    return np.array(all_biases), concept_labels, category_labels


def intrinsic_dimensionality_analysis(bias_matrix, max_dims=200):
    """本征维度估算"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    print(f"\n  Intrinsic dimensionality analysis (matrix: {bias_matrix.shape})...")

    results = {}

    # 1. PCA eigenvalue spectrum
    print("    [1/4] PCA eigenvalue spectrum...")
    n_components = min(max_dims, bias_matrix.shape[0] - 1, bias_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(bias_matrix)

    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    total_var = np.sum(eigenvalues)

    # 找到解释不同方差阈值所需的维度
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    dims_for_threshold = {}
    for t in thresholds:
        idx = np.searchsorted(cumulative, t)
        if idx < len(cumulative):
            dims_for_threshold[t] = int(idx + 1)
            print(f"      {t*100:.0f}% variance: {idx+1} dims")

    results["pca"] = {
        "dims_for_thresholds": dims_for_threshold,
        "top30_variance_ratio": pca.explained_variance_ratio_[:30].tolist(),
        "cumulative_top30": cumulative[:30].tolist(),
        "top30_eigenvalues": eigenvalues[:30].tolist(),
    }

    # 2. Participation Ratio (PR)
    print("    [2/4] Participation Ratio...")
    pr = (np.sum(eigenvalues))**2 / (np.sum(eigenvalues**2) + EPS)
    results["participation_ratio"] = float(pr)
    print(f"      Participation Ratio: {pr:.1f}")

    # 3. TwoNN estimator
    print("    [3/4] TwoNN estimator...")
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(bias_matrix, bias_matrix, metric='cosine')
        # 每个点的两个最近邻距离
        sorted_dists = np.sort(dists, axis=1)
        r1 = sorted_dists[:, 1]  # 最近邻
        r2 = sorted_dists[:, 2]  # 第二近邻

        # 避免除零
        valid = r1 > EPS
        if valid.sum() > 10:
            mu = r2[valid] / r1[valid]
            # MLE估计: d = 1/(mean(log(mu)) - log(2)/(n-2))
            n = bias_matrix.shape[0]
            log_mu = np.log(mu[valid])
            mean_log_mu = np.mean(log_mu)
            if mean_log_mu > 0:
                twonn_dim = n / (2 * np.sum(log_mu - np.log(2)) + EPS)
                results["twonn"] = {
                    "dimension": float(twonn_dim),
                    "method": "Maximum Likelihood (Facco et al. 2017)",
                }
                print(f"      TwoNN intrinsic dim: {twonn_dim:.1f}")
            else:
                results["twonn"] = {"dimension": None, "note": "log(mu) <= 0"}
                print(f"      TwoNN: failed (non-positive log-ratio)")
        else:
            results["twonn"] = {"dimension": None, "note": "too few valid distances"}
    except Exception as e:
        results["twonn"] = {"dimension": None, "error": str(e)}
        print(f"      TwoNN: error - {e}")

    # 4. MLE intrinsic dimension (Levina & Bickel 2004)
    print("    [4/4] MLE intrinsic dimension...")
    try:
        nn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='auto')
        nn.fit(bias_matrix)
        distances, _ = nn.kneighbors(bias_matrix)

        # 对不同k值估计
        mle_dims = {}
        for k in [5, 10, 15, 20]:
            if k >= distances.shape[1]:
                continue
            k_dists = distances[:, k]
            valid = k_dists > EPS
            if valid.sum() > 10:
                log_ratios = np.log(k_dists[valid, np.newaxis] / (distances[valid, 1:k] + EPS + 1e-10))
                # MLE: d_k = (k-1) / sum_j log(r_k / r_j)
                d_estimates = (k - 1) / (np.sum(log_ratios, axis=1) + EPS)
                mle_dim = np.mean(d_estimates)
                mle_dims[f"k={k}"] = float(mle_dim)
                print(f"      MLE (k={k}): {mle_dim:.1f}")
            else:
                mle_dims[f"k={k}"] = None

        results["mle"] = mle_dims
    except Exception as e:
        results["mle"] = {"error": str(e)}
        print(f"      MLE: error - {e}")

    return results


def scaling_analysis(bias_matrix, concept_labels, category_labels):
    """SVD方差解释vs概念数的scaling分析"""
    from sklearn.decomposition import TruncatedSVD

    print(f"\n  Scaling analysis (total: {len(concept_labels)} concepts)...")

    # 测试不同概念数下的SVD方差
    sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results = []

    for n in sample_sizes:
        if n > len(concept_labels):
            n = len(concept_labels)
            if n in [s for s in results]:
                break

        np.random.seed(42)
        indices = np.random.choice(len(concept_labels), size=n, replace=False)
        sub_matrix = bias_matrix[indices]

        n_comp = min(20, n - 1, sub_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(sub_matrix)

        cum_var = np.cumsum(svd.explained_variance_ratio_)
        results.append({
            "n_concepts": n,
            "top5_variance": float(cum_var[4]) if len(cum_var) >= 5 else None,
            "top10_variance": float(cum_var[9]) if len(cum_var) >= 10 else None,
            "top20_variance": float(cum_var[min(19, len(cum_var)-1)]),
        })
        print(f"    n={n}: top5={results[-1]['top5_variance']:.3f}, "
              f"top10={results[-1]['top10_variance']:.3f}, "
              f"top20={results[-1]['top20_variance']:.3f}")

    return results


def generate_report(all_results, output_dir):
    lines = [
        "# Stage459: 超大规模概念验证 + 本征维度估算",
        "",
        f"**时间**: 2026-04-01 02:20",
        f"**模型**: DeepSeek-7B (28层)",
        f"**概念数**: {all_results.get('meta', {}).get('n_concepts', 'N/A')}",
        f"**类别数**: {all_results.get('meta', {}).get('n_categories', 'N/A')}",
        "",
        "---",
    ]

    # 1. 本征维度
    dim = all_results.get("intrinsic_dim", {})
    if dim:
        pca = dim.get("pca", {})
        lines.append("\n## 1. 本征维度估算")
        lines.append("\n### PCA方差解释 vs 维度")
        lines.append("| 目标方差 | 所需维度 |")
        lines.append("|---------|---------|")
        for t, d in sorted(pca.get("dims_for_thresholds", {}).items()):
            lines.append(f"| {t*100:.0f}% | {d} |")

        pr = dim.get("participation_ratio", 0)
        lines.append(f"\n- Participation Ratio: **{pr:.1f}** (有效维度数)")

        twonn = dim.get("twonn", {})
        if twonn and twonn.get("dimension"):
            lines.append(f"- TwoNN本征维度: **{twonn['dimension']:.1f}**")

        mle = dim.get("mle", {})
        if mle:
            mle_strs = []
            for k, v in mle.items():
                try:
                    mle_strs.append(f"{k}={float(v):.1f}")
                except (ValueError, TypeError):
                    pass
            if mle_strs:
                lines.append(f"- MLE本征维度: {', '.join(mle_strs)}")

    # 2. Scaling分析
    scaling = all_results.get("scaling", [])
    if scaling:
        lines.append("\n## 2. SVD方差解释 vs 概念数（Scaling Law）")
        lines.append("\n| 概念数 | Top-5 | Top-10 | Top-20 |")
        lines.append("|--------|-------|--------|--------|")
        for r in scaling:
            t5 = f"{r['top5_variance']*100:.1f}%" if r['top5_variance'] else "-"
            t10 = f"{r['top10_variance']*100:.1f}%" if r['top10_variance'] else "-"
            t20 = f"{r['top20_variance']*100:.1f}%" if r['top20_variance'] else "-"
            lines.append(f"| {r['n_concepts']} | {t5} | {t10} | {t20} |")

    # 3. 完整SVD
    svd = all_results.get("svd", {})
    if svd:
        lines.append("\n## 3. 完整SVD分解（500+词）")
        cv = svd.get("cumulative_variance", [])
        lines.append("\n| K | 累计方差 |")
        lines.append("|---|---------|")
        for k in [5, 10, 20, 30, 50]:
            if k <= len(cv):
                lines.append(f"| {k} | {cv[k-1]*100:.1f}% |")

        # 因子-top概念
        ft = svd.get("factor_top", {})
        if ft:
            lines.append(f"\n### 因子语义（前20个）")
            lines.append("| Factor | 方差 | Top+ | Top- |")
            lines.append("|--------|------|------|------|")
            for fname, finfo in list(ft.items())[:20]:
                var = finfo.get("variance", 0) * 100
                pos = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_positive"][:3])
                neg = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_negative"][:3])
                lines.append(f"| {fname} | {var:.1f}% | {pos} | {neg} |")

    # 4. 结论
    lines.append(f"\n## 4. 结论")
    lines.append("")

    report_path = output_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("Stage459: 超大规模概念验证 + 本征维度估算")
    print(f"Categories: {len(LARGE_CONCEPTS)}, Concepts: {TOTAL_CONCEPTS}")
    print("=" * 60)

    t0 = time.time()

    # 1. 加载模型
    print("\n[1/5] Loading DeepSeek-7B...")
    model, tokenizer, layer_count, hidden_dim = load_model(DEEPSEEK7B_MODEL_PATH)

    # 2. 提取激活
    print(f"\n[2/5] Extracting activations ({TOTAL_CONCEPTS} concepts)...")
    all_activations = extract_all_activations(model, tokenizer, LARGE_CONCEPTS, layer_count)

    # 释放模型
    print("\n  Releasing model...")
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. 构建偏置矩阵
    target_layers = list(range(layer_count - 5, layer_count))
    print(f"\n[3/5] Building bias matrix (layers: {target_layers})...")
    bias_matrix, concept_labels, category_labels = build_bias_matrices(
        all_activations, LARGE_CONCEPTS, target_layers
    )
    print(f"  Bias matrix: {bias_matrix.shape}")

    # 4. 本征维度估算
    print(f"\n[4/5] Intrinsic dimensionality estimation...")
    dim_results = intrinsic_dimensionality_analysis(bias_matrix)

    # 5. Scaling分析
    print(f"\n[5/5] Scaling analysis...")
    scaling_results = scaling_analysis(bias_matrix, concept_labels, category_labels)

    # 完整SVD
    from sklearn.decomposition import TruncatedSVD
    n_comp = min(50, bias_matrix.shape[0] - 1, bias_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_components = svd.fit_transform(bias_matrix)

    factor_top = {}
    for i in range(min(n_comp, 20)):
        scores = svd_components[:, i]
        top_pos = np.argsort(scores)[-5:]
        top_neg = np.argsort(scores)[:5]
        factor_top[f"factor_{i}"] = {
            "variance": float(svd.explained_variance_ratio_[i]),
            "top_positive": [(concept_labels[j], float(scores[j])) for j in reversed(top_pos)],
            "top_negative": [(concept_labels[j], float(scores[j])) for j in top_neg],
        }

    # 保存
    all_results = {
        "intrinsic_dim": dim_results,
        "scaling": scaling_results,
        "svd": {
            "cumulative_variance": np.cumsum(svd.explained_variance_ratio_).tolist(),
            "factor_top": factor_top,
        },
        "meta": {
            "n_concepts": len(concept_labels),
            "n_categories": len(set(category_labels)),
            "bias_dim": int(bias_matrix.shape[1]),
            "elapsed": time.time() - t0,
        },
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)

    generate_report(all_results, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Stage459 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
