# -*- coding: utf-8 -*-
"""
Stage460: 80维高阶SVD因子分析 — 双模型交叉验证
=================================================

核心背景：
  Stage459发现偏置空间本征维度≈35-81，PR=80.9
  但之前只用20维SVD，只看到了冰山一角（22.4%方差）
  现在扩展到80维，发现更多高阶语义因子

核心目标：
1. 80维SVD因子分解，解释50%+方差
2. 80个因子的完整属性关联分析（eta²）
3. 双模型(Qwen3-4B + DeepSeek-7B)对比验证
4. 高阶因子语义发现（第21-80个因子是什么？）
5. 因子层次聚类（发现因子间的层次结构）
6. 80维重建精度测试

模型: Qwen3-4B + DeepSeek-7B (CUDA)
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
    QWEN3_MODEL_PATH,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage460_high_dim_factors_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 大规模概念集（复用Stage459的447词） ====================
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
        "attrs": {"color": 1, "size": 1, "taste": 1, "sweetness": 1},
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
        "attrs": {"size": 1, "speed": 1, "domestic": 1, "habitat": 1},
    },
    "vehicle": {
        "label": "交通工具",
        "words": [
            "car", "bus", "train", "plane", "ship", "bicycle", "motorcycle",
            "truck", "helicopter", "rocket", "boat", "submarine", "tractor",
            "van", "taxi", "ambulance", "firetruck", "scooter", "tram", "cable_car",
            "ferry", "cruiser", "yacht", "canoe", "kayak",
            "skateboard", "rollerskate", "tank", "jet", "glider",
            "blimp", "hovercraft", "spaceship", "satellite", "wagon",
            "sedan", "suv", "pickup", "minivan", "convertible",
        ],
        "attrs": {"speed": 1, "medium": 1, "size": 1},
    },
    "profession": {
        "label": "职业",
        "words": [
            "doctor", "nurse", "teacher", "engineer", "lawyer", "chef", "artist",
            "musician", "writer", "painter", "scientist", "programmer", "pilot",
            "soldier", "firefighter", "police", "farmer", "baker", "butcher", "driver",
            "architect", "carpenter", "plumber", "electrician", "mechanic",
            "surgeon", "dentist", "pharmacist", "veterinarian", "therapist",
            "journalist", "photographer", "designer", "accountant", "manager",
            "director", "producer", "conductor", "librarian", "judge",
        ],
        "attrs": {"domain": 1, "creativity": 1, "social": 1},
    },
    "clothing": {
        "label": "服装",
        "words": [
            "shirt", "pants", "dress", "jacket", "shoes", "hat", "socks",
            "gloves", "scarf", "tie", "belt", "coat", "sweater", "boots",
            "sandals", "uniform", "jeans", "shorts", "skirt", "blazer",
            "apron", "mittens", "slippers", "cardigan", "vest",
            "tuxedo", "briefs", "bra", "camisole", "corset",
        ],
        "attrs": {"warmth": 1, "body_part": 1, "formality": 1},
    },
    "furniture": {
        "label": "家具",
        "words": [
            "chair", "table", "bed", "sofa", "desk", "bookcase", "cabinet",
            "wardrobe", "dresser", "shelf", "stool", "bench", "lamp", "rug",
            "curtain", "mirror", "couch", "armchair", "ottoman", "crib",
        ],
        "attrs": {"softness": 1, "size": 1, "room": 1},
    },
    "food": {
        "label": "食物",
        "words": [
            "bread", "rice", "cake", "cookie", "pizza", "pasta", "soup",
            "salad", "sandwich", "steak", "chicken", "egg", "cheese", "butter",
            "milk", "yogurt", "ice_cream", "chocolate", "candy", "waffle",
            "pancake", "muffin", "croissant", "bagel", "taco",
            "sushi", "dumpling", "noodle", "porridge", "biscuit",
        ],
        "attrs": {"taste": 1, "temperature": 1, "texture": 1},
    },
    "tool": {
        "label": "工具",
        "words": [
            "hammer", "saw", "drill", "wrench", "screwdriver", "pliers",
            "scissors", "knife", "axe", "chisel", "file", "ruler",
            "tape_measure", "level", "compass", "pliers", "wrench",
            "spanner", "clamp", "vise",
        ],
        "attrs": {"precision": 1, "power": 1, "portability": 1},
    },
    "material": {
        "label": "材料",
        "words": [
            "wood", "metal", "glass", "plastic", "stone", "brick", "concrete",
            "leather", "fabric", "paper", "rubber", "ceramic", "marble",
            "granite", "copper", "brass", "bronze", "silver", "gold",
            "iron", "steel", "aluminum", "titanium", "ivory",
        ],
        "attrs": {"hardness": 1, "weight": 1, "value": 1},
    },
    "natural": {
        "label": "自然",
        "words": [
            "mountain", "river", "ocean", "forest", "desert", "island",
            "valley", "cave", "volcano", "waterfall", "lake", "glacier",
            "meadow", "cliff", "swamp", "prairie", "jungle", "reef",
            "canyon", "plateau",
        ],
        "attrs": {"size": 1, "water": 1, "temperature": 1},
    },
    "weather": {
        "label": "天气",
        "words": [
            "rain", "snow", "wind", "storm", "thunder", "lightning", "fog",
            "cloud", "sun", "hail", "tornado", "hurricane", "blizzard",
            "drought", "flood", "mist", "dew", "frost", "sleet", "breeze",
        ],
        "attrs": {"intensity": 1, "temperature": 1, "water": 1},
    },
    "emotion": {
        "label": "情感",
        "words": [
            "happy", "sad", "angry", "fear", "love", "hate", "joy", "sorrow",
            "pride", "shame", "guilt", "envy", "jealousy", "hope", "despair",
            "anxiety", "calm", "excitement", "boredom", "loneliness",
        ],
        "attrs": {"valence": 1, "arousal": 1, "social": 1},
    },
    "color": {
        "label": "颜色",
        "words": [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "gray", "navy", "beige", "ivory",
            "coral", "crimson", "scarlet", "maroon", "olive", "amber",
            "lavender", "turquoise", "magenta", "teal", "bronze",
        ],
        "attrs": {"warmth": 1, "brightness": 1, "saturation": 1},
    },
    "sport": {
        "label": "运动",
        "words": [
            "soccer", "basketball", "tennis", "swimming", "running", "cycling",
            "boxing", "wrestling", "golf", "baseball", "volleyball", "rugby",
            "hockey", "skiing", "surfing", "diving", "archery", "fencing",
            "javelin", "rowing",
        ],
        "attrs": {"team": 1, "contact": 1, "medium": 1},
    },
    "body": {
        "label": "身体部位",
        "words": [
            "head", "hand", "foot", "arm", "leg", "eye", "ear", "nose",
            "mouth", "heart", "brain", "finger", "toe", "knee", "elbow",
            "shoulder", "wrist", "ankle", "neck", "chest",
        ],
        "attrs": {"position": 1, "size": 1, "mobility": 1},
    },
}

# 属性手动标注（Stage456的扩展版）
ATTR_ANNOTATIONS = {}
for cat_name, cat_data in LARGE_CONCEPTS.items():
    for word in cat_data["words"]:
        ATTR_ANNOTATIONS[word] = {}


def sanitize_for_json(obj):
    """递归清理numpy类型以便JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_model(model_path: Path):
    """加载模型和tokenizer"""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA: {want_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True,
        trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
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
    hidden_dim = discover_layers(model)[0].mlp.down_proj.out_features
    print(f"  Layers: {layer_count}, NeuronDim: {neuron_dim}, HiddenDim: {hidden_dim}")

    return model, tokenizer, layer_count, neuron_dim, hidden_dim


def extract_word_activation(model, tokenizer, word, layer_count):
    """提取单个词在所有层的MLP激活（neuron_in）"""
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


def extract_all_activations(model, tokenizer, concepts, layer_count):
    """提取所有概念在各层的激活"""
    all_activations = {}
    total = sum(len(cat["words"]) for cat in concepts.values())
    done = 0

    for cat_name, cat_data in concepts.items():
        print(f"  [{cat_name}]")
        all_activations[cat_name] = {}
        for word in cat_data["words"]:
            try:
                acts = extract_word_activation(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except Exception as e:
                pass

    print(f"  Total: {done}/{total}")
    return all_activations


def build_bias_matrices(all_activations, concepts, target_layers):
    """构建偏置矩阵（概念激活 - 类别基底）"""
    from collections import defaultdict

    all_biases = []
    concept_labels = []
    category_labels = []

    for cat_name, words_acts in all_activations.items():
        if cat_name not in concepts:
            continue

        # 计算类别基底
        words = list(words_acts.keys())
        layer_vecs = defaultdict(list)
        for word in words:
            acts = words_acts[word]
            for l in target_layers:
                if l in acts:
                    layer_vecs[l].append(acts[l])

        basis = {}
        for l in target_layers:
            if layer_vecs[l]:
                basis[l] = np.mean(layer_vecs[l], axis=0)

        # 计算偏置
        for word in words:
            acts = words_acts[word]
            bias_parts = []
            for l in target_layers:
                if l in acts and l in basis:
                    bias = acts[l] - basis[l]
                    bias_parts.append(bias)
            if bias_parts:
                all_biases.append(np.concatenate(bias_parts))
                concept_labels.append(word)
                category_labels.append(cat_name)

    bias_matrix = np.array(all_biases)
    print(f"  Bias matrix: {bias_matrix.shape}")
    return bias_matrix, concept_labels, category_labels


def high_dim_svd_analysis(bias_matrix, concept_labels, category_labels, n_components=80):
    """80维SVD分解 + 完整因子分析"""
    from sklearn.preprocessing import StandardScaler

    results = {}
    scaler = StandardScaler()
    normed = scaler.fit_transform(bias_matrix)

    # PCA (SVD) 分解
    n = min(n_components, min(normed.shape) - 1)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n, random_state=42)
    components = svd.fit_transform(normed)

    # 方差解释比
    var_exp = svd.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)

    results["n_components"] = n
    results["variance_explained"] = var_exp.tolist()
    results["cumulative_variance"] = cum_var.tolist()

    # 关键维度节点
    milestones = {}
    for threshold in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        idx = np.searchsorted(cum_var, threshold)
        if idx < len(cum_var):
            milestones[f"{int(threshold*100)}%"] = int(idx + 1)
            milestones[f"{int(threshold*100)}%_var"] = float(cum_var[idx])
    results["milestones"] = milestones

    # 前80个因子的语义描述
    factor_semantics = []
    for i in range(min(n, 80)):
        scores = components[:, i]
        top_pos_idx = np.argsort(scores)[-5:][::-1]
        top_neg_idx = np.argsort(scores)[:5]
        factor_semantics.append({
            "factor": i,
            "variance": float(var_exp[i]),
            "cum_var": float(cum_var[i]),
            "top_positive": [(concept_labels[j], float(scores[j])) for j in top_pos_idx],
            "top_negative": [(concept_labels[j], float(scores[j])) for j in top_neg_idx],
        })
    results["factor_semantics"] = factor_semantics

    # 因子层次聚类
    from scipy.cluster.hierarchy import linkage, fcluster
    factor_corr = np.corrcoef(components.T)
    Z = linkage(factor_corr, method='average')
    clusters = fcluster(Z, t=20, criterion='maxclust')
    results["factor_clusters"] = clusters.tolist()
    results["n_clusters"] = int(len(set(clusters)))

    return results, components


def factor_attribute_correlation(components, concept_labels, category_labels, concepts):
    """80个因子与属性的关联分析"""
    results = {}

    # 1. 因子-类别关联
    cat_to_idx = defaultdict(list)
    for i, cat in enumerate(category_labels):
        cat_to_idx[cat].append(i)

    factor_cat_corr = {}
    for fi in range(min(50, components.shape[1])):
        scores = components[:, fi]
        cat_means = {}
        for cat, idxs in cat_to_idx.items():
            cat_means[cat] = float(np.mean(scores[idxs]))
        # eta-squared
        grand_mean = np.mean(scores)
        ss_between = sum(len(idxs) * (cat_means[cat] - grand_mean)**2 for cat, idxs in cat_to_idx.items())
        ss_total = np.sum((scores - grand_mean)**2)
        eta_sq = ss_between / (ss_total + 1e-10)
        factor_cat_corr[f"factor_{fi}"] = {
            "eta_squared": float(eta_sq),
            "top_category": max(cat_means, key=cat_means.get),
            "top_cat_mean": float(max(cat_means.values())),
            "bottom_category": min(cat_means, key=cat_means.get),
            "bottom_cat_mean": float(min(cat_means.values())),
        }
    results["factor_category_eta2"] = factor_cat_corr

    # 2. 因子-手动属性关联（使用类别内属性差异）
    # 对有size属性的概念
    size_concepts = []
    for cat_name, cat_data in concepts.items():
        if cat_name in ["fruit", "animal", "vehicle", "furniture", "natural", "body"]:
            for word in cat_data["words"]:
                size_concepts.append(word)

    # 用概念词长作为大小代理（长词=大概念）
    word_lengths = np.array([len(concept_labels[i]) for i in range(len(concept_labels))])

    attr_factors = []
    for fi in range(min(50, components.shape[1])):
        scores = components[:, fi]
        corr = np.corrcoef(scores, word_lengths)[0, 1]
        if abs(corr) > 0.2:
            attr_factors.append({
                "factor": fi,
                "correlation_with_word_length": float(corr),
                "description": "word_length_proxy",
            })

    results["attribute_proxy_factors"] = attr_factors[:20]

    # 3. 类内方差分析（每类概念在因子空间中的分散程度）
    intra_class_dispersion = {}
    for cat, idxs in cat_to_idx.items():
        cat_components = components[idxs]
        cat_mean = np.mean(cat_components, axis=0)
        cat_std = np.std(cat_components, axis=0)
        avg_std = float(np.mean(cat_std))
        max_std = float(np.max(cat_std))
        intra_class_dispersion[cat] = {
            "avg_std": avg_std,
            "max_std": max_std,
            "n_concepts": len(idxs),
        }
    results["intra_class_dispersion"] = intra_class_dispersion

    return results


def reconstruction_analysis(bias_matrix, components, concept_labels, dims=[10, 20, 30, 50, 80]):
    """不同维度的重建精度分析"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    results = {}
    scaler = StandardScaler()
    normed = scaler.fit_transform(bias_matrix)

    for d in dims:
        if d >= min(normed.shape):
            continue
        svd = TruncatedSVD(n_components=d, random_state=42)
        approx = svd.inverse_transform(svd.fit_transform(normed))
        residuals = normed - approx
        mse = float(np.mean(residuals**2))
        cos_sims = []
        for i in range(min(len(normed), 50)):
            orig_norm = np.linalg.norm(normed[i])
            appr_norm = np.linalg.norm(approx[i])
            if orig_norm > 1e-8 and appr_norm > 1e-8:
                cos_sims.append(float(np.dot(normed[i], approx[i]) / (orig_norm * appr_norm)))
        avg_cos = float(np.mean(cos_sims)) if cos_sims else 0
        results[f"dim_{d}"] = {
            "mse": mse,
            "avg_cosine_similarity": avg_cos,
            "var_explained": float(np.sum(svd.explained_variance_ratio_)),
        }

    return results


def cross_model_factor_comparison(comp1, comp2, labels1, labels2, n_factors=20):
    """双模型因子一致性对比"""
    results = {}

    # 找共同概念
    set1 = set(labels1)
    set2 = set(labels2)
    common = sorted(set1 & set2)
    print(f"  Common concepts: {len(common)} (from {len(set1)} x {len(set2)})")

    if len(common) < 10:
        results["error"] = "too few common concepts"
        return results

    idx1 = [labels1.index(w) for w in common]
    idx2 = [labels2.index(w) for w in common]

    # 对比原始偏置相似度
    cos_sims = []
    for i in range(len(common)):
        n1 = np.linalg.norm(comp1[idx1[i]])
        n2 = np.linalg.norm(comp2[idx2[i]])
        if n1 > 1e-8 and n2 > 1e-8:
            cos_sims.append(float(np.dot(comp1[idx1[i]], comp2[idx2[i]]) / (n1 * n2)))
    results["raw_bias_cosine_mean"] = float(np.mean(cos_sims)) if cos_sims else 0
    results["raw_bias_cosine_std"] = float(np.std(cos_sims)) if cos_sims else 0

    # SVD到20维后的因子空间对比
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    n = min(n_factors, min(comp1.shape[1], comp2.shape[1], len(common) - 1))
    svd1 = TruncatedSVD(n_components=n, random_state=42)
    svd2 = TruncatedSVD(n_components=n, random_state=42)

    f1 = svd1.fit_transform(StandardScaler().fit_transform(comp1[idx1]))
    f2 = svd2.fit_transform(StandardScaler().fit_transform(comp2[idx2]))

    # 因子空间余弦相似度
    factor_cos_sims = []
    for i in range(len(common)):
        n1 = np.linalg.norm(f1[i])
        n2 = np.linalg.norm(f2[i])
        if n1 > 1e-8 and n2 > 1e-8:
            factor_cos_sims.append(float(np.dot(f1[i], f2[i]) / (n1 * n2)))
    results["factor_space_cosine_mean"] = float(np.mean(factor_cos_sims)) if factor_cos_sims else 0
    results["factor_space_cosine_std"] = float(np.std(factor_cos_sims)) if factor_cos_sims else 0

    # Procrustes对齐后的相似度
    from scipy.linalg import orthogonal_procrustes
    R, _ = orthogonal_procrustes(f1, f2)
    f1_aligned = f1 @ R
    aligned_cos = []
    for i in range(len(common)):
        n1 = np.linalg.norm(f1_aligned[i])
        n2 = np.linalg.norm(f2[i])
        if n1 > 1e-8 and n2 > 1e-8:
            aligned_cos.append(float(np.dot(f1_aligned[i], f2[i]) / (n1 * n2)))
    results["aligned_cosine_mean"] = float(np.mean(aligned_cos)) if aligned_cos else 0
    results["aligned_cosine_std"] = float(np.std(aligned_cos)) if aligned_cos else 0

    return results


def run_model_experiment(model_name: str, model_path: Path) -> Dict:
    """运行单个模型的完整80维分析"""
    print(f"\n{'='*70}")
    print(f"  Stage460: {model_name} — 80维高阶因子分析")
    print(f"{'='*70}")

    # 1. 加载模型
    print(f"\n[1/6] Loading model...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim, hidden_dim = load_model(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # 2. 提取激活
    print(f"\n[2/6] Extracting activations...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, LARGE_CONCEPTS, layer_count)
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # 选择目标层（最后6层）
    target_layers = list(range(max(0, layer_count - 6), layer_count))
    print(f"  Target layers: {target_layers}")

    # 3. 构建偏置矩阵
    print(f"\n[3/6] Building bias matrices...")
    bias_matrix, concept_labels, category_labels = build_bias_matrices(
        all_activations, LARGE_CONCEPTS, target_layers
    )
    print(f"  Bias matrix: {bias_matrix.shape}")

    # 保存原始偏置矩阵
    np.save(OUTPUT_DIR / f"{model_name}_bias_matrix.npy", bias_matrix)
    with open(OUTPUT_DIR / f"{model_name}_labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": concept_labels, "categories": category_labels}, f, ensure_ascii=False)

    # 释放模型
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 80维SVD分解
    print(f"\n[4/7] 80-dim SVD decomposition...")
    t0 = time.time()
    n_components = min(80, min(bias_matrix.shape) - 1)
    svd_results, svd_components = high_dim_svd_analysis(
        bias_matrix, concept_labels, category_labels, n_components
    )
    print(f"  SVD done in {time.time()-t0:.1f}s")
    print(f"  Top-10 var: {sum(svd_results['variance_explained'][:10])*100:.1f}%")
    print(f"  Top-20 var: {sum(svd_results['variance_explained'][:20])*100:.1f}%")
    print(f"  Top-50 var: {sum(svd_results['variance_explained'][:50])*100:.1f}%")
    print(f"  Top-{n_components} var: {sum(svd_results['variance_explained'][:n_components])*100:.1f}%")
    print(f"  Milestones: {svd_results['milestones']}")

    # 5. 因子-属性关联
    print(f"\n[5/7] Factor-attribute correlation...")
    attr_corr = factor_attribute_correlation(
        svd_components, concept_labels, category_labels, LARGE_CONCEPTS
    )
    # 统计高eta²因子
    high_eta = [(k, v) for k, v in attr_corr["factor_category_eta2"].items()
                if v["eta_squared"] > 0.3]
    print(f"  Factors with eta²>0.3: {len(high_eta)}")
    for k, v in sorted(high_eta, key=lambda x: -x[1]["eta_squared"])[:10]:
        print(f"    {k}: eta²={v['eta_squared']:.3f}, top={v['top_category']}, bottom={v['bottom_category']}")

    # 5. 重建精度
    print(f"\n[6/7] Reconstruction analysis...")
    recon = reconstruction_analysis(bias_matrix, svd_components, concept_labels)
    for k, v in recon.items():
        print(f"    {k}: var={v['var_explained']*100:.1f}%, cos={v['avg_cosine_similarity']:.3f}")

    # 6. 高阶因子（21-80）语义发现
    print(f"\n[7/7] High-order factor discovery (factors 21-{n_components})...")
    high_order_semantics = svd_results["factor_semantics"][20:]
    # 统计有清晰语义的因子
    meaningful = 0
    for fs in high_order_semantics:
        # 如果top+和top-的概念来自不同类别，说明有跨类别语义
        top_pos_words = [w for w, _ in fs["top_positive"]]
        top_neg_words = [w for w, _ in fs["top_negative"]]
        pos_cats = set()
        neg_cats = set()
        for w in top_pos_words + top_neg_words:
            for cat_name, cat_data in LARGE_CONCEPTS.items():
                if w in cat_data["words"]:
                    if w in top_pos_words:
                        pos_cats.add(cat_name)
                    if w in top_neg_words:
                        neg_cats.add(cat_name)
        if len(pos_cats) > 1 or len(neg_cats) > 1 or (pos_cats and neg_cats and pos_cats != neg_cats):
            meaningful += 1
    print(f"  Meaningful high-order factors: {meaningful}/{len(high_order_semantics)}")

    return {
        "model_name": model_name,
        "n_concepts": len(concept_labels),
        "n_components": n_components,
        "neuron_dim": neuron_dim,
        "bias_dim": bias_matrix.shape[1],
        "compression_ratio": bias_matrix.shape[1] / n_components,
        "svd": svd_results,
        "attr_corr": attr_corr,
        "reconstruction": recon,
        "high_order_meaningful": meaningful,
        "high_order_total": len(high_order_semantics),
    }


def build_report(all_results: Dict, cross_model: Dict) -> str:
    """生成完整分析报告"""
    lines = []
    lines.append("# Stage460: 80维高阶SVD因子分析 — 双模型交叉验证")
    lines.append("")
    lines.append(f"**时间**: 2026-04-01 08:50")
    lines.append(f"**模型**: Qwen3-4B (36层) + DeepSeek-7B (28层)")
    lines.append(f"**概念数**: {all_results.get('qwen3', all_results.get('deepseek', {})).get('n_concepts', '?')}")
    lines.append(f"**SVD维度**: 80")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 方差解释里程碑
    lines.append("## 1. 方差解释里程碑（双模型对比）")
    lines.append("")
    lines.append("| 里程碑 | " + " | ".join(r["model_name"] for r in all_results.values()) + " |")
    lines.append("|--------|" + "|".join(["--------"]*len(all_results)) + "|")
    for milestone_key in ["25%", "30%", "40%", "50%", "60%", "70%", "80%"]:
        row = [milestone_key]
        for r in all_results.values():
            ms = r["svd"].get("milestones", {})
            if milestone_key in ms:
                row.append(f"{ms[milestone_key]}维({ms[milestone_key+'_var']*100:.1f}%)")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # 前20/50/80因子方差
    lines.append("")
    lines.append("### 累计方差解释")
    lines.append("")
    lines.append("| 维度 | " + " | ".join(r["model_name"] for r in all_results.values()) + " |")
    lines.append("|------|" + "|".join(["------"]*len(all_results)) + "|")
    for dim_label, dim_count in [("Top-5", 5), ("Top-10", 10), ("Top-20", 20),
                                  ("Top-30", 30), ("Top-50", 50), ("Top-80", 80)]:
        row = [dim_label]
        for r in all_results.values():
            cum = r["svd"]["cumulative_variance"]
            if dim_count <= len(cum):
                row.append(f"{cum[dim_count-1]*100:.1f}%")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # 2. 高阶因子发现
    lines.append("")
    lines.append("## 2. 高阶因子语义发现（Factor 21-80）")
    lines.append("")
    for model_key, r in all_results.items():
        lines.append(f"### {r['model_name']}: {r['high_order_meaningful']}/{r['high_order_total']}有跨类别语义")
        lines.append("")
        semantics = r["svd"]["factor_semantics"][20:40]  # 展示21-40
        for fs in semantics:
            top_pos = ", ".join(f"{w}({s:.2f})" for w, s in fs["top_positive"][:3])
            top_neg = ", ".join(f"{w}({s:.2f})" for w, s in fs["top_negative"][:3])
            lines.append(f"| Factor {fs['factor']} | {fs['variance']*100:.2f}% | {top_pos} | {top_neg} |")
        lines.append("")

    # 3. 因子-类别关联
    lines.append("## 3. 因子-类别关联（eta² > 0.3）")
    lines.append("")
    for model_key, r in all_results.items():
        lines.append(f"### {r['model_name']}")
        lines.append("")
        lines.append("| Factor | eta² | Top类别 | Bottom类别 |")
        lines.append("|--------|-----|---------|------------|")
        sorted_factors = sorted(
            r["attr_corr"]["factor_category_eta2"].items(),
            key=lambda x: -x[1]["eta_squared"]
        )
        for fname, fdata in sorted_factors[:25]:
            if fdata["eta_squared"] > 0.2:
                lines.append(f"| {fname} | {fdata['eta_squared']:.3f} | {fdata['top_category']} | {fdata['bottom_category']} |")
        lines.append("")

    # 4. 重建精度
    lines.append("## 4. 重建精度")
    lines.append("")
    lines.append("| 维度 | " + " | ".join(r["model_name"] for r in all_results.values()) + " |")
    lines.append("|------|" + "|".join(["------"]*len(all_results)) + "|")
    for dim_key in ["dim_10", "dim_20", "dim_30", "dim_50", "dim_80"]:
        row = [dim_key.replace("dim_", "")]
        for r in all_results.values():
            if dim_key in r["reconstruction"]:
                d = r["reconstruction"][dim_key]
                row.append(f"var={d['var_explained']*100:.1f}%, cos={d['avg_cosine_similarity']:.3f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # 5. 类内分散度
    lines.append("")
    lines.append("## 5. 类内分散度（概念多样性指标）")
    lines.append("")
    lines.append("| 类别 | " + " | ".join(r["model_name"] for r in all_results.values()) + " |")
    lines.append("|------|" + "|".join(["------"]*len(all_results)) + "|")
    all_cats = set()
    for r in all_results.values():
        all_cats.update(r["attr_corr"]["intra_class_dispersion"].keys())
    for cat in sorted(all_cats):
        row = [cat]
        for r in all_results.values():
            disp = r["attr_corr"]["intra_class_dispersion"].get(cat, {})
            if disp:
                row.append(f"avg_std={disp['avg_std']:.3f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    # 6. 双模型一致性
    if cross_model:
        lines.append("")
        lines.append("## 6. 双模型因子空间一致性")
        lines.append("")
        lines.append(f"- 原始偏置余弦相似度: {cross_model.get('raw_bias_cosine_mean', 0):.3f} ± {cross_model.get('raw_bias_cosine_std', 0):.3f}")
        lines.append(f"- 20维因子空间余弦: {cross_model.get('factor_space_cosine_mean', 0):.3f} ± {cross_model.get('factor_space_cosine_std', 0):.3f}")
        lines.append(f"- Procrustes对齐后: {cross_model.get('aligned_cosine_mean', 0):.3f} ± {cross_model.get('aligned_cosine_std', 0):.3f}")
        lines.append("")
        if cross_model.get("factor_space_cosine_mean", 0) > cross_model.get("raw_bias_cosine_mean", 0):
            lines.append("**结论**: SVD因子空间一致性 > 原始偏置一致性，SVD起到跨模型对齐作用")
        else:
            lines.append("**注意**: 因子空间一致性未超过原始偏置一致性，需进一步分析")

    # 7. 核心结论
    lines.append("")
    lines.append("## 7. 核心结论")
    lines.append("")
    for model_key, r in all_results.items():
        cum = r["svd"]["cumulative_variance"]
        n = r["n_components"]
        lines.append(f"### {r['model_name']}")
        lines.append(f"- 80维SVD解释方差: {cum[n-1]*100:.1f}%")
        lines.append(f"- 压缩比: {r['compression_ratio']:.0f}:1 ({r['bias_dim']}→{n}维)")
        lines.append(f"- 50%方差所需维度: {r['svd']['milestones'].get('50%', 'N/A')}")
        lines.append(f"- 高阶有意义因子: {r['high_order_meaningful']}/{r['high_order_total']}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Stage460: 80-dim High-Order SVD Factor Analysis")
    print("  Models: Qwen3-4B -> DeepSeek-7B")
    print("=" * 70)

    all_results = {}

    # ===== Qwen3-4B =====
    print("\n\n" + "#" * 70)
    print("# Round 1: Qwen3-4B")
    print("#" * 70)
    t0 = time.time()
    all_results["qwen3"] = run_model_experiment("Qwen3-4B", QWEN3_MODEL_PATH)
    print(f"\n  Qwen3-4B done in {time.time()-t0:.1f}s")

    # ===== DeepSeek-7B =====
    print("\n\n" + "#" * 70)
    print("# Round 2: DeepSeek-7B")
    print("#" * 70)
    t0 = time.time()
    all_results["deepseek"] = run_model_experiment("DeepSeek-7B", DEEPSEEK7B_MODEL_PATH)
    print(f"\n  DeepSeek-7B done in {time.time()-t0:.1f}s")

    # ===== 双模型对比 =====
    print("\n\n" + "#" * 70)
    print("# Cross-Model Comparison")
    print("#" * 70)

    # 加载保存的偏置矩阵进行对比
    cross_model = {}
    try:
        bm1 = np.load(OUTPUT_DIR / "Qwen3-4B_bias_matrix.npy")
        with open(OUTPUT_DIR / "Qwen3-4B_labels.json", "r", encoding="utf-8") as f:
            labels1_data = json.load(f)
        bm2 = np.load(OUTPUT_DIR / "DeepSeek-7B_bias_matrix.npy")
        with open(OUTPUT_DIR / "DeepSeek-7B_labels.json", "r", encoding="utf-8") as f:
            labels2_data = json.load(f)

        cross_model = cross_model_factor_comparison(
            bm1, bm2,
            labels1_data["labels"], labels2_data["labels"],
            n_factors=20
        )
        print(f"  Cross-model analysis done")
        print(f"  Raw cosine: {cross_model.get('raw_bias_cosine_mean', 0):.3f}")
        print(f"  Factor cosine: {cross_model.get('factor_space_cosine_mean', 0):.3f}")
        print(f"  Aligned cosine: {cross_model.get('aligned_cosine_mean', 0):.3f}")
    except Exception as e:
        print(f"  Cross-model comparison failed: {e}")

    # ===== 保存 =====
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json({"results": all_results, "cross_model": cross_model}),
                  f, ensure_ascii=False, indent=2)

    report = build_report(all_results, cross_model)
    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 摘要
    print("\n\n" + "=" * 70)
    print("  Stage460 Summary")
    print("=" * 70)
    for key, r in all_results.items():
        cum = r["svd"]["cumulative_variance"]
        print(f"\n  {r['model_name']}:")
        print(f"    Concepts: {r['n_concepts']}, Components: {r['n_components']}")
        print(f"    Top-20 variance: {cum[19]*100:.1f}%")
        print(f"    Top-50 variance: {cum[min(49, len(cum)-1)]*100:.1f}%")
        print(f"    Top-{r['n_components']} variance: {cum[r['n_components']-1]*100:.1f}%")
        print(f"    50% var at dim: {r['svd']['milestones'].get('50%', 'N/A')}")
        print(f"    Compression: {r['compression_ratio']:.0f}:1")

    print(f"\n  Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
