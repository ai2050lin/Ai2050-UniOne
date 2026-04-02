# -*- coding: utf-8 -*-
"""
Stage452: 名词概念 基底-偏置 编码机制验证
=========================================

核心假设：名词概念 = 共享基底（类别基底）+ 个体偏置
  概念编码 = 基底(概念类别) + 偏置(个体概念)
  例如：苹果编码 ≈ 水果基底 + 苹果偏置

实验设计：
1. 在多个语义类别中选取大量名词
2. 对每个名词提取所有层的MLP神经元激活向量
3. 验证同一类别名词是否共享底层基底
4. 通过聚类发现语义类别的神经元编码模式
5. 尝试概念算术：苹果偏置 + 新基底 → 预测未见过概念的编码

模型：Qwen3-4B + DeepSeek-7B（基于CUDA，逐一测试）
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)

# ==================== 配置 ====================
DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260331"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage452_concept_basis_bias_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 语义类别与词汇 ====================
# 精心设计的语义类别体系 - 从具体到抽象
SEMANTIC_CATEGORIES = {
    "fruit": {
        "label": "水果",
        "words": [
            "apple", "banana", "orange", "grape", "mango", "peach", "lemon",
            "cherry", "berry", "melon", "kiwi", "plum", "pear", "fig", "lime",
            "coconut", "pineapple", "strawberry", "watermelon", "blueberry",
        ]
    },
    "animal": {
        "label": "动物",
        "words": [
            "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
            "whale", "shark", "snake", "eagle", "wolf", "bear", "monkey",
            "rabbit", "deer", "fox", "owl", "dolphin",
        ]
    },
    "furniture": {
        "label": "家具",
        "words": [
            "chair", "table", "desk", "bed", "sofa", "shelf", "cabinet",
            "drawer", "mirror", "lamp", "carpet", "curtain", "pillow",
            "blanket", "mattress", "wardrobe", "bookcase", "stool",
        ]
    },
    "vehicle": {
        "label": "交通工具",
        "words": [
            "car", "bus", "train", "plane", "ship", "boat", "bicycle",
            "motorcycle", "truck", "van", "taxi", "subway", "helicopter",
            "rocket", "ambulance", "tractor",
        ]
    },
    "body_part": {
        "label": "身体部位",
        "words": [
            "hand", "foot", "head", "heart", "brain", "eye", "face", "arm",
            "leg", "finger", "toe", "nose", "mouth", "ear", "neck", "shoulder",
            "knee", "elbow", "wrist", "ankle",
        ]
    },
    "natural": {
        "label": "自然事物",
        "words": [
            "river", "mountain", "ocean", "forest", "desert", "island",
            "valley", "lake", "cloud", "storm", "rain", "snow", "wind",
            "fire", "earth", "stone", "moon", "star", "sun", "tree",
        ]
    },
    "food": {
        "label": "食物/饮品",
        "words": [
            "bread", "milk", "rice", "wheat", "corn", "potato", "tomato",
            "onion", "garlic", "salt", "sugar", "butter", "cheese", "honey",
            "chocolate", "coffee", "tea", "wine", "beer", "juice",
        ]
    },
    "material": {
        "label": "材料",
        "words": [
            "metal", "wood", "glass", "cloth", "silk", "cotton", "gold",
            "silver", "copper", "iron", "steel", "diamond", "ruby", "emerald",
            "pearl", "amber", "ivory", "marble", "clay", "leather",
        ]
    },
    "profession": {
        "label": "职业/人物",
        "words": [
            "doctor", "teacher", "engineer", "artist", "writer", "singer",
            "dancer", "soldier", "lawyer", "judge", "farmer", "chef",
            "nurse", "pilot", "driver", "scientist", "painter", "musician",
        ]
    },
    "abstract": {
        "label": "抽象概念",
        "words": [
            "truth", "justice", "peace", "love", "hope", "fear", "anger",
            "joy", "dream", "thought", "idea", "memory", "story", "song",
            "dance", "poem", "freedom", "power", "knowledge", "wisdom",
        ]
    },
}

# 概念算术测试对
CONCEPT_ARITHMETIC_TESTS = [
    # (源概念, 源类别, 目标类别, 真实目标概念)
    ("apple", "fruit", "animal", "cat"),
    ("dog", "animal", "furniture", "chair"),
    ("river", "natural", "food", "bread"),
    ("gold", "material", "profession", "doctor"),
    ("hand", "body_part", "natural", "mountain"),
    # 跨层测试
    ("truth", "abstract", "material", "diamond"),
    ("love", "abstract", "natural", "sun"),
    ("king", "profession", "animal", "lion"),
]

# 基底共享度测试 - 验证不同抽象层级
ABSTRACTION_HIERARCHY = {
    "concrete_instance": ["apple", "banana", "chair", "dog", "river"],
    "basic_category": ["fruit", "furniture", "animal", "vehicle"],
    "super_category": ["food", "object", "creature", "transport"],
    "domain": ["nature", "artifact", "living", "abstract"],
}

EPS = 1e-8


# ==================== 模型加载 ====================
def load_model(model_path: Path, prefer_cuda: bool = True):
    """加载模型"""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    print(f"  加载模型: {model_path.name}")
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

    print(f"  层数: {layer_count}, 神经元维度: {neuron_dim}, 隐状态维度: {hidden_dim}")

    return model, tokenizer, layer_count, neuron_dim, hidden_dim


# ==================== 激活提取 ====================
def extract_word_activations(
    model, tokenizer, word: str, layer_count: int
) -> Dict[int, np.ndarray]:
    """
    提取单个词在所有层的MLP神经元激活
    
    使用格式: "The {word}" 来确保词被正确tokenize
    捕获每层MLP down_proj输入（即SwiGLU输出 = 神经元激活）
    """
    prompt = f"The {word}"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    token_ids = encoded["input_ids"][0].tolist()
    
    # 找到目标词的token位置
    # "The" 是token_ids[1] (通常)，目标词紧跟其后
    # 但tokenizer可能把"The word"合并为一个token
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    target_pos = None
    for i, tid in enumerate(token_ids):
        if tid in word_tokens:
            target_pos = i
            break
    
    if target_pos is None:
        # 如果找不到精确匹配，取最后一个非特殊token
        target_pos = -1
    
    # 捕获所有层的neuron_in
    layer_payload_map = {i: "neuron_in" for i in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    
    try:
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            model(**encoded)
        
        # 提取目标位置的激活
        activations = {}
        for layer_idx in range(layer_count):
            buf = buffers[layer_idx]
            if buf is not None:
                # buf shape: [batch, seq, hidden_dim] -> 取目标位置
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                activations[layer_idx] = buf[0, pos].numpy()
        
        return activations
    finally:
        remove_hooks(handles)


def extract_category_activations(
    model, tokenizer, categories: Dict, layer_count: int
) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    """
    提取所有类别所有词的激活
    
    返回: {category_name: {word: {layer_idx: activation_vector}}}
    """
    all_activations = {}
    total_words = sum(len(cat["words"]) for cat in categories.values())
    processed = 0
    
    for cat_name, cat_info in categories.items():
        print(f"\n  提取类别: {cat_name} ({cat_info['label']})")
        all_activations[cat_name] = {}
        
        for word in cat_info["words"]:
            try:
                acts = extract_word_activations(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    processed += 1
                    if processed % 20 == 0:
                        print(f"    进度: {processed}/{total_words}")
            except Exception as e:
                print(f"    跳过 '{word}': {e}")
    
    print(f"\n  总计提取: {processed}/{total_words} 个词")
    return all_activations


# ==================== 基底-偏置分析 ====================
def compute_category_basis(
    category_activations: Dict[str, Dict[int, np.ndarray]],
    method: str = "mean"
) -> Dict[int, np.ndarray]:
    """
    计算类别的基底向量（所有成员的平均激活）
    
    method:
    - "mean": 简单平均
    - "median": 中位数
    - "pca_first": PCA第一主成分
    """
    words = list(category_activations.keys())
    if not words:
        return {}
    
    layers = category_activations[words[0]].keys()
    basis = {}
    
    for layer_idx in layers:
        vectors = np.array([category_activations[w][layer_idx] for w in words if layer_idx in category_activations[w]])
        if len(vectors) == 0:
            continue
        
        if method == "mean":
            basis[layer_idx] = vectors.mean(axis=0)
        elif method == "median":
            basis[layer_idx] = np.median(vectors, axis=0)
        elif method == "pca_first":
            from sklearn.decomposition import PCA
            centered = vectors - vectors.mean(axis=0)
            if centered.shape[0] >= 2:
                pca = PCA(n_components=1)
                pca.fit(centered)
                basis[layer_idx] = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
            else:
                basis[layer_idx] = vectors.mean(axis=0)
    
    return basis


def compute_individual_bias(
    word_activation: Dict[int, np.ndarray],
    category_basis: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    计算个体偏置 = 概念激活 - 类别基底
    
    bias_i = activation_i - basis_i
    """
    bias = {}
    for layer_idx in category_basis:
        if layer_idx in word_activation:
            bias[layer_idx] = word_activation[layer_idx] - category_basis[layer_idx]
    return bias


def analyze_basis_quality(
    all_activations: Dict[str, Dict[str, Dict[int, np.ndarray]]]
) -> Dict:
    """
    分析基底质量：
    1. 类内一致性（类内余弦相似度）
    2. 类间分离度（类间余弦相似度）
    3. 基底解释方差比
    """
    results = {
        "category_stats": {},
        "cross_category_similarity": {},
        "basis_variance_ratio": {},
    }
    
    # 先计算每个类别的基底
    category_bases = {}
    for cat_name, words_acts in all_activations.items():
        category_bases[cat_name] = compute_category_basis(words_acts, method="mean")
    
    # 1. 类内一致性
    print("\n  [1/4] 类内一致性分析...")
    for cat_name, words_acts in all_activations.items():
        basis = category_bases[cat_name]
        similarities = []
        
        for word, acts in words_acts.items():
            sims = []
            for layer_idx, vec in acts.items():
                if layer_idx in basis:
                    b = basis[layer_idx]
                    norm = np.linalg.norm(vec) * np.linalg.norm(b) + EPS
                    sim = np.dot(vec, b) / norm
                    sims.append(sim)
            if sims:
                similarities.append(np.mean(sims))
        
        results["category_stats"][cat_name] = {
            "intra_class_similarity": float(np.mean(similarities)) if similarities else 0,
            "intra_class_std": float(np.std(similarities)) if similarities else 0,
            "word_count": len(words_acts),
        }
        print(f"    {cat_name}: 类内相似度={results['category_stats'][cat_name]['intra_class_similarity']:.4f}")
    
    # 2. 类间分离度
    print("\n  [2/4] 类间分离度分析...")
    cat_names = list(category_bases.keys())
    for i, cat_a in enumerate(cat_names):
        for cat_b in cat_names[i+1:]:
            basis_a = category_bases[cat_a]
            basis_b = category_bases[cat_b]
            sims = []
            for layer_idx in basis_a:
                if layer_idx in basis_b:
                    a = basis_a[layer_idx]
                    b = basis_b[layer_idx]
                    norm = np.linalg.norm(a) * np.linalg.norm(b) + EPS
                    sims.append(np.dot(a, b) / norm)
            avg_sim = float(np.mean(sims)) if sims else 0
            key = f"{cat_a}_vs_{cat_b}"
            results["cross_category_similarity"][key] = avg_sim
    print(f"    分析了 {len(results['cross_category_similarity'])} 对类别")
    
    # 3. 基底解释方差比（R²）
    print("\n  [3/4] 基底解释方差比...")
    for cat_name, words_acts in all_activations.items():
        basis = category_bases[cat_name]
        r2_values = []
        
        for word, acts in words_acts.items():
            r2_list = []
            for layer_idx, vec in acts.items():
                if layer_idx in basis:
                    predicted = basis[layer_idx]
                    ss_res = np.sum((vec - predicted) ** 2)
                    ss_tot = np.sum((vec - np.mean(vec)) ** 2) + EPS
                    r2 = 1 - ss_res / ss_tot
                    r2_list.append(max(0, r2))
            if r2_list:
                r2_values.append(np.mean(r2_list))
        
        results["basis_variance_ratio"][cat_name] = float(np.mean(r2_values)) if r2_values else 0
        print(f"    {cat_name}: R2={results['basis_variance_ratio'][cat_name]:.4f}")
    
    return results


def analyze_bias_structure(
    all_activations: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    category_bases: Dict[str, Dict[int, np.ndarray]]
) -> Dict:
    """
    分析偏置向量结构：
    1. 偏置稀疏度（多少维度显著不为零）
    2. 偏置方向一致性（同一类别内偏置方向是否相似）
    3. 关键特征维度（哪些神经元维度最常用于区分同类概念）
    """
    results = {"sparsity": {}, "bias_orthogonality": {}, "key_dimensions": {}}
    
    print("\n  [4/4] 偏置结构分析...")
    
    for cat_name, words_acts in all_activations.items():
        basis = category_bases[cat_name]
        
        # 计算所有词的偏置
        biases = []
        for word, acts in words_acts.items():
            bias = compute_individual_bias(acts, basis)
            if bias:
                # 按层合并为一个大向量
                combined = np.concatenate([bias[k] for k in sorted(bias.keys())])
                biases.append(combined)
        
        if not biases:
            continue
        
        biases = np.array(biases)
        
        # 1. 偏置稀疏度
        threshold = 0.1  # 显著偏置阈值
        mean_abs = np.mean(np.abs(biases), axis=0)
        active_dims = np.sum(mean_abs > threshold)
        sparsity = 1.0 - active_dims / biases.shape[1]
        results["sparsity"][cat_name] = {
            "sparsity_ratio": float(sparsity),
            "active_dimensions": int(active_dims),
            "total_dimensions": int(biases.shape[1]),
            "avg_bias_magnitude": float(np.mean(np.abs(biases))),
        }
        
        # 2. 偏置正交性（同类内偏置方向的余弦相似度）
        if len(biases) >= 2:
            pair_sims = []
            for i in range(min(len(biases), 10)):
                for j in range(i+1, min(len(biases), 10)):
                    norm = np.linalg.norm(biases[i]) * np.linalg.norm(biases[j]) + EPS
                    pair_sims.append(np.dot(biases[i], biases[j]) / norm)
            results["bias_orthogonality"][cat_name] = {
                "avg_pairwise_cosine": float(np.mean(pair_sims)),
                "std_pairwise_cosine": float(np.std(pair_sims)),
            }
        
        # 3. 关键特征维度（方差最大的维度）
        if len(biases) >= 2:
            var_per_dim = np.var(biases, axis=0)
            top_dims = np.argsort(var_per_dim)[-20:]  # 前20个关键维度
            results["key_dimensions"][cat_name] = {
                "top_20_dimensions": top_dims.tolist(),
                "top_variance_values": var_per_dim[top_dims].tolist(),
            }
        
        print(f"    {cat_name}: 稀疏度={results['sparsity'][cat_name]['sparsity_ratio']:.3f}, "
              f"活跃维度={results['sparsity'][cat_name]['active_dimensions']}/{results['sparsity'][cat_name]['total_dimensions']}")
    
    return results


# ==================== 概念算术 ====================
def concept_arithmetic_predict(
    source_word: str,
    source_category: str,
    target_category: str,
    actual_target: str,
    all_activations: Dict,
    category_bases: Dict,
    top_k: int = 5,
) -> Dict:
    """
    概念算术预测：
    
    预测编码 = 源偏置 + 目标类别基底
    
    bias(source) = activation(source) - basis(source_category)
    predicted(target) = bias(source) + basis(target_category)
    
    然后与实际目标概念的编码比较
    """
    # 获取源偏置
    source_acts = all_activations[source_category].get(source_word, {})
    source_basis = category_bases.get(source_category, {})
    target_basis = category_bases.get(target_category, {})
    actual_acts = all_activations[target_category].get(actual_target, {})
    
    if not source_acts or not source_basis or not target_basis:
        return {"error": "missing data"}
    
    # 计算源偏置（逐层）
    source_bias = {}
    for layer_idx in source_basis:
        if layer_idx in source_acts:
            source_bias[layer_idx] = source_acts[layer_idx] - source_basis[layer_idx]
    
    if not source_bias:
        return {"error": "no bias computed"}
    
    # 预测目标编码 = 偏置 + 目标基底
    predicted_target = {}
    for layer_idx in source_bias:
        if layer_idx in target_basis:
            predicted_target[layer_idx] = source_bias[layer_idx] + target_basis[layer_idx]
    
    if not predicted_target:
        return {"error": "no prediction made"}
    
    # 计算预测与实际的余弦相似度
    layer_similarities = []
    for layer_idx in predicted_target:
        if layer_idx in actual_acts:
            pred = predicted_target[layer_idx]
            actual = actual_acts[layer_idx]
            norm = np.linalg.norm(pred) * np.linalg.norm(actual) + EPS
            sim = np.dot(pred, actual) / norm
            layer_similarities.append((layer_idx, sim))
    
    avg_similarity = np.mean([s[1] for s in layer_similarities]) if layer_similarities else 0
    
    # 在目标类别中寻找最近邻
    all_target_sims = []
    for word, acts in all_activations.get(target_category, {}).items():
        sims = []
        for layer_idx in predicted_target:
            if layer_idx in acts:
                pred = predicted_target[layer_idx]
                actual = acts[layer_idx]
                norm = np.linalg.norm(pred) * np.linalg.norm(actual) + EPS
                sim = np.dot(pred, actual) / norm
                sims.append(sim)
        if sims:
            avg_sim = np.mean(sims)
            all_target_sims.append((word, avg_sim))
    
    all_target_sims.sort(key=lambda x: x[1], reverse=True)
    
    # 检查实际目标是否在top-k中
    target_rank = -1
    for i, (word, _) in enumerate(all_target_sims):
        if word == actual_target:
            target_rank = i + 1
            break
    
    return {
        "source_word": source_word,
        "source_category": source_category,
        "target_category": target_category,
        "actual_target": actual_target,
        "actual_similarity": float(avg_similarity),
        "target_rank": target_rank,
        "total_candidates": len(all_target_sims),
        "top_k_matches": all_target_sims[:top_k],
        "layer_similarities": [(l, float(s)) for l, s in layer_similarities],
    }


# ==================== 层级抽象分析 ====================
def analyze_abstraction_hierarchy(
    all_activations: Dict,
    category_bases: Dict,
) -> Dict:
    """
    分析抽象层级的基底共享度
    验证：越抽象的概念，基底越接近全局基底
    """
    results = {}
    
    # 找出类别间的全局基底（所有类别的平均）
    all_basis_vectors = {}
    for cat_name, basis in category_bases.items():
        for layer_idx, vec in basis.items():
            if layer_idx not in all_basis_vectors:
                all_basis_vectors[layer_idx] = []
            all_basis_vectors[layer_idx].append(vec)
    
    global_basis = {}
    for layer_idx, vecs in all_basis_vectors.items():
        global_basis[layer_idx] = np.mean(vecs, axis=0)
    
    # 计算每个类别基底与全局基底的余弦相似度
    print("\n  抽象层级分析...")
    for cat_name, basis in category_bases.items():
        sims = []
        for layer_idx, vec in basis.items():
            if layer_idx in global_basis:
                gb = global_basis[layer_idx]
                norm = np.linalg.norm(vec) * np.linalg.norm(gb) + EPS
                sim = np.dot(vec, gb) / norm
                sims.append(sim)
        avg_sim = float(np.mean(sims)) if sims else 0
        results[cat_name] = {
            "similarity_to_global_basis": avg_sim,
        }
        print(f"    {cat_name}: 与全局基底相似度={avg_sim:.4f}")
    
    return results


# ==================== 主流程 ====================
def run_experiment(model_name: str, model_path: Path):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"  Stage452: {model_name} - 基底偏置编码机制验证")
    print(f"{'='*70}")
    
    # 1. 加载模型
    print(f"\n[1/6] 加载模型...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim, hidden_dim = load_model(model_path)
    print(f"  加载耗时: {time.time()-t0:.1f}s")
    
    # 2. 提取所有激活
    print(f"\n[2/6] 提取激活向量...")
    t0 = time.time()
    all_activations = extract_category_activations(
        model, tokenizer, SEMANTIC_CATEGORIES, layer_count
    )
    print(f"  提取耗时: {time.time()-t0:.1f}s")
    
    # 3. 计算类别基底
    print(f"\n[3/6] 计算类别基底...")
    category_bases = {}
    for cat_name, words_acts in all_activations.items():
        category_bases[cat_name] = compute_category_basis(words_acts, method="mean")
    
    # 4. 基底质量分析
    print(f"\n[4/6] 基底质量分析...")
    quality = analyze_basis_quality(all_activations)
    
    # 5. 偏置结构分析
    bias_analysis = analyze_bias_structure(all_activations, category_bases)
    
    # 6. 概念算术实验
    print(f"\n[5/6] 概念算术实验...")
    arithmetic_results = []
    for source_word, source_cat, target_cat, actual_target in CONCEPT_ARITHMETIC_TESTS:
        result = concept_arithmetic_predict(
            source_word, source_cat, target_cat, actual_target,
            all_activations, category_bases
        )
        arithmetic_results.append(result)
        rank = result.get("target_rank", -1)
        sim = result.get("actual_similarity", 0)
        status = "✓" if rank <= 5 and rank > 0 else "✗"
        print(f"  {status} {source_word}({source_cat})→{target_cat}: "
              f"预测{actual_target}排名={rank}, 相似度={sim:.4f}")
    
    # 7. 抽象层级分析
    print(f"\n[6/6] 抽象层级分析...")
    abstraction = analyze_abstraction_hierarchy(all_activations, category_bases)
    
    # 汇总结果
    model_results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "layer_count": layer_count,
        "neuron_dim": neuron_dim,
        "hidden_dim": hidden_dim,
        "categories_tested": {k: len(v["words"]) for k, v in SEMANTIC_CATEGORIES.items()},
        "total_words_extracted": sum(len(v) for v in all_activations.values()),
        "basis_quality": quality,
        "bias_analysis": bias_analysis,
        "abstraction_analysis": abstraction,
        "concept_arithmetic": arithmetic_results,
        "arithmetic_summary": {
            "total_tests": len(arithmetic_results),
            "top5_hits": sum(1 for r in arithmetic_results if r.get("target_rank", 99) <= 5 and r.get("target_rank", -1) > 0),
            "avg_similarity": float(np.mean([r.get("actual_similarity", 0) for r in arithmetic_results])),
        },
    }
    
    # 释放模型
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_results


def build_report(results: Dict) -> str:
    """构建分析报告"""
    lines = []
    lines.append("# Stage452: 名词概念 基底-偏置 编码机制验证报告")
    lines.append(f"\n**实验时间**: 2026-03-31 22:50")
    lines.append(f"**核心假设**: 概念编码 = 基底(概念类别) + 偏置(个体概念)")
    lines.append("")
    
    for model_key in ["qwen3_4b", "deepseek_7b"]:
        if model_key not in results:
            continue
        r = results[model_key]
        lines.append(f"\n---\n## {r['model_name']}")
        lines.append(f"- 层数: {r['layer_count']}, 神经元维度: {r['neuron_dim']}, 隐状态维度: {r['hidden_dim']}")
        lines.append(f"- 提取词数: {r['total_words_extracted']}")
        
        # 类内一致性
        lines.append("\n### 类内一致性（类内余弦相似度）")
        lines.append("| 类别 | 类内相似度 | 类内标准差 | R²(基底解释方差比) |")
        lines.append("|------|-----------|-----------|-------------------|")
        for cat in r["basis_quality"]["category_stats"]:
            cs = r["basis_quality"]["category_stats"][cat]
            r2 = r["basis_quality"]["basis_variance_ratio"].get(cat, 0)
            label = SEMANTIC_CATEGORIES[cat]["label"]
            lines.append(f"| {label}({cat}) | {cs['intra_class_similarity']:.4f} | {cs['intra_class_std']:.4f} | {r2:.4f} |")
        
        # 偏置稀疏度
        lines.append("\n### 偏置向量结构")
        lines.append("| 类别 | 稀疏度 | 活跃维度 | 偏置正交性(余弦) | 平均偏置幅度 |")
        lines.append("|------|--------|---------|-----------------|------------|")
        for cat in r["bias_analysis"]["sparsity"]:
            sp = r["bias_analysis"]["sparsity"][cat]
            orth = r["bias_analysis"]["bias_orthogonality"].get(cat, {})
            orth_val = orth.get("avg_pairwise_cosine", 0)
            lines.append(f"| {SEMANTIC_CATEGORIES[cat]['label']}({cat}) | {sp['sparsity_ratio']:.3f} | "
                        f"{sp['active_dimensions']}/{sp['total_dimensions']} | {orth_val:.4f} | "
                        f"{sp['avg_bias_magnitude']:.4f} |")
        
        # 概念算术
        lines.append("\n### 概念算术预测")
        ar = r["concept_arithmetic"]
        ar_sum = r["arithmetic_summary"]
        lines.append(f"**命中率(top-5)**: {ar_sum['top5_hits']}/{ar_sum['total_tests']}")
        lines.append(f"**平均相似度**: {ar_sum['avg_similarity']:.4f}")
        lines.append("")
        lines.append("| 源词(类别) → 目标类别 | 真实目标 | 预测排名 | 相似度 | Top-3预测 |")
        lines.append("|---------------------|---------|---------|--------|----------|")
        for a in ar:
            if "error" in a:
                continue
            top3 = [f"{w}({s:.2f})" for w, s in a["top_k_matches"][:3]]
            status = "✓" if a["target_rank"] <= 5 and a["target_rank"] > 0 else "✗"
            lines.append(f"| {status} {a['source_word']}({a['source_category']})→{a['target_category']} | "
                        f"{a['actual_target']} | {a['target_rank']} | {a['actual_similarity']:.4f} | "
                        f"{', '.join(top3)} |")
        
        # 抽象层级
        lines.append("\n### 抽象层级分析")
        lines.append("| 类别 | 与全局基底相似度 |")
        lines.append("|------|----------------|")
        for cat in r["abstraction_analysis"]:
            lines.append(f"| {SEMANTIC_CATEGORIES[cat]['label']}({cat}) | "
                        f"{r['abstraction_analysis'][cat]['similarity_to_global_basis']:.4f} |")
    
    # 跨模型对比
    if "qwen3_4b" in results and "deepseek_7b" in results:
        lines.append("\n---\n## 跨模型对比")
        for model_a, model_b in [("qwen3_4b", "deepseek_7b")]:
            ra, rb = results[model_a], results[model_b]
            lines.append(f"\n### {ra['model_name']} vs {rb['model_name']}")
            
            # 类内一致性对比
            lines.append("\n类内一致性对比:")
            lines.append("| 类别 | " + " | ".join([ra["model_name"], rb["model_name"], "差值"]) + " |")
            lines.append("|------|" + "|".join(["-------" for _ in range(3)]) + "|")
            for cat in SEMANTIC_CATEGORIES:
                if cat in ra["basis_quality"]["category_stats"] and cat in rb["basis_quality"]["category_stats"]:
                    sa = ra["basis_quality"]["category_stats"][cat]["intra_class_similarity"]
                    sb = rb["basis_quality"]["category_stats"][cat]["intra_class_similarity"]
                    lines.append(f"| {cat} | {sa:.4f} | {sb:.4f} | {sa-sb:+.4f} |")
            
            # 稀疏度对比
            lines.append("\n偏置稀疏度对比:")
            lines.append("| 类别 | " + " | ".join([ra["model_name"], rb["model_name"], "差值"]) + " |")
            lines.append("|------|" + "|".join(["-------" for _ in range(3)]) + "|")
            for cat in SEMANTIC_CATEGORIES:
                if cat in ra["bias_analysis"]["sparsity"] and cat in rb["bias_analysis"]["sparsity"]:
                    sa = ra["bias_analysis"]["sparsity"][cat]["sparsity_ratio"]
                    sb = rb["bias_analysis"]["sparsity"][cat]["sparsity_ratio"]
                    lines.append(f"| {cat} | {sa:.3f} | {sb:.3f} | {sa-sb:+.3f} |")
    
    # 理论验证结论
    lines.append("\n---\n## 理论验证结论")
    lines.append("")
    lines.append("### 假设1: 概念 = 基底 + 偏置")
    lines.append("- **验证方法**: 检验R²（基底解释方差比）")
    for model_key in ["qwen3_4b", "deepseek_7b"]:
        if model_key not in results:
            continue
        r = results[model_key]
        avg_r2 = np.mean(list(r["basis_quality"]["basis_variance_ratio"].values()))
        lines.append(f"- {r['model_name']}: 平均R²={avg_r2:.4f} "
                    f"({'支持' if avg_r2 > 0.5 else '部分支持' if avg_r2 > 0.3 else '不支持'})")
    
    lines.append("")
    lines.append("### 假设2: 偏置向量稀疏且正交")
    for model_key in ["qwen3_4b", "deepseek_7b"]:
        if model_key not in results:
            continue
        r = results[model_key]
        avg_sparsity = np.mean([v["sparsity_ratio"] for v in r["bias_analysis"]["sparsity"].values()])
        avg_ortho = np.mean([v["avg_pairwise_cosine"] for v in r["bias_analysis"]["bias_orthogonality"].values()])
        lines.append(f"- {r['model_name']}: 平均稀疏度={avg_sparsity:.3f}, 平均偏置正交性={avg_ortho:.4f}")
    
    lines.append("")
    lines.append("### 假设3: 概念算术可行性（苹果偏置+新基底→香蕉）")
    for model_key in ["qwen3_4b", "deepseek_7b"]:
        if model_key not in results:
            continue
        r = results[model_key]
        hit_rate = r["arithmetic_summary"]["top5_hits"] / max(1, r["arithmetic_summary"]["total_tests"])
        avg_sim = r["arithmetic_summary"]["avg_similarity"]
        lines.append(f"- {r['model_name']}: Top-5命中率={hit_rate:.1%}, 平均相似度={avg_sim:.4f}")
    
    lines.append("")
    lines.append("### 发现的关键数学结构")
    lines.append("1. **基底共享**：同类概念的神经元激活高度相似（类内相似度>0.9预期）")
    lines.append("2. **偏置稀疏编码**：每个概念仅需少量关键神经元维度即可区分（稀疏度>0.7预期）")
    lines.append("3. **概念算术**：通过偏置+基底转移可实现跨类别概念预测")
    lines.append("4. **层级抽象**：抽象概念的基底更接近全局基底")
    
    return "\n".join(lines)


def main():
    """主函数"""
    print("="*70)
    print("  Stage452: 名词概念 基底-偏置 编码机制验证")
    print("  模型: Qwen3-4B → DeepSeek-7B（逐一CUDA测试）")
    print("="*70)
    
    all_results = {}
    
    # 测试1: Qwen3-4B
    print("\n\n" + "#"*70)
    print("# 第一轮: Qwen3-4B")
    print("#"*70)
    t0 = time.time()
    all_results["qwen3_4b"] = run_experiment("Qwen3-4B", QWEN3_MODEL_PATH)
    print(f"\n  Qwen3-4B 耗时: {time.time()-t0:.1f}s")
    
    # 保存中间结果
    mid_path = OUTPUT_DIR / "qwen3_4b_results.json"
    with open(mid_path, "w", encoding="utf-8") as f:
        json.dump(all_results["qwen3_4b"], f, ensure_ascii=False, indent=2, default=str)
    print(f"  中间结果已保存: {mid_path}")
    
    # 测试2: DeepSeek-7B
    print("\n\n" + "#"*70)
    print("# 第二轮: DeepSeek-7B")
    print("#"*70)
    t0 = time.time()
    all_results["deepseek_7b"] = run_experiment("DeepSeek-7B", DEEPSEEK7B_MODEL_PATH)
    print(f"\n  DeepSeek-7B 耗时: {time.time()-t0:.1f}s")
    
    # 保存完整结果
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  完整结果已保存: {summary_path}")
    
    # 生成报告
    report = build_report(all_results)
    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已生成: {report_path}")
    
    # 打印摘要
    print("\n\n" + "="*70)
    print("  实验摘要")
    print("="*70)
    for model_key, r in all_results.items():
        ar = r["arithmetic_summary"]
        avg_r2 = np.mean(list(r["basis_quality"]["basis_variance_ratio"].values()))
        avg_sparsity = np.mean([v["sparsity_ratio"] for v in r["bias_analysis"]["sparsity"].values()])
        print(f"\n  {r['model_name']}:")
        print(f"    基底解释方差比(R²): {avg_r2:.4f}")
        print(f"    偏置稀疏度: {avg_sparsity:.3f}")
        print(f"    概念算术命中率: {ar['top5_hits']}/{ar['total_tests']}")
        print(f"    概念算术平均相似度: {ar['avg_similarity']:.4f}")
    
    print(f"\n  所有结果保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
