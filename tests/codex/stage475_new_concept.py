#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage475: 新概念写入实验（简化版）

目标：验证"大脑如何随时添加新记忆"
方法：直接分析embedding空间的几何结构，不需要微调

核心问题：
1. 新概念可能占据的方向是什么？
2. 与已有概念的关系是什么？

简化版直接分析embedding空间，避免模型加载和微调的复杂性
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage475_simple_{time.strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
QWEN3_MODEL_PATH = Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c")
DEEPSEEK_MODEL_PATH = Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60")


def load_embedding(model_path: Path) -> tuple:
    """加载embedding权重和tokenizer"""
    from transformers import AutoTokenizer

    # 设置离线模式
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )

    embed_key = "model.embed_tokens.weight"
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            if embed_key in handle.keys():
                embeddings = handle.get_tensor(embed_key).detach().cpu().float().numpy()
                break

    return embeddings, tokenizer


def analyze_embedding_space(embeddings: np.ndarray, tokenizer, sample_size: int = 500) -> Dict:
    """分析embedding空间的结构"""
    vocab_size, emb_dim = embeddings.shape
    print(f"词汇表大小: {vocab_size}, embedding维度: {emb_dim}")

    # 随机采样分析
    random.seed(42)
    indices = random.sample(range(vocab_size), min(sample_size, vocab_size))
    sampled = embeddings[indices]

    # 1. 计算每个embedding的范数分布
    norms = np.linalg.norm(sampled, axis=1)
    norm_stats = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }

    # 2. 计算采样之间的余弦相似度
    normalized = sampled / (norms[:, np.newaxis] + 1e-8)
    similarity_matrix = normalized @ normalized.T
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarity = np.mean(similarity_matrix)

    # 3. SVD分析
    centered = sampled - sampled.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1

    # 4. 找到与随机方向最"正交"和最"平行"的概念
    # 用PCA第一个主成分作为语义轴
    pc1 = Vt[0]

    # 找出在这个轴上得分最高和最低的概念
    projections = sampled @ pc1
    sorted_idx = np.argsort(projections)

    extreme_concepts = {
        "highest": [],
        "lowest": [],
    }

    for idx in sorted_idx[-5:][::-1]:
        word = tokenizer.decode([indices[idx]]).strip()
        if word and len(word) < 20:
            extreme_concepts["highest"].append({
                "word": word,
                "projection": float(projections[idx]),
            })

    for idx in sorted_idx[:5]:
        word = tokenizer.decode([indices[idx]]).strip()
        if word and len(word) < 20:
            extreme_concepts["lowest"].append({
                "word": word,
                "projection": float(projections[idx]),
            })

    return {
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "norm_stats": norm_stats,
        "avg_similarity": float(avg_similarity),
        "svd_analysis": {
            "singular_values": S[:20].tolist(),
            "dim_90": int(dim_90),
            "dim_95": int(dim_95),
            "cumulative_variance": cumvar[:50].tolist(),
        },
        "extreme_concepts": extreme_concepts,
    }


def simulate_new_concept(embeddings: np.ndarray, tokenizer) -> Dict:
    """
    模拟新概念写入：
    假设新概念是已有概念的线性组合
    分析：如果添加"florp"新概念，它可能占据什么位置？
    """
    vocab_size, emb_dim = embeddings.shape

    # 随机选3个概念作为"父概念"
    random.seed(123)
    parent_indices = random.sample(range(vocab_size), 3)
    parent_embeddings = embeddings[parent_indices]

    # 解码父概念
    parent_words = []
    for idx in parent_indices:
        word = tokenizer.decode([idx]).strip()
        if not word:
            word = f"[{idx}]"
        parent_words.append(word)

    print(f"模拟父概念: {parent_words}")

    # 新概念 = 父概念的加权平均 + 噪声
    weights = np.array([0.5, 0.3, 0.2])
    new_embedding = parent_embeddings.T @ weights

    # 添加小噪声模拟"创新"
    noise = np.random.randn(emb_dim) * 0.1
    new_embedding = new_embedding + noise

    # 归一化
    new_embedding = new_embedding / np.linalg.norm(new_embedding)

    # 分析新概念与已有概念的相似度
    similarity = embeddings @ new_embedding
    top_k = 20
    top_indices = np.argsort(similarity)[::-1][:top_k]

    similar_words = []
    for idx in top_indices:
        word = tokenizer.decode([idx]).strip()
        if word and len(word) < 20:
            similar_words.append({
                "word": word,
                "similarity": float(similarity[idx]),
            })

    # 检查是否与父概念相似
    parent_similarities = [float(similarity[idx]) for idx in parent_indices]

    return {
        "parent_concepts": parent_words,
        "parent_similarities": parent_similarities,
        "avg_parent_similarity": float(np.mean(parent_similarities)),
        "top_similar_concepts": similar_words[:10],
        "conclusion": "如果新概念是线性组合，会与父概念高度相似" if np.mean(parent_similarities) > 0.5 else "新概念可能占据独立方向",
    }


def find_independent_directions(embeddings: np.ndarray, tokenizer, n_directions: int = 10) -> Dict:
    """找出embedding空间中相对独立的方向"""
    random.seed(42)

    # 随机选一些概念
    n_samples = min(200, embeddings.shape[0])
    indices = random.sample(range(embeddings.shape[0]), n_samples)
    sampled = embeddings[indices]

    # 对每对概念计算余弦相似度
    normalized = sampled / (np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8)
    similarity = normalized @ normalized.T

    # 找出与其他概念相似度最低的概念（相对独立）
    np.fill_diagonal(similarity, 1)  # 排除自己
    avg_sim = np.mean(similarity, axis=1)
    most_independent_idx = np.argsort(avg_sim)[:n_directions]

    independent_concepts = []
    for idx in most_independent_idx:
        word = tokenizer.decode([indices[idx]]).strip()
        if not word:
            word = f"[{indices[idx]}]"
        independent_concepts.append({
            "word": word,
            "index": int(indices[idx]),
            "avg_similarity": float(avg_sim[idx]),
        })

    return {
        "independent_concepts": independent_concepts,
        "note": "这些概念与其他概念的相似度最低，可能是编码中的'独立单元'",
    }


def run_experiment(model_name: str):
    """运行新概念写入实验"""
    print(f"\n{'='*60}")
    print(f"Stage475: 新概念写入实验（简化版） - {model_name.upper()}")
    print(f"{'='*60}")

    if model_name == "qwen3":
        model_path = QWEN3_MODEL_PATH
    else:
        model_path = DEEPSEEK_MODEL_PATH

    output_file = OUTPUT_DIR / f"stage475_results_{model_name}.json"

    # 加载embedding
    print(f"正在加载 {model_name} 的embedding...")
    embeddings, tokenizer = load_embedding(model_path)
    print(f"加载完成: {embeddings.shape}")

    # 分析embedding空间
    print("\n分析embedding空间结构...")
    space_analysis = analyze_embedding_space(embeddings, tokenizer)

    # 模拟新概念写入
    print("\n模拟新概念写入...")
    new_concept_sim = simulate_new_concept(embeddings, tokenizer)

    # 找独立方向
    print("\n寻找独立编码方向...")
    independent_dirs = find_independent_directions(embeddings, tokenizer)

    results = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_space_analysis": space_analysis,
        "new_concept_simulation": new_concept_sim,
        "independent_directions": independent_dirs,
    }

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: {output_file}")
    print(f"\n关键发现:")
    print(f"  - 平均相似度: {space_analysis['avg_similarity']:.4f}")
    print(f"  - 90%方差维度: {space_analysis['svd_analysis']['dim_90']}")
    print(f"  - 独立概念数: {len(independent_dirs['independent_concepts'])}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python stage475_new_concept.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ["qwen3", "deepseek"]:
        print("模型必须是 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)