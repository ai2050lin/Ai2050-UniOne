#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage477: 概念干扰矩阵 — 447×447干扰分析

目标：回答"编码的基本单位是什么"
方法：构建概念间的干扰矩阵，发现编码的"独立单元"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage477_interference_{time.strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
QWEN3_MODEL_PATH = Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c")
DEEPSEEK_MODEL_PATH = Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60")

# 447概念列表
CONCEPTS_447 = [
    # 动物（50）
    "cat", "dog", "tiger", "lion", "elephant", "horse", "bird", "fish", "monkey", "bear",
    "wolf", "fox", "rabbit", "mouse", "snake", "frog", "duck", "chicken", "pig", "cow",
    "sheep", "goat", "deer", "whale", "dolphin", "shark", "turtle", "crocodile", "penguin", "polar bear",
    "panda", "koala", "giraffe", "zebra", "hippo", "rhino", "camel", "llama", "sloth", "otter",
    "beaver", "squirrel", "chipmunk", "raccoon", "skunk", "mole", "hedgehog", "bat", "owl", "eagle",
    # 水果（20）
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple", "mango", "peach", "cherry",
    "lemon", "lime", "pear", "plum", "kiwi", "fig", "papaya", "coconut", "avocado", "blueberry",
    # 食物（30）
    "bread", "rice", "pasta", "pizza", "hamburger", "sandwich", "soup", "salad", "meat", "fish",
    "chicken", "egg", "milk", "cheese", "butter", "oil", "salt", "sugar", "spice", "pepper",
    "cake", "cookie", "chocolate", "ice cream", "coffee", "tea", "juice", "water", "wine", "beer",
    # 颜色（20）
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "gold", "silver", "cyan", "magenta", "violet", "indigo", "beige", "tan", "maroon",
    # 形状（20）
    "circle", "square", "triangle", "rectangle", "oval", "diamond", "star", "heart", "cube", "sphere",
    "cone", "cylinder", "pyramid", "line", "curve", "angle", "dot", "ring", "spiral", "wave",
    # 数字（20）
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion", "zero", "first", "second", "third", "half", "double",
    # 时间（20）
    "morning", "afternoon", "evening", "night", "day", "week", "month", "year", "hour", "minute",
    "second", "spring", "summer", "autumn", "winter", "today", "tomorrow", "yesterday", "now", "then",
    # 地点（20）
    "home", "school", "office", "hospital", "store", "park", "beach", "mountain", "river", "lake",
    "ocean", "desert", "forest", "garden", "farm", "city", "village", "street", "house", "room",
    # 人物（20）
    "man", "woman", "child", "boy", "girl", "father", "mother", "brother", "sister", "friend",
    "teacher", "doctor", "nurse", "police", "soldier", "artist", "writer", "scientist", "engineer", "chef",
    # 职业（20）
    "driver", "pilot", "sailor", "farmer", "builder", "painter", "musician", "dancer", "actor", "athlete",
    "architect", "lawyer", "judge", "banker", "shopkeeper", "plumber", "electrician", "mechanic", "photographer", "journalist",
    # 情感（20）
    "happy", "sad", "angry", "fear", "love", "hate", "joy", "peace", "hope", "dream",
    "surprise", "excitement", "calm", "anxiety", "jealousy", "pride", "shame", "guilt", "regret", "envy",
    # 动作（20）
    "run", "walk", "jump", "fly", "swim", "climb", "crawl", "dance", "sing", "play",
    "work", "study", "read", "write", "draw", "cook", "clean", "fight", "sleep", "eat",
    # 物体（20）
    "chair", "table", "bed", "door", "window", "floor", "wall", "roof", "book", "pen",
    "phone", "computer", "TV", "radio", "clock", "lamp", "desk", "shelf", "box", "bag",
    # 交通工具（15）
    "car", "bus", "train", "plane", "ship", "boat", "bike", "motorcycle", "truck", "taxi",
    "ambulance", "firetruck", "subway", "helicopter", "rocket",
    # 自然（20）
    "sun", "moon", "star", "sky", "earth", "fire", "water", "air", "tree", "flower",
    "grass", "leaf", "rock", "sand", "cloud", "rain", "snow", "wind", "storm", "light",
    # 抽象概念（20）
    "life", "death", "truth", "beauty", "justice", "freedom", "peace", "war", "hope", "faith",
    "reason", "knowledge", "wisdom", "power", "money", "success", "failure", "health", "disease", "god",
    # 科技（15）
    "internet", "software", "data", "robot", "AI", "science", "technology", "space", "satellite", "telescope",
    "microscope", "camera", "laser", "nuclear", "electricity",
    # 语言（10）
    "word", "sentence", "grammar", "language", "English", "Chinese", "Spanish", "French", "German", "Japanese",
    # 状态（20）
    "hot", "cold", "warm", "cool", "dry", "wet", "clean", "dirty", "safe", "dangerous",
    "fast", "slow", "big", "small", "tall", "short", "long", "wide", "narrow", "thick",
    # 材质（10）
    "metal", "wood", "stone", "plastic", "paper", "glass", "cloth", "wool", "cotton", "silk",
]

CONCEPTS_447 = CONCEPTS_447[:447]
print(f"概念数量: {len(CONCEPTS_447)}")


def load_model_and_tokenizer(model_name: str):
    """加载embedding和tokenizer"""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer

    if model_name == "qwen3":
        model_path = QWEN3_MODEL_PATH
    else:
        model_path = DEEPSEEK_MODEL_PATH

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


def get_concept_embeddings(concepts: list, embeddings: np.ndarray, tokenizer) -> np.ndarray:
    """提取概念的embedding"""
    activations = []
    for concept in tqdm(concepts, desc="提取激活"):
        tokens = tokenizer.encode(concept, add_special_tokens=False)
        if not tokens:
            activations.append(np.zeros(embeddings.shape[1]))
            continue

        token_id = tokens[0]
        if token_id < embeddings.shape[0]:
            activations.append(embeddings[token_id])
        else:
            activations.append(np.zeros(embeddings.shape[1]))

    return np.array(activations)


def compute_interference_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算概念间的干扰矩阵（余弦相似度）"""
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # 余弦相似度矩阵
    similarity = normalized @ normalized.T

    # 干扰 = 1 - |相似度|
    interference = 1 - np.abs(similarity)
    np.fill_diagonal(interference, 0)

    return interference, similarity


def analyze_clusters(similarity: np.ndarray, concepts: list, n_clusters: int = 20):
    """层次聚类分析"""
    # 距离矩阵：将相似度转换为距离
    # 使用 1 - |similarity|，确保非负
    distance_matrix = 1 - np.abs(similarity)
    # 确保对角线为0且所有值非负
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 1)

    # 检查是否有NaN或Inf
    if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
        print("警告：距离矩阵包含NaN或Inf，替换为0")
        distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # 层次聚类
    condensed = squareform(distance_matrix)
    
    # 检查condensed是否有效
    if np.any(np.isnan(condensed)) or np.any(np.isinf(condensed)):
        print("警告：condensed矩阵包含NaN或Inf")
        condensed = np.nan_to_num(condensed, nan=0.0)
    
    linkage_matrix = linkage(condensed, method="average")  # 用average代替ward避免问题

    # 切割
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # 分析每个簇
    clusters = {}
    for i in range(1, n_clusters + 1):
        indices = np.where(cluster_labels == i)[0]
        cluster_concepts = [concepts[j] for j in indices]

        # 簇内相似度
        sub_sim = similarity[np.ix_(indices, indices)]
        np.fill_diagonal(sub_sim, 0)
        avg_similarity = np.mean(np.abs(sub_sim)) if len(indices) > 1 else 0

        clusters[f"cluster_{i}"] = {
            "size": len(indices),
            "concepts": cluster_concepts[:10],
            "avg_similarity": float(avg_similarity),
        }

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
    }


def find_independent_units(interference: np.ndarray, concepts: list, threshold: float = 0.3):
    """找出编码的独立单元"""
    n = len(concepts)
    independent_units = []

    for i in range(n):
        row = interference[i, :]
        max_interference = np.max(row)
        avg_interference = np.mean(row)

        if max_interference < threshold:
            independent_units.append({
                "concept": concepts[i],
                "index": i,
                "max_interference": float(max_interference),
                "avg_interference": float(avg_interference),
            })

    independent_units.sort(key=lambda x: x["max_interference"])
    return independent_units


def run_experiment(model_name: str):
    """运行实验"""
    print(f"\n{'='*60}")
    print(f"Stage477: 概念干扰矩阵 - {model_name.upper()}")
    print(f"{'='*60}")

    output_file = OUTPUT_DIR / f"stage477_results_{model_name}.json"

    # 加载数据
    print(f"正在加载 {model_name} 数据...")
    embeddings, tokenizer = load_model_and_tokenizer(model_name)
    print(f"Embedding: {embeddings.shape}")

    # 提取概念embedding
    print("提取447概念embedding...")
    concept_embeddings = get_concept_embeddings(CONCEPTS_447, embeddings, tokenizer)
    print(f"概念矩阵: {concept_embeddings.shape}")

    # 计算干扰矩阵
    print("计算干扰矩阵...")
    interference, similarity = compute_interference_matrix(concept_embeddings)
    print(f"干扰矩阵: {interference.shape}")

    # 聚类分析
    print("聚类分析...")
    cluster_results = analyze_clusters(similarity, CONCEPTS_447, n_clusters=20)

    # 找独立单元
    print("寻找独立单元...")
    independent_units = find_independent_units(interference, CONCEPTS_447, threshold=0.3)

    # 统计
    summary = {
        "total_concepts": len(CONCEPTS_447),
        "embedding_dim": int(embeddings.shape[1]),
        "mean_interference": float(np.mean(interference)),
        "std_interference": float(np.std(interference)),
    }

    # 最大干扰对
    max_i, max_j = np.unravel_index(np.argmax(interference), interference.shape)
    summary["max_interference_pair"] = {
        "concept_1": CONCEPTS_447[max_i],
        "concept_2": CONCEPTS_447[max_j],
        "value": float(interference[max_i, max_j]),
    }

    results = {
        "model": model_name,
        "summary": summary,
        "cluster_analysis": cluster_results,
        "independent_units": independent_units[:30],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: {output_file}")
    print(f"\n关键发现:")
    print(f"  - 平均干扰度: {summary['mean_interference']:.4f}")
    print(f"  - 独立单元数: {len(independent_units)} (阈值<0.3)")
    print(f"  - 最大干扰对: {summary['max_interference_pair']['concept_1']} - {summary['max_interference_pair']['concept_2']}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python stage477_interference_matrix.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ["qwen3", "deepseek"]:
        print("模型必须是 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)