#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage476: 500概念IB验证 — 验证17维稳定性

目标：验证17维是否在更大规模（500概念）下仍然成立
方法：扩展到500概念，重新做SVD/IB分析
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "codex_temp" / f"stage476_500_concept_{time.strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
QWEN3_MODEL_PATH = Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c")
DEEPSEEK_MODEL_PATH = Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60")

# 500概念列表
CONCEPTS_500 = [
    # 动物
    "cat", "dog", "tiger", "lion", "elephant", "horse", "bird", "fish", "monkey", "bear",
    "wolf", "fox", "rabbit", "mouse", "snake", "frog", "duck", "chicken", "pig", "cow",
    "sheep", "goat", "deer", "whale", "dolphin", "shark", "turtle", "crocodile", "penguin", "polar bear",
    "panda", "koala", "giraffe", "zebra", "hippo", "rhino", "camel", "llama", "sloth", "otter",
    "beaver", "squirrel", "chipmunk", "raccoon", "skunk", "mole", "hedgehog", "bat", "owl", "eagle",
    # 水果
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple", "mango", "peach", "cherry",
    "lemon", "lime", "pear", "plum", "kiwi", "fig", "papaya", "coconut", "avocado", "blueberry",
    # 食物
    "bread", "rice", "pasta", "pizza", "hamburger", "sandwich", "soup", "salad", "meat", "fish",
    "chicken", "egg", "milk", "cheese", "butter", "oil", "salt", "sugar", "spice", "pepper",
    "cake", "cookie", "chocolate", "ice cream", "coffee", "tea", "juice", "water", "wine", "beer",
    # 颜色
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "gold", "silver", "cyan", "magenta", "violet", "indigo", "beige", "tan", "maroon",
    # 形状
    "circle", "square", "triangle", "rectangle", "oval", "diamond", "star", "heart", "cube", "sphere",
    "cone", "cylinder", "pyramid", "line", "curve", "angle", "dot", "ring", "spiral", "wave",
    # 数字
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion", "zero", "first", "second", "third", "half", "double",
    # 时间
    "morning", "afternoon", "evening", "night", "day", "week", "month", "year", "hour", "minute",
    "second", "spring", "summer", "autumn", "winter", "today", "tomorrow", "yesterday", "now", "then",
    # 地点
    "home", "school", "office", "hospital", "store", "park", "beach", "mountain", "river", "lake",
    "ocean", "desert", "forest", "garden", "farm", "city", "village", "street", "house", "room",
    # 人物
    "man", "woman", "child", "boy", "girl", "father", "mother", "brother", "sister", "friend",
    "teacher", "doctor", "nurse", "police", "soldier", "artist", "writer", "scientist", "engineer", "chef",
    # 职业
    "driver", "pilot", "sailor", "farmer", "builder", "painter", "musician", "dancer", "actor", "athlete",
    "architect", "lawyer", "judge", "banker", "shopkeeper", "plumber", "electrician", "mechanic", "photographer", "journalist",
    # 情感
    "happy", "sad", "angry", "fear", "love", "hate", "joy", "peace", "hope", "dream",
    "surprise", "excitement", "calm", "anxiety", "jealousy", "pride", "shame", "guilt", "regret", "envy",
    # 动作
    "run", "walk", "jump", "fly", "swim", "climb", "crawl", "dance", "sing", "play",
    "work", "study", "read", "write", "draw", "cook", "clean", "fight", "sleep", "eat",
    # 物体
    "chair", "table", "bed", "door", "window", "floor", "wall", "roof", "book", "pen",
    "phone", "computer", "TV", "radio", "clock", "lamp", "desk", "shelf", "box", "bag",
    # 交通工具
    "car", "bus", "train", "plane", "ship", "boat", "bike", "motorcycle", "truck", "taxi",
    "ambulance", "firetruck", "subway", "helicopter", "rocket",
    # 自然
    "sun", "moon", "star", "sky", "earth", "fire", "water", "air", "tree", "flower",
    "grass", "leaf", "rock", "sand", "cloud", "rain", "snow", "wind", "storm", "light",
    # 抽象概念
    "life", "death", "truth", "beauty", "justice", "freedom", "peace", "war", "hope", "faith",
    "reason", "knowledge", "wisdom", "power", "money", "success", "failure", "health", "disease", "god",
    # 科技
    "internet", "software", "data", "robot", "AI", "science", "technology", "space", "satellite", "telescope",
    "microscope", "camera", "laser", "nuclear", "electricity",
    # 语言
    "word", "sentence", "grammar", "language", "English", "Chinese", "Spanish", "French", "German", "Japanese",
    # 状态
    "hot", "cold", "warm", "cool", "dry", "wet", "clean", "dirty", "safe", "dangerous",
    "fast", "slow", "big", "small", "tall", "short", "long", "wide", "narrow", "thick",
    # 材质
    "metal", "wood", "stone", "plastic", "paper", "glass", "cloth", "wool", "cotton", "silk",
    # 新增概念
    "galaxy", "universe", "planet", "comet", "asteroid", "nebula", "blackhole", "gravity", "orbit", "moon",
    "mars", "venus", "jupiter", "saturn", "mercury", "neptune", "uranus", "pluto", "sunrise", "sunset",
    "horizon", "equator", "pole", "continent", "island", "volcano", "earthquake", "tsunami", "hurricane", "tornado",
]

CONCEPTS_500 = CONCEPTS_500[:500]
print(f"概念数量: {len(CONCEPTS_500)}")


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


def compute_svd_analysis(X: np.ndarray) -> Dict:
    """SVD分析"""
    # 归一化
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-8)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # 累积方差
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1
    dim_99 = np.searchsorted(cumvar, 0.99) + 1

    # 找"肘部" - 曲率最大的点
    curvature = np.diff(cumvar, 2)
    elbow_dim = np.argmax(curvature) + 2 if len(curvature) > 0 else 10

    return {
        "singular_values": S[:30].tolist(),
        "cumulative_variance": cumvar[:50].tolist(),
        "dim_90": int(dim_90),
        "dim_95": int(dim_95),
        "dim_99": int(dim_99),
        "elbow_dim": int(elbow_dim),
    }


def run_experiment(model_name: str):
    """运行实验"""
    print(f"\n{'='*60}")
    print(f"Stage476: 500概念IB验证 - {model_name.upper()}")
    print(f"{'='*60}")

    output_file = OUTPUT_DIR / f"stage476_results_{model_name}.json"

    # 加载数据
    print(f"正在加载 {model_name} 数据...")
    embeddings, tokenizer = load_model_and_tokenizer(model_name)
    print(f"Embedding: {embeddings.shape}")

    # 提取概念embedding
    print("提取500概念embedding...")
    concept_embeddings = get_concept_embeddings(CONCEPTS_500, embeddings, tokenizer)
    print(f"概念矩阵: {concept_embeddings.shape}")

    # SVD分析
    print("SVD分析...")
    svd_results = compute_svd_analysis(concept_embeddings)

    # 与17维对比
    # 之前Stage450发现17维IB饱和，现在验证500概念是否仍为17维
    optimal_dim = svd_results["dim_95"]  # 用95%方差作为"压缩维度"

    results = {
        "model": model_name,
        "num_concepts": len(CONCEPTS_500),
        "embedding_dim": int(embeddings.shape[1]),
        "svd_analysis": svd_results,
        "comparison": {
            "stage450_optimal_dim": 17,  # 之前的结果
            "stage476_optimal_dim": optimal_dim,
            "deviation": abs(optimal_dim - 17),
            "verdict": "稳定" if abs(optimal_dim - 17) <= 5 else "变化",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: {output_file}")
    print(f"\n关键发现:")
    print(f"  - 90%方差维度: {svd_results['dim_90']}")
    print(f"  - 95%方差维度: {svd_results['dim_95']}")
    print(f"  - 与17维偏差: {abs(optimal_dim - 17)}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python stage476_500_concept_ib.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ["qwen3", "deepseek"]:
        print("模型必须是 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)