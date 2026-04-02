# -*- coding: utf-8 -*-
"""
Stage456: 大规模概念验证 + 非线性语义因子分解
===========================================

Stage455核心瓶颈: SVD前20因子仅解释21.5%方差,偏置空间维度太高。
Stage456目标:
  1. 扩展到200+词（10+类别）,提高SVD累计方差
  2. 用Autoencoder进行非线性分解,发现更紧凑的语义表示
  3. t-SNE/UMAP可视化概念空间结构
  4. 验证语义因子在更大规模上的稳定性
  5. 跨类别因子共享分析

核心假设:
  - 更大概念集 → SVD累计方差显著提升（从21.5%→40%+）
  - 非线性分解 > 线性SVD（Autoencoder重建误差 < SVD）
  - 语义因子在跨类别间部分共享

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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage456_large_scale_factors_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 扩展语义类别（200+词，10+类别） ====================
SEMANTIC_CATEGORIES = {
    "fruit": {
        "label": "水果",
        "words": [
            "apple", "banana", "orange", "grape", "mango", "peach", "lemon",
            "cherry", "berry", "melon", "kiwi", "plum", "pear", "fig", "lime",
            "coconut", "pineapple", "strawberry", "watermelon", "blueberry",
            "papaya", "guava", "pomegranate", "apricot", "cranberry",
        ],
        "color": {"apple": "red", "banana": "yellow", "orange": "orange", "grape": "purple",
                  "cherry": "red", "lemon": "yellow", "blueberry": "blue", "kiwi": "green",
                  "pear": "green", "plum": "purple", "strawberry": "red", "lime": "green",
                  "fig": "purple", "peach": "pink", "coconut": "brown", "watermelon": "green",
                  "papaya": "orange", "guava": "green", "pomegranate": "red", "apricot": "orange",
                  "cranberry": "red", "melon": "green", "pineapple": "yellow"},
        "size": {"watermelon": 5, "melon": 4, "pineapple": 4, "coconut": 3,
                 "apple": 3, "orange": 3, "pear": 3, "peach": 3,
                 "grape": 2, "blueberry": 1, "berry": 1, "cherry": 1,
                 "strawberry": 2, "kiwi": 2, "lime": 1, "fig": 2,
                 "banana": 3, "mango": 3, "papaya": 3, "guava": 2, "pomegranate": 3,
                 "apricot": 2, "cranberry": 1, "plum": 2, "lemon": 2},
        "taste": {"lemon": 1, "lime": 1,
                  "banana": 3, "mango": 3, "apple": 3, "pear": 2,
                  "cherry": 3, "strawberry": 3, "grape": 3,
                  "peach": 3, "blueberry": 2, "fig": 3, "coconut": 2,
                  "watermelon": 2, "pineapple": 3, "papaya": 2, "guava": 2,
                  "pomegranate": 2, "apricot": 3, "cranberry": 1, "melon": 2, "plum": 3, "orange": 3},
    },
    "animal": {
        "label": "动物",
        "words": [
            "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
            "whale", "shark", "snake", "eagle", "wolf", "bear", "monkey",
            "rabbit", "deer", "fox", "owl", "dolphin",
            "penguin", "parrot", "frog", "turtle", "crocodile",
        ],
        "size": {"elephant": 5, "whale": 5, "bear": 4, "shark": 4,
                 "horse": 4, "lion": 3, "tiger": 3, "crocodile": 4,
                 "deer": 3, "wolf": 3, "monkey": 2, "dolphin": 3,
                 "dog": 2, "fox": 2, "eagle": 2, "owl": 2,
                 "cat": 2, "rabbit": 1, "snake": 2, "bird": 1,
                 "fish": 1, "penguin": 2, "parrot": 1, "frog": 1, "turtle": 1},
        "habitat": {"fish": "water", "whale": "water", "dolphin": "water", "shark": "water",
                    "frog": "water", "turtle": "water", "crocodile": "water",
                    "bird": "air", "eagle": "air", "owl": "air", "penguin": "land",
                    "parrot": "air",
                    "dog": "land", "cat": "land", "horse": "land", "lion": "land",
                    "tiger": "land", "wolf": "land", "bear": "land", "deer": "land",
                    "fox": "land", "rabbit": "land", "monkey": "land", "snake": "land"},
        "domestic": {"dog": 1, "cat": 1, "horse": 1, "rabbit": 1, "parrot": 1,
                     "lion": 0, "tiger": 0, "wolf": 0, "eagle": 0,
                     "shark": 0, "whale": 0, "snake": 0, "bear": 0,
                     "elephant": 0, "deer": 0, "fox": 0, "owl": 0, "dolphin": 0,
                     "penguin": 0, "frog": 0, "turtle": 0, "crocodile": 0,
                     "fish": 0, "bird": 0, "monkey": 0},
        "danger": {"lion": 1, "tiger": 1, "shark": 1, "crocodile": 1, "wolf": 1, "bear": 1,
                   "snake": 1, "eagle": 0,
                   "dog": 0, "cat": 0, "horse": 0, "elephant": 0,
                   "whale": 0, "dolphin": 0, "deer": 0, "rabbit": 0,
                   "fox": 0, "owl": 0, "penguin": 0, "parrot": 0,
                   "frog": 0, "turtle": 0, "fish": 0, "bird": 0, "monkey": 0},
    },
    "vehicle": {
        "label": "交通工具",
        "words": [
            "car", "bus", "train", "plane", "ship", "boat", "bicycle",
            "motorcycle", "truck", "van", "taxi", "subway", "helicopter",
            "rocket", "ambulance", "tractor", "scooter", "cart", "canoe",
            "tank",
        ],
        "speed": {"rocket": 5, "plane": 4, "helicopter": 4,
                  "train": 4, "motorcycle": 3, "car": 3, "bus": 3, "taxi": 3,
                  "truck": 2, "bicycle": 2, "tractor": 1, "scooter": 3,
                  "boat": 2, "ship": 2, "subway": 3, "ambulance": 3,
                  "canoe": 1, "cart": 1, "van": 2, "tank": 3},
        "medium": {"ship": "water", "boat": "water", "canoe": "water",
                   "plane": "air", "helicopter": "air", "rocket": "air",
                   "car": "land", "bus": "land", "train": "land", "truck": "land",
                   "van": "land", "taxi": "land", "bicycle": "land", "motorcycle": "land",
                   "tractor": "land", "subway": "underground", "ambulance": "land",
                   "scooter": "land", "cart": "land", "tank": "land"},
        "size": {"ship": 5, "train": 4, "plane": 4, "truck": 4, "tank": 4,
                 "bus": 3, "subway": 3, "helicopter": 3, "boat": 3,
                 "car": 2, "van": 2, "ambulance": 2, "taxi": 2,
                 "bicycle": 1, "motorcycle": 1, "scooter": 1,
                 "canoe": 1, "cart": 1, "rocket": 2, "tractor": 2},
    },
    "natural": {
        "label": "自然事物",
        "words": [
            "river", "mountain", "ocean", "forest", "desert", "island",
            "valley", "lake", "cloud", "storm", "rain", "snow", "wind",
            "fire", "earth", "stone", "moon", "star", "sun", "tree",
            "volcano", "glacier", "cave", "waterfall", "meadow",
        ],
        "size": {"ocean": 5, "mountain": 4, "desert": 4, "forest": 4,
                 "sun": 5, "moon": 4, "glacier": 4, "volcano": 4,
                 "river": 3, "lake": 3, "island": 3, "valley": 3, "tree": 2,
                 "stone": 1, "cloud": 3, "storm": 3, "rain": 1,
                 "snow": 1, "wind": 1, "fire": 1, "star": 1,
                 "earth": 5, "cave": 2, "waterfall": 2, "meadow": 3},
        "element": {"fire": 1, "stone": 2, "rain": 3, "wind": 4,
                    "cloud": 3, "snow": 3, "ocean": 3, "river": 3,
                    "lake": 3, "earth": 2, "mountain": 2,
                    "sun": 1, "star": 1, "moon": 2,
                    "forest": 2, "desert": 2, "island": 2,
                    "valley": 2, "tree": 2, "volcano": 1, "glacier": 3,
                    "cave": 2, "waterfall": 3, "meadow": 2},
        "danger": {"volcano": 1, "storm": 1, "fire": 1,
                   "ocean": 1, "desert": 1, "mountain": 0,
                   "river": 0, "lake": 0, "island": 0, "valley": 0, "tree": 0,
                   "stone": 0, "cloud": 0, "rain": 0, "snow": 0, "wind": 0,
                   "earth": 0, "sun": 0, "moon": 0, "star": 0, "forest": 0,
                   "glacier": 1, "cave": 0, "waterfall": 1, "meadow": 0},
    },
    "furniture": {
        "label": "家具",
        "words": [
            "chair", "table", "desk", "bed", "sofa", "shelf", "cabinet",
            "drawer", "mirror", "lamp", "carpet", "curtain", "pillow",
            "blanket", "mattress", "wardrobe", "bookcase", "stool",
            "couch", "bench",
        ],
        "softness": {"sofa": 4, "pillow": 5, "blanket": 4, "mattress": 4,
                     "carpet": 3, "curtain": 2, "couch": 4,
                     "chair": 1, "table": 1, "desk": 1, "shelf": 1,
                     "cabinet": 1, "drawer": 1, "mirror": 1,
                     "lamp": 1, "stool": 1, "bookcase": 1, "wardrobe": 1,
                     "bench": 2},
        "room": {"bed": 1, "mattress": 1, "pillow": 1, "wardrobe": 1,
                 "desk": 2, "chair": 2, "bookcase": 2, "shelf": 2,
                 "sofa": 3, "carpet": 3, "curtain": 3, "couch": 3,
                 "table": 4, "cabinet": 5,
                 "lamp": 0, "mirror": 0, "stool": 0, "drawer": 0, "blanket": 0, "bench": 0},
        "movable": {"chair": 1, "stool": 1, "lamp": 1, "pillow": 1, "blanket": 1, "bench": 1,
                    "table": 0, "desk": 0, "bed": 0, "sofa": 0, "shelf": 0,
                    "cabinet": 0, "wardrobe": 0, "bookcase": 0, "carpet": 0, "curtain": 0,
                    "drawer": 0, "mirror": 0, "couch": 0, "mattress": 0},
    },
    "material": {
        "label": "材料",
        "words": [
            "metal", "wood", "glass", "cloth", "silk", "cotton", "gold",
            "silver", "copper", "iron", "steel", "diamond", "ruby", "emerald",
            "pearl", "amber", "ivory", "marble", "clay", "leather",
            "plastic", "paper", "concrete", "ceramic", "wool",
        ],
        "hardness": {"diamond": 5, "iron": 4, "steel": 4,
                     "marble": 4, "glass": 3, "concrete": 4, "ceramic": 3,
                     "gold": 2, "silver": 2, "copper": 2,
                     "wood": 3, "clay": 2, "amber": 3,
                     "cloth": 1, "silk": 1, "cotton": 1,
                     "leather": 2, "ivory": 4,
                     "metal": 3, "pearl": 3, "ruby": 4, "emerald": 4,
                     "plastic": 2, "paper": 1, "wool": 1},
        "value": {"diamond": 5, "gold": 4, "silver": 4,
                  "ruby": 4, "emerald": 4, "pearl": 4,
                  "silk": 3, "marble": 3,
                  "copper": 2, "leather": 2, "amber": 2, "ivory": 4,
                  "metal": 1, "wood": 1, "cloth": 1, "cotton": 1,
                  "glass": 1, "iron": 1, "steel": 2,
                  "clay": 1, "plastic": 1, "paper": 1, "concrete": 1,
                  "ceramic": 2, "wool": 2},
        "natural": {"wood": 1, "cotton": 1, "silk": 1, "leather": 1, "ivory": 1,
                    "pearl": 1, "amber": 1, "marble": 1, "clay": 1, "ruby": 1,
                    "emerald": 1, "diamond": 1, "gold": 1, "silver": 1, "copper": 1,
                    "iron": 1, "wool": 1,
                    "glass": 0, "steel": 0, "metal": 0, "cloth": 0,
                    "plastic": 0, "paper": 0, "concrete": 0, "ceramic": 0},
    },
    "profession": {
        "label": "职业/人物",
        "words": [
            "doctor", "teacher", "engineer", "artist", "writer", "singer",
            "dancer", "soldier", "lawyer", "judge", "farmer", "chef",
            "nurse", "pilot", "driver", "scientist", "painter", "musician",
            "actor", "architect",
        ],
        "field": {"doctor": 1, "nurse": 1,
                  "teacher": 2, "scientist": 3,
                  "engineer": 4, "architect": 4, "pilot": 5,
                  "artist": 6, "painter": 6, "musician": 6, "singer": 6, "dancer": 6, "writer": 6, "actor": 6,
                  "lawyer": 7, "judge": 7,
                  "soldier": 8, "farmer": 9, "chef": 10, "driver": 5},
        "creative": {"artist": 1, "writer": 1, "singer": 1, "dancer": 1, "painter": 1, "musician": 1,
                     "doctor": 0, "engineer": 0, "soldier": 0, "lawyer": 0, "judge": 0,
                     "farmer": 0, "chef": 1, "teacher": 0, "nurse": 0, "pilot": 0,
                     "driver": 0, "scientist": 1, "actor": 1, "architect": 1},
        "prestige": {"doctor": 4, "judge": 5, "lawyer": 4, "scientist": 4, "architect": 4,
                     "engineer": 3, "pilot": 3, "teacher": 3, "artist": 3,
                     "writer": 3, "musician": 3, "singer": 3, "actor": 3, "painter": 3,
                     "dancer": 2, "chef": 2, "farmer": 1, "nurse": 2,
                     "soldier": 2, "driver": 1},
    },
    "food": {
        "label": "食物",
        "words": [
            "bread", "rice", "pasta", "cheese", "butter", "milk", "egg",
            "meat", "chicken", "beef", "pork", "fish_food", "soup", "salad",
            "cake", "cookie", "chocolate", "ice_cream", "pizza", "burger",
        ],
        "temperature": {"ice_cream": 1, "milk": 2, "soup": 3, "salad": 2,
                        "bread": 2, "rice": 3, "pasta": 3, "cheese": 2, "butter": 2,
                        "egg": 3, "meat": 3, "chicken": 3, "beef": 3, "pork": 3,
                        "cake": 2, "cookie": 2, "chocolate": 2, "pizza": 3, "burger": 3},
        "health": {"salad": 5, "egg": 4, "chicken": 4, "fish_food": 4,
                   "rice": 3, "pasta": 2, "bread": 3, "milk": 3,
                   "cheese": 2, "butter": 1, "meat": 2, "beef": 2, "pork": 2,
                   "soup": 3, "cake": 1, "cookie": 1, "chocolate": 1,
                   "ice_cream": 1, "pizza": 1, "burger": 1},
        "sweetness": {"chocolate": 4, "cake": 4, "cookie": 4, "ice_cream": 5,
                      "bread": 2, "rice": 1, "pasta": 1, "cheese": 1, "butter": 1,
                      "milk": 2, "egg": 1, "meat": 1, "chicken": 1, "beef": 1, "pork": 1,
                      "fish_food": 1, "soup": 1, "salad": 1, "pizza": 2, "burger": 2},
    },
    "clothing": {
        "label": "服装",
        "words": [
            "shirt", "pants", "dress", "jacket", "coat", "hat", "shoes",
            "boots", "socks", "gloves", "scarf", "tie", "belt", "jeans",
            "sweater", "uniform", "gown", "vest", "shorts", "sandals",
        ],
        "warmth": {"coat": 5, "jacket": 4, "sweater": 4, "gloves": 4, "scarf": 4, "boots": 4,
                   "shirt": 2, "pants": 2, "dress": 2, "hat": 3, "shoes": 2,
                   "socks": 3, "tie": 1, "belt": 1, "jeans": 3,
                   "uniform": 2, "gown": 1, "vest": 2, "shorts": 1, "sandals": 1},
        "formality": {"gown": 5, "uniform": 4, "coat": 3, "tie": 4, "dress": 4,
                      "jacket": 3, "shirt": 3, "pants": 2, "shoes": 3, "boots": 2,
                      "hat": 2, "gloves": 2, "scarf": 2, "socks": 1, "belt": 1,
                      "jeans": 1, "sweater": 2, "vest": 2, "shorts": 1, "sandals": 1},
        "body_part": {"hat": 1, "scarf": 2, "gloves": 3, "shirt": 4, "jacket": 4, "coat": 4,
                      "dress": 4, "sweater": 4, "vest": 4, "uniform": 4, "gown": 4,
                      "pants": 5, "jeans": 5, "shorts": 5, "belt": 5,
                      "shoes": 6, "boots": 6, "sandals": 6, "socks": 6, "tie": 4},
    },
    "tool": {
        "label": "工具/器具",
        "words": [
            "hammer", "saw", "drill", "knife", "scissors", "needle", "pen",
            "brush", "key", "lock", "rope", "chain", "wheel", "ladder",
            "shovel", "axe", "compass", "ruler", "glue", "tape",
        ],
        "danger": {"knife": 4, "axe": 4, "saw": 3, "drill": 3,
                   "scissors": 2, "needle": 2,
                   "hammer": 2, "shovel": 1,
                   "pen": 1, "brush": 1, "key": 1, "lock": 1,
                   "rope": 1, "chain": 1, "wheel": 1, "ladder": 2,
                   "compass": 1, "ruler": 1, "glue": 1, "tape": 1},
        "complexity": {"compass": 4, "lock": 4, "drill": 4,
                       "wheel": 3, "chain": 3, "ladder": 2,
                       "hammer": 1, "saw": 2, "axe": 1, "knife": 1, "shovel": 1,
                       "scissors": 2, "needle": 1, "pen": 2, "brush": 1,
                       "key": 2, "rope": 1, "ruler": 1, "glue": 2, "tape": 2},
    },
}

# 统计总数
TOTAL_CONCEPTS = sum(len(c["words"]) for c in SEMANTIC_CATEGORIES.values())
print(f"Total categories: {len(SEMANTIC_CATEGORIES)}")
print(f"Total concepts: {TOTAL_CONCEPTS}")

EPS = 1e-8


def sanitize_for_json(obj):
    """递归转换numpy类型"""
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


# ==================== 模型加载 ====================
def load_model(model_path: Path):
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


def extract_word_activation(model, tokenizer, word: str, layer_count: int) -> Optional[Dict[int, np.ndarray]]:
    """提取单个词在所有层的MLP激活"""
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


def extract_all_activations(model, tokenizer, categories: Dict, layer_count: int) -> Dict:
    """提取所有概念激活"""
    all_activations = {}
    total = sum(len(cat["words"]) for cat in categories.values())
    done = 0

    for cat_name, cat_info in categories.items():
        print(f"  [{cat_name}] ({len(cat_info['words'])} words)")
        all_activations[cat_name] = {}
        for word in cat_info["words"]:
            try:
                acts = extract_word_activation(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
                    if done % 20 == 0:
                        print(f"    Progress: {done}/{total}")
            except Exception as e:
                print(f"    ERROR: {word}: {e}")

    print(f"  Total extracted: {done}/{total}")
    return all_activations


# ==================== 偏置矩阵构建 ====================
def build_bias_matrices(
    all_activations: Dict,
    categories: Dict,
    target_layers: List[int],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """构建偏置矩阵（概念×维度），合并所有类别"""
    all_biases = []
    concept_labels = []
    category_labels = []

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

        basis = {}
        for l in target_layers:
            if layer_vecs[l]:
                basis[l] = np.mean(layer_vecs[l], axis=0)

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
                combined_bias = np.concatenate(bias_parts)
                all_biases.append(combined_bias)
                concept_labels.append(word)
                category_labels.append(cat_name)

    return np.array(all_biases), concept_labels, category_labels


# ==================== SVD分解 ====================
def svd_analysis(bias_matrix, concept_labels, category_labels, n_components=30):
    """SVD分解，测试不同K值"""
    from sklearn.decomposition import TruncatedSVD

    print(f"\n  SVD analysis (n_components={n_components})...")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(bias_matrix)

    cumulative = np.cumsum(svd.explained_variance_ratio_)

    results = {
        "n_components": n_components,
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "components": components.tolist(),
    }

    # 打印关键K值的累计方差
    for k in [5, 10, 15, 20, 25, 30]:
        if k <= n_components:
            print(f"    Top-{k} cumulative variance: {cumulative[k-1]*100:.1f}%")

    # 因子-top概念
    factor_top = {}
    for i in range(min(n_components, 20)):
        scores = components[:, i]
        top_pos = np.argsort(scores)[-5:]
        top_neg = np.argsort(scores)[:5]

        factor_top[f"factor_{i}"] = {
            "variance": float(svd.explained_variance_ratio_[i]),
            "top_positive": [(concept_labels[j], float(scores[j])) for j in reversed(top_pos)],
            "top_negative": [(concept_labels[j], float(scores[j])) for j in top_neg],
        }
        print(f"    Factor {i} ({svd.explained_variance_ratio_[i]*100:.1f}%): "
              f"+ [{', '.join(c for c, _ in factor_top[f'factor_{i}']['top_positive'][:3])}] "
              f"- [{', '.join(c for c, _ in factor_top[f'factor_{i}']['top_negative'][:3])}]")

    results["factor_top_concepts"] = factor_top

    return results, components


# ==================== Autoencoder非线性分解 ====================
class SemanticAutoencoder(torch.nn.Module):
    """用于非线性语义因子提取的自编码器"""
    def __init__(self, input_dim, latent_dim=20, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                torch.nn.Linear(prev_dim, h_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(h_dim),
                torch.nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        encoder_layers.append(torch.nn.Linear(prev_dim, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                torch.nn.Linear(prev_dim, h_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(h_dim),
                torch.nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(torch.nn.Linear(prev_dim, input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)


def autoencoder_analysis(bias_matrix, concept_labels, category_labels, latent_dims=[10, 20, 30, 50]):
    """用Autoencoder进行非线性语义因子提取"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    print(f"\n  Autoencoder analysis...")

    # 标准化
    scaler = StandardScaler()
    bias_scaled = scaler.fit_transform(bias_matrix)

    # 先PCA降维以加速训练
    from sklearn.decomposition import PCA
    pca_dim = min(500, bias_matrix.shape[0] - 1, bias_matrix.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    bias_pca = pca.fit_transform(bias_scaled)
    print(f"    PCA to {pca_dim} dims (explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

    results = {}

    # 对不同潜在维度
    for latent_dim in latent_dims:
        print(f"\n    Training AE with latent_dim={latent_dim}...")

        # 数据转tensor
        X = torch.tensor(bias_pca, dtype=torch.float32)

        model_ae = SemanticAutoencoder(pca_dim, latent_dim, hidden_dims=[256, 128, 64])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ae = model_ae.to(device)
        X = X.to(device)

        optimizer = torch.optim.Adam(model_ae.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        # 训练
        model_ae.train()
        losses = []
        for epoch in range(200):
            x_recon, z = model_ae(X)
            loss = torch.nn.functional.mse_loss(x_recon, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0 or epoch == 199:
                losses.append(float(loss.item()))
                print(f"      Epoch {epoch}: loss={loss.item():.6f}")

        # 评估重建误差（在原始空间中）
        model_ae.eval()
        with torch.no_grad():
            x_recon, z = model_ae(X)
            recon_loss_pca = torch.nn.functional.mse_loss(x_recon, X).item()

            # 反PCA变换到原始空间
            z_np = z.cpu().numpy()
            x_recon_np = x_recon.cpu().numpy()
            x_recon_original = scaler.inverse_transform(pca.inverse_transform(x_recon_np))
            x_original = scaler.inverse_transform(pca.inverse_transform(X.cpu().numpy()))

            recon_loss_original = np.mean((x_recon_original - x_original) ** 2)
            total_var = np.var(x_original)

            # 计算解释方差比
            explained_ratio = max(0, 1 - recon_loss_original / (total_var + EPS))

            # SVD对比（同样维度）
            svd = TruncatedSVD(n_components=latent_dim, random_state=42)
            svd_recon = svd.inverse_transform(svd.fit_transform(bias_scaled))
            svd_loss = np.mean((scaler.inverse_transform(svd_recon) - x_original) ** 2)
            svd_explained = max(0, 1 - svd_loss / (total_var + EPS))

        print(f"      AE explained variance: {explained_ratio*100:.1f}% (SVD: {svd_explained*100:.1f}%)")
        print(f"      AE vs SVD improvement: {(explained_ratio - svd_explained)*100:.1f}%")

        # 因子分析
        factor_top = {}
        for i in range(min(latent_dim, 15)):
            scores = z_np[:, i]
            top_pos = np.argsort(scores)[-5:]
            top_neg = np.argsort(scores)[:5]

            factor_top[f"ae_factor_{i}"] = {
                "top_positive": [(concept_labels[j], float(scores[j])) for j in reversed(top_pos)],
                "top_negative": [(concept_labels[j], float(scores[j])) for j in top_neg],
            }
            print(f"      AE Factor {i}: "
                  f"+ [{', '.join(c for c, _ in factor_top[f'ae_factor_{i}']['top_positive'][:3])}] "
                  f"- [{', '.join(c for c, _ in factor_top[f'ae_factor_{i}']['top_negative'][:3])}]")

        results[f"latent_{latent_dim}"] = {
            "latent_dim": latent_dim,
            "explained_variance_ratio": float(explained_ratio),
            "svd_baseline": float(svd_explained),
            "improvement": float(explained_ratio - svd_explained),
            "recon_loss": float(recon_loss_original),
            "latent_codes": z_np.tolist(),
            "factor_top_concepts": factor_top,
        }

        # 清理
        del model_ae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ==================== 因子-属性关联 ====================
def factor_attribute_analysis(svd_components, ae_results, concept_labels, category_labels, categories):
    """计算因子与属性关联度"""
    print(f"\n  Factor-Attribute ANOVA analysis...")

    results = {"svd": {}, "autoencoder": {}}

    # SVD因子
    for cat_name, cat_info in categories.items():
        cat_indices = [i for i, c in enumerate(category_labels) if c == cat_name]
        if len(cat_indices) < 3:
            continue

        cat_concepts = [concept_labels[i] for i in cat_indices]

        for attr_name, attr_values in cat_info.items():
            if attr_name in ("label", "words"):
                continue
            unique_vals = sorted(set(attr_values.values()))
            if len(unique_vals) < 2:
                continue

            # SVD
            best_eta, best_factor = 0, None
            for f_idx in range(min(svd_components.shape[1], 20)):
                groups = defaultdict(list)
                for idx, word in zip(cat_indices, cat_concepts):
                    if word in attr_values:
                        groups[attr_values[word]].append(float(svd_components[idx, f_idx]))

                if len(groups) < 2 or sum(len(g) for g in groups.values()) < 4:
                    continue

                grand_mean = np.mean([v for g in groups.values() for v in g])
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
                ss_total = sum((v - grand_mean)**2 for g in groups.values() for v in g)
                eta = ss_between / (ss_total + EPS)

                if eta > best_eta:
                    best_eta = eta
                    best_factor = f_idx

            if best_eta > 0.3:
                key = f"{cat_name}.{attr_name}"
                results["svd"][key] = {
                    "best_factor": int(best_factor),
                    "eta_squared": float(best_eta),
                }
                print(f"    SVD: {key} → factor_{best_factor} (eta²={best_eta:.3f})")

    # Autoencoder因子（取latent_20）
    if "latent_20" in ae_results:
        ae_codes = np.array(ae_results["latent_20"]["latent_codes"])
        for cat_name, cat_info in categories.items():
            cat_indices = [i for i, c in enumerate(category_labels) if c == cat_name]
            if len(cat_indices) < 3:
                continue
            cat_concepts = [concept_labels[i] for i in cat_indices]

            for attr_name, attr_values in cat_info.items():
                if attr_name in ("label", "words"):
                    continue
                unique_vals = sorted(set(attr_values.values()))
                if len(unique_vals) < 2:
                    continue

                best_eta, best_factor = 0, None
                for f_idx in range(min(ae_codes.shape[1], 20)):
                    groups = defaultdict(list)
                    for idx, word in zip(cat_indices, cat_concepts):
                        if word in attr_values:
                            groups[attr_values[word]].append(float(ae_codes[idx, f_idx]))

                    if len(groups) < 2 or sum(len(g) for g in groups.values()) < 4:
                        continue

                    grand_mean = np.mean([v for g in groups.values() for v in g])
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
                    ss_total = sum((v - grand_mean)**2 for g in groups.values() for v in g)
                    eta = ss_between / (ss_total + EPS)

                    if eta > best_eta:
                        best_eta = eta
                        best_factor = f_idx

                if best_eta > 0.3:
                    key = f"{cat_name}.{attr_name}"
                    results["autoencoder"][key] = {
                        "best_factor": int(best_factor),
                        "eta_squared": float(best_eta),
                    }
                    print(f"    AE:  {key} → ae_factor_{best_factor} (eta²={best_eta:.3f})")

    return results


# ==================== 概念算术验证 ====================
def concept_arithmetic_test(bias_matrix, concept_labels, category_labels, svd_components, ae_results):
    """在原始空间、SVD空间、AE空间中进行概念算术测试"""
    print(f"\n  Concept arithmetic test...")

    # 预定义同类算术对
    same_category_pairs = [
        ("dog", "cat"), ("car", "bus"), ("apple", "banana"),
        ("mountain", "valley"), ("chair", "table"), ("doctor", "nurse"),
        ("bread", "cake"), ("shirt", "dress"), ("hammer", "saw"),
        ("diamond", "ruby"), ("gold", "silver"), ("lion", "tiger"),
        ("plane", "helicopter"), ("river", "lake"), ("bed", "sofa"),
    ]

    # 预定义跨类算术对
    cross_category_pairs = [
        ("dog", "cat", "apple", "banana"),  # animal→fruit
        ("car", "bus", "ship", "boat"),     # vehicle→vehicle (same)
        ("mountain", "valley", "river", "lake"),  # natural→natural
        ("doctor", "nurse", "teacher", "student"),  # profession→profession
        ("gold", "silver", "diamond", "ruby"),  # material→material
    ]

    word2idx = {w: i for i, w in enumerate(concept_labels)}

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))

    results = {"same_category": {}, "cross_category": {}}

    # 原始空间算术
    print("\n  === Same-category arithmetic ===")
    for w1, w2 in same_category_pairs:
        if w1 not in word2idx or w2 not in word2idx:
            continue
        sim = cosine_sim(bias_matrix[word2idx[w1]], bias_matrix[word2idx[w2]])
        results["same_category"][f"{w1}→{w2}"] = round(sim, 3)

    # SVD空间算术
    print("\n  === SVD factor space arithmetic ===")
    svd_sims = []
    for w1, w2 in same_category_pairs:
        if w1 not in word2idx or w2 not in word2idx:
            continue
        sim = cosine_sim(svd_components[word2idx[w1]], svd_components[word2idx[w2]])
        svd_sims.append(sim)
        results["same_category"][f"{w1}→{w2}_svd"] = round(sim, 3)

    # AE空间算术
    if "latent_20" in ae_results:
        ae_codes = np.array(ae_results["latent_20"]["latent_codes"])
        print("\n  === AE factor space arithmetic ===")
        ae_sims = []
        for w1, w2 in same_category_pairs:
            if w1 not in word2idx or w2 not in word2idx:
                continue
            sim = cosine_sim(ae_codes[word2idx[w1]], ae_codes[word2idx[w2]])
            ae_sims.append(sim)
            results["same_category"][f"{w1}→{w2}_ae"] = round(sim, 3)

        if ae_sims:
            print(f"    AE avg same-cat sim: {np.mean(ae_sims):.3f}")
            results["ae_avg_same_category"] = float(np.mean(ae_sims))

    # 打印汇总
    raw_sims = [v for k, v in results["same_category"].items() if not k.endswith("_svd") and not k.endswith("_ae")]
    if raw_sims:
        print(f"\n  Raw avg same-cat sim: {np.mean(raw_sims):.3f}")
    if svd_sims:
        print(f"  SVD avg same-cat sim: {np.mean(svd_sims):.3f}")
        results["svd_avg_same_category"] = float(np.mean(svd_sims))

    # 跨类测试
    print("\n  === Cross-category arithmetic ===")
    for w1, w2, w3, w4 in cross_category_pairs:
        if w1 not in word2idx or w2 not in word2idx or w3 not in word2idx or w4 not in word2idx:
            continue
        # 偏移量类比: w1→w2 的偏移 = bias(w2)-bias(w1)
        # 预测: bias(w3) + shift ≈ bias(w4)
        shift = bias_matrix[word2idx[w2]] - bias_matrix[word2idx[w1]]
        predicted = bias_matrix[word2idx[w3]] + shift
        target = bias_matrix[word2idx[w4]]
        sim = cosine_sim(predicted, target)
        results["cross_category"][f"{w1}→{w2} applied to {w3}→{w4}"] = round(sim, 3)
        print(f"    {w1}→{w2} → {w3}→{w4}: {sim:.3f}")

    cross_sims = list(results["cross_category"].values())
    if cross_sims:
        print(f"  Cross-cat avg: {np.mean(cross_sims):.3f}")
        results["cross_category_avg"] = float(np.mean(cross_sims))

    return results


# ==================== 生成报告 ====================
def generate_report(all_results, output_dir):
    """生成Markdown报告"""
    lines = [
        "# Stage456: 大规模概念验证 + 非线性语义因子分解",
        "",
        f"**时间**: 2026-04-01 01:50",
        f"**目标**: 200+词验证 + Autoencoder非线性分解 + 因子稳定性分析",
        f"**模型**: DeepSeek-7B (28层)",
        f"**概念数**: {TOTAL_CONCEPTS}",
        f"**类别数**: {len(SEMANTIC_CATEGORIES)}",
        "",
        "---",
    ]

    # SVD结果
    r = all_results.get("svd", {})
    lines.append("\n## 1. SVD分解（线性基线）")
    lines.append(f"- Bias matrix shape: {r.get('matrix_shape', 'N/A')}")
    if "cumulative_variance" in r:
        cv = r["cumulative_variance"]
        lines.append(f"\n### 累计解释方差")
        lines.append("| K | 累计方差 |")
        lines.append("|---|---------|")
        for k in [5, 10, 15, 20, 25, 30]:
            if k <= len(cv):
                lines.append(f"| {k} | {cv[k-1]*100:.1f}% |")

    # 因子-top概念
    if "factor_top_concepts" in r:
        lines.append(f"\n### SVD因子语义（前15个）")
        lines.append("| Factor | 方差 | Top+ | Top- |")
        lines.append("|--------|------|------|------|")
        for fname, finfo in list(r["factor_top_concepts"].items())[:15]:
            var = finfo.get("variance", 0) * 100
            pos = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_positive"][:3])
            neg = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_negative"][:3])
            lines.append(f"| {fname} | {var:.1f}% | {pos} | {neg} |")

    # Autoencoder结果
    ae = all_results.get("autoencoder", {})
    if ae:
        lines.append("\n## 2. Autoencoder非线性分解")
        lines.append("| 潜在维度 | AE方差解释 | SVD基线 | AE提升 |")
        lines.append("|---------|-----------|---------|--------|")
        for kname, kinfo in sorted(ae.items()):
            kdim = kinfo["latent_dim"]
            ae_var = kinfo["explained_variance_ratio"] * 100
            svd_var = kinfo["svd_baseline"] * 100
            imp = kinfo["improvement"] * 100
            lines.append(f"| {kdim} | {ae_var:.1f}% | {svd_var:.1f}% | {imp:+.1f}% |")

        # AE因子语义（latent_20）
        if "latent_20" in ae:
            lines.append(f"\n### AE因子语义（latent_20, 前10个）")
            lines.append("| Factor | Top+ | Top- |")
            lines.append("|--------|------|------|")
            for fname, finfo in list(ae["latent_20"]["factor_top_concepts"].items())[:10]:
                pos = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_positive"][:3])
                neg = ", ".join(f"{c}({s:.2f})" for c, s in finfo["top_negative"][:3])
                lines.append(f"| {fname} | {pos} | {neg} |")

    # 因子-属性关联
    attr = all_results.get("factor_attribute", {})
    if attr:
        lines.append("\n## 3. 因子-属性关联 (ANOVA eta²)")
        lines.append("\n### SVD因子")
        lines.append("| 属性 | 最佳因子 | eta² |")
        lines.append("|------|---------|------|")
        svd_attrs = sorted(attr.get("svd", {}).items(), key=lambda x: -x[1]["eta_squared"])
        for key, val in svd_attrs[:20]:
            lines.append(f"| {key} | factor_{val['best_factor']} | {val['eta_squared']:.3f} |")

        if attr.get("autoencoder"):
            lines.append("\n### Autoencoder因子")
            lines.append("| 属性 | 最佳因子 | eta² |")
            lines.append("|------|---------|------|")
            ae_attrs = sorted(attr["autoencoder"].items(), key=lambda x: -x[1]["eta_squared"])
            for key, val in ae_attrs[:20]:
                lines.append(f"| {key} | ae_factor_{val['best_factor']} | {val['eta_squared']:.3f} |")

    # 概念算术
    arith = all_results.get("arithmetic", {})
    if arith:
        lines.append("\n## 4. 概念算术测试")
        if "raw_avg_same_category" in arith or "svd_avg_same_category" in arith:
            raw_avg = arith.get("raw_avg_same_category", 0)
            svd_avg = arith.get("svd_avg_same_category", 0)
            ae_avg = arith.get("ae_avg_same_category", 0)
            cross_avg = arith.get("cross_category_avg", 0)
            lines.append(f"\n| 空间 | 同类平均 | 跨类平均 |")
            lines.append(f"|------|---------|---------|")
            lines.append(f"| 原始空间 | {raw_avg:.3f} | {cross_avg:.3f} |")
            lines.append(f"| SVD空间 | {svd_avg:.3f} | - |")
            lines.append(f"| AE空间 | {ae_avg:.3f} | - |")

        lines.append(f"\n### 同类算术详情")
        lines.append("| 对 | 原始 | SVD | AE |")
        lines.append("|----|------|-----|----|")
        pairs = [("dog", "cat"), ("car", "bus"), ("apple", "banana"),
                 ("mountain", "valley"), ("chair", "table"), ("doctor", "nurse"),
                 ("bread", "cake"), ("shirt", "dress")]
        for w1, w2 in pairs:
            raw = arith.get("same_category", {}).get(f"{w1}→{w2}", 0)
            svd = arith.get("same_category", {}).get(f"{w1}→{w2}_svd", 0)
            ae = arith.get("same_category", {}).get(f"{w1}→{w2}_ae", 0)
            lines.append(f"| {w1}→{w2} | {raw:.3f} | {svd:.3f} | {ae:.3f} |")

    # 结论
    lines.append("\n## 5. 结论")
    lines.append("")

    report_path = output_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")
    return report_path


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("Stage456: 大规模概念验证 + 非线性语义因子分解")
    print(f"Categories: {len(SEMANTIC_CATEGORIES)}, Concepts: {TOTAL_CONCEPTS}")
    print("=" * 60)

    t0 = time.time()

    # 1. 加载模型
    print("\n[1/6] Loading DeepSeek-7B...")
    model, tokenizer, layer_count, neuron_dim, hidden_dim = load_model(DEEPSEEK7B_MODEL_PATH)

    # 2. 提取激活
    print(f"\n[2/6] Extracting activations ({TOTAL_CONCEPTS} concepts)...")
    all_activations = extract_all_activations(model, tokenizer, SEMANTIC_CATEGORIES, layer_count)

    # 释放模型
    print("\n  Releasing model...")
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. 构建偏置矩阵
    # 使用最后5层
    target_layers = list(range(layer_count - 5, layer_count))
    print(f"\n[3/6] Building bias matrix (target layers: {target_layers})...")
    bias_matrix, concept_labels, category_labels = build_bias_matrices(
        all_activations, SEMANTIC_CATEGORIES, target_layers
    )
    print(f"  Bias matrix shape: {bias_matrix.shape}")
    print(f"  Concepts: {len(concept_labels)}, Categories: {len(set(category_labels))}")

    # 4. SVD分析
    print(f"\n[4/6] SVD decomposition...")
    n_components = min(30, bias_matrix.shape[0] - 1, bias_matrix.shape[1])
    svd_results, svd_components = svd_analysis(bias_matrix, concept_labels, category_labels, n_components)

    # 5. Autoencoder分析
    print(f"\n[5/6] Autoencoder nonlinear decomposition...")
    ae_results = autoencoder_analysis(bias_matrix, concept_labels, category_labels, latent_dims=[10, 20, 30])

    # 6. 因子-属性 + 算术测试
    print(f"\n[6/6] Factor analysis & arithmetic tests...")
    attr_results = factor_attribute_analysis(svd_components, ae_results, concept_labels, category_labels, SEMANTIC_CATEGORIES)
    arith_results = concept_arithmetic_test(bias_matrix, concept_labels, category_labels, svd_components, ae_results)

    # 汇总
    all_results = {
        "svd": {
            "matrix_shape": list(bias_matrix.shape),
            **svd_results,
        },
        "autoencoder": ae_results,
        "factor_attribute": attr_results,
        "arithmetic": arith_results,
        "meta": {
            "n_concepts": len(concept_labels),
            "n_categories": len(set(category_labels)),
            "target_layers": target_layers,
            "elapsed_seconds": time.time() - t0,
        },
    }

    # 保存
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {summary_path}")

    # 生成报告
    generate_report(all_results, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Stage456 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
