"""
CCML(Phase 33): 关键验证 — 在跳到"流形+函数"之前先排除更简单的解释
=============================================================================
Phase 32批评分析:

  ❌ 最大逻辑跳跃: 跨层一致性 ≠ 函数存在性证明
     替代解释A: h_L2 = A·h_L1 (线性变换) → w_L2 = A^{-T} w_L1 → 权重不同但预测一致
     替代解释B: 高维空间probe不唯一 → cos≈0.5只是说明解不唯一

  ❌ KL实验误读: probe方向≠因果方向, KL相等只说明probe方向不特殊

  ❌ 流形证据不足: 没有测intrinsic dim, 没有测geodesic, 没有测曲率

  ❌ Lasso过度纠偏: 从信任Ridge到信任Lasso, 两边都极端

Phase 33核心任务(3个关键验证, 必须按顺序):
  33A: ★★★★★★★★★★★ 层间变换线性性测试 (最关键!)
    → 拟合 h_L2 = A·h_L1 + b, 测R²
    → 如果R²很高(>0.99) → 线性变换足够解释 → 不需要"非线性函数"
    → 如果R²低 → 确实有非线性 → 流形框架有道理
    → 同时检查: A是否可逆? A^{-T}w_L1 ≈ w_L2? (验证替代解释A)

  33B: ★★★★★★★★★ 内在维度估计
    → MLE (Maximum Likelihood Estimation) 方法
    → 局部PCA rank分析
    → 回答: 语义空间到底是几维的?

  33C: ★★★★★★★ 梯度归因: 找真正的因果方向
    → 对edible属性, 计算d(logit_edible)/d(h) (梯度方向)
    → 对比: 梯度方向 vs probe方向 vs 随机方向
    → 如果梯度方向有特殊因果效应 → 找到了真正的因果机制
    → 如果梯度方向也和随机方向等效 → 因果方向是分布式的
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from collections import defaultdict

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U)


def compute_cos(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# 数据定义
# ============================================================================

CONCEPT_DATASET = {
    "apple":      {"edible":1, "animacy":0, "size":0.5, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "orange":     {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "banana":     {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "strawberry": {"edible":1, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "grape":      {"edible":1, "animacy":0, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "cherry":     {"edible":1, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "lemon":      {"edible":1, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "mango":      {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "peach":      {"edible":1, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "pear":       {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "watermelon": {"edible":1, "animacy":0, "size":0.9, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.3, "function_food":1},
    "pineapple":  {"edible":1, "animacy":0, "size":0.7, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "blueberry":  {"edible":1, "animacy":0, "size":0.1, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "coconut":    {"edible":1, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.3, "function_food":1},
    "tomato":     {"edible":1, "animacy":0, "size":0.3, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "kiwi":       {"edible":1, "animacy":0, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "plum":       {"edible":1, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "fig":        {"edible":1, "animacy":0, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":1},
    "lime":       {"edible":1, "animacy":0, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":1},
    "melon":      {"edible":1, "animacy":0, "size":0.7, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.3, "function_food":1},
    "dog":        {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":0},
    "cat":        {"edible":0, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":0},
    "elephant":   {"edible":0, "animacy":1, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.9, "function_food":0},
    "eagle":      {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.7, "function_food":0},
    "salmon":     {"edible":1, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.3, "function_food":1},
    "horse":      {"edible":0, "animacy":1, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.9, "function_food":0},
    "cow":        {"edible":1, "animacy":1, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.9, "function_food":1},
    "pig":        {"edible":1, "animacy":1, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.9, "function_food":1},
    "bird":       {"edible":0, "animacy":1, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":0},
    "fish":       {"edible":1, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.2, "function_food":1},
    "snake":      {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "frog":       {"edible":0, "animacy":1, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "bee":        {"edible":0, "animacy":1, "size":0.1, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.7, "function_food":0},
    "ant":        {"edible":0, "animacy":1, "size":0.05,"color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.7, "function_food":0},
    "bear":       {"edible":0, "animacy":1, "size":0.9, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.9, "function_food":0},
    "rabbit":     {"edible":0, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":0},
    "deer":       {"edible":0, "animacy":1, "size":0.7, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":0},
    "whale":      {"edible":0, "animacy":1, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":0},
    "chicken":    {"edible":1, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.8, "function_food":1},
    "shark":      {"edible":0, "animacy":1, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.3, "function_food":0},
    "hammer":     {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.2, "function_food":0},
    "knife":      {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.2, "function_food":0},
    "chair":      {"edible":0, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "shirt":      {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "car":        {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "book":       {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "shoe":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.2, "function_food":0},
    "ball":       {"edible":0, "animacy":0, "size":0.3, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.3, "function_food":0},
    "cup":        {"edible":0, "animacy":0, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.2, "function_food":0},
    "pen":        {"edible":0, "animacy":0, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "table":      {"edible":0, "animacy":0, "size":0.7, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "door":       {"edible":0, "animacy":0, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "rock":       {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.2, "function_food":0},
    "key":        {"edible":0, "animacy":0, "size":0.1, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "plate":      {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.3, "function_food":0},
    "bottle":     {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "clock":      {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.3, "function_food":0},
    "lamp":       {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "tree":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "flower":     {"edible":0, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0,
                   "material_organic":1, "shape_round":1, "temperature_cold":0.5, "function_food":0},
    "cloud":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "water":      {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":1},
    "fire":       {"edible":0, "animacy":0, "size":0.5, "color_red":1, "color_green":0, "color_yellow":1,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.0, "function_food":0},
    "grass":      {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":0},
    "sand":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.5, "function_food":0},
    "snow":       {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.0, "function_food":0},
    "rain":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.1, "function_food":0},
    "mountain":   {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "river":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "ocean":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.2, "function_food":0},
    "sun":        {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.0, "function_food":0},
    "moon":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.2, "function_food":0},
    "star":       {"edible":0, "animacy":0, "size":0.1, "color_red":0, "color_green":0, "color_yellow":1,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.0, "function_food":0},
    "stone":      {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.3, "function_food":0},
    "leaf":       {"edible":0, "animacy":0, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0,
                   "material_organic":1, "shape_round":0, "temperature_cold":0.5, "function_food":0},
    "ice":        {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.0, "function_food":0},
    "wind":       {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.2, "function_food":0},
    "earth":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":1, "temperature_cold":0.3, "function_food":0},
}

ATTR_NAMES = ["edible", "animacy", "size", "color_red", "color_green", "color_yellow",
              "material_organic", "shape_round", "temperature_cold", "function_food"]

CONTEXT_TEMPLATES = [
    "The {word} is here",
    "I see a {word}",
    "The {word} was found",
    "Look at that {word}",
    "A {word} appeared",
    "This {word} looks nice",
    "Every {word} has features",
]


def find_token_index(tokens, target_word):
    target_lower = target_word.lower().strip()
    for i, t in enumerate(tokens):
        if t.lower().strip() == target_lower:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:3] == target_lower[:3]:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:2] == target_lower[:2]:
            return i
    return -1


def collect_hs_with_template(model, tokenizer, device, words, layer_idx, template):
    layers = get_layers(model)
    if layer_idx >= len(layers):
        return None, None
    
    target_layer = layers[layer_idx]
    all_hs = []
    valid_words = []
    
    for word in words:
        sent = template.format(word=word)
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        
        dep_idx = find_token_index(tokens_list, word)
        if dep_idx < 0:
            continue
        
        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()
        
        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()
        
        if 'h' in captured:
            h_vec = captured['h'][0, dep_idx, :]
            all_hs.append(h_vec)
            valid_words.append(word)
    
    if len(all_hs) == 0:
        return None, None
    return np.array(all_hs), valid_words


# ============================================================================
# 33A: 层间变换线性性测试 (最关键实验!)
# ============================================================================

def expA_layer_transformation_linearity(model_name, model, tokenizer, device, layers_to_test):
    """
    33A: 测试 h_L2 = A·h_L1 + b 是否成立
    
    如果R²很高 → 线性变换足够解释层间关系 → 不需要"非线性函数"
    如果R²低 → 确实有非线性 → 流形框架有道理
    
    同时验证: A^{-T}w_L1 ≈ w_L2 (替代解释A)
    """
    print(f"\n{'='*70}")
    print(f"33A: 层间变换线性性测试 — 最关键的实验!")
    print(f"     如果h_L2 = A·h_L1 + b (R²>0.99) → 不需要非线性函数假说")
    print(f"     如果R²低 → 确认非线性 → 流形框架有道理")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "A", "experiment": "layer_transformation_linearity", "pairs": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    
    # 收集多上下文的hidden states (增加样本量)
    layer_H = {}
    layer_valid_words = {}
    
    for layer_idx in layers_to_test:
        all_hs = []
        all_words = []
        
        for template in CONTEXT_TEMPLATES:
            H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, template)
            if H is not None:
                all_hs.append(H)
                all_words.extend(valid_words)
        
        if len(all_hs) > 0:
            layer_H[layer_idx] = np.vstack(all_hs)
            layer_valid_words[layer_idx] = all_words
        
        print(f"  Layer {layer_idx}: {layer_H[layer_idx].shape[0]} samples" if layer_idx in layer_H else f"  Layer {layer_idx}: FAILED")
    
    # 对每对相邻层, 测试线性变换
    layer_indices = sorted(layer_H.keys())
    
    for i in range(len(layer_indices) - 1):
        l1 = layer_indices[i]
        l2 = layer_indices[i + 1]
        
        H1 = layer_H[l1]
        H2 = layer_H[l2]
        
        n = min(H1.shape[0], H2.shape[0])
        if n < 20:
            print(f"  L{l1}→L{l2}: 样本不足 ({n}), 跳过")
            continue
        
        H1 = H1[:n]
        H2 = H2[:n]
        
        d = H1.shape[1]
        
        print(f"\n  --- L{l1} → L{l2} (n={n}, d={d}) ---")
        
        # ===== 1. 线性拟合: h_L2 = A·h_L1 + b =====
        # 使用Ridge回归, 逐维度拟合
        H1_mean = np.mean(H1, axis=0, keepdims=True)
        H2_mean = np.mean(H2, axis=0, keepdims=True)
        H1_centered = H1 - H1_mean
        H2_centered = H2 - H2_mean
        
        # 方式1: 整体Ridge (所有维度一起)
        # A的每一行 = Ridge预测H2的一个维度
        # 这等价于: H2_centered = A @ H1_centered.T
        
        # 用SVD求解 (更高效)
        # A = H2_centered.T @ H1_centered @ pinv(H1_centered.T @ H1_centered)
        # 加正则化: A = H2_centered.T @ H1_centered @ (H1_centered.T @ H1_centered + λI)^{-1}
        
        lam = 1.0  # Ridge正则化参数
        XtX = H1_centered.T @ H1_centered + lam * np.eye(d)
        XtY = H1_centered.T @ H2_centered
        A = np.linalg.solve(XtX, XtY).T  # [d, d]
        b = (H2_mean - H1_mean @ A.T).flatten()
        
        # 预测
        H2_pred = (A @ H1_centered.T).T + H2_mean
        
        # R² (逐维度)
        r2_per_dim = []
        for dim in range(d):
            ss_res = np.sum((H2_centered[:, dim] - (H2_pred - H2_mean)[:, dim]) ** 2)
            ss_tot = np.sum(H2_centered[:, dim] ** 2)
            r2_dim = 1 - ss_res / max(ss_tot, 1e-10)
            r2_per_dim.append(r2_dim)
        
        r2_overall = 1 - np.sum((H2_centered - (H2_pred - H2_mean)) ** 2) / max(np.sum(H2_centered ** 2), 1e-10)
        
        # 残差分析
        residuals = H2_centered - (H2_pred - H2_mean)
        residual_norms = np.linalg.norm(residuals, axis=1)
        h2_norms = np.linalg.norm(H2_centered, axis=1)
        relative_errors = residual_norms / np.maximum(h2_norms, 1e-10)
        
        print(f"    线性变换 R² (整体): {r2_overall:.6f}")
        print(f"    R² (逐维度): mean={np.mean(r2_per_dim):.6f}, "
              f"min={np.min(r2_per_dim):.6f}, max={np.max(r2_per_dim):.6f}")
        print(f"    相对误差: mean={np.mean(relative_errors):.6f}, "
              f"median={np.median(relative_errors):.6f}, "
              f"90th={np.percentile(relative_errors, 90):.6f}")
        
        # ===== 2. A的性质分析 =====
        # A是否可逆?
        s_A = np.linalg.svd(A, compute_uv=False)
        cond_A = s_A[0] / max(s_A[-1], 1e-10)
        print(f"    A的condition number: {cond_A:.2f}")
        print(f"    A的奇异值: top5={s_A[:5].tolist()}, bottom5={s_A[-5:].tolist()}")
        
        # ===== 3. 替代解释A验证: A^{-T}w_L1 ≈ w_L2? =====
        print(f"\n    ★★★ 替代解释A验证: A^{{-T}}w_L1 ≈ w_L2? ★★★")
        
        V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
        
        # 在L1和L2分别训练probe (用第一个模板)
        H1_single, vw1 = collect_hs_with_template(model, tokenizer, device, concepts, l1, CONTEXT_TEMPLATES[0])
        H2_single, vw2 = collect_hs_with_template(model, tokenizer, device, concepts, l2, CONTEXT_TEMPLATES[0])
        
        if H1_single is not None and H2_single is not None and len(vw1) >= 20 and len(vw2) >= 20:
            # 找共同词
            common = sorted(set(vw1) & set(vw2))
            if len(common) >= 15:
                idx1 = [vw1.index(w) for w in common]
                idx2 = [vw2.index(w) for w in common]
                
                H1_c = H1_single[idx1]
                H2_c = H2_single[idx2]
                V_c = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in common])
                
                H1m = np.mean(H1_c, axis=0, keepdims=True)
                H2m = np.mean(H2_c, axis=0, keepdims=True)
                Vm = np.mean(V_c, axis=0, keepdims=True)
                
                # 训练probe
                n_attr = len(ATTR_NAMES)
                W1 = np.zeros((n_attr, d))
                W2 = np.zeros((n_attr, d))
                
                for ai, attr in enumerate(ATTR_NAMES):
                    ridge1 = Ridge(alpha=1.0)
                    ridge1.fit(H1_c - H1m, V_c[:, ai] - Vm[0, ai])
                    W1[ai] = ridge1.coef_
                    
                    ridge2 = Ridge(alpha=1.0)
                    ridge2.fit(H2_c - H2m, V_c[:, ai] - Vm[0, ai])
                    W2[ai] = ridge2.coef_
                
                # 验证: A^{-T} w_L1 ≈ w_L2?
                try:
                    A_inv_T = np.linalg.inv(A).T  # A^{-T}
                    
                    attr_transform_coss = {}
                    for ai, attr in enumerate(ATTR_NAMES):
                        w1 = W1[ai]
                        w2_actual = W2[ai]
                        w2_predicted = A_inv_T @ w1  # 替代解释A的预测
                        
                        cos_actual = compute_cos(w1, w2_actual)
                        cos_predicted = compute_cos(w2_predicted, w2_actual)
                        
                        attr_transform_coss[attr] = {
                            "cos_w1_w2": float(cos_actual),
                            "cos_AinvTw1_w2": float(cos_predicted),
                        }
                        
                        print(f"      {attr}: cos(w1,w2)={cos_actual:.4f}, "
                              f"cos(A^{{-T}}w1, w2)={cos_predicted:.4f} "
                              f"{'✓ 替代解释成立!' if cos_predicted > 0.8 else '✗ 替代解释不成立'}")
                except np.linalg.LinAlgError:
                    print(f"      A不可逆, 无法计算A^{{-T}}")
                    attr_transform_coss = {"error": "A not invertible"}
                
                # ===== 4. 残差是否有结构? =====
                # 如果残差与属性相关 → 非线性部分编码了语义
                print(f"\n    ★★★ 残差与属性的相关性 ★★★")
                
                # 用残差预测属性
                residuals_common = H2_c - (A @ (H1_c - H1m).T).T - H2m
                
                residual_attr_r2 = {}
                for ai, attr in enumerate(ATTR_NAMES[:3]):  # 只测3个属性
                    ridge_res = Ridge(alpha=1.0)
                    ridge_res.fit(residuals_common, V_c[:, ai] - Vm[0, ai])
                    pred_res = ridge_res.predict(residuals_common)
                    ss_res = np.sum((V_c[:, ai] - Vm[0, ai] - pred_res) ** 2)
                    ss_tot = np.sum((V_c[:, ai] - Vm[0, ai]) ** 2)
                    r2_res = 1 - ss_res / max(ss_tot, 1e-10)
                    residual_attr_r2[attr] = float(r2_res)
                    print(f"      残差→{attr}: R²={r2_res:.4f} {'← 残差编码了语义!' if r2_res > 0.1 else ''}")
                
                # ===== 5. 非线性检验: 二次项 =====
                print(f"\n    ★★★ 非线性检验: 添加二次项是否提升R²? ★★★")
                
                # 构造二次特征: [h, h²的对角部分]
                H1_quad = np.hstack([H1_c - H1m, (H1_c - H1m) ** 2])
                
                for ai, attr in enumerate(["edible", "animacy", "size"]):
                    # 线性模型
                    ridge_lin = Ridge(alpha=1.0)
                    ridge_lin.fit(H1_c - H1m, V_c[:, ai] - Vm[0, ai])
                    r2_lin = 1 - np.sum((V_c[:, ai] - Vm[0, ai] - ridge_lin.predict(H1_c - H1m)) ** 2) / max(
                        np.sum((V_c[:, ai] - Vm[0, ai]) ** 2), 1e-10)
                    
                    # 二次模型
                    ridge_quad = Ridge(alpha=1.0)
                    ridge_quad.fit(H1_quad, V_c[:, ai] - Vm[0, ai])
                    r2_quad = 1 - np.sum((V_c[:, ai] - Vm[0, ai] - ridge_quad.predict(H1_quad)) ** 2) / max(
                        np.sum((V_c[:, ai] - Vm[0, ai]) ** 2), 1e-10)
                    
                    print(f"      {attr}: 线性R²={r2_lin:.4f}, 二次R²={r2_quad:.4f}, "
                          f"提升={r2_quad - r2_lin:.4f} "
                          f"{'← 非线性显著!' if r2_quad - r2_lin > 0.05 else ''}")
        else:
            attr_transform_coss = {"error": "insufficient common words"}
            residual_attr_r2 = {}
        
        pair_key = f"L{l1}_L{l2}"
        results["pairs"][pair_key] = {
            "r2_overall": float(r2_overall),
            "r2_per_dim_mean": float(np.mean(r2_per_dim)),
            "r2_per_dim_min": float(np.min(r2_per_dim)),
            "relative_error_mean": float(np.mean(relative_errors)),
            "relative_error_median": float(np.median(relative_errors)),
            "condition_number_A": float(cond_A),
            "singular_values_top5": [float(x) for x in s_A[:5]],
            "singular_values_bottom5": [float(x) for x in s_A[-5:]],
            "attr_transform_coss": attr_transform_coss,
            "residual_attr_r2": residual_attr_r2,
            "n_samples": n,
        }
    
    return results


# ============================================================================
# 33B: 内在维度估计
# ============================================================================

def expB_intrinsic_dimension(model_name, model, tokenizer, device, layers_to_test):
    """
    33B: 估计语义空间的内在维度
    
    方法1: MLE (Maximum Likelihood Estimation) — Levina & Bickel (2004)
    方法2: 局部PCA rank分析
    方法3: TWO-NN (Facco et al. 2017)
    
    回答: 语义空间到底是几维的?
    """
    print(f"\n{'='*70}")
    print(f"33B: 内在维度估计 — 语义空间到底是几维的?")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "B", "experiment": "intrinsic_dimension", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集多上下文的hidden states
        all_hs = []
        for template in CONTEXT_TEMPLATES:
            H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, template)
            if H is not None:
                all_hs.append(H)
        
        if len(all_hs) == 0:
            continue
        
        H = np.vstack(all_hs)
        n, d = H.shape
        print(f"    样本数: {n}, 维度: {d}")
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        
        # ===== 方法1: MLE =====
        # 对每个点, 找k近邻, 估计局部维度
        # d_mle(i) = [1/(k-1)] * Σ_{j=1}^{k-1} log(T_k/T_j)^{-1}
        # 其中T_j是第j近邻的距离
        
        from scipy.spatial.distance import pdist, squareform
        
        # 子采样 (如果太大)
        if n > 300:
            idx = np.random.choice(n, 300, replace=False)
            H_sub = H_centered[idx]
        else:
            H_sub = H_centered
        
        n_sub = H_sub.shape[0]
        
        # 计算距离矩阵
        D = squareform(pdist(H_sub, 'euclidean'))
        
        k_values = [5, 10, 15, 20]
        mle_dims = {}
        
        for k in k_values:
            if k >= n_sub:
                continue
            
            local_dims = []
            for i in range(n_sub):
                # 找k+1个最近邻 (包括自身)
                dists = np.sort(D[i])
                dists = dists[1:k+1]  # 排除自身 (距离=0)
                
                if dists[-1] < 1e-10:
                    continue
                
                # MLE估计
                dim_estimates = []
                for j in range(1, k):
                    if dists[j-1] < 1e-10:
                        continue
                    ratio = dists[k-1] / dists[j-1]
                    if ratio > 1e10:
                        continue
                    dim_estimates.append(np.log(ratio))
                
                if len(dim_estimates) > 0:
                    local_d = (k - 1) / sum(dim_estimates)
                    if 0 < local_d < d:  # 合理范围
                        local_dims.append(local_d)
            
            if len(local_dims) > 0:
                mle_dims[str(k)] = {
                    "mean": float(np.mean(local_dims)),
                    "median": float(np.median(local_dims)),
                    "std": float(np.std(local_dims)),
                    "n_valid": len(local_dims),
                }
                print(f"    MLE (k={k}): mean={np.mean(local_dims):.1f}, "
                      f"median={np.median(local_dims):.1f}, std={np.std(local_dims):.1f}")
            else:
                mle_dims[str(k)] = {"error": "no valid estimates"}
        
        # ===== 方法2: 局部PCA rank =====
        # 在每个点的邻域做PCA, 看需要多少主成分解释95%方差
        
        k_pca = 15
        local_ranks = []
        
        for i in range(min(n_sub, 100)):  # 最多100个点
            dists_idx = np.argsort(D[i])[1:k_pca+1]
            neighbors = H_sub[dists_idx]
            
            if len(neighbors) < 5:
                continue
            
            # 局部PCA
            n_local = min(len(neighbors), d - 1)
            pca_local = PCA(n_components=n_local)
            pca_local.fit(neighbors)
            
            cumvar = np.cumsum(pca_local.explained_variance_ratio_)
            rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
            local_ranks.append(rank_95)
        
        if len(local_ranks) > 0:
            print(f"    局部PCA rank (95%方差): mean={np.mean(local_ranks):.1f}, "
                  f"median={np.median(local_ranks):.1f}, std={np.std(local_ranks):.1f}")
        
        # ===== 方法3: TWO-NN =====
        # 基于最近两个近邻距离的比值估计维度
        # d = 1 / (mean(log(r2/r1)))  where r1, r2 are distances to 1st and 2nd NN
        
        two_nn_dims = []
        for i in range(n_sub):
            dists_sorted = np.sort(D[i])
            r1 = dists_sorted[1]  # 第1近邻 (索引0是自身)
            r2 = dists_sorted[2]  # 第2近邻
            
            if r1 < 1e-10 or r2 / r1 > 1e6:
                continue
            
            mu = r2 / r1
            if mu > 1:
                d_est = 1.0 / np.log(mu)
                if 0 < d_est < d:
                    two_nn_dims.append(d_est)
        
        if len(two_nn_dims) > 0:
            print(f"    TWO-NN: mean={np.mean(two_nn_dims):.1f}, "
                  f"median={np.median(two_nn_dims):.1f}")
        
        # ===== 方法4: 全局PCA方差解释 =====
        pca_global = PCA()
        pca_global.fit(H_centered)
        cumvar_global = np.cumsum(pca_global.explained_variance_ratio_)
        
        dim_50 = int(np.searchsorted(cumvar_global, 0.50) + 1)
        dim_90 = int(np.searchsorted(cumvar_global, 0.90) + 1)
        dim_95 = int(np.searchsorted(cumvar_global, 0.95) + 1)
        dim_99 = int(np.searchsorted(cumvar_global, 0.99) + 1)
        
        print(f"    全局PCA: 50%→{dim_50}维, 90%→{dim_90}维, 95%→{dim_95}维, 99%→{dim_99}维")
        
        # ===== 方法5: 概念vs随机基线 =====
        # 如果概念集有内在结构 → 内在维度应该远小于d
        # 对比: 真实概念 vs 随机高斯点
        
        # 随机基线 (同等方差的高斯)
        H_cov = np.cov(H_centered.T)
        H_random = np.random.multivariate_normal(np.zeros(d), H_cov, size=n_sub)
        
        D_random = squareform(pdist(H_random, 'euclidean'))
        
        # MLE on random
        k_mle = 10
        random_dims = []
        for i in range(min(n_sub, 100)):
            dists = np.sort(D_random[i])
            dists = dists[1:k_mle+1]
            if dists[-1] < 1e-10:
                continue
            dim_estimates = []
            for j in range(1, k_mle):
                if dists[j-1] < 1e-10:
                    continue
                ratio = dists[k_mle-1] / dists[j-1]
                if ratio > 1e10:
                    continue
                dim_estimates.append(np.log(ratio))
            if len(dim_estimates) > 0:
                local_d = (k_mle - 1) / sum(dim_estimates)
                if 0 < local_d < d:
                    random_dims.append(local_d)
        
        if len(random_dims) > 0:
            print(f"    随机基线 MLE (k={k_mle}): mean={np.mean(random_dims):.1f}")
        
        results["layers"][str(layer_idx)] = {
            "mle_dims": mle_dims,
            "local_pca_rank": {
                "mean": float(np.mean(local_ranks)) if local_ranks else None,
                "median": float(np.median(local_ranks)) if local_ranks else None,
                "values": [int(x) for x in local_ranks[:50]],
            },
            "two_nn": {
                "mean": float(np.mean(two_nn_dims)) if two_nn_dims else None,
                "median": float(np.median(two_nn_dims)) if two_nn_dims else None,
            },
            "global_pca": {
                "dim_50": dim_50, "dim_90": dim_90, "dim_95": dim_95, "dim_99": dim_99,
            },
            "random_baseline_mle": {
                "mean": float(np.mean(random_dims)) if random_dims else None,
            },
            "n_samples": n,
            "d_model": d,
        }
    
    return results


# ============================================================================
# 33C: 梯度归因 — 找真正的因果方向
# ============================================================================

def expC_gradient_attribution(model_name, model, tokenizer, device, layers_to_test):
    """
    33C: 用梯度找到真正的因果方向
    
    方法: 对edible属性, 计算 d(P(edible|context))/d(h_L)
    - P(edible) 可以用"edible" token的logit来代理
    - 梯度方向 = 模型认为"让这个概念更可食用"的方向
    - 对比: 梯度方向 vs probe方向 vs 随机方向
    """
    print(f"\n{'='*70}")
    print(f"33C: 梯度归因 — 找真正的因果方向")
    print(f"     对比: 梯度方向 vs probe方向 vs 随机方向")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "C", "experiment": "gradient_attribution", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    # 找edible相关的token
    # 用"edible"和"food"作为目标token
    edible_tokens = ["edible", "food", "eat", "delicious", "tasty"]
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        layers = get_layers(model)
        if layer_idx >= len(layers):
            continue
        
        target_layer = layers[layer_idx]
        
        # 先收集hidden states并训练probe (用第一个模板)
        H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, CONTEXT_TEMPLATES[0])
        if H is None or len(H) < 30:
            continue
        
        valid_indices = [concepts.index(c) for c in valid_words if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        n_attr = len(ATTR_NAMES)
        d_model = H_centered.shape[1]
        
        # 训练Ridge probe
        W_probes = np.zeros((n_attr, d_model))
        probe_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            probe_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # ===== 梯度计算 =====
        # 对每个概念, 计算d(edible_logit)/d(h_L)
        
        # 找edible token的id
        edible_token_ids = []
        for tok_str in edible_tokens:
            tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            edible_token_ids.extend(tok_ids)
        edible_token_ids = list(set(edible_token_ids))
        print(f"    Edible token IDs: {edible_token_ids}")
        
        gradient_directions = {}  # {concept: {dir_name: grad_proj}}
        probe_directions = {}
        
        # 定义probe方向和随机方向 (在循环前)
        edible_probe = W_probes[0] / (np.linalg.norm(W_probes[0]) + 1e-10)
        animacy_probe = W_probes[1] / (np.linalg.norm(W_probes[1]) + 1e-10)
        np.random.seed(42)
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-10)
        
        # 用几个典型概念
        test_concepts = ["apple", "dog", "hammer"]
        
        for concept in test_concepts:
            if concept not in valid_words:
                continue
            
            # 用数值梯度(有限差分), 避免OOM
            # d(logit_edible)/d(h) ≈ [logit(h+ε·e_i) - logit(h-ε·e_i)] / (2ε)
            # 但d=2560-4096, 逐维度差分太慢
            # 替代: 用probe方向和随机方向的有限差分验证因果效力
            
            c_idx = valid_words.index(concept)
            h_orig = H[c_idx]
            
            # 计算原始edible logit
            h_tensor = torch.tensor(h_orig, dtype=model.lm_head.weight.dtype, 
                                   device=device).unsqueeze(0)
            with torch.no_grad():
                logits_orig = model.lm_head(h_tensor)
            if len(edible_token_ids) > 0:
                edible_logit_orig = float(logits_orig[0, edible_token_ids].mean())
            else:
                edible_logit_orig = 0.0
            
            # 沿各个方向做有限差分, 得到该方向的"梯度投影"
            # grad·dir ≈ [logit(h+ε·dir) - logit(h-ε·dir)] / (2ε)
            eps_fd = 0.01 * float(np.linalg.norm(h_orig - H_mean[0]))
            
            directions = {
                "probe_edible": edible_probe,
                "probe_animacy": animacy_probe,
                "random": rand_dir,
            }
            
            for dir_name, dir_vec in directions.items():
                h_plus = h_orig + eps_fd * dir_vec
                h_minus = h_orig - eps_fd * dir_vec
                
                h_plus_t = torch.tensor(h_plus, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                h_minus_t = torch.tensor(h_minus, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    logits_plus = model.lm_head(h_plus_t)
                    logits_minus = model.lm_head(h_minus_t)
                
                if len(edible_token_ids) > 0:
                    logit_plus = float(logits_plus[0, edible_token_ids].mean())
                    logit_minus = float(logits_minus[0, edible_token_ids].mean())
                else:
                    logit_plus = 0.0
                    logit_minus = 0.0
                
                grad_proj = (logit_plus - logit_minus) / (2 * eps_fd)
                
                # 存储为"等效梯度投影"
                if concept not in gradient_directions:
                    gradient_directions[concept] = {}
                gradient_directions[concept][dir_name] = float(grad_proj)
        
        # ===== 分析梯度方向 vs probe方向 =====
        print(f"\n    ★★★ 梯度投影 vs Probe方向 (数值有限差分) ★★★")
        
        grad_probe_coss = {}
        grad_random_coss = {}
        
        
        # 梯度投影比较 (数值梯度版本)
        print(f"\n    ★★★ 梯度投影 vs Probe方向 (数值有限差分) ★★★")
        for concept, grad_projs in gradient_directions.items():
            probe_ed = grad_projs.get("probe_edible", 0)
            probe_an = grad_projs.get("probe_animacy", 0)
            rand_proj = grad_projs.get("random", 0)
            print(f"      {concept}: ∂logit/∂(probe_edible)={probe_ed:.4f}, "
                  f"∂logit/∂(probe_animacy)={probe_an:.4f}, ∂logit/∂(random)={rand_proj:.4f}")
        
        # ===== 因果效力测试 =====
        print(f"\n    ★★★ 因果效力测试: 沿不同方向扰动对edible logit的影响 ★★★")
        
        eps_values = [0.1, 0.5, 1.0, 2.0]
        
        causal_test = {}
        
        for concept in ["apple", "dog", "hammer"]:
            if concept not in valid_words:
                continue
            
            c_idx = valid_words.index(concept)
            h_orig = H[c_idx]
            
            # 方向列表 (没有autograd梯度, 只用probe和随机)
            directions = {
                "probe_edible": edible_probe,
                "probe_animacy": animacy_probe,
                "random": rand_dir,
                "random2": np.random.randn(d_model) / (np.linalg.norm(np.random.randn(d_model)) + 1e-10),
            }
            
            # 先计算原始logit
            h_tensor_orig = torch.tensor(h_orig, dtype=model.lm_head.weight.dtype, 
                                        device=device).unsqueeze(0)
            with torch.no_grad():
                logits_orig = model.lm_head(h_tensor_orig)
            if len(edible_token_ids) > 0:
                edible_logit_orig = float(logits_orig[0, edible_token_ids].mean())
            else:
                edible_logit_orig = 0
            
            concept_causal = {}
            
            for dir_name, dir_vec in directions.items():
                logit_changes = []
                
                for eps in eps_values:
                    h_pert = h_orig + eps * dir_vec * np.linalg.norm(h_orig - H_mean[0])
                    
                    h_tensor = torch.tensor(h_pert, dtype=model.lm_head.weight.dtype, 
                                           device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits_pert = model.lm_head(h_tensor)
                    
                    if len(edible_token_ids) > 0:
                        edible_logit_pert = float(logits_pert[0, edible_token_ids].mean())
                    else:
                        edible_logit_pert = 0
                    
                    delta = edible_logit_pert - edible_logit_orig
                    logit_changes.append(float(delta))
                
                concept_causal[dir_name] = {
                    "eps_values": eps_values,
                    "edible_logit_changes": logit_changes,
                }
            
            causal_test[concept] = concept_causal
            
            # 打印
            print(f"      {concept} (edible={CONCEPT_DATASET[concept]['edible']}):")
            for dir_name, data in concept_causal.items():
                print(f"        {dir_name}: Δlogit = {data['edible_logit_changes']}")
        
        results["layers"][str(layer_idx)] = {
            "probe_r2": probe_r2,
            "gradient_projections": {k: v for k, v in gradient_directions.items()} if isinstance(list(gradient_directions.values())[0] if gradient_directions else {}, dict) else {},
            "grad_probe_coss": grad_probe_coss,
            "grad_random_coss": grad_random_coss,
            "causal_test": causal_test,
            "edible_token_ids": edible_token_ids,
        }
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 33: 关键验证")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=层间线性性, 2=内在维度, 3=梯度归因")
    args = parser.parse_args()
    
    model_name = args.model
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    layers_to_test = [0, n_layers//4, n_layers//2, 3*n_layers//4]
    layers_to_test = sorted(set(layers_to_test))
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    print(f"测试层: {layers_to_test}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.exp in [0, 1]:
        resA = expA_layer_transformation_linearity(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccml_expA_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpA] 层间线性性结果已保存")
    
    if args.exp in [0, 2]:
        resB = expB_intrinsic_dimension(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccml_expB_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpB] 内在维度结果已保存")
    
    if args.exp in [0, 3]:
        # 梯度测试只用2个层,避免OOM
        grad_layers = [layers_to_test[0], layers_to_test[-1]] if len(layers_to_test) > 2 else layers_to_test
        resC = expC_gradient_attribution(model_name, model, tokenizer, device, grad_layers)
        with open(os.path.join(output_dir, f"ccml_expC_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resC, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpC] 梯度归因结果已保存")
    
    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
