"""
CCMI(Phase 29): 潜在语义结构破解——CCA/Sparse/非线性结构分析
=============================================================================
Phase 28核心发现(经批判修正):
  ★★★ "PCA提升" ≠ "属性空间存在"!
    → PCA按方差排序, 找到的是"语义主轴", 不是"属性坐标系"
    → 需要用CCA/LDA等监督方法找最对齐属性的子空间

  ★★★ R²≈0.15的三个替代解释:
    → (A) hidden state大部分是非语义信息(token, freq, syntax, position)
    → (B) 人工标签v ≠ 模型内部变量z
    → (C) 强非线性关系

  ★★★ 属性应是连续latent features, 不是离散标签

  ★★★ 三个独立问题:
    → (1) 属性是否存在  ✔ 有证据(干预+分类)
    → (2) 属性是否线性  ❌ 被否定
    → (3) 属性是否独立  ❌ 观测空间被否定

Phase 29核心任务(3个最关键实验):
  29A: ★★★★★★★★★★★ CCA监督子空间 (比PCA关键!)
    → 用CCA找最对齐属性的子空间
    → 在CCA空间中检验edible跨条件cos
    → 如果cos>0.8 → 接近"属性坐标系"!

  29B: ★★★★★★★★★ Sparse Coding反推潜在z
    → 用sparse coding/autoencoder在h上找低维潜在结构
    → 检验潜在维度是否可解释为语义属性
    → 关键: 不是用人工标签回归h, 而是从h中反推出结构!

  29C: ★★★★★★★★★★ 非线性回归的结构分析
    → MLP: h = MLP(v) + ε
    → 重点不是R², 而是MLP学到的结构:
      - Jacobian ∂h/∂v 是否稳定?
      - 哪些属性发生coupling?
      - 交互项有多大?
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
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U)

# 修正版compute_cos
def compute_cos(v1, v2):
    """计算两个向量的余弦相似度 (双归一化)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# 数据定义: 扩展概念集 + 更丰富的属性
# ============================================================================

# 80个概念, 标注12个属性 (比Phase 28的8个更丰富)
# 新增: shape(形状), temperature(温度), function_category(功能类别)
CONCEPT_DATASET = {
    # 水果(20个, 可食=1, animate=0)
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
    # 动物(20个)
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
    # 工具/物品(20个)
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
    "wall":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0,
                   "material_organic":0, "shape_round":0, "temperature_cold":0.3, "function_food":0},
    "window":     {"edible":0, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0,
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
    # 自然物(20个)
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

# 条件解耦数据 (大样本, 每组10+概念)
CONDITIONAL_DATA = {
    "red_edible":   ["apple", "strawberry", "cherry", "tomato", "watermelon",
                     "flower", "fire", "ball"],
    "red_inedible": ["hammer", "knife", "shirt", "car", "book",
                     "door", "wall", "key"],
    "green_edible": ["grape", "pear", "kiwi", "lime", "melon",
                     "water", "lettuce"],
    "green_inedible": ["grass", "tree", "frog", "leaf", "moss",
                       "fern", "cactus", "emerald"],
}

TEMPLATE = "The {word} is here"


# ============================================================================
# 工具函数
# ============================================================================

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
    for i, t in enumerate(tokens):
        if len(t.strip()) > 0 and t.lower().strip()[0] == target_lower[0]:
            return i
    return -1


def collect_hs_for_words(model, tokenizer, device, words, layer_idx):
    """收集一组词在指定层的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        return None, None
    
    target_layer = layers[layer_idx]
    all_hs = []
    valid_words = []
    
    for word in words:
        sent = TEMPLATE.format(word=word)
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
# 29A: CCA监督子空间 — 找最对齐属性的子空间
# ============================================================================

def expA_cca_supervised_subspace(model_name, model, tokenizer, device, layers_to_test):
    """29A: 用CCA找最对齐属性的子空间, 比PCA更关键"""
    print(f"\n{'='*70}")
    print(f"ExpA: CCA监督子空间 — 找最对齐属性的子空间 (比PCA关键!)")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "A", "experiment": "cca_supervised_subspace", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        H, valid_concepts = collect_hs_for_words(model, tokenizer, device, concepts, layer_idx)
        
        if H is None or len(H) < 30:
            print(f"    数据不足({len(H) if H is not None else 0}), 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_concepts if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        layer_result = {}
        
        # ===== Step 1: CCA =====
        # CCA找H和V之间的最大相关子空间
        n_components_cca = min(len(ATTR_NAMES), V_centered.shape[1], H_centered.shape[1], len(H) - 1)
        n_components_cca = min(n_components_cca, len(ATTR_NAMES))
        
        try:
            cca = CCA(n_components=n_components_cca)
            H_cca, V_cca_proj = cca.fit_transform(H_centered, V_centered)  # [n, n_comp]
        except Exception as e:
            print(f"    CCA失败: {e}, 跳过")
            continue
        
        # CCA相关系数
        cca_corrs = []
        for i in range(n_components_cca):
            corr = np.corrcoef(H_cca[:, i], V_cca_proj[:, i])[0, 1]
            cca_corrs.append(float(abs(corr)))
        
        print(f"    CCA相关系数: {[f'{c:.3f}' for c in cca_corrs]}")
        
        # ===== Step 2: 在CCA空间中做属性回归 =====
        ridge_cca = Ridge(alpha=1.0)
        ridge_cca.fit(V_centered, H_cca)
        H_cca_pred = ridge_cca.predict(V_centered)
        
        ss_res = np.sum((H_cca - H_cca_pred) ** 2)
        ss_tot = np.sum(H_cca ** 2)
        r2_cca = 1 - ss_res / max(ss_tot, 1e-10)
        
        A_cca = ridge_cca.coef_  # [n_comp_cca, n_attr]
        n_attr_cca = min(A_cca.shape[1], len(ATTR_NAMES))
        
        # CCA空间中属性方向间的余弦
        cos_matrix_cca = np.zeros((n_attr_cca, n_attr_cca))
        for i in range(n_attr_cca):
            for j in range(n_attr_cca):
                cos_matrix_cca[i, j] = compute_cos(A_cca[:, i], A_cca[:, j])
        
        # ===== Step 3: 条件解耦 — 在CCA空间中 =====
        cond_data_cca = {}
        for group_name, words in CONDITIONAL_DATA.items():
            hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
            if hs is not None and len(hs) >= 3:
                hs_centered = hs - H_mean
                hs_cca = cca.transform(hs_centered)
                cond_data_cca[group_name] = {"hs_cca": hs_cca, "words": valid}
        
        # 计算edible跨条件cos (在CCA空间中)
        cross_condition_cos_cca = None
        if "red_edible" in cond_data_cca and "green_edible" in cond_data_cca:
            re = cond_data_cca["red_edible"]["hs_cca"]
            ge = cond_data_cca["green_edible"]["hs_cca"]
            ri = cond_data_cca.get("red_inedible", {}).get("hs_cca")
            gi = cond_data_cca.get("green_inedible", {}).get("hs_cca")
            
            if ri is not None and gi is not None:
                # edible方向 = 可食质心 - 不可食质心
                edible_red = np.mean(re, axis=0) - np.mean(ri, axis=0)
                edible_green = np.mean(ge, axis=0) - np.mean(gi, axis=0)
                cross_condition_cos_cca = compute_cos(edible_red, edible_green)
        
        # color跨条件cos (在CCA空间中)
        cross_condition_cos_color_cca = None
        if all(k in cond_data_cca for k in ["red_edible", "red_inedible", "green_edible", "green_inedible"]):
            red_e = cond_data_cca["red_edible"]["hs_cca"]
            red_i = cond_data_cca["red_inedible"]["hs_cca"]
            green_e = cond_data_cca["green_edible"]["hs_cca"]
            green_i = cond_data_cca["green_inedible"]["hs_cca"]
            
            color_edible = np.mean(red_e, axis=0) - np.mean(green_e, axis=0)
            color_inedible = np.mean(red_i, axis=0) - np.mean(green_i, axis=0)
            cross_condition_cos_color_cca = compute_cos(color_edible, color_inedible)
        
        # ===== Step 4: 对比PCA =====
        # PCA (用相同维度数)
        pca = PCA(n_components=n_components_cca)
        H_pca = pca.fit_transform(H_centered)
        
        # PCA空间中条件解耦
        cond_data_pca = {}
        for group_name, words in CONDITIONAL_DATA.items():
            hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
            if hs is not None and len(hs) >= 3:
                hs_centered = hs - H_mean
                hs_pca = pca.transform(hs_centered)
                cond_data_pca[group_name] = {"hs_pca": hs_pca, "words": valid}
        
        cross_condition_cos_pca = None
        if all(k in cond_data_pca for k in ["red_edible", "green_edible", "red_inedible", "green_inedible"]):
            re = cond_data_pca["red_edible"]["hs_pca"]
            ge = cond_data_pca["green_edible"]["hs_pca"]
            ri = cond_data_pca["red_inedible"]["hs_pca"]
            gi = cond_data_pca["green_inedible"]["hs_pca"]
            
            edible_red = np.mean(re, axis=0) - np.mean(ri, axis=0)
            edible_green = np.mean(ge, axis=0) - np.mean(gi, axis=0)
            cross_condition_cos_pca = compute_cos(edible_red, edible_green)
        
        # ===== Step 5: 对比原始空间 =====
        cond_data_raw = {}
        for group_name, words in CONDITIONAL_DATA.items():
            hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
            if hs is not None and len(hs) >= 3:
                hs_centered = hs - H_mean
                cond_data_raw[group_name] = {"hs": hs_centered, "words": valid}
        
        cross_condition_cos_raw = None
        if all(k in cond_data_raw for k in ["red_edible", "green_edible", "red_inedible", "green_inedible"]):
            re = cond_data_raw["red_edible"]["hs"]
            ge = cond_data_raw["green_edible"]["hs"]
            ri = cond_data_raw["red_inedible"]["hs"]
            gi = cond_data_raw["green_inedible"]["hs"]
            
            edible_red = np.mean(re, axis=0) - np.mean(ri, axis=0)
            edible_green = np.mean(ge, axis=0) - np.mean(gi, axis=0)
            cross_condition_cos_raw = compute_cos(edible_red, edible_green)
        
        # ===== Step 6: ICA对比 =====
        n_components_ica = min(n_components_cca, len(H) - 1)
        try:
            ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=500)
            H_ica = ica.fit_transform(H_centered)
            
            cond_data_ica = {}
            for group_name, words in CONDITIONAL_DATA.items():
                hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
                if hs is not None and len(hs) >= 3:
                    hs_centered = hs - H_mean
                    hs_ica = ica.transform(hs_centered)
                    cond_data_ica[group_name] = {"hs_ica": hs_ica, "words": valid}
            
            cross_condition_cos_ica = None
            if all(k in cond_data_ica for k in ["red_edible", "green_edible", "red_inedible", "green_inedible"]):
                re = cond_data_ica["red_edible"]["hs_ica"]
                ge = cond_data_ica["green_edible"]["hs_ica"]
                ri = cond_data_ica["red_inedible"]["hs_ica"]
                gi = cond_data_ica["green_inedible"]["hs_ica"]
                
                edible_red = np.mean(re, axis=0) - np.mean(ri, axis=0)
                edible_green = np.mean(ge, axis=0) - np.mean(gi, axis=0)
                cross_condition_cos_ica = compute_cos(edible_red, edible_green)
        except Exception as e:
            print(f"    ICA失败: {e}")
            cross_condition_cos_ica = None
        
        # 汇总
        layer_result = {
            "n_concepts": len(valid_concepts),
            "n_cca_components": n_components_cca,
            "cca_correlations": cca_corrs,
            "r2_in_cca_space": float(r2_cca),
            "cos_matrix_in_cca_space": {ATTR_NAMES[i]: {ATTR_NAMES[j]: float(cos_matrix_cca[i, j]) 
                for j in range(n_attr_cca)} for i in range(n_attr_cca)},
            "edible_cross_condition_cos": {
                "raw_space": cross_condition_cos_raw,
                "pca_space": cross_condition_cos_pca,
                "cca_space": cross_condition_cos_cca,
                "ica_space": cross_condition_cos_ica,
            },
            "color_cross_condition_cos_cca": cross_condition_cos_color_cca,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    R²(CCA空间): {r2_cca:.4f}")
        print(f"    edible跨条件cos: raw={cross_condition_cos_raw:.4f}, "
              f"PCA={cross_condition_cos_pca:.4f}, "
              f"CCA={cross_condition_cos_cca:.4f}" + 
              (f", ICA={cross_condition_cos_ica:.4f}" if cross_condition_cos_ica is not None else ""))
        print(f"    color跨条件cos(CCA): {cross_condition_cos_color_cca:.4f}" if cross_condition_cos_color_cca else "    color跨条件cos(CCA): N/A")
        
        # CCA空间中属性余弦
        print(f"    CCA空间属性余弦(edible↔animacy): {cos_matrix_cca[0,1]:.4f}")
        print(f"    CCA空间属性余弦(edible↔color_red): {cos_matrix_cca[0,3]:.4f}")
    
    return results


# ============================================================================
# 29B: Sparse Coding反推潜在z
# ============================================================================

def expB_sparse_coding(model_name, model, tokenizer, device, layers_to_test):
    """29B: 用sparse coding/autoencoder从h中反推低维潜在结构"""
    print(f"\n{'='*70}")
    print(f"ExpB: Sparse Coding反推潜在z — 从h中找可解释的低维结构")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "B", "experiment": "sparse_coding", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        H, valid_concepts = collect_hs_for_words(model, tokenizer, device, concepts, layer_idx)
        
        if H is None or len(H) < 30:
            print(f"    数据不足, 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_concepts if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        layer_result = {}
        
        # ===== Method 1: Dictionary Learning (Sparse Coding) =====
        # 找一组过完备基, 使h可以稀疏表示
        from sklearn.decomposition import DictionaryLearning
        
        n_dict = 20  # 字典大小 (远大于属性数10, 看是否能自然对齐)
        
        try:
            dict_learner = DictionaryLearning(n_components=n_dict, alpha=1.0, 
                                               max_iter=500, random_state=42,
                                               transform_algorithm='lasso_lars')
            Z_sparse = dict_learner.fit_transform(H_centered)  # [n, n_dict]
            D = dict_learner.components_  # [n_dict, d_model]
        except Exception as e:
            print(f"    DictionaryLearning失败: {e}, 用PCA替代")
            pca = PCA(n_components=n_dict)
            Z_sparse = pca.fit_transform(H_centered)
            D = pca.components_
        
        # 分析: Z_sparse的每个维度是否与属性对齐?
        # 方法: 计算Z_sparse的每列与V_centered的每列的相关性
        attr_alignment = np.zeros((n_dict, len(ATTR_NAMES)))
        for i in range(n_dict):
            for j in range(len(ATTR_NAMES)):
                if np.std(Z_sparse[:, i]) > 1e-10 and np.std(V_centered[:, j]) > 1e-10:
                    corr = np.corrcoef(Z_sparse[:, i], V_centered[:, j])[0, 1]
                    attr_alignment[i, j] = abs(corr)
        
        # 每个字典维度的最佳对齐属性
        best_alignment_per_dim = []
        for i in range(n_dict):
            best_attr_idx = np.argmax(attr_alignment[i])
            best_corr = attr_alignment[i, best_attr_idx]
            best_alignment_per_dim.append({
                "dim": i,
                "best_attr": ATTR_NAMES[best_attr_idx],
                "correlation": float(best_corr),
            })
        
        # 每个属性被哪些维度覆盖
        attr_coverage = {}
        for j, attr in enumerate(ATTR_NAMES):
            top_dims = sorted(range(n_dict), key=lambda i: -attr_alignment[i, j])[:3]
            attr_coverage[attr] = {
                "top_dims": top_dims,
                "top_corrs": [float(attr_alignment[i, j]) for i in top_dims],
                "max_corr": float(np.max(attr_alignment[:, j])),
            }
        
        # ===== Method 2: Non-negative Matrix Factorization =====
        from sklearn.decomposition import NMF
        
        # NMF需要非负输入, 将H_centered做最小-最大归一化
        H_nmf = H - H.min()
        
        try:
            nmf = NMF(n_components=n_dict, random_state=42, max_iter=500)
            Z_nmf = nmf.fit_transform(H_nmf)  # [n, n_dict]
            W_nmf = nmf.components_  # [n_dict, d_model]
            
            # NMF维度与属性对齐
            attr_alignment_nmf = np.zeros((n_dict, len(ATTR_NAMES)))
            for i in range(n_dict):
                for j in range(len(ATTR_NAMES)):
                    if np.std(Z_nmf[:, i]) > 1e-10 and np.std(V_centered[:, j]) > 1e-10:
                        corr = np.corrcoef(Z_nmf[:, i], V_centered[:, j])[0, 1]
                        attr_alignment_nmf[i, j] = abs(corr)
            
            attr_coverage_nmf = {}
            for j, attr in enumerate(ATTR_NAMES):
                attr_coverage_nmf[attr] = {
                    "max_corr": float(np.max(attr_alignment_nmf[:, j])),
                }
        except Exception as e:
            print(f"    NMF失败: {e}")
            attr_coverage_nmf = {}
        
        # ===== Method 3: PCA (对比) =====
        pca = PCA(n_components=n_dict)
        Z_pca = pca.fit_transform(H_centered)
        
        attr_alignment_pca = np.zeros((n_dict, len(ATTR_NAMES)))
        for i in range(n_dict):
            for j in range(len(ATTR_NAMES)):
                if np.std(Z_pca[:, i]) > 1e-10 and np.std(V_centered[:, j]) > 1e-10:
                    corr = np.corrcoef(Z_pca[:, i], V_centered[:, j])[0, 1]
                    attr_alignment_pca[i, j] = abs(corr)
        
        attr_coverage_pca = {}
        for j, attr in enumerate(ATTR_NAMES):
            attr_coverage_pca[attr] = {
                "max_corr": float(np.max(attr_alignment_pca[:, j])),
            }
        
        # ===== 关键对比: 三种方法的属性对齐度 =====
        alignment_comparison = {}
        for attr in ATTR_NAMES:
            alignment_comparison[attr] = {
                "sparse_max_corr": attr_coverage.get(attr, {}).get("max_corr", 0),
                "nmf_max_corr": attr_coverage_nmf.get(attr, {}).get("max_corr", 0),
                "pca_max_corr": attr_coverage_pca.get(attr, {}).get("max_corr", 0),
            }
        
        # Sparse coding的重构质量
        H_recon_sparse = Z_sparse @ D
        ss_res = np.sum((H_centered - H_recon_sparse) ** 2)
        ss_tot = np.sum(H_centered ** 2)
        r2_sparse = 1 - ss_res / max(ss_tot, 1e-10)
        
        # Z_sparse的稀疏度 (每个样本平均非零比例)
        sparsity = np.mean(np.abs(Z_sparse) < 0.01 * np.max(np.abs(Z_sparse)))
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "n_dict_components": n_dict,
            "r2_sparse_reconstruction": float(r2_sparse),
            "sparsity": float(sparsity),
            "alignment_comparison": alignment_comparison,
            "best_alignment_per_dim": best_alignment_per_dim[:10],
            "attr_coverage_sparse": attr_coverage,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    Sparse重构R²: {r2_sparse:.4f}, 稀疏度: {sparsity:.4f}")
        print(f"    属性对齐度 (max |corr|):")
        for attr in ATTR_NAMES:
            vals = alignment_comparison[attr]
            print(f"      {attr}: sparse={vals['sparse_max_corr']:.3f}, "
                  f"nmf={vals['nmf_max_corr']:.3f}, pca={vals['pca_max_corr']:.3f}")
    
    return results


# ============================================================================
# 29C: 非线性回归结构分析 — Jacobian + 交互项
# ============================================================================

def expC_nonlinear_structure(model_name, model, tokenizer, device, layers_to_test):
    """29C: MLP非线性回归, 分析Jacobian稳定性和交互项"""
    print(f"\n{'='*70}")
    print(f"ExpC: 非线性回归结构分析 — Jacobian稳定性 + 交互项")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "C", "experiment": "nonlinear_structure", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        H, valid_concepts = collect_hs_for_words(model, tokenizer, device, concepts, layer_idx)
        
        if H is None or len(H) < 30:
            print(f"    数据不足, 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_concepts if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        layer_result = {}
        
        # ===== Step 1: 构造增强特征 (含交互项) =====
        # v_aug = [v_1, v_2, ..., v_k, v_1*v_2, v_1*v_3, ..., v_1*v_k, v_2*v_3, ...]
        n_attr = len(ATTR_NAMES)
        V_interactions = []
        interaction_names = []
        
        # 原始属性
        for i in range(n_attr):
            V_interactions.append(V_centered[:, i])
            interaction_names.append(ATTR_NAMES[i])
        
        # 二阶交互项 (只取重要的对: edible×color, edible×animacy, animacy×size等)
        important_pairs = [
            (0, 1, "edible×animacy"),
            (0, 3, "edible×color_red"),
            (0, 4, "edible×color_green"),
            (0, 5, "edible×color_yellow"),
            (1, 2, "animacy×size"),
            (1, 6, "animacy×material"),
            (3, 4, "color_red×green"),
            (3, 5, "color_red×yellow"),
        ]
        
        for i, j, name in important_pairs:
            if i < n_attr and j < n_attr:
                V_interactions.append(V_centered[:, i] * V_centered[:, j])
                interaction_names.append(name)
        
        V_aug = np.column_stack(V_interactions)  # [n, n_attr + n_interactions]
        
        print(f"    增强特征数: {V_aug.shape[1]} (原始{n_attr} + 交互{len(important_pairs)})")
        
        # ===== Step 2: 线性回归 (基准) =====
        ridge_linear = Ridge(alpha=1.0)
        ridge_linear.fit(V_centered, H_centered)
        H_pred_linear = ridge_linear.predict(V_centered)
        
        ss_res_lin = np.sum((H_centered - H_pred_linear) ** 2)
        ss_tot = np.sum(H_centered ** 2)
        r2_linear = 1 - ss_res_lin / max(ss_tot, 1e-10)
        
        # ===== Step 3: 增强线性回归 (含交互项) =====
        ridge_aug = Ridge(alpha=1.0)
        ridge_aug.fit(V_aug, H_centered)
        H_pred_aug = ridge_aug.predict(V_aug)
        
        ss_res_aug = np.sum((H_centered - H_pred_aug) ** 2)
        r2_augmented = 1 - ss_res_aug / max(ss_tot, 1e-10)
        
        # 交互项的贡献
        interaction_contribution = r2_augmented - r2_linear
        
        # 交互项系数
        A_aug = ridge_aug.coef_  # [d_model, n_features]
        interaction_coefs = {}
        for idx, name in enumerate(interaction_names[n_attr:]):
            # 交互项在所有d_model维度上的平均绝对系数
            mean_abs_coef = float(np.mean(np.abs(A_aug[:, n_attr + idx])))
            interaction_coefs[name] = mean_abs_coef
        
        # ===== Step 4: MLP回归 (PyTorch) =====
        # 小MLP: v → hidden(64) → hidden(32) → h
        import torch.nn as nn
        
        n_hidden1 = 64
        n_hidden2 = 32
        d_model = H_centered.shape[1]
        
        mlp = nn.Sequential(
            nn.Linear(n_attr, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, d_model),
        )
        
        V_tensor = torch.FloatTensor(V_centered)
        H_tensor = torch.FloatTensor(H_centered)
        
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
        
        # 训练
        mlp.train()
        best_loss = float('inf')
        patience = 50
        no_improve = 0
        
        for epoch in range(1000):
            optimizer.zero_grad()
            pred = mlp(V_tensor)
            loss = torch.nn.functional.mse_loss(pred, H_tensor)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        # MLP R²
        mlp.eval()
        with torch.no_grad():
            H_pred_mlp = mlp(V_tensor).numpy()
        
        ss_res_mlp = np.sum((H_centered - H_pred_mlp) ** 2)
        r2_mlp = 1 - ss_res_mlp / max(ss_tot, 1e-10)
        
        mlp_advantage = r2_mlp - r2_linear
        
        # ===== Step 5: Jacobian分析 =====
        # 计算∂h/∂v 在不同概念处的Jacobian
        # Jacobian稳定性 = 不同概念处Jacobian的变化程度
        
        jacobians = []
        for i in range(len(V_centered)):
            v_single = torch.FloatTensor(V_centered[i:i+1]).requires_grad_(True)
            h_single = mlp(v_single)
            
            # 对每个v维度求梯度 (取h的PC1方向)
            # 简化: 只看h的前10个主成分方向的梯度
            pca_h = PCA(n_components=min(10, d_model))
            pca_h.fit(H_centered)
            top_directions = pca_h.components_[:3]  # 前3个PC方向
            
            jac_row = []
            for pc_idx in range(min(3, len(top_directions))):
                direction = torch.FloatTensor(top_directions[pc_idx])
                proj = torch.dot(h_single.squeeze(), direction)
                grad = torch.autograd.grad(proj, v_single, retain_graph=True)[0]
                jac_row.append(grad.squeeze().detach().numpy())
            
            jacobians.append(np.array(jac_row))  # [3, n_attr]
        
        jacobians = np.array(jacobians)  # [n_concepts, 3, n_attr]
        
        # Jacobian稳定性: 不同概念处Jacobian的变化
        # 对每个PC方向, 计算Jacobian在概念间的变异系数
        jacobian_stability = {}
        for pc_idx in range(min(3, jacobians.shape[1])):
            jac_pc = jacobians[:, pc_idx, :]  # [n_concepts, n_attr]
            mean_jac = np.mean(jac_pc, axis=0)
            std_jac = np.std(jac_pc, axis=0)
            cv = std_jac / (np.abs(mean_jac) + 1e-10)  # 变异系数
            
            # 哪些属性的Jacobian最不稳定?
            stability_per_attr = {ATTR_NAMES[j]: float(cv[j]) for j in range(n_attr)}
            jacobian_stability[f"PC{pc_idx}"] = {
                "mean_cv": float(np.mean(cv)),
                "stability_per_attr": stability_per_attr,
            }
        
        # Jacobian方向的一致性: 不同概念处Jacobian方向的余弦
        if jacobians.shape[0] >= 2:
            jac_cos_pairs = []
            for i in range(min(20, jacobians.shape[0])):
                for j in range(i+1, min(20, jacobians.shape[0])):
                    cos_val = compute_cos(jacobians[i, 0], jacobians[j, 0])
                    jac_cos_pairs.append(cos_val)
            mean_jac_cos = float(np.mean(jac_cos_pairs))
        else:
            mean_jac_cos = 0.0
        
        # ===== Step 6: 属性交互分析 =====
        # 通过MLP的数值微分找交互项
        # 交互 = f(v1=1, v2=1) - f(v1=1, v2=0) - f(v1=0, v2=1) + f(v1=0, v2=0)
        
        # 对edible×color_red交互
        interaction_effects = {}
        for attr_i, attr_j, name in important_pairs:
            # v_base: 所有属性取均值
            v_base = torch.FloatTensor(np.zeros((1, n_attr)))
            
            # 分别扰动
            v_i = v_base.clone(); v_i[0, attr_i] = 1.0
            v_j = v_base.clone(); v_j[0, attr_j] = 1.0
            v_ij = v_base.clone(); v_ij[0, attr_i] = 1.0; v_ij[0, attr_j] = 1.0
            
            with torch.no_grad():
                h_base = mlp(v_base).numpy()
                h_i = mlp(v_i).numpy()
                h_j = mlp(v_j).numpy()
                h_ij = mlp(v_ij).numpy()
            
            # 交互效应
            interaction = h_ij - h_i - h_j + h_base
            interaction_norm = float(np.linalg.norm(interaction))
            
            # 相对于主效应的比例
            main_i_norm = float(np.linalg.norm(h_i - h_base))
            main_j_norm = float(np.linalg.norm(h_j - h_base))
            main_total = main_i_norm + main_j_norm + 1e-10
            interaction_ratio = interaction_norm / main_total
            
            interaction_effects[name] = {
                "interaction_norm": interaction_norm,
                "main_i_norm": main_i_norm,
                "main_j_norm": main_j_norm,
                "interaction_ratio": float(interaction_ratio),
            }
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "r2_linear": float(r2_linear),
            "r2_augmented": float(r2_augmented),
            "r2_mlp": float(r2_mlp),
            "interaction_contribution": float(interaction_contribution),
            "mlp_advantage": float(mlp_advantage),
            "interaction_coefs": interaction_coefs,
            "jacobian_stability": jacobian_stability,
            "mean_jacobian_cos": mean_jac_cos,
            "interaction_effects": interaction_effects,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    R²: 线性={r2_linear:.4f}, 增强线性={r2_augmented:.4f}, MLP={r2_mlp:.4f}")
        print(f"    交互项贡献: {interaction_contribution:.4f}, MLP优势: {mlp_advantage:.4f}")
        print(f"    Jacobian方向一致性(cos): {mean_jac_cos:.4f}")
        print(f"    交互效应:")
        for name, info in interaction_effects.items():
            print(f"      {name}: ratio={info['interaction_ratio']:.4f} "
                  f"(norm={info['interaction_norm']:.4f})")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 29: 潜在语义结构破解")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=CCA, 2=Sparse, 3=非线性")
    args = parser.parse_args()
    
    model_name = args.model
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 测试层: 浅/中/深
    layers_to_test = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-2]
    layers_to_test = sorted(set(layers_to_test))
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {info.d_model}")
    print(f"测试层: {layers_to_test}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    if args.exp in [0, 1]:
        resA = expA_cca_supervised_subspace(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmi_expA_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpA] 结果已保存")
    
    if args.exp in [0, 2]:
        resB = expB_sparse_coding(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmi_expB_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpB] 结果已保存")
    
    if args.exp in [0, 3]:
        resC = expC_nonlinear_structure(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmi_expC_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resC, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpC] 结果已保存")
    
    # 释放GPU
    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
