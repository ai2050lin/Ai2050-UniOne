"""
CCMJ(Phase 30): 子空间编码 + Probe分析 + Logit空间解码
=============================================================================
Phase 29核心发现(经批判修正):
  ★★★ "max|corr|<0.7" ≠ "不存在属性坐标系"!
    → 属性可能是多维子空间(distributed subspace), 不是单维轴
    → 需要看k维子空间能解释多少属性, 而不是单维corr

  ★★★ MLP R²高 ≠ 证明非线性机制!
    → MLP可能在学隐式z, 而不是证明f是非线性的
    → 需要区分"表示方式"和"生成机制"

  ★★★ CCA不如PCA ≠ 属性不重要!
    → CCA优化correlation, 不优化cos一致性
    → CCA不好的原因是属性标签v ≠ 模型内部变量z

  ★★★ 没分析h→logits(关键缺口)!
    → 属性可能在h中是分布式的, 但在W_U投影后被"解码成清晰方向"
    → 编码空间 ≠ 解码空间

最关键洞察:
  属性 = "可读的函数方向", 而不是"表示的基底"
  h不是用属性构造的, 但可以被属性函数解码

Phase 30核心任务(3个最关键实验):
  30A: ★★★★★★★★★★★ 子空间属性编码
    → 不再看max|corr|, 看k维子空间能解释多少属性方差
    → 用SVD/PCA on probe weights分析属性子空间维度
    → 如果K<5 → 属性近似低维; 如果K>20 → 属性高度分布式

  30B: ★★★★★★★★★ Probe权重结构分析
    → 训练线性分类器(probe): h → 属性
    → 分析权重向量w的稀疏性/集中度/重叠度
    → 不同属性的w是否重叠? 集中在哪些PC?

  30C: ★★★★★★★★★★★ h→logits的属性投影(关键缺口!)
    → 分析 W_U @ h 中的属性结构
    → 如果属性在logit空间更线性 → 模型是"解码时"显式使用属性
    → 比较: hidden空间 vs logit空间的属性可分性
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy.sparse.linalg import svds
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
# 数据定义 (与Phase 29相同)
# ============================================================================

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
# 30A: 子空间属性编码 — k维子空间能解释多少属性
# ============================================================================

def expA_subspace_encoding(model_name, model, tokenizer, device, layers_to_test):
    """30A: 不看单维corr, 看k维子空间能解释多少属性方差"""
    print(f"\n{'='*70}")
    print(f"ExpA: 子空间属性编码 — k维子空间能解释多少属性方差?")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "A", "experiment": "subspace_encoding", "layers": {}}
    
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
        
        # ===== Step 1: 子空间维度分析 =====
        # 对每个属性, 训练线性回归 h → v_attr
        # 然后对回归权重做SVD, 看需要多少维度解释95%权重方差
        
        n_attr = len(ATTR_NAMES)
        d_model = H_centered.shape[1]
        
        # 方法: Ridge回归 h → v_attr, 得到权重 w_attr [d_model]
        # 然后对 W = [w_1, w_2, ..., w_k] 做SVD
        # 看前k个奇异值能解释多少
        
        W_probes = np.zeros((n_attr, d_model))  # [n_attr, d_model]
        probe_r2 = {}
        
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
            
            # 单属性R²
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            probe_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # SVD of probe weight matrix
        # W_probes: [n_attr, d_model]
        U_w, s_w, Vt_w = np.linalg.svd(W_probes, full_matrices=False)
        
        # 前k个奇异值解释的方差比例
        total_var = np.sum(s_w ** 2)
        cumvar = np.cumsum(s_w ** 2) / total_var
        
        # 找到解释90%方差需要的维度数
        k_90 = np.searchsorted(cumvar, 0.90) + 1
        k_95 = np.searchsorted(cumvar, 0.95) + 1
        k_99 = np.searchsorted(cumvar, 0.99) + 1
        
        print(f"    Probe权重SVD: 前3奇异值占比={[f'{c:.3f}' for c in cumvar[:3]]}")
        print(f"    子空间维度: k_90={k_90}, k_95={k_95}, k_99={k_99}")
        print(f"    单属性R²:")
        for attr in ATTR_NAMES:
            print(f"      {attr}: {probe_r2[attr]:.4f}")
        
        # ===== Step 2: k维子空间的属性解释力 =====
        # 用前k个PC方向投影H, 然后在这个子空间中做属性回归
        # 看k=1,2,3,...,20时, 子空间能解释多少属性方差
        
        pca = PCA(n_components=min(30, d_model, len(H)-1))
        H_pca = pca.fit_transform(H_centered)
        
        subspace_r2 = {}
        for k in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
            if k > H_pca.shape[1]:
                continue
            H_sub = H_pca[:, :k]
            
            # 对每个属性回归
            attr_r2_k = {}
            for i, attr in enumerate(ATTR_NAMES):
                ridge = Ridge(alpha=1.0)
                ridge.fit(H_sub, V_centered[:, i])
                pred = ridge.predict(H_sub)
                ss_res = np.sum((V_centered[:, i] - pred) ** 2)
                ss_tot = np.sum(V_centered[:, i] ** 2)
                attr_r2_k[attr] = 1 - ss_res / max(ss_tot, 1e-10)
            
            subspace_r2[f"k={k}"] = attr_r2_k
        
        # ===== Step 3: 属性子空间与PCA对齐度 =====
        # 对每个属性, 找到解释它最好的前k个PC方向
        # 这些方向是否集中? 还是分散?
        
        attr_pc_alignment = {}
        for i, attr in enumerate(ATTR_NAMES):
            # 在30维PCA空间中回归
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_pca, V_centered[:, i])
            w_pc = ridge.coef_  # [30]
            
            # w_pc在哪些PC上权重最大?
            abs_w = np.abs(w_pc)
            total_abs = np.sum(abs_w) + 1e-10
            
            # 前3个最大权重的PC index
            top3_pc = np.argsort(abs_w)[-3:][::-1]
            top3_weights = abs_w[top3_pc] / total_abs
            
            # Gini系数 (衡量集中度, 0=均匀, 1=集中)
            sorted_w = np.sort(abs_w)
            n = len(sorted_w)
            cumsum = np.cumsum(sorted_w)
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_w)) / (n * np.sum(sorted_w) + 1e-10) - (n + 1) / n
            
            attr_pc_alignment[attr] = {
                "top3_pc": top3_pc.tolist(),
                "top3_weights": [float(w) for w in top3_weights],
                "gini": float(gini),
                "top3_cumulative_weight": float(np.sum(top3_weights)),
            }
        
        # ===== Step 4: 子空间R² vs 原始R² =====
        # 对比: k=10子空间 vs 全空间 的R²
        full_r2 = subspace_r2.get("k=30", subspace_r2.get("k=20", subspace_r2.get("k=15", {})))
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "probe_r2_per_attr": probe_r2,
            "subspace_dims": {"k_90": int(k_90), "k_95": int(k_95), "k_99": int(k_99)},
            "singular_values": [float(s) for s in s_w[:min(10, len(s_w))]],
            "cumulative_variance": [float(c) for c in cumvar[:min(10, len(cumvar))]],
            "subspace_r2": subspace_r2,
            "attr_pc_alignment": attr_pc_alignment,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印关键结果
        print(f"    子空间编码效率:")
        for k_str in ["k=1", "k=3", "k=5", "k=10"]:
            if k_str in subspace_r2:
                mean_r2 = np.mean(list(subspace_r2[k_str].values()))
                print(f"      {k_str}: mean R²={mean_r2:.4f}")
        
        print(f"    属性PC集中度(Gini):")
        for attr in ATTR_NAMES:
            gini = attr_pc_alignment[attr]["gini"]
            top3_w = attr_pc_alignment[attr]["top3_cumulative_weight"]
            print(f"      {attr}: Gini={gini:.3f}, top3权重={top3_w:.3f}")
    
    return results


# ============================================================================
# 30B: Probe权重结构分析
# ============================================================================

def expB_probe_structure(model_name, model, tokenizer, device, layers_to_test):
    """30B: 分析线性probe的权重结构——稀疏性/集中度/重叠度"""
    print(f"\n{'='*70}")
    print(f"ExpB: Probe权重结构分析 — 稀疏性/集中度/重叠度")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "B", "experiment": "probe_structure", "layers": {}}
    
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
        
        n_attr = len(ATTR_NAMES)
        d_model = H_centered.shape[1]
        
        # ===== Step 1: 训练线性probe =====
        # 对每个属性: h → v_attr (Ridge回归)
        W_probes = np.zeros((n_attr, d_model))
        probe_acc = {}
        
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
            
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            probe_acc[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # ===== Step 2: 权重稀疏性 =====
        # 稀疏性 = 权重向量中有多少比例接近0?
        # 方法: Gini系数 + 有效维度数(1/sum(p_i^2))
        
        sparsity_metrics = {}
        for i, attr in enumerate(ATTR_NAMES):
            w = np.abs(W_probes[i])
            total = np.sum(w) + 1e-10
            p = w / total  # 概率分布
            
            # Gini系数
            sorted_p = np.sort(p)
            n = len(sorted_p)
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_p)) / (n * np.sum(sorted_p) + 1e-10) - (n + 1) / n
            
            # 有效维度
            eff_dim = 1.0 / (np.sum(p ** 2) + 1e-10)
            
            # L1/L2 比 (越大越稀疏)
            l1 = np.sum(w)
            l2 = np.linalg.norm(W_probes[i])
            l1_l2_ratio = l1 / (l2 + 1e-10)
            
            sparsity_metrics[attr] = {
                "gini": float(gini),
                "effective_dim": float(eff_dim),
                "l1_l2_ratio": float(l1_l2_ratio),
            }
        
        # ===== Step 3: 权重集中度 =====
        # 权重在H的哪些PC方向上集中?
        
        pca_h = PCA(n_components=min(50, d_model, len(H)-1))
        H_pca = pca_h.fit_transform(H_centered)
        PCs = pca_h.components_  # [n_pc, d_model]
        
        concentration = {}
        for i, attr in enumerate(ATTR_NAMES):
            w = W_probes[i]
            
            # w在PC上的投影
            proj_on_pcs = PCs @ w  # [n_pc]
            abs_proj = np.abs(proj_on_pcs)
            total_proj = np.sum(abs_proj ** 2) + 1e-10
            
            # 前3个PC解释的比例
            top3_idx = np.argsort(abs_proj)[-3:][::-1]
            top3_energy = np.sum(abs_proj[top3_idx] ** 2) / total_proj
            
            concentration[attr] = {
                "top3_pc_indices": top3_idx.tolist(),
                "top3_pc_energy": float(top3_energy),
                "pc_proj_top5": [float(abs_proj[j]) for j in np.argsort(abs_proj)[-5:][::-1]],
            }
        
        # ===== Step 4: 权重重叠度 =====
        # 不同属性的probe权重向量之间的余弦相似度
        # 如果重叠度高(cos>0.5) → 属性共享编码空间
        # 如果重叠度低(cos<0.2) → 属性使用不同方向
        
        overlap_matrix = np.zeros((n_attr, n_attr))
        for i in range(n_attr):
            for j in range(n_attr):
                overlap_matrix[i, j] = compute_cos(W_probes[i], W_probes[j])
        
        # 平均非对角余弦
        off_diag = overlap_matrix[np.triu_indices(n_attr, k=1)]
        mean_overlap = float(np.mean(np.abs(off_diag)))
        max_overlap = float(np.max(np.abs(off_diag)))
        
        # 找重叠最高和最低的属性对
        triu_idx = np.triu_indices(n_attr, k=1)
        abs_vals = np.abs(off_diag)
        max_idx = np.argmax(abs_vals)
        min_idx = np.argmin(abs_vals)
        
        # ===== Step 5: 二值属性分类器(对比) =====
        # 对edible, animacy等二值属性, 训练LogisticRegression
        # 看分类准确率和权重
        
        binary_attrs = ["edible", "animacy", "material_organic", "shape_round", "function_food"]
        binary_results = {}
        
        for attr in binary_attrs:
            if attr not in ATTR_NAMES:
                continue
            attr_idx = ATTR_NAMES.index(attr)
            y = (V_valid[:, attr_idx] > 0.5).astype(int)
            
            if len(np.unique(y)) < 2:
                continue
            
            lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
            try:
                scores = cross_val_score(lr, H_centered, y, cv=min(5, len(np.unique(y))), scoring='accuracy')
                binary_results[attr] = {
                    "cv_accuracy": float(np.mean(scores)),
                    "cv_std": float(np.std(scores)),
                }
            except Exception as e:
                binary_results[attr] = {"error": str(e)}
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "probe_r2": probe_acc,
            "sparsity_metrics": sparsity_metrics,
            "concentration": concentration,
            "overlap_matrix": {ATTR_NAMES[i]: {ATTR_NAMES[j]: float(overlap_matrix[i, j]) 
                for j in range(n_attr)} for i in range(n_attr)},
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "binary_classification": binary_results,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    Probe R²:")
        for attr in ATTR_NAMES:
            print(f"      {attr}: {probe_acc[attr]:.4f}")
        
        print(f"    稀疏性(Gini):")
        for attr in ATTR_NAMES:
            g = sparsity_metrics[attr]["gini"]
            ed = sparsity_metrics[attr]["effective_dim"]
            print(f"      {attr}: Gini={g:.3f}, 有效维度={ed:.1f}")
        
        print(f"    PC集中度(top3 energy):")
        for attr in ATTR_NAMES:
            e = concentration[attr]["top3_pc_energy"]
            print(f"      {attr}: {e:.4f}")
        
        print(f"    权重重叠: mean|cos|={mean_overlap:.4f}, max|cos|={max_overlap:.4f}")
        print(f"    二值分类:")
        for attr, res in binary_results.items():
            if "cv_accuracy" in res:
                print(f"      {attr}: acc={res['cv_accuracy']:.4f}")
    
    return results


# ============================================================================
# 30C: h→logits的属性投影 — 编码空间 vs 解码空间
# ============================================================================

def expC_logit_decode(model_name, model, tokenizer, device, layers_to_test):
    """30C: 分析W_U@h中的属性结构——编码空间vs解码空间"""
    print(f"\n{'='*70}")
    print(f"ExpC: h→logits的属性投影 — 编码空间 vs 解码空间 (关键缺口!)")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "C", "experiment": "logit_decode", "layers": {}}
    
    # 获取W_U
    W_U = get_W_U(model)  # [vocab_size, d_model]
    vocab_size, d_model = W_U.shape
    print(f"    W_U shape: {W_U.shape}")
    
    # 对W_U做SVD, 获取行空间基
    k_svd = min(100, d_model, vocab_size - 2)
    U_wut, s_wut, _ = svds(W_U.T.astype(np.float32), k=k_svd)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k] — W_U行空间基
    
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
        
        n_attr = len(ATTR_NAMES)
        
        # ===== Step 1: Logit空间中的属性回归 =====
        # 对比: hidden空间 vs logit空间的属性R²
        
        L = (W_U @ H_centered.T).T  # [n_concepts, vocab_size] — logit空间
        L_mean = np.mean(L, axis=0, keepdims=True)
        L_centered = L - L_mean
        
        # hidden空间R² (全空间)
        hidden_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            hidden_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # logit空间R² (全空间)
        logit_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(L_centered, V_centered[:, i])
            pred = ridge.predict(L_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            logit_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # ===== Step 2: W_U行空间中的属性投影 =====
        # H在W_U行空间中的投影 vs 原始H
        # 关键问题: 属性信号是否在W_U行空间中?
        
        H_in_WU = H_centered @ U_wut @ U_wut.T  # H在W_U行空间中的投影
        H_out_WU = H_centered - H_in_WU  # H在W_U正交补中的部分
        
        # W_U行空间中的R²
        wu_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_in_WU, V_centered[:, i])
            pred = ridge.predict(H_in_WU)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            wu_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # W_U正交补中的R²
        orth_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_out_WU, V_centered[:, i])
            pred = ridge.predict(H_out_WU)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            orth_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # ===== Step 3: 属性方向在logit空间的可分性 =====
        # 在logit空间中, 属性是否更容易区分?
        
        # edible在logit空间中的方向
        edible_idx = V_centered[:, 0] > 0.5
        inedible_idx = ~edible_idx
        
        # hidden空间: edible方向
        if np.sum(edible_idx) > 2 and np.sum(inedible_idx) > 2:
            edible_dir_hidden = np.mean(H_centered[edible_idx], axis=0) - np.mean(H_centered[inedible_idx], axis=0)
            edible_dir_logit = np.mean(L_centered[edible_idx], axis=0) - np.mean(L_centered[inedible_idx], axis=0)
            
            # 两个方向的质量: 信噪比 (方向norm / 类内方差)
            var_within_edible = np.mean(np.var(H_centered[edible_idx], axis=0))
            var_within_inedible = np.mean(np.var(H_centered[inedible_idx], axis=0))
            snr_hidden = np.linalg.norm(edible_dir_hidden) / (np.sqrt(var_within_edible + var_within_inedible) + 1e-10)
            
            var_within_edible_L = np.mean(np.var(L_centered[edible_idx], axis=0))
            var_within_inedible_L = np.mean(np.var(L_centered[inedible_idx], axis=0))
            snr_logit = np.linalg.norm(edible_dir_logit) / (np.sqrt(var_within_edible_L + var_within_inedible_L) + 1e-10)
        else:
            snr_hidden = 0
            snr_logit = 0
        
        # animacy方向
        anim_idx = V_centered[:, 1] > 0.5
        inanim_idx = ~anim_idx
        
        if np.sum(anim_idx) > 2 and np.sum(inanim_idx) > 2:
            anim_dir_hidden = np.mean(H_centered[anim_idx], axis=0) - np.mean(H_centered[inanim_idx], axis=0)
            anim_dir_logit = np.mean(L_centered[anim_idx], axis=0) - np.mean(L_centered[inanim_idx], axis=0)
            
            var_within_anim = np.mean(np.var(H_centered[anim_idx], axis=0))
            var_within_inanim = np.mean(np.var(H_centered[inanim_idx], axis=0))
            snr_anim_hidden = np.linalg.norm(anim_dir_hidden) / (np.sqrt(var_within_anim + var_within_inanim) + 1e-10)
            
            var_within_anim_L = np.mean(np.var(L_centered[anim_idx], axis=0))
            var_within_inanim_L = np.mean(np.var(L_centered[inanim_idx], axis=0))
            snr_anim_logit = np.linalg.norm(anim_dir_logit) / (np.sqrt(var_within_anim_L + var_within_inanim_L) + 1e-10)
        else:
            snr_anim_hidden = 0
            snr_anim_logit = 0
        
        # ===== Step 4: 属性在logit空间中的"清晰度" =====
        # 在logit空间中, 属性方向之间的余弦更低 → 更正交 → 更清晰
        
        # hidden空间中属性方向间余弦
        attr_dirs_hidden = {}
        for i, attr in enumerate(ATTR_NAMES):
            high = V_centered[:, i] > np.median(V_centered[:, i])
            low = ~high
            if np.sum(high) > 2 and np.sum(low) > 2:
                attr_dirs_hidden[attr] = np.mean(H_centered[high], axis=0) - np.mean(H_centered[low], axis=0)
        
        attr_dirs_logit = {}
        for i, attr in enumerate(ATTR_NAMES):
            high = V_centered[:, i] > np.median(V_centered[:, i])
            low = ~high
            if np.sum(high) > 2 and np.sum(low) > 2:
                attr_dirs_logit[attr] = np.mean(L_centered[high], axis=0) - np.mean(L_centered[low], axis=0)
        
        # 计算方向间余弦
        hidden_attr_cos = {}
        logit_attr_cos = {}
        attr_list = list(attr_dirs_hidden.keys())
        
        for a1 in attr_list:
            for a2 in attr_list:
                if a1 >= a2:
                    continue
                cos_h = compute_cos(attr_dirs_hidden[a1], attr_dirs_hidden[a2])
                cos_l = compute_cos(attr_dirs_logit[a1], attr_dirs_logit[a2])
                hidden_attr_cos[f"{a1}×{a2}"] = float(cos_h)
                logit_attr_cos[f"{a1}×{a2}"] = float(cos_l)
        
        mean_hidden_cos = float(np.mean(np.abs(list(hidden_attr_cos.values())))) if hidden_attr_cos else 0
        mean_logit_cos = float(np.mean(np.abs(list(logit_attr_cos.values())))) if logit_attr_cos else 0
        
        # ===== Step 5: 关键指标 —— logit空间属性线性度提升比 =====
        logit_gain = {}
        for attr in ATTR_NAMES:
            if attr in hidden_r2 and attr in logit_r2:
                if abs(hidden_r2[attr]) > 0.01:
                    logit_gain[attr] = logit_r2[attr] / hidden_r2[attr]
                else:
                    logit_gain[attr] = 0
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "hidden_r2": hidden_r2,
            "logit_r2": logit_r2,
            "wu_row_space_r2": wu_r2,
            "wu_orthogonal_r2": orth_r2,
            "snr": {
                "edible": {"hidden": float(snr_hidden), "logit": float(snr_logit)},
                "animacy": {"hidden": float(snr_anim_hidden), "logit": float(snr_anim_logit)},
            },
            "attr_direction_cos": {
                "hidden_mean": mean_hidden_cos,
                "logit_mean": mean_logit_cos,
                "hidden_pairs": hidden_attr_cos,
                "logit_pairs": logit_attr_cos,
            },
            "logit_gain_ratio": logit_gain,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    R²对比 (hidden vs logit):")
        for attr in ATTR_NAMES:
            h_r2 = hidden_r2.get(attr, 0)
            l_r2 = logit_r2.get(attr, 0)
            gain = logit_gain.get(attr, 0)
            print(f"      {attr}: hidden={h_r2:.4f}, logit={l_r2:.4f}, gain={gain:.2f}x")
        
        print(f"    W_U行空间R² vs 正交补R²:")
        for attr in ATTR_NAMES:
            w = wu_r2.get(attr, 0)
            o = orth_r2.get(attr, 0)
            print(f"      {attr}: W_U行空间={w:.4f}, 正交补={o:.4f}")
        
        print(f"    SNR (hidden vs logit):")
        print(f"      edible: hidden={snr_hidden:.4f}, logit={snr_logit:.4f}")
        print(f"      animacy: hidden={snr_anim_hidden:.4f}, logit={snr_anim_logit:.4f}")
        
        print(f"    属性方向间余弦:")
        print(f"      hidden mean|cos|={mean_hidden_cos:.4f}")
        print(f"      logit mean|cos|={mean_logit_cos:.4f}")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 30: 子空间编码 + Probe分析 + Logit解码")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=子空间, 2=Probe, 3=Logit")
    args = parser.parse_args()
    
    model_name = args.model
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 测试层
    layers_to_test = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-2]
    layers_to_test = sorted(set(layers_to_test))
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {info.d_model}")
    print(f"测试层: {layers_to_test}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "glm5_temp")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    if args.exp in [0, 1]:
        resA = expA_subspace_encoding(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmj_expA_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpA] 结果已保存")
    
    if args.exp in [0, 2]:
        resB = expB_probe_structure(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmj_expB_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpB] 结果已保存")
    
    if args.exp in [0, 3]:
        resC = expC_logit_decode(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmj_expC_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resC, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpC] 结果已保存")
    
    # 释放GPU
    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
