"""
CCMK(Phase 31): 子空间三重验证 — 稳定性/因果性/生成性
=============================================================================
Phase 30核心发现(经批判修正):
  ★★★ "7维子空间" ≠ "语义本身只有7维"!
    → SVD(probe权重)的秩=7, 只说明probe线性相关
    → 真实语义变量维度可能>7
    → 属性=子空间方向, 但子空间秩≠语义维度

  ★★★ W_U不能从正交空间恢复信息!
    → 如果h_orth ⟂ Row(W_U), 则 W_U·h_orth = 0
    → logit R²高的原因: SNR放大 + label leakage + 维度变化效应
    → W_U是"任务对齐投影", 不是"语义解码器"

  ★★★ Ridge高估有效维度!
    → Ridge回归分散权重, 不鼓励稀疏
    → 需要:Lasso/sparse probe验证真实维度

  ★★★ "子空间存在" ≠ "语义变量存在"!
    → 需要三重验证: 稳定性(跨上下文) + 因果性(干预) + 生成性(插值)

Phase 31核心任务(3个最关键验证实验):
  31A: ★★★★★★★★★★★ 子空间跨上下文稳定性 (最重要!)
    → 同一概念在不同上下文中, 子空间坐标是否稳定?
    → 如果不稳定 → 子空间是"上下文相关投影", 不是"语义空间"
    → 如果稳定 → 子空间可能真正编码了语义变量

  31B: ★★★★★★★★★ 子空间因果干预
    → 在7D子空间中干预, 是否能改变logits?
    → 如果子空间干预无效 → 子空间不是因果机制变量
    → 如果子空间干预有效 → 子空间可能是语义变量

  31C: ★★★★★★★ 子空间生成性验证 + Lasso维度
    → 在子空间中插值, decode成token, 看结果是否合理
    → 用Lasso重新训练probe, 看真实有效维度
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
from sklearn.linear_model import Ridge, Lasso, LassoCV
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
# 数据定义 (与Phase 30相同)
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

# 多个上下文模板 (关键: 不同的句法结构, 但保持概念不变)
CONTEXT_TEMPLATES = [
    "The {word} is here",           # 原始模板
    "I see a {word}",               # 简单观察
    "The {word} was found",         # 被动语态
    "Look at that {word}",          # 指代
    "A {word} appeared",            # 出现
    "This {word} looks nice",       # 评价
    "Every {word} has features",    # 一般性描述
]

# 用于干预实验的概念样本 (覆盖不同属性组合)
INTERVENTION_CONCEPTS = [
    "apple", "dog", "hammer", "tree",  # 典型概念
    "salmon", "cow", "ball", "fire",   # 属性混合概念
    "strawberry", "elephant", "rock", "water",  # 更多覆盖
]


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


def collect_hs_with_template(model, tokenizer, device, words, layer_idx, template):
    """收集一组词在指定层的hidden states (使用指定模板)"""
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


def get_subspace_basis(H_centered, n_components=10):
    """获取属性子空间的PCA基"""
    pca = PCA(n_components=min(n_components, H_centered.shape[1], H_centered.shape[0]-1))
    pca.fit(H_centered)
    return pca


# ============================================================================
# 31A: 子空间跨上下文稳定性 — 决定性验证!
# ============================================================================

def expA_context_stability(model_name, model, tokenizer, device, layers_to_test):
    """31A: 同一概念在不同上下文中, 子空间坐标是否稳定?"""
    print(f"\n{'='*70}")
    print(f"ExpA: 子空间跨上下文稳定性 — 同一概念, 不同上下文, 坐标是否稳定?")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "A", "experiment": "context_stability", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集每个模板下的hidden states
        all_hs_by_context = {}
        common_words = None
        
        for template in CONTEXT_TEMPLATES:
            H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, template)
            if H is None:
                continue
            all_hs_by_context[template] = (H, valid_words)
            if common_words is None:
                common_words = set(valid_words)
            else:
                common_words = common_words & set(valid_words)
        
        if len(all_hs_by_context) < 3 or common_words is None or len(common_words) < 20:
            print(f"    上下文不足({len(all_hs_by_context)})或共同词不足({len(common_words) if common_words else 0}), 跳过")
            continue
        
        common_words = sorted(common_words)
        print(f"    共同词数: {len(common_words)}, 上下文数: {len(all_hs_by_context)}")
        
        # ===== Step 1: 属性子空间 (用第一个上下文训练) =====
        ref_template = CONTEXT_TEMPLATES[0]
        H_ref, words_ref = all_hs_by_context[ref_template]
        
        # 取共同词的hidden states
        ref_indices = [words_ref.index(w) for w in common_words]
        H_ref_common = H_ref[ref_indices]
        
        H_ref_mean = np.mean(H_ref_common, axis=0, keepdims=True)
        H_ref_centered = H_ref_common - H_ref_mean
        
        # 训练属性probe和PCA子空间
        V_common = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in common_words])
        V_mean = np.mean(V_common, axis=0, keepdims=True)
        V_centered = V_common - V_mean
        
        n_attr = len(ATTR_NAMES)
        d_model = H_ref_centered.shape[1]
        
        # 训练Ridge probe
        W_probes = np.zeros((n_attr, d_model))
        probe_r2 = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_ref_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
            pred = ridge.predict(H_ref_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            probe_r2[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        # SVD获取子空间基
        U_w, s_w, Vt_w = np.linalg.svd(W_probes, full_matrices=False)
        total_var = np.sum(s_w ** 2)
        cumvar = np.cumsum(s_w ** 2) / total_var
        k_95 = np.searchsorted(cumvar, 0.95) + 1
        
        # 子空间投影矩阵 (前k_95维)
        subspace_basis = Vt_w[:k_95]  # [k_95, d_model] — 属性子空间基
        P_sub = subspace_basis.T @ subspace_basis  # 投影矩阵 [d_model, d_model]
        
        print(f"    子空间维度 k_95={k_95}")
        print(f"    参考上下文probe R²: " + ", ".join([f"{a}={probe_r2[a]:.3f}" for a in ["edible", "animacy", "size"]]))
        
        # ===== Step 2: 跨上下文稳定性分析 =====
        # 关键指标:
        # A) 同一概念在不同上下文中的子空间坐标的方差
        # B) 子空间坐标的"上下文方差/属性方差"比 → 越小越稳定
        # C) PCA空间中的稳定性 (对比)
        
        # 方法: 对每个概念, 在不同上下文中投影到子空间, 看坐标变化
        
        # 先用PCA做同样分析 (作为基准)
        pca = PCA(n_components=min(k_95, d_model, len(H_ref_common)-1))
        pca.fit(H_ref_centered)
        
        # 对每个上下文, 提取共同词的子空间坐标
        coords_by_context = {}  # {template: [n_words, k_95]}
        coords_pca_by_context = {}
        
        for template, (H_ctx, words_ctx) in all_hs_by_context.items():
            ctx_indices = [words_ctx.index(w) for w in common_words]
            H_ctx_common = H_ctx[ctx_indices]
            
            # 投影到属性子空间 (相对于参考均值)
            H_ctx_centered = H_ctx_common - H_ref_mean  # 用参考均值中心化!
            coords_sub = H_ctx_centered @ P_sub  # [n_words, k_95] 但这是回到原空间
            # 更准确: 直接用子空间基做投影
            coords_sub = H_ctx_centered @ subspace_basis.T  # [n_words, k_95]
            
            # PCA空间投影
            coords_pca = pca.transform(H_ctx_centered)[:, :k_95]
            
            coords_by_context[template] = coords_sub
            coords_pca_by_context[template] = coords_pca
        
        # ===== Step 3: 计算稳定性指标 =====
        
        # 3a) 同一概念跨上下文的坐标方差 (子空间)
        n_words = len(common_words)
        n_ctx = len(coords_by_context)
        
        # 堆叠所有上下文的坐标
        coords_stack = np.stack([coords_by_context[t] for t in CONTEXT_TEMPLATES 
                                  if t in coords_by_context], axis=0)  # [n_ctx, n_words, k_95]
        coords_pca_stack = np.stack([coords_pca_by_context[t] for t in CONTEXT_TEMPLATES 
                                      if t in coords_pca_by_context], axis=0)
        
        # 每个概念跨上下文的方差
        var_across_ctx = np.var(coords_stack, axis=0)  # [n_words, k_95]
        var_across_ctx_pca = np.var(coords_pca_stack, axis=0)
        
        # 每个概念跨上下文的方差 (总)
        total_var_across_ctx = np.sum(var_across_ctx, axis=1)  # [n_words]
        total_var_across_ctx_pca = np.sum(var_across_ctx_pca, axis=1)
        
        # 概念间的方差 (属性方差)
        mean_coords = np.mean(coords_stack, axis=0)  # [n_words, k_95]
        var_across_words = np.var(mean_coords, axis=0)  # [k_95]
        total_var_across_words = np.sum(var_across_words)
        
        # ★★★ 关键指标: 上下文方差 / 属性方差 ★★★
        mean_ctx_var = float(np.mean(total_var_across_ctx))
        mean_word_var = float(total_var_across_words / n_words) if n_words > 0 else 0
        var_ratio = mean_ctx_var / max(mean_word_var, 1e-10)
        
        # PCA空间同样的ratio
        mean_ctx_var_pca = float(np.mean(total_var_across_ctx_pca))
        mean_coords_pca = np.mean(coords_pca_stack, axis=0)
        var_across_words_pca = np.var(mean_coords_pca, axis=0)
        mean_word_var_pca = float(np.sum(var_across_words_pca) / n_words) if n_words > 0 else 0
        var_ratio_pca = mean_ctx_var_pca / max(mean_word_var_pca, 1e-10)
        
        # 3b) 同一概念跨上下文的余弦相似度
        cos_across_ctx = []
        for w_idx in range(min(n_words, 30)):  # 限制计算量
            word_coords = coords_stack[:, w_idx, :]  # [n_ctx, k_95]
            for i in range(n_ctx):
                for j in range(i+1, n_ctx):
                    cos = compute_cos(word_coords[i], word_coords[j])
                    cos_across_ctx.append(cos)
        
        mean_cos_across_ctx = float(np.mean(cos_across_ctx)) if cos_across_ctx else 0
        
        # PCA空间余弦
        cos_across_ctx_pca = []
        for w_idx in range(min(n_words, 30)):
            word_coords = coords_pca_stack[:, w_idx, :]
            for i in range(n_ctx):
                for j in range(i+1, n_ctx):
                    cos = compute_cos(word_coords[i], word_coords[j])
                    cos_across_ctx_pca.append(cos)
        
        mean_cos_across_ctx_pca = float(np.mean(cos_across_ctx_pca)) if cos_across_ctx_pca else 0
        
        # 3c) 全空间(hidden state)跨上下文余弦
        cos_full_space = []
        hs_by_ctx = {}
        for template, (H_ctx, words_ctx) in all_hs_by_context.items():
            ctx_indices = [words_ctx.index(w) for w in common_words]
            H_ctx_common = H_ctx[ctx_indices]
            hs_by_ctx[template] = H_ctx_common
        
        hs_stack = np.stack([hs_by_ctx[t] for t in CONTEXT_TEMPLATES 
                              if t in hs_by_ctx], axis=0)  # [n_ctx, n_words, d_model]
        
        for w_idx in range(min(n_words, 30)):
            word_hs = hs_stack[:, w_idx, :]  # [n_ctx, d_model]
            # 中心化后计算
            word_hs_centered = word_hs - np.mean(word_hs, axis=0, keepdims=True)
            for i in range(n_ctx):
                for j in range(i+1, n_ctx):
                    cos = compute_cos(word_hs_centered[i], word_hs_centered[j])
                    cos_full_space.append(cos)
        
        mean_cos_full = float(np.mean(cos_full_space)) if cos_full_space else 0
        
        # 3d) 属性解码的跨上下文稳定性
        # 在每个上下文中, 用ref上下文训练的probe解码属性
        decode_stability = {}
        for i, attr in enumerate(ATTR_NAMES):
            w_probe = W_probes[i]  # [d_model]
            
            # 在每个上下文中预测属性
            preds_by_ctx = []
            for template in CONTEXT_TEMPLATES:
                if template not in hs_by_ctx:
                    continue
                H_ctx = hs_by_ctx[template]
                pred = H_ctx_centered_proj = (H_ctx - H_ref_mean) @ w_probe
                preds_by_ctx.append(pred)
            
            if len(preds_by_ctx) < 3:
                continue
            
            preds_stack = np.stack(preds_by_ctx, axis=0)  # [n_ctx, n_words]
            # 每个概念跨上下文的预测方差
            var_pred = np.var(preds_stack, axis=0)  # [n_words]
            # 预测值的总方差
            var_total = np.var(preds_stack)
            
            # 稳定性 = 1 - (跨上下文方差/总方差)
            stability = 1.0 - float(np.mean(var_pred)) / max(float(var_total), 1e-10)
            decode_stability[attr] = float(stability)
        
        # 3e) 子空间 vs 全空间 的稳定性提升
        # 如果子空间更稳定 → 子空间滤除了"上下文噪声"
        stability_improvement = mean_cos_across_ctx - mean_cos_full
        
        layer_result = {
            "n_concepts": len(common_words),
            "n_contexts": len(all_hs_by_context),
            "k_95": int(k_95),
            "probe_r2_ref": probe_r2,
            "subspace_var_ratio": float(var_ratio),
            "pca_var_ratio": float(var_ratio_pca),
            "subspace_mean_cos_across_ctx": mean_cos_across_ctx,
            "pca_mean_cos_across_ctx": mean_cos_across_ctx_pca,
            "fullspace_mean_cos_across_ctx": mean_cos_full,
            "stability_improvement": float(stability_improvement),
            "decode_stability": decode_stability,
            "var_across_ctx_per_dim": [float(np.mean(var_across_ctx[:, d])) for d in range(k_95)],
            "var_across_words_per_dim": [float(var_across_words[d]) for d in range(k_95)],
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印关键结果
        print(f"    ★★★ 关键指标 ★★★")
        print(f"    上下文方差/属性方差: 子空间={var_ratio:.4f}, PCA={var_ratio_pca:.4f}")
        print(f"    跨上下文余弦: 子空间={mean_cos_across_ctx:.4f}, PCA={mean_cos_across_ctx_pca:.4f}, 全空间={mean_cos_full:.4f}")
        print(f"    子空间稳定性提升: {stability_improvement:.4f}")
        print(f"    属性解码稳定性:")
        for attr in ["edible", "animacy", "size", "color_red"]:
            if attr in decode_stability:
                print(f"      {attr}: {decode_stability[attr]:.4f}")
    
    return results


# ============================================================================
# 31B: 子空间因果干预 — 子空间干预能否改变输出?
# ============================================================================

def expB_causal_intervention(model_name, model, tokenizer, device, layers_to_test):
    """31B: 在子空间中干预, 看是否能改变logits"""
    print(f"\n{'='*70}")
    print(f"ExpB: 子空间因果干预 — 在7D子空间中干预, logits是否变化?")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "B", "experiment": "causal_intervention", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集hidden states
        H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, CONTEXT_TEMPLATES[0])
        if H is None or len(H) < 30:
            print(f"    数据不足, 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_words if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        n_attr = len(ATTR_NAMES)
        d_model = H_centered.shape[1]
        
        # ===== Step 1: 训练probe并获取子空间 =====
        W_probes = np.zeros((n_attr, d_model))
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
        
        # SVD获取子空间
        U_w, s_w, Vt_w = np.linalg.svd(W_probes, full_matrices=False)
        total_var = np.sum(s_w ** 2)
        cumvar = np.cumsum(s_w ** 2) / total_var
        k_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        
        subspace_basis = Vt_w[:k_95]  # [k_95, d_model]
        
        # 正交空间基
        orth_basis = Vt_w[k_95:]  # [d_model-k_95, d_model]
        
        print(f"    子空间维度 k_95={k_95}")
        
        # ===== Step 2: 干预实验 =====
        # 对每个干预概念:
        # A) 原始hidden state → logits
        # B) 在子空间中添加属性方向 → logits
        # C) 在正交空间中添加同样大小的方向 → logits
        # D) 在全空间中添加方向 → logits
        # 
        # 如果子空间干预有效但正交干预无效 → 子空间是因果机制变量
        
        intervention_results = []
        
        # 属性方向 (子空间中的)
        # edible: 在子空间中edible=1 vs edible=0的均值差
        edible_idx = V_centered[:, 0] > 0.5
        inedible_idx = ~edible_idx
        
        if np.sum(edible_idx) < 3 or np.sum(inedible_idx) < 3:
            print(f"    edible/inedible样本不足, 跳过")
            continue
        
        edible_dir_full = np.mean(H_centered[edible_idx], axis=0) - np.mean(H_centered[inedible_idx], axis=0)
        animacy_dir_full = np.mean(H_centered[V_centered[:, 1] > 0.5], axis=0) - np.mean(H_centered[V_centered[:, 1] <= 0.5], axis=0)
        
        # 投影到子空间和正交空间
        edible_sub = subspace_basis.T @ (subspace_basis @ edible_dir_full)  # 子空间部分
        edible_orth = edible_dir_full - edible_sub  # 正交空间部分
        
        animacy_sub = subspace_basis.T @ (subspace_basis @ animacy_dir_full)
        animacy_orth = animacy_dir_full - animacy_sub
        
        # 归一化干预强度
        scale = np.linalg.norm(edible_dir_full) + 1e-10
        beta = 2.0  # 干预强度倍数
        
        # 对干预概念做前向传播
        for concept in INTERVENTION_CONCEPTS:
            if concept not in valid_words:
                continue
            
            concept_idx = valid_words.index(concept)
            h_orig = H[concept_idx]  # 原始(未中心化)
            
            sent = CONTEXT_TEMPLATES[0].format(word=concept)
            toks = tokenizer(sent, return_tensors="pt").to(device)
            tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, concept)
            if dep_idx < 0:
                continue
            
            # 四种干预
            interventions = {
                "baseline": h_orig,
                "subspace_edible": h_orig + beta * edible_sub,
                "orth_edible": h_orig + beta * edible_orth,
                "full_edible": h_orig + beta * edible_dir_full,
                "subspace_animacy": h_orig + beta * animacy_sub,
                "orth_animacy": h_orig + beta * animacy_orth,
                "full_animacy": h_orig + beta * animacy_dir_full,
            }
            
            # 对每种干预, 替换hidden state, 计算logits
            logits_by_intervention = {}
            
            for intv_name, h_intv in interventions.items():
                # 直接用W_U计算logits (不需要前向传播)
                # h_intv: [d_model]
                h_tensor = torch.tensor(h_intv, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                logits = model.lm_head(h_tensor)
                logits_np = logits.detach().float().cpu().numpy()[0]  # [vocab_size]
                logits_by_intervention[intv_name] = logits_np
            
            # 计算logit变化
            baseline_logits = logits_by_intervention["baseline"]
            
            concept_intv = {
                "concept": concept,
                "concept_attrs": {attr: float(CONCEPT_DATASET[concept][attr]) for attr in ATTR_NAMES},
            }
            
            for intv_name in ["subspace_edible", "orth_edible", "full_edible",
                              "subspace_animacy", "orth_animacy", "full_animacy"]:
                diff = logits_by_intervention[intv_name] - baseline_logits
                # top-5变化的token
                top5_idx = np.argsort(np.abs(diff))[-5:][::-1]
                top5_tokens = [safe_decode(tokenizer, idx) for idx in top5_idx]
                top5_diffs = [float(diff[idx]) for idx in top5_idx]
                
                # 总变化量
                total_change = float(np.linalg.norm(diff))
                
                # 子空间干预 vs 正交干预 的有效性比
                concept_intv[intv_name] = {
                    "total_logit_change": total_change,
                    "top5_tokens": top5_tokens,
                    "top5_diffs": top5_diffs,
                }
            
            intervention_results.append(concept_intv)
        
        # ===== Step 3: 统计子空间 vs 正交干预效果 =====
        sub_edible_changes = [r["subspace_edible"]["total_logit_change"] for r in intervention_results]
        orth_edible_changes = [r["orth_edible"]["total_logit_change"] for r in intervention_results]
        full_edible_changes = [r["full_edible"]["total_logit_change"] for r in intervention_results]
        
        sub_anim_changes = [r["subspace_animacy"]["total_logit_change"] for r in intervention_results]
        orth_anim_changes = [r["orth_animacy"]["total_logit_change"] for r in intervention_results]
        full_anim_changes = [r["full_animacy"]["total_logit_change"] for r in intervention_results]
        
        # 子空间干预占比 = sub_change / full_change
        sub_edible_ratio = np.mean(sub_edible_changes) / max(np.mean(full_edible_changes), 1e-10)
        orth_edible_ratio = np.mean(orth_edible_changes) / max(np.mean(full_edible_changes), 1e-10)
        sub_anim_ratio = np.mean(sub_anim_changes) / max(np.mean(full_anim_changes), 1e-10)
        orth_anim_ratio = np.mean(orth_anim_changes) / max(np.mean(full_anim_changes), 1e-10)
        
        layer_result = {
            "k_95": int(k_95),
            "n_intervention_concepts": len(intervention_results),
            "edible_subspace_ratio": float(sub_edible_ratio),
            "edible_orth_ratio": float(orth_edible_ratio),
            "animacy_subspace_ratio": float(sub_anim_ratio),
            "animacy_orth_ratio": float(orth_anim_ratio),
            "mean_changes": {
                "sub_edible": float(np.mean(sub_edible_changes)),
                "orth_edible": float(np.mean(orth_edible_changes)),
                "full_edible": float(np.mean(full_edible_changes)),
                "sub_animacy": float(np.mean(sub_anim_changes)),
                "orth_animacy": float(np.mean(orth_anim_changes)),
                "full_animacy": float(np.mean(full_anim_changes)),
            },
            "sample_interventions": intervention_results[:4],  # 只保存前4个样本
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    ★★★ 因果干预效果 ★★★")
        print(f"    Edible: 子空间占比={sub_edible_ratio:.4f}, 正交占比={orth_edible_ratio:.4f}")
        print(f"    Animacy: 子空间占比={sub_anim_ratio:.4f}, 正交占比={orth_anim_ratio:.4f}")
        print(f"    均值变化: sub_edible={np.mean(sub_edible_changes):.4f}, orth_edible={np.mean(orth_edible_changes):.4f}, full={np.mean(full_edible_changes):.4f}")
        
        # 打印一个样本的top tokens
        if len(intervention_results) > 0:
            r = intervention_results[0]
            print(f"    样本 '{r['concept']}' 干预后top变化token:")
            for intv_name in ["subspace_edible", "orth_edible", "full_edible"]:
                top = r[intv_name]["top5_tokens"][:3]
                print(f"      {intv_name}: {top}")
    
    return results


# ============================================================================
# 31C: Lasso维度验证 + 子空间生成性
# ============================================================================

def expC_lasso_dimension(model_name, model, tokenizer, device, layers_to_test):
    """31C: Lasso probe验证真实维度 + 子空间插值生成"""
    print(f"\n{'='*70}")
    print(f"ExpC: Lasso维度验证 + 子空间插值生成性")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "C", "experiment": "lasso_dimension", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, CONTEXT_TEMPLATES[0])
        if H is None or len(H) < 30:
            print(f"    数据不足, 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_words if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        n_attr = len(ATTR_NAMES)
        d_model = H_centered.shape[1]
        
        # ===== Step 1: Lasso probe — 真实有效维度 =====
        # 对每个属性, 用Lasso回归, 看有多少非零权重
        
        lasso_results = {}
        ridge_results = {}
        
        # 先降维到PCA空间 (Lasso在高维不稳定)
        n_pca = min(100, d_model, len(H)-1)
        pca = PCA(n_components=n_pca)
        H_pca = pca.fit_transform(H_centered)
        
        for i, attr in enumerate(ATTR_NAMES):
            # Ridge (对比基准)
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_pca, V_centered[:, i])
            pred_ridge = ridge.predict(H_pca)
            ss_res = np.sum((V_centered[:, i] - pred_ridge) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            ridge_r2 = 1 - ss_res / max(ss_tot, 1e-10)
            ridge_nnz = np.sum(np.abs(ridge.coef_) > 1e-6)  # 几乎全部非零
            
            # Lasso (稀疏)
            # 用LassoCV自动选alpha
            try:
                lasso_cv = LassoCV(cv=min(5, len(H_pca)//10), max_iter=5000, n_alphas=20)
                lasso_cv.fit(H_pca, V_centered[:, i])
                best_alpha = float(lasso_cv.alpha_)
                
                lasso = Lasso(alpha=best_alpha, max_iter=5000)
                lasso.fit(H_pca, V_centered[:, i])
                
                pred_lasso = lasso.predict(H_pca)
                ss_res_l = np.sum((V_centered[:, i] - pred_lasso) ** 2)
                lasso_r2 = 1 - ss_res_l / max(ss_tot, 1e-10)
                lasso_nnz = int(np.sum(np.abs(lasso.coef_) > 1e-6))
                lasso_sparsity = 1.0 - lasso_nnz / n_pca
                
            except Exception as e:
                lasso_r2 = 0
                lasso_nnz = n_pca
                lasso_sparsity = 0
                best_alpha = 0
            
            lasso_results[attr] = {
                "r2": float(lasso_r2),
                "n_nonzero": lasso_nnz,
                "sparsity": float(lasso_sparsity),
                "best_alpha": float(best_alpha),
            }
            ridge_results[attr] = {
                "r2": float(ridge_r2),
                "n_nonzero": int(ridge_nnz),
            }
        
        # ===== Step 2: Lasso权重的有效维度 =====
        # 对Lasso权重做SVD, 看真正的秩
        
        W_lasso = np.zeros((n_attr, n_pca))
        for i, attr in enumerate(ATTR_NAMES):
            try:
                lasso = Lasso(alpha=lasso_results[attr]["best_alpha"], max_iter=5000)
                lasso.fit(H_pca, V_centered[:, i])
                W_lasso[i] = lasso.coef_
            except:
                pass
        
        U_l, s_l, Vt_l = np.linalg.svd(W_lasso, full_matrices=False)
        total_var_l = np.sum(s_l ** 2)
        cumvar_l = np.cumsum(s_l ** 2) / max(total_var_l, 1e-10)
        
        k_95_lasso = int(np.searchsorted(cumvar_l, 0.95) + 1) if total_var_l > 1e-10 else n_attr
        
        # ===== 计算子空间基 (用于后续插值和删除实验) =====
        W_probes_ridge = np.zeros((n_attr, d_model))
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes_ridge[i] = ridge.coef_
        _, _, Vt_ridge = np.linalg.svd(W_probes_ridge, full_matrices=False)
        total_var_ridge = np.sum(Vt_ridge ** 2)
        subspace_basis = Vt_ridge[:k_95_lasso]  # 使用Lasso确定的维度
        
        # ===== Step 3: 子空间插值生成性 =====
        # 在两个概念之间插值, decode成token, 看结果是否语义合理
        
        interpolation_results = []
        test_pairs = [
            ("apple", "dog"),      # fruit → animal
            ("hammer", "tree"),    # tool → plant
            ("apple", "orange"),   # same category
            ("dog", "elephant"),   # same category (animals)
        ]
        
        for w1, w2 in test_pairs:
            if w1 not in valid_words or w2 not in valid_words:
                continue
            
            idx1 = valid_words.index(w1)
            idx2 = valid_words.index(w2)
            h1 = H[idx1]
            h2 = H[idx2]
            
            # 在子空间中插值
            # 先投影到子空间
            h1_sub = subspace_basis.T @ (subspace_basis @ (h1 - H_mean[0])) + H_mean[0]
            h2_sub = subspace_basis.T @ (subspace_basis @ (h2 - H_mean[0])) + H_mean[0]
            
            interp_tokens = []
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                h_interp = (1 - alpha) * h1_sub + alpha * h2_sub
                logits = model.lm_head(torch.tensor(h_interp, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0))
                top_token_idx = torch.argmax(logits[0]).item()
                top_token = safe_decode(tokenizer, top_token_idx)
                top_prob = float(torch.softmax(logits[0], dim=0)[top_token_idx])
                
                # 也看top-3
                top3_idx = torch.topk(logits[0], 3).indices.tolist()
                top3_tokens = [safe_decode(tokenizer, idx) for idx in top3_idx]
                
                interp_tokens.append({
                    "alpha": alpha,
                    "top_token": top_token,
                    "top_prob": top_prob,
                    "top3_tokens": top3_tokens,
                })
            
            interpolation_results.append({
                "word1": w1, "word2": w2,
                "interpolation": interp_tokens,
            })
        
        # ===== Step 4: 子空间删除实验 =====
        # 把hidden state投影到正交空间(删掉子空间), 看logits变化
        
        deletion_results = []
        for concept in INTERVENTION_CONCEPTS[:6]:
            if concept not in valid_words:
                continue
            
            idx = valid_words.index(concept)
            h_orig = H[idx]
            
            # 删除子空间
            h_no_sub = h_orig - H_mean[0]
            h_no_sub = h_no_sub - subspace_basis.T @ (subspace_basis @ h_no_sub)  # 去掉子空间
            h_no_sub = h_no_sub + H_mean[0]
            
            # 计算logits
            logits_orig = model.lm_head(torch.tensor(h_orig, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0))
            logits_no_sub = model.lm_head(torch.tensor(h_no_sub, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0))
            
            # top token变化
            top_orig = safe_decode(tokenizer, torch.argmax(logits_orig[0]).item())
            top_no_sub = safe_decode(tokenizer, torch.argmax(logits_no_sub[0]).item())
            
            # logit变化量
            diff = (logits_no_sub - logits_orig).detach().float().cpu().numpy()[0]
            total_change = float(np.linalg.norm(diff))
            
            # 概率分布变化
            prob_orig = torch.softmax(logits_orig[0], dim=0).detach().float().cpu().numpy()
            prob_no_sub = torch.softmax(logits_no_sub[0], dim=0).detach().float().cpu().numpy()
            kl_div = float(np.sum(prob_no_sub * np.log((prob_no_sub + 1e-10) / (prob_orig + 1e-10))))
            
            deletion_results.append({
                "concept": concept,
                "top_token_orig": top_orig,
                "top_token_no_sub": top_no_sub,
                "total_logit_change": total_change,
                "kl_divergence": kl_div,
            })
        
        layer_result = {
            "n_concepts": len(valid_words),
            "n_pca_dims": n_pca,
            "lasso_results": lasso_results,
            "ridge_results": ridge_results,
            "lasso_subspace_k95": int(k_95_lasso),
            "lasso_singular_values": [float(s) for s in s_l[:min(10, len(s_l))]],
            "lasso_cumvar": [float(c) for c in cumvar_l[:min(10, len(cumvar_l))]],
            "interpolation_results": interpolation_results,
            "deletion_results": deletion_results,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印
        print(f"    ★★★ Lasso vs Ridge ★★★")
        for attr in ["edible", "animacy", "size", "color_red"]:
            if attr in lasso_results:
                lr = lasso_results[attr]
                rr = ridge_results[attr]
                print(f"      {attr}: Lasso R²={lr['r2']:.4f}(nnz={lr['n_nonzero']}), Ridge R²={rr['r2']:.4f}(nnz={rr['n_nonzero']})")
        
        print(f"    Lasso子空间k_95={k_95_lasso}")
        
        print(f"    插值生成:")
        for ir in interpolation_results[:2]:
            tokens = [step["top_token"] for step in ir["interpolation"]]
            print(f"      {ir['word1']}→{ir['word2']}: {tokens}")
        
        print(f"    子空间删除效果:")
        for dr in deletion_results[:4]:
            print(f"      {dr['concept']}: {dr['top_token_orig']}→{dr['top_token_no_sub']}, KL={dr['kl_divergence']:.4f}")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 31: 子空间三重验证 — 稳定性/因果性/生成性")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=稳定性, 2=因果, 3=Lasso")
    args = parser.parse_args()
    
    model_name = args.model
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 测试层 (聚焦浅层和中层)
    layers_to_test = [0, n_layers//6, n_layers//3, n_layers//2]
    layers_to_test = sorted(set(layers_to_test))
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {info.d_model}")
    print(f"测试层: {layers_to_test}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    if args.exp in [0, 1]:
        resA = expA_context_stability(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmk_expA_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpA] 结果已保存")
    
    if args.exp in [0, 2]:
        resB = expB_causal_intervention(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmk_expB_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpB] 结果已保存")
    
    if args.exp in [0, 3]:
        resC = expC_lasso_dimension(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccmk_expC_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resC, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpC] 结果已保存")
    
    # 释放GPU
    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
