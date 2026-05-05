"""
CCML(Phase 32): 流形几何验证 — 从"线性坐标"到"非线性流形"
=============================================================================
Phase 31经批判修正后的核心认知:
  ❌ "删除子空间无影响 → 子空间不重要" → ✅ "存在冗余编码"
  ❌ "正交干预有效 → 子空间失败" → ✅ "分布式+非正交编码"
  ❌ "插值失败 → 子空间无效" → ✅ "子空间=流形切空间(tangent space)"
  ❌ "Lasso=3-5维 → 真实维度" → ✅ "Lasso只捕捉最强线性可读部分"
  
  ★★★ 核心洞察: 属性不是"坐标"而是"函数" g(h) ★★★
  → Probe可读 = 函数存在
  → 无单一方向 = 非线性
  → 子空间有效 = 局部线性(切空间)
  → 插值失败 = 离开流形
  → 正交空间有信息 = 非正交编码

Phase 32核心任务(3个关键验证):
  32A: ★★★★★★★★★★★ 尺度依赖线性性 (曲率测试!)
    → 在不同尺度上测试probe的线性预测是否成立
    → 如果小尺度线性、大尺度非线性 → 切空间假设成立 → 流形存在
    → 如果始终线性 → 不需要流形假设
    → 关键指标: probe预测误差随扰动尺度的变化曲线

  32B: ★★★★★★★★★ 跨层Probe一致性
    → 不同层的probe权重是否不同但预测一致?
    → 如果w_L1 ≠ w_L2 但 pred一致 → 属性=函数(不同点有不同切空间)
    → 如果w_L1 ≈ w_L2 → 属性=坐标(全局线性)
    → 关键指标: probe权重的跨层余弦 + 预测一致性

  32C: ★★★★★★★ 流形插值 (autoencoder latent插值)
    → 用autoencoder学习hidden states的低维流形
    → 在latent空间插值, 解码回hidden, 再decode成token
    → 如果latent插值产生有意义概念 → 流形假设确认
    → 对比线性插值(已知失败)
    → 关键指标: 插值点的token语义合理性
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
from sklearn.linear_model import Ridge
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
# 数据定义 (与Phase 31相同)
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

INTERVENTION_CONCEPTS = [
    "apple", "dog", "hammer", "tree",
    "salmon", "cow", "ball", "fire",
    "strawberry", "elephant", "rock", "water",
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
# 32A: 尺度依赖线性性 (曲率测试) — 最关键的实验!
# ============================================================================

def expA_scale_dependent_linearity(model_name, model, tokenizer, device, layers_to_test):
    """
    32A: 在不同扰动尺度上测试probe线性预测是否成立
    
    核心思想:
    - 如果语义编码是流形M, probe找到了切空间T_z M
    - 在切空间方向的小扰动 → 停在流形附近 → probe预测准确
    - 大扰动 → 离开流形 → probe预测偏差大
    
    实验:
    - 取一个概念h₀
    - 沿probe方向 w_edible 添加扰动: h₀ + ε·w
    - 对每个ε, 用probe预测属性值, 同时decode成token看语义
    - 如果小ε准确大ε偏差 → 切空间假设成立 → 流形存在
    """
    print(f"\n{'='*70}")
    print(f"32A: 尺度依赖线性性 (曲率测试)")
    print(f"     如果小尺度线性、大尺度非线性 → 切空间假设成立 → 流形存在")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "A", "experiment": "scale_dependent_linearity", "layers": {}}
    
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
        
        # SVD获取子空间
        U_w, s_w, Vt_w = np.linalg.svd(W_probes, full_matrices=False)
        total_var = np.sum(s_w ** 2)
        cumvar = np.cumsum(s_w ** 2) / total_var
        k_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        subspace_basis = Vt_w[:k_95]
        
        print(f"    子空间维度 k_95={k_95}")
        
        # ===== 尺度依赖测试 =====
        # 对多个测试概念, 沿不同方向扰动
        
        # 定义测试方向:
        # 1) probe方向 (edible, animacy) — 应该在子空间内
        # 2) 子空间主成分方向 — 子空间内
        # 3) 随机方向 — 可能在正交空间
        
        test_scales = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        
        # 属性方向 (归一化)
        edible_dir = W_probes[0] / (np.linalg.norm(W_probes[0]) + 1e-10)
        animacy_dir = W_probes[1] / (np.linalg.norm(W_probes[1]) + 1e-10)
        size_dir = W_probes[2] / (np.linalg.norm(W_probes[2]) + 1e-10)
        
        # 子空间主成分方向
        pc1_dir = Vt_w[0] / (np.linalg.norm(Vt_w[0]) + 1e-10)
        pc2_dir = Vt_w[1] / (np.linalg.norm(Vt_w[1]) + 1e-10) if k_95 > 1 else pc1_dir
        
        # 正交空间随机方向
        np.random.seed(42)
        rand_orth = np.random.randn(d_model)
        rand_orth = rand_orth - subspace_basis.T @ (subspace_basis @ rand_orth)  # 投影到正交空间
        rand_orth = rand_orth / (np.linalg.norm(rand_orth) + 1e-10)
        
        directions = {
            "edible_probe": edible_dir,
            "animacy_probe": animacy_dir,
            "size_probe": size_dir,
            "subspace_pc1": pc1_dir,
            "subspace_pc2": pc2_dir,
            "orth_random": rand_orth,
        }
        
        # 测试概念样本 (覆盖不同类别)
        test_concepts = ["apple", "dog", "hammer", "tree", "salmon", "ball"]
        
        scale_results = {}
        
        for dir_name, dir_vec in directions.items():
            dir_scale_results = []
            
            for concept in test_concepts:
                if concept not in valid_words:
                    continue
                
                c_idx = valid_words.index(concept)
                h_orig = H[c_idx]  # 未中心化
                h_orig_centered = H_centered[c_idx]
                
                concept_scales = []
                
                for eps in test_scales:
                    # 扰动后的hidden state
                    h_perturbed = h_orig + eps * dir_vec * np.linalg.norm(h_orig_centered)  # 按原始范数缩放
                    
                    # 用probe预测属性 (在中心化空间)
                    h_pert_centered = h_perturbed - H_mean[0]
                    preds = {}
                    for i, attr in enumerate(ATTR_NAMES):
                        pred_val = float(np.dot(W_probes[i], h_pert_centered))
                        preds[attr] = pred_val
                    
                    # Decode成top token
                    h_tensor = torch.tensor(h_perturbed, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.lm_head(h_tensor)
                    top5_idx = torch.topk(logits[0], 5).indices.tolist()
                    top5_tokens = [safe_decode(tokenizer, idx) for idx in top5_idx]
                    top5_probs = torch.softmax(logits[0], dim=0)[top5_idx].tolist()
                    
                    # 计算与原始logits的KL散度
                    logits_orig = model.lm_head(torch.tensor(h_orig, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0))
                    prob_orig = torch.softmax(logits_orig[0], dim=0).detach().float().cpu().numpy()
                    prob_pert = torch.softmax(logits[0], dim=0).detach().float().cpu().numpy()
                    kl = float(np.sum(prob_pert * np.log((prob_pert + 1e-10) / (prob_orig + 1e-10))))
                    
                    concept_scales.append({
                        "epsilon": eps,
                        "probe_predictions": preds,
                        "top5_tokens": top5_tokens,
                        "top5_probs": [float(p) for p in top5_probs],
                        "kl_from_original": kl,
                    })
                
                dir_scale_results.append({
                    "concept": concept,
                    "original_attrs": {attr: float(CONCEPT_DATASET[concept][attr]) for attr in ATTR_NAMES},
                    "scales": concept_scales,
                })
            
            scale_results[dir_name] = dir_scale_results
        
        # ===== 关键指标: 曲率 =====
        # 曲率 = |线性预测误差| / ε²
        # 如果曲率 ≈ 0 → 线性空间
        # 如果曲率 > 0 → 非线性流形
        
        curvature_analysis = {}
        for dir_name in ["edible_probe", "animacy_probe", "subspace_pc1", "orth_random"]:
            if dir_name not in scale_results:
                continue
            
            curvatures_by_concept = []
            for cr in scale_results[dir_name]:
                # 找到probe预测的变化率
                # 线性预测: pred(ε) = pred(0) + ε·||w||² (沿probe方向)
                # 如果是线性的, pred(ε) 应该线性增长
                # 如果是流形的, pred(ε) 在大ε时偏离线性
                
                preds_edible = [s["probe_predictions"]["edible"] for s in cr["scales"]]
                epsilons = [s["epsilon"] for s in cr["scales"]]
                
                # 归一化: 以ε=0.01为基准, 看随ε增长的偏差
                if len(preds_edible) < 3:
                    continue
                
                pred_base = preds_edible[0]  # ε=0.01的预测
                eps_base = epsilons[0]
                
                # 线性外推: pred_linear(ε) = pred_base * (ε / eps_base)
                deviations = []
                for i, (pred, eps) in enumerate(zip(preds_edible, epsilons)):
                    if eps < 1e-10:
                        continue
                    pred_linear = pred_base * (eps / eps_base)
                    deviation = abs(pred - pred_linear)
                    relative_dev = deviation / max(abs(pred_linear), 1e-10)
                    deviations.append({
                        "epsilon": eps,
                        "pred_actual": float(pred),
                        "pred_linear": float(pred_linear),
                        "relative_deviation": float(relative_dev),
                    })
                
                curvatures_by_concept.append({
                    "concept": cr["concept"],
                    "deviations": deviations,
                })
            
            curvature_analysis[dir_name] = curvatures_by_concept
        
        # ===== KL散度分析 =====
        # KL随ε的变化: 如果线性, KL应与ε²成正比
        # 如果流形, KL在大ε时增长更快 (离开流形导致输出崩溃)
        
        kl_analysis = {}
        for dir_name in ["edible_probe", "animacy_probe", "subspace_pc1", "orth_random"]:
            if dir_name not in scale_results:
                continue
            
            kl_by_scale = defaultdict(list)
            for cr in scale_results[dir_name]:
                for s in cr["scales"]:
                    kl_by_scale[s["epsilon"]].append(s["kl_from_original"])
            
            kl_means = {str(eps): float(np.mean(kls)) for eps, kls in kl_by_scale.items()}
            kl_analysis[dir_name] = kl_means
        
        # ===== 语义token变化分析 =====
        # 在不同尺度下, top token是否保持语义一致?
        semantic_analysis = {}
        for dir_name in ["edible_probe", "animacy_probe"]:
            if dir_name not in scale_results:
                continue
            
            for cr in scale_results[dir_name][:3]:  # 只分析前3个概念
                concept = cr["concept"]
                tokens_at_scales = [(s["epsilon"], s["top5_tokens"][:3]) for s in cr["scales"]]
                semantic_analysis[f"{dir_name}_{concept}"] = tokens_at_scales
        
        layer_result = {
            "k_95": int(k_95),
            "probe_r2": probe_r2,
            "test_scales": test_scales,
            "curvature_analysis": curvature_analysis,
            "kl_analysis": kl_analysis,
            "semantic_analysis": semantic_analysis,
            "n_test_concepts": len(test_concepts),
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印关键结果
        print(f"    ★★★ 曲率测试结果 ★★★")
        
        # 打印probe方向的偏差
        for dir_name in ["edible_probe", "animacy_probe", "orth_random"]:
            if dir_name not in curvature_analysis:
                continue
            ca = curvature_analysis[dir_name]
            if len(ca) > 0:
                # 取第一个概念, 打印偏差随ε的变化
                dev = ca[0]["deviations"]
                print(f"    {dir_name} (概念: {ca[0]['concept']}):")
                for d in dev[::2]:  # 每隔一个打印
                    print(f"      ε={d['epsilon']:.2f}: 实际={d['pred_actual']:.4f}, "
                          f"线性={d['pred_linear']:.4f}, 偏差={d['relative_deviation']:.4f}")
        
        # 打印KL散度
        print(f"    KL散度随ε变化:")
        for dir_name in ["edible_probe", "animacy_probe", "orth_random"]:
            if dir_name in kl_analysis:
                kl = kl_analysis[dir_name]
                kl_str = ", ".join([f"ε={eps}:KL={v:.4f}" for eps, v in sorted(kl.items(), key=lambda x: float(x[0]))[:5]])
                print(f"      {dir_name}: {kl_str}")
        
        # 打印语义变化
        print(f"    语义token变化 (edible方向):")
        for key, vals in list(semantic_analysis.items())[:2]:
            if "edible" in key:
                tokens_str = " → ".join([f"ε={eps}:[{','.join(t[:2])}]" for eps, t in vals[:5]])
                print(f"      {key}: {tokens_str}")
    
    return results


# ============================================================================
# 32B: 跨层Probe一致性 — 属性是"函数"还是"坐标"?
# ============================================================================

def expB_cross_layer_probe_consistency(model_name, model, tokenizer, device, layers_to_test):
    """
    32B: 不同层的probe权重是否不同但预测一致?
    
    如果语义是"坐标" → 不同层应该有相同方向
    如果语义是"函数" → 不同层可以有不同方向, 但预测一致
    → 因为函数在不同点有不同的切空间
    
    实验:
    - 在每个层训练Ridge probe
    - 比较不同层probe权重的余弦相似度
    - 比较不同层probe预测的相关性
    - 如果权重不同但预测一致 → 属性=函数(不同切空间)
    """
    print(f"\n{'='*70}")
    print(f"32B: 跨层Probe一致性 — 属性是'函数'还是'坐标'?")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "B", "experiment": "cross_layer_probe_consistency", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    V = np.array([[CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    # 在每个层收集hidden states并训练probe
    layer_probes = {}  # {layer_idx: {attr: w_probe}}
    layer_H = {}       # {layer_idx: (H, valid_words)}
    layer_r2 = {}
    
    for layer_idx in layers_to_test:
        print(f"\n  收集Layer {layer_idx}...")
        
        H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, CONTEXT_TEMPLATES[0])
        if H is None or len(H) < 30:
            continue
        
        valid_indices = [concepts.index(c) for c in valid_words if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        V_mean = np.mean(V_valid, axis=0, keepdims=True)
        V_centered = V_valid - V_mean
        
        d_model = H_centered.shape[1]
        n_attr = len(ATTR_NAMES)
        
        W_probes = np.zeros((n_attr, d_model))
        r2_dict = {}
        for i, attr in enumerate(ATTR_NAMES):
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_centered, V_centered[:, i])
            W_probes[i] = ridge.coef_
            pred = ridge.predict(H_centered)
            ss_res = np.sum((V_centered[:, i] - pred) ** 2)
            ss_tot = np.sum(V_centered[:, i] ** 2)
            r2_dict[attr] = 1 - ss_res / max(ss_tot, 1e-10)
        
        layer_probes[layer_idx] = W_probes
        layer_H[layer_idx] = (H, valid_words, H_mean, V_mean)
        layer_r2[layer_idx] = r2_dict
        
        print(f"    R²: edible={r2_dict['edible']:.3f}, animacy={r2_dict['animacy']:.3f}")
    
    # ===== 跨层Probe权重余弦相似度 =====
    print(f"\n  ★★★ 跨层Probe权重余弦相似度 ★★★")
    
    attr_cos_across_layers = {}
    for attr_idx, attr in enumerate(ATTR_NAMES):
        layer_indices = sorted(layer_probes.keys())
        cos_matrix = np.zeros((len(layer_indices), len(layer_indices)))
        
        for i, l1 in enumerate(layer_indices):
            for j, l2 in enumerate(layer_indices):
                cos_matrix[i, j] = compute_cos(layer_probes[l1][attr_idx], layer_probes[l2][attr_idx])
        
        attr_cos_across_layers[attr] = {
            "layer_indices": layer_indices,
            "cos_matrix": cos_matrix.tolist(),
            "mean_cos_off_diagonal": float(np.mean(cos_matrix[np.triu_indices(len(layer_indices), k=1)])),
            "min_cos": float(np.min(cos_matrix[np.triu_indices(len(layer_indices), k=1)])),
        }
        
        print(f"    {attr}: 跨层平均cos={attr_cos_across_layers[attr]['mean_cos_off_diagonal']:.4f}, "
              f"最低cos={attr_cos_across_layers[attr]['min_cos']:.4f}")
    
    # ===== 跨层预测一致性 =====
    # 用层L的probe在层L'的数据上预测, 看预测是否一致
    print(f"\n  ★★★ 跨层预测一致性 ★★★")
    
    # 找到所有层共有的词
    common_words = None
    for layer_idx in layer_H:
        _, vw, _, _ = layer_H[layer_idx]
        if common_words is None:
            common_words = set(vw)
        else:
            common_words = common_words & set(vw)
    
    if common_words is None or len(common_words) < 20:
        print(f"    共同词不足, 无法比较")
    else:
        common_words = sorted(common_words)
        layer_indices = sorted(layer_probes.keys())
        
        cross_pred_consistency = {}
        
        for attr_idx, attr in enumerate(["edible", "animacy", "size"]):
            # 对每对层, 比较预测值的相关性
            pred_corrs = {}
            for i, l1 in enumerate(layer_indices):
                for j, l2 in enumerate(layer_indices):
                    if l1 >= l2:
                        continue
                    
                    H1, vw1, H_mean1, V_mean1 = layer_H[l1]
                    H2, vw2, H_mean2, V_mean2 = layer_H[l2]
                    
                    # 取共同词
                    idx1 = [vw1.index(w) for w in common_words]
                    idx2 = [vw2.index(w) for w in common_words]
                    
                    H1_common = H1[idx1]
                    H2_common = H2[idx2]
                    
                    # 用各自的probe预测
                    pred1 = (H1_common - H_mean1[0]) @ layer_probes[l1][attr_idx]
                    pred2 = (H2_common - H_mean2[0]) @ layer_probes[l2][attr_idx]
                    
                    # 相关系数
                    corr = np.corrcoef(pred1, pred2)[0, 1]
                    pred_corrs[f"L{l1}_L{l2}"] = float(corr)
            
            cross_pred_consistency[attr] = pred_corrs
            
            print(f"    {attr}: 跨层预测相关性 = {pred_corrs}")
    
    # ===== 子空间跨层对齐 =====
    # 不同层的子空间基是否一致?
    print(f"\n  ★★★ 子空间跨层对齐 ★★★")
    
    subspace_alignment = {}
    layer_indices = sorted(layer_probes.keys())
    
    for i, l1 in enumerate(layer_indices):
        for j, l2 in enumerate(layer_indices):
            if l1 >= l2:
                continue
            
            W1 = layer_probes[l1]
            W2 = layer_probes[l2]
            
            # SVD获取子空间
            _, _, Vt1 = np.linalg.svd(W1, full_matrices=False)
            _, _, Vt2 = np.linalg.svd(W2, full_matrices=False)
            
            k = min(7, Vt1.shape[0], Vt2.shape[0])
            sub1 = Vt1[:k]
            sub2 = Vt2[:k]
            
            # 子空间对齐度 = ||P1·P2||_F / k (越大越对齐)
            # 用子空间余弦相似度
            # 对sub1的每个向量, 找sub2中最近的
            alignment_scores = []
            for v1 in sub1:
                max_cos = max([compute_cos(v1, v2) for v2 in sub2])
                alignment_scores.append(max_cos)
            
            mean_alignment = float(np.mean(alignment_scores))
            
            # CCA相似度 (简化版: 用子空间主角度)
            # S_sub = sub1 @ sub2.T 的SVD
            S = sub1 @ sub2.T
            _, s_vals, _ = np.linalg.svd(S)
            principal_angles = np.arccos(np.clip(s_vals, -1, 1))
            mean_angle = float(np.mean(principal_angles))
            
            key = f"L{l1}_L{l2}"
            subspace_alignment[key] = {
                "mean_vector_alignment": mean_alignment,
                "principal_angles_rad": [float(a) for a in principal_angles],
                "mean_principal_angle_rad": mean_angle,
                "mean_principal_angle_deg": float(np.degrees(mean_angle)),
            }
            
            print(f"    {key}: 向量对齐={mean_alignment:.4f}, "
                  f"主角度={np.degrees(mean_angle):.1f}°")
    
    results["attr_cos_across_layers"] = attr_cos_across_layers
    results["cross_pred_consistency"] = cross_pred_consistency
    results["subspace_alignment"] = subspace_alignment
    results["layer_r2"] = {str(k): v for k, v in layer_r2.items()}
    results["common_words_count"] = len(common_words) if common_words else 0
    
    return results


# ============================================================================
# 32C: 流形插值 — Autoencoder Latent插值 vs 线性插值
# ============================================================================

class SimpleAutoencoder(nn.Module):
    """简单自编码器, 学习hidden states的低维流形"""
    def __init__(self, d_model, d_latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def expC_manifold_interpolation(model_name, model, tokenizer, device, layers_to_test):
    """
    32C: 用Autoencoder学习流形, 在latent空间插值, 对比线性插值
    
    如果latent插值产生有意义概念 → 流形存在
    如果线性插值也有效 → 不需要流形假设(不太可能, Phase 31已证伪)
    """
    print(f"\n{'='*70}")
    print(f"32C: 流形插值 — Autoencoder Latent插值 vs 线性插值")
    print(f"{'='*70}")
    
    results = {"model": model_name, "exp": "C", "experiment": "manifold_interpolation", "layers": {}}
    
    concepts = list(CONCEPT_DATASET.keys())
    
    # 只测试浅层和深层
    for layer_idx in layers_to_test[:2]:  # 只测2层,节省时间
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集大量hidden states (用于训练AE)
        all_hs_by_context = []
        all_words_by_context = []
        
        for template in CONTEXT_TEMPLATES:
            H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, template)
            if H is not None:
                all_hs_by_context.append(H)
                all_words_by_context.append(valid_words)
        
        if len(all_hs_by_context) == 0:
            continue
        
        # 合并所有上下文的hidden states
        H_all = np.vstack(all_hs_by_context)
        print(f"    总样本数: {H_all.shape[0]}")
        
        # 用第一个模板的hidden states作为主要测试集
        H, valid_words = collect_hs_with_template(model, tokenizer, device, concepts, layer_idx, CONTEXT_TEMPLATES[0])
        if H is None:
            continue
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        
        # ===== 训练Autoencoder =====
        d_model = H_all.shape[1]
        
        # 尝试不同的latent维度
        latent_dims = [4, 8, 16]
        ae_results = {}
        
        for d_latent in latent_dims:
            print(f"\n    训练Autoencoder (d_latent={d_latent})...")
            
            ae = SimpleAutoencoder(d_model, d_latent).to(device)
            optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
            
            # 准备训练数据
            H_all_mean = np.mean(H_all, axis=0, keepdims=True)
            H_train = torch.tensor(H_all - H_all_mean, dtype=torch.float32).to(device)
            
            # 训练
            n_epochs = 200
            batch_size = 32
            n_samples = H_train.shape[0]
            
            ae.train()
            for epoch in range(n_epochs):
                perm = torch.randperm(n_samples)
                epoch_loss = 0
                n_batches = 0
                
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch = H_train[perm[start:end]]
                    
                    recon, z = ae(batch)
                    loss = nn.MSELoss()(recon, batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                if (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/n_batches:.6f}")
            
            # 评估重建质量
            ae.eval()
            with torch.no_grad():
                H_test = torch.tensor(H_centered, dtype=torch.float32).to(device)
                recon, z_test = ae(H_test)
                recon_np = recon.cpu().numpy()
                z_np = z_test.cpu().numpy()
            
            # 重建R²
            ss_res = np.sum((H_centered - recon_np) ** 2)
            ss_tot = np.sum(H_centered ** 2)
            recon_r2 = 1 - ss_res / max(ss_tot, 1e-10)
            
            print(f"    重建R²: {recon_r2:.4f}")
            
            # ===== 插值实验 =====
            test_pairs = [
                ("apple", "dog"),
                ("hammer", "tree"),
                ("apple", "orange"),
                ("dog", "elephant"),
            ]
            
            interpolation_results = []
            
            for w1, w2 in test_pairs:
                if w1 not in valid_words or w2 not in valid_words:
                    continue
                
                idx1 = valid_words.index(w1)
                idx2 = valid_words.index(w2)
                
                z1 = z_np[idx1]
                z2 = z_np[idx2]
                h1_orig = H[idx1]
                h2_orig = H[idx2]
                
                # 1) 线性插值 (在原始空间)
                linear_interp = []
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    h_interp = (1 - alpha) * h1_orig + alpha * h2_orig
                    h_tensor = torch.tensor(h_interp, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.lm_head(h_tensor)
                    top5_idx = torch.topk(logits[0], 5).indices.tolist()
                    top5_tokens = [safe_decode(tokenizer, idx) for idx in top5_idx]
                    top5_probs = torch.softmax(logits[0], dim=0)[top5_idx].tolist()
                    
                    linear_interp.append({
                        "alpha": alpha,
                        "top5_tokens": top5_tokens,
                        "top5_probs": [float(p) for p in top5_probs],
                    })
                
                # 2) AE latent插值
                ae_interp = []
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    z_tensor = torch.tensor(z_interp, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        h_recon = ae.decoder(z_tensor)
                    h_recon_np = h_recon.cpu().numpy()[0] + H_mean[0]
                    
                    h_tensor = torch.tensor(h_recon_np, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.lm_head(h_tensor)
                    top5_idx = torch.topk(logits[0], 5).indices.tolist()
                    top5_tokens = [safe_decode(tokenizer, idx) for idx in top5_idx]
                    top5_probs = torch.softmax(logits[0], dim=0)[top5_idx].tolist()
                    
                    ae_interp.append({
                        "alpha": alpha,
                        "top5_tokens": top5_tokens,
                        "top5_probs": [float(p) for p in top5_probs],
                    })
                
                # 3) PCA空间插值 (作为基线)
                n_pca = min(d_latent * 2, d_model, len(H_centered) - 1)
                pca = PCA(n_components=n_pca)
                H_pca = pca.fit_transform(H_centered)
                
                pca_interp = []
                h1_pca = H_pca[idx1]
                h2_pca = H_pca[idx2]
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    h_pca_interp = (1 - alpha) * h1_pca + alpha * h2_pca
                    h_orig_space = pca.inverse_transform(h_pca_interp.reshape(1, -1))[0] + H_mean[0]
                    
                    h_tensor = torch.tensor(h_orig_space, dtype=model.lm_head.weight.dtype, device=device).unsqueeze(0)
                    with torch.no_grad():
                        logits = model.lm_head(h_tensor)
                    top5_idx = torch.topk(logits[0], 5).indices.tolist()
                    top5_tokens = [safe_decode(tokenizer, idx) for idx in top5_idx]
                    top5_probs = torch.softmax(logits[0], dim=0)[top5_idx].tolist()
                    
                    pca_interp.append({
                        "alpha": alpha,
                        "top5_tokens": top5_tokens,
                        "top5_probs": [float(p) for p in top5_probs],
                    })
                
                interpolation_results.append({
                    "word1": w1, "word2": w2,
                    "linear_interp": linear_interp,
                    "ae_interp": ae_interp,
                    "pca_interp": pca_interp,
                })
            
            ae_results[str(d_latent)] = {
                "recon_r2": float(recon_r2),
                "d_latent": d_latent,
                "interpolation_results": interpolation_results,
            }
            
            # 打印结果
            print(f"\n    ★★★ 插值结果 (d_latent={d_latent}) ★★★")
            for ir in interpolation_results[:2]:
                print(f"      {ir['word1']} → {ir['word2']}:")
                linear_tokens = [s["top5_tokens"][0] for s in ir["linear_interp"]]
                ae_tokens = [s["top5_tokens"][0] for s in ir["ae_interp"]]
                pca_tokens = [s["top5_tokens"][0] for s in ir["pca_interp"]]
                print(f"        线性: {linear_tokens}")
                print(f"        AE:   {ae_tokens}")
                print(f"        PCA:  {pca_tokens}")
        
        layer_result = {
            "d_model": d_model,
            "n_train_samples": H_all.shape[0],
            "ae_results": ae_results,
        }
        
        results["layers"][str(layer_idx)] = layer_result
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 32: 流形几何验证")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                        help="实验编号: 0=全部, 1=曲率, 2=跨层, 3=AE插值")
    args = parser.parse_args()
    
    model_name = args.model
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    layers_to_test = [0, n_layers//3, n_layers//2]
    layers_to_test = sorted(set(layers_to_test))
    print(f"模型: {model_name}, 层数: {n_layers}, d_model: {info.d_model}")
    print(f"测试层: {layers_to_test}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.exp in [0, 1]:
        resA = expA_scale_dependent_linearity(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccml_expA_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resA, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpA] 曲率测试结果已保存")
    
    if args.exp in [0, 2]:
        resB = expB_cross_layer_probe_consistency(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccml_expB_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resB, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpB] 跨层一致性结果已保存")
    
    if args.exp in [0, 3]:
        resC = expC_manifold_interpolation(model_name, model, tokenizer, device, layers_to_test)
        with open(os.path.join(output_dir, f"ccml_expC_{model_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(resC, f, ensure_ascii=False, indent=2)
        print(f"\n[ExpC] AE插值结果已保存")
    
    release_model(model)
    print(f"\n模型 {model_name} 已释放")


if __name__ == "__main__":
    main()
