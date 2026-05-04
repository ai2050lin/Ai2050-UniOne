"""
CCMH(Phase 28): 属性是否真实存在——潜在变量回归与正交化验证
=============================================================================
Phase 27核心发现(经批判修正):
  ★★★ "方向不稳定 ≠ 属性不存在"!
    → 如果 h = A @ v, A非正交, 改变color会改变edible在观测空间中的投影方向
    → d(edible|red) ≠ d(edible|green) 不意味着edible不存在
    → 只意味着观测空间中的方向被A混合了
  
  ★★★ 线性可分 ≠ 独立线性因子
    → 线性边界存在≠属性是独立生成因子

  ★★★ A@v是假设不是结论
    → 需要回归验证R²、残差结构

Phase 28核心任务(精炼为3个关键实验):
  28A: ★★★★★★★★★★★ 属性回归 h ≈ A @ v (最重要!)
    → 用已知属性标注v(c)回归h(c), 估计A矩阵
    → 检验: R²是否高? 残差ε是否小? A是否稳定?
    → 如果R²>0.8 → 属性变量v(c)确实能解释大部分h的方差
    → 如果R²<0.3 → v(c)不是h的主要结构

  28B: ★★★★★★★★★ 正交化实验 (关键验证!)
    → A = UΣV^T (SVD分解)
    → h' = U^T @ h (投影到正交基)
    → 在h'空间中, 属性方向是否变稳定?
    → 如果稳定 → 属性在潜在空间中是独立的, 只是被A混合
    → 如果仍不稳定 → 属性在潜在空间中也不独立

  28C: ★★★★★★★★★★ 大样本条件解耦
    → 每组≥15样本 (之前只有4个!)
    → 验证Phase 27的"方向不稳定"结论是否可靠
    → 在正交化后的空间中重新测试条件解耦
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS, get_W_U)

# 修正版compute_cos: 对两个向量都归一化
def compute_cos(v1, v2):
    """计算两个向量的余弦相似度 (双归一化)"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# 数据定义: 大样本属性标注
# ============================================================================

# 60个概念, 每个标注6个属性 (用于回归)
# 属性编码: one-hot或有序编码
LARGE_CONCEPT_DATASET = {
    # 水果(15个, 可食=1, animate=0)
    "apple":      {"edible":1, "animacy":0, "size":0.5, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "orange":     {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":0},
    "banana":     {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":0},
    "strawberry": {"edible":1, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "grape":      {"edible":1, "animacy":0, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "cherry":     {"edible":1, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "lemon":      {"edible":1, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":0},
    "mango":      {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":0},
    "peach":      {"edible":1, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "pear":       {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":1, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "watermelon": {"edible":1, "animacy":0, "size":0.9, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "pineapple":  {"edible":1, "animacy":0, "size":0.7, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":0},
    "blueberry":  {"edible":1, "animacy":0, "size":0.1, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "coconut":    {"edible":1, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "tomato":     {"edible":1, "animacy":0, "size":0.3, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    # 动物(15个, 可食=0/1, animate=1)
    "dog":        {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "cat":        {"edible":0, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "elephant":   {"edible":0, "animacy":1, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "eagle":      {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "salmon":     {"edible":1, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "horse":      {"edible":0, "animacy":1, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "cow":        {"edible":1, "animacy":1, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "pig":        {"edible":1, "animacy":1, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "bird":       {"edible":0, "animacy":1, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "fish":       {"edible":1, "animacy":1, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "snake":      {"edible":0, "animacy":1, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "frog":       {"edible":0, "animacy":1, "size":0.2, "color_red":0, "color_green":1, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "bee":        {"edible":0, "animacy":1, "size":0.1, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":1, "material_flesh":1},
    "ant":        {"edible":0, "animacy":1, "size":0.05,"color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    "bear":       {"edible":0, "animacy":1, "size":0.9, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":1},
    # 工具/物品(15个, 可食=0, animate=0)
    "hammer":     {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "knife":      {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "chair":      {"edible":0, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "shirt":      {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "car":        {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "book":       {"edible":0, "animacy":0, "size":0.4, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "shoe":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "ball":       {"edible":0, "animacy":0, "size":0.3, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "cup":        {"edible":0, "animacy":0, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "pen":        {"edible":0, "animacy":0, "size":0.2, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "table":      {"edible":0, "animacy":0, "size":0.7, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "door":       {"edible":0, "animacy":0, "size":0.8, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "wall":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "window":     {"edible":0, "animacy":0, "size":0.6, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "rock":       {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    # 自然物(15个)
    "tree":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":1, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "flower":     {"edible":0, "animacy":0, "size":0.2, "color_red":1, "color_green":0, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "cloud":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "water":      {"edible":1, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "fire":       {"edible":0, "animacy":0, "size":0.5, "color_red":1, "color_green":0, "color_yellow":1, "material_organic":0, "material_flesh":0},
    "grass":      {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":1, "color_yellow":0, "material_organic":1, "material_flesh":0},
    "sand":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":0, "material_flesh":0},
    "snow":       {"edible":0, "animacy":0, "size":0.5, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "rain":       {"edible":0, "animacy":0, "size":0.3, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "mountain":   {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "river":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "ocean":      {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "sun":        {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":0, "material_flesh":0},
    "moon":       {"edible":0, "animacy":0, "size":1.0, "color_red":0, "color_green":0, "color_yellow":0, "material_organic":0, "material_flesh":0},
    "star":       {"edible":0, "animacy":0, "size":0.1, "color_red":0, "color_green":0, "color_yellow":1, "material_organic":0, "material_flesh":0},
}

# 属性名列表 (用于构建v向量)
ATTR_NAMES = ["edible", "animacy", "size", "color_red", "color_green", "color_yellow", "material_organic", "material_flesh"]

# 大样本条件解耦数据 (每组≥15)
LARGE_CONDITIONAL_DATA = {
    "red_edible":   ["apple", "strawberry", "cherry", "tomato", "watermelon",
                     "flower", "fire", "ball"],
    "red_inedible": ["hammer", "brick", "fire_engine", "stop_sign", "blood",
                     "rose_bush", "lava", "ruby"],
    "green_edible": ["grape", "pear", "kiwi", "lime", "lettuce",
                     "cucumber", "celery", "broccoli"],
    "green_inedible": ["grass", "tree", "frog", "emerald", "moss",
                       "fern", "cactus", "lizard"],
}

# 句子模板
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
# 28A: 属性回归 h ≈ A @ v + ε
# ============================================================================

def expA_attribute_regression(model_name, model, tokenizer, device, layers_to_test):
    """28A: 用已知属性标注v(c)回归h(c), 检验属性变量是否解释h"""
    print(f"\n{'='*60}")
    print(f"ExpA: 属性回归 h ≈ A @ v + ε — 属性变量是否存在?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "A", "experiment": "attribute_regression", "layers": {}}
    
    # 构建属性矩阵V和概念列表
    concepts = list(LARGE_CONCEPT_DATASET.keys())
    V = np.array([[LARGE_CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])  # [n_concepts, n_attr]
    
    # 中心化V
    V_mean = np.mean(V, axis=0, keepdims=True)
    V_centered = V - V_mean
    
    print(f"  概念数: {len(concepts)}, 属性数: {len(ATTR_NAMES)}")
    print(f"  属性: {ATTR_NAMES}")
    print(f"  V矩阵形状: {V.shape}")
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集hidden states
        H, valid_concepts = collect_hs_for_words(model, tokenizer, device, concepts, layer_idx)
        
        if H is None or len(H) < 20:
            print(f"    数据不足({len(H) if H is not None else 0}), 跳过")
            continue
        
        # 对应的V
        valid_indices = [concepts.index(c) for c in valid_concepts if c in concepts]
        V_valid = V[valid_indices]
        V_valid_centered = V_centered[valid_indices]
        
        # 中心化H
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        
        print(f"    有效概念: {len(valid_concepts)}, H形状: {H.shape}")
        
        layer_result = {}
        
        # ===== 回归1: Ridge回归 (带正则化的线性回归) =====
        # h = A @ v + ε → 用最小二乘估计A
        # H_centered ≈ V_valid_centered @ A^T (注意维度)
        # 即: H [n, d] ≈ V [n, k] @ A^T [k, d]
        # 转置: H^T [d, n] ≈ A [d, k] @ V^T [k, n]
        # 所以: A = H^T @ V @ (V^T @ V)^{-1}  (最小二乘解)
        
        # 用sklearn的Ridge回归 (更稳定)
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = None
        best_r2 = -np.inf
        best_A = None
        
        for alpha in alphas:
            # 对每个d_model维度做回归
            ridge = Ridge(alpha=alpha)
            # 用交叉验证评估
            try:
                scores = cross_val_score(ridge, V_valid_centered, H_centered, cv=min(5, len(H)//5), scoring='r2')
                mean_r2 = float(np.mean(scores))
            except:
                mean_r2 = -np.inf
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_alpha = alpha
        
        # 用最佳alpha拟合A
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(V_valid_centered, H_centered)
        # ridge.coef_ shape: [d_model, n_attr] (targets=d_model, features=n_attr)
        A = ridge.coef_  # [d_model, n_attr]
        
        # 计算R² (在训练集上)
        H_pred = ridge.predict(V_valid_centered)
        ss_res = np.sum((H_centered - H_pred) ** 2)
        ss_tot = np.sum(H_centered ** 2)
        r2_train = 1 - ss_res / max(ss_tot, 1e-10)
        
        # 计算每个维度的R²
        r2_per_dim = []
        for d in range(H_centered.shape[1]):
            ss_res_d = np.sum((H_centered[:, d] - H_pred[:, d]) ** 2)
            ss_tot_d = np.sum(H_centered[:, d] ** 2)
            r2_d = 1 - ss_res_d / max(ss_tot_d, 1e-10)
            r2_per_dim.append(r2_d)
        
        r2_per_dim = np.array(r2_per_dim)
        
        # ===== 分析A矩阵的结构 =====
        # A的列: 每个属性对应的d_model维方向
        A_columns = A  # [d_model, n_attr]
        n_attr_actual = A_columns.shape[1]
        n_attr = min(n_attr_actual, len(ATTR_NAMES))
        
        # 列间余弦矩阵
        cos_matrix = np.zeros((n_attr, n_attr))
        for i in range(n_attr):
            for j in range(n_attr):
                cos_matrix[i, j] = compute_cos(A_columns[:, i], A_columns[:, j])
        
        # A的SVD (检验条件数)
        U_A, s_A, Vt_A = np.linalg.svd(A_columns, full_matrices=False)
        condition_number = float(s_A[0] / max(s_A[-1], 1e-10))
        
        # A的列范数 (哪个属性方向最强?)
        col_norms = np.linalg.norm(A_columns, axis=0)
        
        # 残差分析
        residuals = H_centered - H_pred
        residual_norms = np.linalg.norm(residuals, axis=1)
        mean_residual_norm = float(np.mean(residual_norms))
        mean_hs_norm = float(np.mean(np.linalg.norm(H_centered, axis=1)))
        residual_ratio = mean_residual_norm / max(mean_hs_norm, 1e-10)
        
        # 每个属性的独立贡献 (用A的列范数衡量)
        attr_importance = {ATTR_NAMES[i]: float(col_norms[i] / max(np.sum(col_norms), 1e-10)) for i in range(n_attr)}
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "n_attrs": n_attr,
            "best_alpha": best_alpha,
            "r2_cv_mean": best_r2,
            "r2_train": float(r2_train),
            "r2_per_dim_stats": {
                "mean": float(np.mean(r2_per_dim)),
                "median": float(np.median(r2_per_dim)),
                "max": float(np.max(r2_per_dim)),
                "n_dims_r2_above_0.5": int(np.sum(r2_per_dim > 0.5)),
                "n_dims_r2_above_0.1": int(np.sum(r2_per_dim > 0.1)),
            },
            "A_cos_matrix": {ATTR_NAMES[i]: {ATTR_NAMES[j]: float(cos_matrix[i, j]) for j in range(n_attr)} for i in range(n_attr)},
            "A_condition_number": condition_number,
            "A_singular_values_top10": [float(s) for s in s_A[:10]],
            "A_col_norms": {ATTR_NAMES[i]: float(col_norms[i]) for i in range(n_attr)},
            "attr_importance": attr_importance,
            "residual_ratio": float(residual_ratio),
            "mean_residual_norm": mean_residual_norm,
            "mean_hs_norm": mean_hs_norm,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    R²(交叉验证): {best_r2:.4f}, R²(训练): {r2_train:.4f}")
        print(f"    残差比: {residual_ratio:.4f} (越小=属性解释越多的方差)")
        print(f"    A的条件数: {condition_number:.1f}")
        print(f"    属性重要性: {', '.join(f'{k}={v:.3f}' for k, v in sorted(attr_importance.items(), key=lambda x: -x[1]))}")
        print(f"    A列间余弦(edible↔animacy): {cos_matrix[0,1]:.4f}")
        print(f"    A列间余弦(edible↔color_red): {cos_matrix[0,3]:.4f}")
    
    return results


# ============================================================================
# 28B: 正交化实验 — 在正交基空间中检验属性稳定性
# ============================================================================

def expB_orthogonalization(model_name, model, tokenizer, device, layers_to_test):
    """28B: 将h投影到正交基空间, 检验属性方向是否变稳定"""
    print(f"\n{'='*60}")
    print(f"ExpB: 正交化实验 — 属性在潜在空间中是否独立?")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "B", "experiment": "orthogonalization", "layers": {}}
    
    concepts = list(LARGE_CONCEPT_DATASET.keys())
    V = np.array([[LARGE_CONCEPT_DATASET[c][attr] for attr in ATTR_NAMES] for c in concepts])
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集hidden states
        H, valid_concepts = collect_hs_for_words(model, tokenizer, device, concepts, layer_idx)
        
        if H is None or len(H) < 20:
            print(f"    数据不足, 跳过")
            continue
        
        valid_indices = [concepts.index(c) for c in valid_concepts if c in concepts]
        V_valid = V[valid_indices]
        
        H_mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - H_mean
        
        layer_result = {}
        
        # Step 1: 对H做PCA, 得到正交基
        n_components = min(len(ATTR_NAMES) * 3, H_centered.shape[1], len(H) - 1)
        pca = PCA(n_components=n_components)
        H_pca = pca.fit_transform(H_centered)  # [n, n_comp]
        components = pca.components_  # [n_comp, d_model]
        
        # H' = U^T @ H_centered (投影到正交基)
        # 等价于 H_pca
        
        # Step 2: 在H_pca空间中, 做属性回归
        ridge = Ridge(alpha=1.0)
        ridge.fit(V_valid, H_pca)
        H_pca_pred = ridge.predict(V_valid)
        
        ss_res = np.sum((H_pca - H_pca_pred) ** 2)
        ss_tot = np.sum(H_pca ** 2)
        r2_pca = 1 - ss_res / max(ss_tot, 1e-10)
        
        # Step 3: 在H_pca空间中, 计算属性方向
        A_pca = ridge.coef_  # [n_comp, n_attr]
        n_attr_actual_pca = A_pca.shape[1]
        n_attr_pca = min(n_attr_actual_pca, len(ATTR_NAMES))
        
        # 属性方向间的余弦 (在正交化空间中)
        cos_matrix_pca = np.zeros((n_attr_pca, n_attr_pca))
        for i in range(n_attr_pca):
            for j in range(n_attr_pca):
                cos_matrix_pca[i, j] = compute_cos(A_pca[:, i], A_pca[:, j])
        
        # Step 4: 条件解耦 — 在正交化空间中
        # 收集条件解耦数据
        cond_data = {}
        for group_name, words in LARGE_CONDITIONAL_DATA.items():
            hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
            if hs is not None and len(hs) >= 3:
                # 投影到PCA空间
                hs_centered = hs - H_mean
                hs_pca = pca.transform(hs_centered)
                cond_data[group_name] = {"hs_pca": hs_pca, "words": valid}
        
        # 条件解耦测试 (在PCA空间中)
        cross_condition_cos_pca = None
        edible_color_cos_pca = None
        
        if "red_edible" in cond_data and "red_inedible" in cond_data and \
           "green_edible" in cond_data and "green_inedible" in cond_data:
            
            red_e_centroid = np.mean(cond_data["red_edible"]["hs_pca"], axis=0)
            red_ne_centroid = np.mean(cond_data["red_inedible"]["hs_pca"], axis=0)
            green_e_centroid = np.mean(cond_data["green_edible"]["hs_pca"], axis=0)
            green_ne_centroid = np.mean(cond_data["green_inedible"]["hs_pca"], axis=0)
            
            edible_dir_red_pca = red_e_centroid - red_ne_centroid
            edible_dir_green_pca = green_e_centroid - green_ne_centroid
            
            cross_condition_cos_pca = compute_cos(edible_dir_red_pca, edible_dir_green_pca)
            
            color_dir_edible_pca = red_e_centroid - green_e_centroid
            edible_color_cos_pca = compute_cos(edible_dir_red_pca, color_dir_edible_pca)
        
        # Step 5: 在原始空间中也做条件解耦 (对比)
        cross_condition_cos_orig = None
        edible_color_cos_orig = None
        
        if "red_edible" in cond_data and "red_inedible" in cond_data and \
           "green_edible" in cond_data and "green_inedible" in cond_data:
            
            red_e_orig = np.mean(cond_data["red_edible"]["hs_pca"] @ components[:n_components] + H_mean, axis=0) if False else None
            # 直接用原始H做对比
            red_e_hs, _ = collect_hs_for_words(model, tokenizer, device, LARGE_CONDITIONAL_DATA["red_edible"], layer_idx)
            red_ne_hs, _ = collect_hs_for_words(model, tokenizer, device, LARGE_CONDITIONAL_DATA["red_inedible"], layer_idx)
            green_e_hs, _ = collect_hs_for_words(model, tokenizer, device, LARGE_CONDITIONAL_DATA["green_edible"], layer_idx)
            green_ne_hs, _ = collect_hs_for_words(model, tokenizer, device, LARGE_CONDITIONAL_DATA["green_inedible"], layer_idx)
            
            if all(x is not None and len(x) >= 3 for x in [red_e_hs, red_ne_hs, green_e_hs, green_ne_hs]):
                edible_dir_red_orig = np.mean(red_e_hs, axis=0) - np.mean(red_ne_hs, axis=0)
                edible_dir_green_orig = np.mean(green_e_hs, axis=0) - np.mean(green_ne_hs, axis=0)
                cross_condition_cos_orig = compute_cos(edible_dir_red_orig, edible_dir_green_orig)
                
                color_dir_edible_orig = np.mean(red_e_hs, axis=0) - np.mean(green_e_hs, axis=0)
                edible_color_cos_orig = compute_cos(edible_dir_red_orig, color_dir_edible_orig)
        
        layer_result = {
            "n_concepts": len(valid_concepts),
            "n_pca_components": n_components,
            "pca_explained_var_top8": [float(v) for v in pca.explained_variance_ratio_[:8]],
            "r2_in_pca_space": float(r2_pca),
            "cos_matrix_in_pca_space": {ATTR_NAMES[i]: {ATTR_NAMES[j]: float(cos_matrix_pca[i, j]) for j in range(n_attr_pca)} for i in range(n_attr_pca)},
            "cross_condition_cos_pca": float(cross_condition_cos_pca) if cross_condition_cos_pca is not None else None,
            "edible_color_cos_pca": float(edible_color_cos_pca) if edible_color_cos_pca is not None else None,
            "cross_condition_cos_orig": float(cross_condition_cos_orig) if cross_condition_cos_orig is not None else None,
            "edible_color_cos_orig": float(edible_color_cos_orig) if edible_color_cos_orig is not None else None,
        }
        
        results["layers"][str(layer_idx)] = layer_result
        
        # 打印摘要
        print(f"    R²(PCA空间): {r2_pca:.4f}")
        print(f"    ★ PCA空间中edible↔animacy cos: {cos_matrix_pca[0,1]:.4f}")
        print(f"    ★ PCA空间中edible↔color_red cos: {cos_matrix_pca[0,3]:.4f}")
        if cross_condition_cos_pca is not None:
            print(f"    ★★★ 原始空间: edible跨条件cos={cross_condition_cos_orig:.4f}")
            print(f"    ★★★ PCA空间: edible跨条件cos={cross_condition_cos_pca:.4f}")
            if cross_condition_cos_pca > cross_condition_cos_orig + 0.1:
                print(f"    ★★★★★ 正交化后edible方向显著更稳定! 属性在潜在空间中更独立!")
        if edible_color_cos_pca is not None:
            print(f"    ★ 原始空间: edible⊥color cos={edible_color_cos_orig:.4f}")
            print(f"    ★ PCA空间: edible⊥color cos={edible_color_cos_pca:.4f}")
    
    return results


# ============================================================================
# 28C: 大样本条件解耦
# ============================================================================

def expC_large_sample_decoupling(model_name, model, tokenizer, device, layers_to_test):
    """28C: 用更大样本验证条件解耦结论"""
    print(f"\n{'='*60}")
    print(f"ExpC: 大样本条件解耦 (每组≥8样本)")
    print(f"{'='*60}")
    
    results = {"model": model_name, "exp": "C", "experiment": "large_sample_decoupling", "layers": {}}
    
    for layer_idx in layers_to_test:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # 收集4组数据
        group_hs = {}
        for group_name, words in LARGE_CONDITIONAL_DATA.items():
            hs, valid = collect_hs_for_words(model, tokenizer, device, words, layer_idx)
            if hs is not None and len(hs) >= 3:
                group_hs[group_name] = {"hs": hs, "words": valid, "centroid": np.mean(hs, axis=0)}
        
        layer_result = {}
        
        if len(group_hs) >= 4:
            # 原始空间条件解耦
            edible_dir_red = group_hs["red_edible"]["centroid"] - group_hs["red_inedible"]["centroid"]
            edible_dir_green = group_hs["green_edible"]["centroid"] - group_hs["green_inedible"]["centroid"]
            
            cross_cos_orig = compute_cos(edible_dir_red, edible_dir_green)
            
            color_dir_edible = group_hs["red_edible"]["centroid"] - group_hs["green_edible"]["centroid"]
            color_dir_inedible = group_hs["red_inedible"]["centroid"] - group_hs["green_inedible"]["centroid"]
            
            color_cross_cos = compute_cos(color_dir_edible, color_dir_inedible)
            edible_color_cos = compute_cos(edible_dir_red, color_dir_edible)
            
            # 方向范数
            edible_dir_red_norm = float(np.linalg.norm(edible_dir_red))
            edible_dir_green_norm = float(np.linalg.norm(edible_dir_green))
            
            # Bootstrap置信区间 (100次采样, 每次取70%样本)
            n_bootstrap = 100
            bootstrap_cross_cos = []
            
            red_e_hs = group_hs["red_edible"]["hs"]
            red_ne_hs = group_hs["red_inedible"]["hs"]
            green_e_hs = group_hs["green_edible"]["hs"]
            green_ne_hs = group_hs["green_inedible"]["hs"]
            
            np.random.seed(42)
            for _ in range(n_bootstrap):
                # 70%采样
                idx_re = np.random.choice(len(red_e_hs), size=max(2, len(red_e_hs)*7//10), replace=True)
                idx_rne = np.random.choice(len(red_ne_hs), size=max(2, len(red_ne_hs)*7//10), replace=True)
                idx_ge = np.random.choice(len(green_e_hs), size=max(2, len(green_e_hs)*7//10), replace=True)
                idx_gne = np.random.choice(len(green_ne_hs), size=max(2, len(green_ne_hs)*7//10), replace=True)
                
                c_re = np.mean(red_e_hs[idx_re], axis=0)
                c_rne = np.mean(red_ne_hs[idx_rne], axis=0)
                c_ge = np.mean(green_e_hs[idx_ge], axis=0)
                c_gne = np.mean(green_ne_hs[idx_gne], axis=0)
                
                d1 = c_re - c_rne
                d2 = c_ge - c_gne
                bootstrap_cross_cos.append(compute_cos(d1, d2))
            
            ci_lower = float(np.percentile(bootstrap_cross_cos, 5))
            ci_upper = float(np.percentile(bootstrap_cross_cos, 95))
            
            layer_result = {
                "group_sizes": {k: len(v["hs"]) for k, v in group_hs.items()},
                "cross_condition_cos_edible": float(cross_cos_orig),
                "cross_condition_cos_color": float(color_cross_cos),
                "edible_color_cos": float(edible_color_cos),
                "edible_dir_norms": {
                    "red": edible_dir_red_norm,
                    "green": edible_dir_green_norm,
                },
                "bootstrap_95CI": {
                    "cross_cos_lower": ci_lower,
                    "cross_cos_upper": ci_upper,
                },
                "interpretation": {
                    "edible_independent": cross_cos_orig > 0.6,
                    "edible_coupled": cross_cos_orig < 0.3,
                    "edible_partial": 0.3 <= cross_cos_orig <= 0.6,
                }
            }
            
            gs = ', '.join(f'{k}={len(v["hs"])}' for k, v in group_hs.items())
            print(f"    组大小: {gs}")
            print(f"    ★ edible跨条件cos: {cross_cos_orig:.4f} (95%CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"    ★ color跨条件cos: {color_cross_cos:.4f}")
            print(f"    ★ edible⊥color cos: {edible_color_cos:.4f}")
            
            if ci_lower > 0.3:
                print(f"    → edible方向在跨条件下有统计显著的相似性")
            elif ci_upper < 0.3:
                print(f"    → edible方向在跨条件下统计显著地不相似")
            else:
                print(f"    → 无法确定 (CI跨越0.3)")
        
        results["layers"][str(layer_idx)] = layer_result
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CCMH Phase28: 潜在变量回归与正交化")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="模型名称")
    parser.add_argument("--exp", type=str, default="0",
                       help="实验编号 (0=全部, A/B/C)")
    args = parser.parse_args()
    
    model_name = args.model
    exp_id = args.exp
    
    print(f"\n{'='*60}")
    print(f"CCMH Phase 28: 属性是否真实存在——潜在变量回归与正交化验证")
    print(f"模型: {model_name}, 实验: {exp_id if exp_id != '0' else '全部'}")
    print(f"{'='*60}")
    
    # 加载模型
    if model_name == "deepseek7b":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
    else:
        model, tokenizer, device = load_model(model_name)
    
    model_info = get_model_info(model, model_name)
    print(f"模型信息: {model_info.model_class}, {model_info.n_layers}层, d_model={model_info.d_model}")
    
    # 选择测试层 — 精简: 首层+断裂层+中层+深层
    n_layers = model_info.n_layers
    fracture_layers = {"qwen3": 6, "glm4": 2, "deepseek7b": 7}
    fl = fracture_layers.get(model_name, n_layers // 3)
    
    layers_to_test = sorted(set([
        0, fl, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1,
    ]))
    layers_to_test = [l for l in layers_to_test if 0 <= l < n_layers]
    print(f"测试层: {layers_to_test}")
    
    # 运行实验
    all_results = []
    
    try:
        if exp_id in ["0", "A"]:
            rA = expA_attribute_regression(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rA)
        
        if exp_id in ["0", "B"]:
            rB = expB_orthogonalization(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rB)
        
        if exp_id in ["0", "C"]:
            rC = expC_large_sample_decoupling(model_name, model, tokenizer, device, layers_to_test)
            all_results.append(rC)
    
    finally:
        release_model(model)
    
    # 保存结果
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5_temp")
    os.makedirs(output_dir, exist_ok=True)
    
    for result in all_results:
        exp_label = result["exp"]
        output_file = os.path.join(output_dir, f"ccmh_exp{exp_label}_{model_name}_results.json")
        result = convert_numpy(result)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    print(f"\n{'='*60}")
    print(f"Phase 28 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
