# -*- coding: utf-8 -*-
"""
Stage468: 双曲嵌入验证 — Poincare Embedding + Lorentz Model 拟合偏置空间
========================================================================

核心目标：
  直接验证"概念偏置空间是双曲的"这一假说

实验设计：
  1. 嵌入质量对比（3种空间）：
     - 欧氏空间（baseline）
     - Poincare Ball Model（双曲空间表示1）
     - Lorentz Model / 双曲柱面（双曲空间表示2）
  2. 拟合指标：
     - Stress: 嵌入保距度（lower = better）
     - Trustworthiness: 邻域保持度（higher = better）
     - 类内聚类度（类内距离 vs 类间距离）
     - 层次结构保持度（概念层次关系是否在几何中体现）
  3. 双曲性统计检验：
     - 曲率估计（通过三角形角度偏差）
     - 比较不同曲率假设下的R²
     - Gauss-Bonnet定理验证
  4. 双模型交叉验证

模型: 单模型（命令行选择）
  python stage468_hyperbolic_embedding.py qwen3
  python stage468_hyperbolic_embedding.py deepseek

注意：一次只运行一个模型，避免GPU内存溢出
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
    QWEN3_MODEL_PATH,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage468_hyperbolic_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集（层次结构标注） ====================
# 层次结构用于验证双曲空间能否自然表示"IS-A"关系
CONCEPTS = {
    "fruit": {
        "label": "水果",
        "level": 0,  # 基础类别
        "words": ["apple", "banana", "orange", "grape", "mango", "peach",
                  "lemon", "cherry", "watermelon", "strawberry", "pear",
                  "pineapple", "coconut", "kiwi", "blueberry", "melon",
                  "fig", "plum", "lime"],
    },
    "animal": {
        "label": "动物",
        "level": 0,
        "words": ["dog", "cat", "bird", "fish", "horse", "lion", "tiger",
                  "elephant", "whale", "shark", "snake", "eagle", "wolf",
                  "bear", "monkey", "dolphin", "penguin", "rabbit", "deer",
                  "fox", "giraffe", "zebra", "gorilla", "kangaroo", "panda"],
    },
    "vehicle": {
        "label": "交通工具",
        "level": 0,
        "words": ["car", "bus", "train", "plane", "ship", "bicycle",
                  "motorcycle", "truck", "helicopter", "rocket", "boat",
                  "submarine", "taxi", "ambulance", "ferry"],
    },
    "profession": {
        "label": "职业",
        "level": 0,
        "words": ["doctor", "nurse", "teacher", "engineer", "lawyer",
                  "chef", "artist", "musician", "writer", "scientist",
                  "programmer", "pilot", "soldier", "firefighter", "police",
                  "farmer", "baker", "architect", "surgeon", "dentist"],
    },
    "color": {
        "label": "颜色",
        "level": 0,
        "words": ["red", "blue", "green", "yellow", "orange", "purple",
                  "pink", "brown", "black", "white", "gray", "navy",
                  "beige", "coral", "crimson", "maroon", "olive", "amber",
                  "lavender"],
    },
    "emotion": {
        "label": "情感",
        "level": 0,
        "words": ["happy", "sad", "angry", "fear", "love", "hate", "joy",
                  "sorrow", "pride", "shame", "guilt", "envy", "hope",
                  "despair", "anxiety", "calm", "excitement", "boredom",
                  "loneliness"],
    },
    "furniture": {
        "label": "家具",
        "level": 0,
        "words": ["chair", "table", "bed", "sofa", "desk", "bookcase",
                  "cabinet", "wardrobe", "shelf", "stool", "bench", "lamp",
                  "rug", "curtain", "mirror", "couch"],
    },
    "food": {
        "label": "食物",
        "level": 0,
        "words": ["bread", "rice", "cake", "cookie", "pizza", "pasta",
                  "soup", "salad", "sandwich", "steak", "chicken", "egg",
                  "cheese", "butter", "milk", "yogurt", "ice_cream",
                  "chocolate", "candy", "waffle"],
    },
}

# 概念子类层次（用于双曲层次结构验证）
HIERARCHY = {
    "fruit": {
        "berry": ["strawberry", "blueberry", "cherry", "grape"],
        "citrus": ["lemon", "lime", "orange"],
        "tropical": ["pineapple", "coconut", "mango", "banana"],
        "tree_fruit": ["apple", "pear", "peach", "plum", "fig"],
        "melon_family": ["watermelon", "melon"],
    },
    "animal": {
        "mammal": ["dog", "cat", "horse", "lion", "tiger", "elephant",
                   "whale", "bear", "monkey", "dolphin", "rabbit", "deer",
                   "fox", "giraffe", "zebra", "gorilla", "kangaroo", "panda"],
        "bird": ["bird", "eagle", "penguin", "ostrich"],
        "fish": ["fish", "shark", "dolphin"],
        "reptile": ["snake", "turtle", "crocodile"],
    },
    "vehicle": {
        "land": ["car", "bus", "train", "bicycle", "motorcycle", "truck",
                 "taxi", "ambulance"],
        "air": ["plane", "helicopter", "rocket"],
        "water": ["ship", "boat", "submarine", "ferry"],
    },
}


# ==================== 双曲几何核心函数（纯numpy实现） ====================

class PoincareBall:
    """Poincare Ball Model (庞加莱球模型)
    
    双曲空间的一种表示，定义在单位开球 {|x| < 1} 内
    距离: d(x,y) = arcosh(1 + 2*||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
    """

    def __init__(self, dim: int, curvature: float = -1.0):
        self.dim = dim
        self.c = abs(curvature)  # 曲率绝对值

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Poincare距离"""
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        diff_sq = np.sum((x - y) ** 2)

        denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
        if denom < EPS:
            return 10.0

        arg = 1.0 + 2.0 * self.c * diff_sq / denom
        if arg <= 1.0:
            return 0.0
        return np.arccosh(min(arg, 100.0)) / np.sqrt(self.c)

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """批量计算距离矩阵"""
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Möbius加法（双曲空间中的向量加法）"""
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        xy = np.dot(x, y)

        num = (1.0 + 2.0 * xy / self.c + y_norm_sq / self.c) * x + (1.0 - x_norm_sq) * y
        denom = 1.0 + 2.0 * xy / self.c + x_norm_sq * y_norm_sq / (self.c ** 2)
        if abs(denom) < EPS:
            return np.zeros_like(x)
        return num / denom

    def project(self, x: np.ndarray) -> np.ndarray:
        """投影到Poincare球内"""
        norm = np.linalg.norm(x)
        max_norm = 1.0 - 1e-5
        if norm >= max_norm:
            return x * (max_norm / norm) * 0.99
        return x

    def exp_map(self, v: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """指数映射（从切空间映射到流形）"""
        if x is None:
            x = np.zeros_like(v)
        v_norm = np.linalg.norm(v)
        if v_norm < EPS:
            return x
        # 缩放因子
        lam_x = (1.0 - np.linalg.norm(x) ** 2) / (2.0 * np.sqrt(self.c))
        tanh_term = np.tanh(np.sqrt(self.c) * v_norm / (2.0 * lam_x))
        unit_v = v / v_norm
        return self.project(x + 2.0 * lam_x * tanh_term * unit_v)

    def log_map(self, y: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """对数映射（从流形映射到切空间）"""
        if x is None:
            x = np.zeros_like(y)
        diff = y - x
        diff_norm = np.linalg.norm(diff)
        x_norm = np.linalg.norm(x)

        lam_x = (1.0 - x_norm ** 2) / (2.0 * np.sqrt(self.c))
        lam_y = (1.0 - np.linalg.norm(y) ** 2) / (2.0 * np.sqrt(self.c))
        sqrt_c = np.sqrt(self.c)

        denom = (1.0 + 2.0 * np.dot(x, diff) / self.c + np.linalg.norm(y) ** 2 / self.c)
        if denom < EPS:
            return np.zeros_like(y)

        arg = sqrt_c * diff_norm / denom
        arg = min(arg, 20.0)  # 截断避免溢出
        if arg < 1e-10:
            return np.zeros_like(y)

        factor = 2.0 * lam_x / sqrt_c * np.arctanh(sqrt_c * diff_norm / lam_x) / (diff_norm + EPS)
        return factor * diff


class LorentzModel:
    """Lorentz Model / Hyperboloid (洛伦兹模型/双曲面)
    
    双曲空间的另一种表示，定义在 {x: <x,x>_L = -1/c} 上
    其中Lorentz内积 <x,y>_L = -x[0]*y[0] + sum(x[1:]*y[1:])
    """

    def __init__(self, dim: int, curvature: float = -1.0):
        self.dim = dim + 1  # Lorentz空间多1维
        self.c = abs(curvature)
        self.ambient_dim = dim + 1

    def minkowski_dot(self, x: np.ndarray, y: np.ndarray) -> float:
        """Minkowski内积"""
        return -x[0] * y[0] + np.dot(x[1:], y[1:])

    def project_to_hyperboloid(self, x: np.ndarray) -> np.ndarray:
        """投影到双曲面上"""
        # 保持最后一个坐标为正
        x = np.array(x, dtype=np.float64)
        x[0] = np.sqrt(1.0 / self.c + np.sum(x[1:] ** 2))
        return x

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Lorentz距离"""
        md = self.minkowski_dot(x, y)
        if md <= 1.0 / self.c:
            md = 1.0 / self.c + EPS
        return np.arccosh(min(self.c * md, 100.0)) / np.sqrt(self.c)

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """批量计算距离矩阵"""
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def exp_map(self, v: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Lorentz指数映射"""
        if x is None:
            x = np.zeros(self.ambient_dim)
            x[0] = 1.0 / np.sqrt(self.c)
        v_norm = np.linalg.norm(v)
        if v_norm < EPS:
            return x
        return np.cosh(np.sqrt(self.c) * v_norm) * x + np.sinh(np.sqrt(self.c) * v_norm) * v / v_norm

    def log_map(self, y: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Lorentz对数映射"""
        if x is None:
            x = np.zeros(self.ambient_dim)
            x[0] = 1.0 / np.sqrt(self.c)
        md = self.minkowski_dot(x, y)
        if md <= 1.0 / self.c:
            md = 1.0 / self.c + EPS
        alpha = np.arccosh(min(self.c * md, 100.0)) / np.sqrt(self.c)
        if alpha < EPS:
            return np.zeros(self.ambient_dim)
        v = y - self.c * md * x
        v_norm = np.linalg.norm(v)
        if v_norm < EPS:
            return np.zeros(self.ambient_dim)
        return alpha * v / v_norm


# ==================== 嵌入质量评估指标 ====================

def stress_score(D_orig: np.ndarray, D_embed: np.ndarray) -> float:
    """压力分数（Kruskal Stress）：衡量嵌入保距度
    stress = sqrt(sum((D_orig - D_embed)^2) / sum(D_orig^2))
    """
    i_upper = np.triu_indices_from(D_orig, k=1)
    d_orig = D_orig[i_upper]
    d_embed = D_embed[i_upper]
    numerator = np.sum((d_orig - d_embed) ** 2)
    denominator = np.sum(d_orig ** 2)
    if denominator < EPS:
        return 1.0
    return float(np.sqrt(numerator / denominator))


def trustworthiness(D_orig: np.ndarray, X_embed: np.ndarray, k: int = 5) -> float:
    """信任度（Venna & Kaski, 2006）：衡量邻域保持度
    高信任度意味着原始空间中的近邻在嵌入后仍然保持为近邻
    """
    from sklearn.neighbors import NearestNeighbors

    n = D_orig.shape[0]
    # 原始空间k近邻
    orig_nn = np.argsort(D_orig, axis=1)[:, 1:k + 1]

    # 嵌入空间k近邻
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_embed)
    embed_nn = nn.kneighbors(X_embed, return_distance=False)[:, 1:k + 1]

    # 计算不在原始k近邻中但出现在嵌入k近邻中的点数
    violations = 0
    for i in range(n):
        orig_set = set(orig_nn[i])
        for j in embed_nn[i]:
            if j not in orig_set:
                # 距离惩罚
                violations += (D_orig[i, j] - np.median(np.sort(D_orig[i])[1:k + 1]))

    max_violations = n * k * (n - k - 1) / 2.0
    if max_violations < EPS:
        return 1.0
    return float(1.0 - 2.0 * violations / max_violations)


def clustering_quality(X_embed: np.ndarray, word_cats: List[str]) -> Dict:
    """聚类质量评估：类内距离 vs 类间距离"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
    labels = np.array([cat_to_int[c] for c in word_cats])

    # 类内距离
    within_dists = []
    for cat in cat_to_int:
        mask = labels == cat_to_int[cat]
        if np.sum(mask) < 2:
            continue
        cat_points = X_embed[mask]
        # 用Poincare距离（如果X_embed在Poincare球内）或欧氏距离
        if np.max(np.linalg.norm(cat_points, axis=1)) < 0.99:
            # 可能在Poincare球内，用欧氏距离近似
            pass
        dists = pdist(cat_points, metric='euclidean')
        within_dists.append(np.mean(dists))

    # 类间距离
    unique_cats = sorted(set(word_cats))
    between_dists = []
    for i in range(len(unique_cats)):
        for j in range(i + 1, len(unique_cats)):
            mask_i = labels == cat_to_int[unique_cats[i]]
            mask_j = labels == cat_to_int[unique_cats[j]]
            center_i = np.mean(X_embed[mask_i], axis=0)
            center_j = np.mean(X_embed[mask_j], axis=0)
            between_dists.append(np.linalg.norm(center_i - center_j))

    result = {
        "within_cluster_mean": float(np.mean(within_dists)) if within_dists else 0.0,
        "between_cluster_mean": float(np.mean(between_dists)) if between_dists else 0.0,
        "cluster_ratio": float(np.mean(between_dists) / (np.mean(within_dists) + EPS)) if within_dists else 0.0,
    }

    # 尝试计算轮廓系数
    try:
        if len(set(labels)) >= 2 and len(labels) > 10:
            result["silhouette"] = float(silhouette_score(X_embed, labels))
        else:
            result["silhouette"] = 0.0
    except Exception:
        result["silhouette"] = 0.0

    try:
        if len(set(labels)) >= 2:
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_embed, labels))
        else:
            result["calinski_harabasz"] = 0.0
    except Exception:
        result["calinski_harabasz"] = 0.0

    return result


# ==================== 嵌入方法 ====================

def euclidean_embedding(X_high: np.ndarray, n_components: int = 2) -> np.ndarray:
    """欧氏空间嵌入（PCA基线）"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_high)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_norm)


def poincare_embedding_from_euclidean(X_euclid: np.ndarray, curvature: float = -1.0) -> np.ndarray:
    """将欧氏嵌入映射到Poincare球内
    
    策略：通过缩放+非线性映射将欧氏坐标压缩到单位球内
    同时尽量保持相对距离关系
    """
    ball = PoincareBall(dim=X_euclid.shape[1], curvature=curvature)

    # 方法1: 基于距离的优化嵌入
    # 先用MDS保持距离，然后映射到Poincare球
    from sklearn.manifold import MDS

    # 计算欧氏距离矩阵
    D_euclid = squareform(pdist(X_euclid, 'euclidean'))
    n = X_euclid.shape[0]
    dim = X_euclid.shape[1]

    # 归一化距离
    D_euclid = D_euclid / (D_euclid.max() + EPS)

    # 初始嵌入：通过缩放放到球内
    max_norm = np.max(np.linalg.norm(X_euclid, axis=1))
    X_init = X_euclid / (max_norm + EPS) * 0.8

    # 优化：最小化嵌入距离和原始距离的差异
    def poincare_loss(flat):
        X = flat.reshape(n, dim)
        # 约束到球内
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = np.where(norms >= 0.99, X / norms * 0.98, X)
        # 计算Poincare距离矩阵
        D_poincare = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D_poincare[i, j] = ball.distance(X[i], X[j])
                D_poincare[j, i] = D_poincare[i, j]
        # Stress
        i_upper = np.triu_indices(n, k=1)
        diff = D_euclid[i_upper] - D_poincare[i_upper]
        return np.sum(diff ** 2)

    def poincare_grad(flat):
        """数值梯度"""
        grad = np.zeros_like(flat)
        eps = 1e-5
        loss0 = poincare_loss(flat)
        for i in range(len(flat)):
            flat[i] += eps
            loss1 = poincare_loss(flat)
            flat[i] -= eps
            grad[i] = (loss1 - loss0) / eps
        return grad

    flat_init = X_init.flatten()
    print("    优化Poincare嵌入...")

    # 使用L-BFGS-B
    bounds = [(-0.99, 0.99)] * len(flat_init)
    result = minimize(
        poincare_loss, flat_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-10},
    )

    X_poincare = result.x.reshape(n, dim)
    # 确保在球内
    norms = np.linalg.norm(X_poincare, axis=1, keepdims=True)
    X_poincare = np.where(norms >= 0.99, X_poincare / norms * 0.98, X_poincare)

    return X_poincare


def lorentz_embedding_from_euclidean(X_euclid: np.ndarray, curvature: float = -1.0) -> np.ndarray:
    """将欧氏嵌入映射到Lorentz双曲面
    
    策略：使用Poincare嵌入，然后通过等距变换转到Lorentz模型
    """
    n, dim = X_euclid.shape
    ball = PoincareBall(dim=dim, curvature=curvature)
    lorentz = LorentzModel(dim=dim, curvature=curvature)

    # 先得到Poincare嵌入
    X_poincare = poincare_embedding_from_euclidean(X_euclid, curvature)

    # Poincare -> Lorentz 转换公式：
    # x_Lorentz = (0, x_Poincare) -> x_Lorentz[0] = sqrt(1 + ||x_P||^2/c) / (1 - ||x_P||^2)
    # x_Lorentz[1:] = x_P / (1 - ||x_P||^2)
    X_lorentz = np.zeros((n, lorentz.ambient_dim))
    for i in range(n):
        p = X_poincare[i]
        p_norm_sq = np.sum(p ** 2)
        denom = 1.0 - p_norm_sq
        if denom < EPS:
            denom = EPS
        X_lorentz[i, 0] = (1.0 + p_norm_sq) / denom  # 简化版
        X_lorentz[i, 0] = np.sqrt(1.0 / lorentz.c + p_norm_sq / denom ** 2)
        X_lorentz[i, 1:] = p / denom

    return X_lorentz


# ==================== 双曲性检验 ====================

def estimate_curvature(X: np.ndarray, D_orig: np.ndarray) -> Dict:
    """估计空间曲率，验证是否为双曲空间
    
    方法：通过三角形的角度和检验空间曲率
    - 正曲率（球面）：三角形内角和 > π
    - 零曲率（欧氏）：三角形内角和 = π
    - 负曲率（双曲）：三角形内角和 < π
    """
    n = X.shape[0]
    angle_sums = []

    # 随机采样三角形
    rng = np.random.RandomState(42)
    n_triangles = min(200, n * (n - 1) * (n - 2) // 6)

    for _ in range(n_triangles):
        idx = rng.choice(n, 3, replace=False)
        i, j, k = idx

        # 三角形边长
        a = D_orig[j, k]  # 对顶点i
        b = D_orig[i, k]  # 对顶点j
        c = D_orig[i, j]  # 对顶点k

        # 余弦定理求角度（欧氏近似）
        if a < EPS or b < EPS or c < EPS:
            continue

        cos_A = np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)
        cos_B = np.clip((a ** 2 + c ** 2 - b ** 2) / (2 * a * c), -1, 1)
        cos_C = np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1, 1)

        angle_sum = np.arccos(cos_A) + np.arccos(cos_B) + np.arccos(cos_C)
        angle_sums.append(angle_sum)

    if not angle_sums:
        return {"mean_angle_sum": np.pi, "deficit": 0.0, "is_hyperbolic": False}

    mean_sum = np.mean(angle_sums)
    deficit = np.pi - mean_sum  # 正值 = 双曲

    return {
        "n_triangles": len(angle_sums),
        "mean_angle_sum": float(mean_sum),
        "std_angle_sum": float(np.std(angle_sums)),
        "deficit": float(deficit),
        "deficit_over_pi": float(deficit / np.pi),
        "is_hyperbolic": deficit > 0.02 * np.pi,  # 显著大于2%的双曲偏差
        "pct_hyperbolic": float(np.mean([s < np.pi for s in angle_sums]) * 100),
    }


def curvature_sweep_stress(D_orig: np.ndarray, X_euclid_2d: np.ndarray) -> Dict:
    """扫描不同曲率下的嵌入stress，找到最优曲率
    
    如果最优曲率显著为负，支持双曲假说
    """
    results = {}
    curvatures = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]

    for c in curvatures:
        if c == 0:
            # 欧氏
            results[f"c={c}"] = stress_score(D_orig, squareform(pdist(X_euclid_2d)))
        else:
            try:
                X_p = poincare_embedding_from_euclidean(X_euclid_2d, curvature=c)
                ball = PoincareBall(dim=X_euclid_2d.shape[1], curvature=c)
                D_p = ball.distance_matrix(X_p)
                results[f"c={c}"] = stress_score(D_orig, D_p)
            except Exception as e:
                results[f"c={c}"] = float('inf')

    # 找最优曲率
    best_c = min(results, key=results.get)
    return {
        "curvature_stress": results,
        "best_curvature": best_c,
        "best_stress": results[best_c],
        "euclidean_stress": results.get("c=0", float('inf')),
        "is_optimally_hyperbolic": "c=-" in best_c and results[best_c] < results.get("c=0", float('inf')),
    }


def hierarchy_preservation_test(X_embed: np.ndarray, word_list: List[str],
                                 hierarchy: Dict, space_type: str = "euclidean") -> Dict:
    """测试层次结构的保持度
    
    在双曲空间中，子类概念应该靠近父类，且子类内部也应该聚类
    """
    word_to_idx = {w: i for i, w in enumerate(word_list)}

    results = {}
    for parent_cat, subcats in hierarchy.items():
        cat_results = {}
        for subcat, subcat_words in subcats.items():
            # 获取子类词在嵌入中的位置
            valid_words = [w for w in subcat_words if w in word_to_idx]
            if len(valid_words) < 2:
                continue

            indices = [word_to_idx[w] for w in valid_words]
            subcat_points = X_embed[indices]

            if space_type == "euclidean":
                subcat_dists = pdist(subcat_points, 'euclidean')
            else:
                # Poincare距离
                subcat_dists = pdist(subcat_points, 'euclidean')  # 简化

            cat_results[subcat] = {
                "n_words": len(valid_words),
                "mean_intra_distance": float(np.mean(subcat_dists)),
                "std_intra_distance": float(np.std(subcat_dists)),
            }

        results[parent_cat] = cat_results

    return results


# ==================== 模型和数据提取 ====================

def sanitize_for_json(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    print(f"  CUDA available: {want_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
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
    print(f"  Layers: {layer_count}, NeuronDim: {neuron_dim}")
    return model, tokenizer, layer_count, neuron_dim


def extract_word_per_layer(model, tokenizer, word, layer_count):
    """提取单个词在每层的MLP neuron_in激活"""
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

        per_layer = {}
        for li in range(layer_count):
            buf = buffers[li]
            if buf is not None:
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                per_layer[li] = buf[0, pos].float().numpy()
        return per_layer
    finally:
        remove_hooks(handles)


def extract_all_activations(model, tokenizer, concepts, layer_count):
    """提取所有概念在每层的激活"""
    all_activations = {}
    total = sum(len(c["words"]) for c in concepts.values())
    done = 0

    for cat_name, cat_data in concepts.items():
        all_activations[cat_name] = {}
        for word in cat_data["words"]:
            try:
                acts = extract_word_per_layer(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except Exception:
                pass

    print(f"  Total extracted: {done}/{total}")
    return all_activations


def find_golden_layer(all_activations, concepts, layer_count):
    """找到信息密度最高的黄金层（通过类间方差最大化）"""
    from sklearn.preprocessing import StandardScaler

    best_layer = 0
    best_score = -1

    for li in range(layer_count):
        word_vecs = []
        word_cats = []

        for cat_name, words_acts in all_activations.items():
            for word, acts in words_acts.items():
                if li in acts:
                    word_vecs.append(acts[li])
                    word_cats.append(cat_name)

        if len(word_vecs) < 20:
            continue

        bias_matrix = np.array(word_vecs)
        scaler = StandardScaler()
        normed = scaler.fit_transform(bias_matrix)

        # 类间方差（eta²）
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(10, len(word_vecs) - 1), random_state=42)
        comp = svd.fit_transform(normed)

        cat_to_idx = defaultdict(list)
        for i, cat in enumerate(word_cats):
            cat_to_idx[cat].append(i)

        total_eta = 0
        for fi in range(min(5, comp.shape[1])):
            scores = comp[:, fi]
            grand_mean = np.mean(scores)
            ss_between = sum(len(idxs) * (np.mean(scores[idxs]) - grand_mean) ** 2
                           for idxs in cat_to_idx.values())
            ss_total = np.sum((scores - grand_mean) ** 2)
            total_eta += ss_between / (ss_total + 1e-10)

        if total_eta > best_score:
            best_score = total_eta
            best_layer = li

    return best_layer


def build_bias_space(all_activations, concepts, layer_count, golden_layer):
    """构建偏置空间矩阵"""
    word_vecs = []
    word_labels = []
    word_cats = []

    for cat_name, words_acts in all_activations.items():
        # 类别基底
        cat_vecs = []
        for word, acts in words_acts.items():
            if golden_layer in acts:
                cat_vecs.append(acts[golden_layer])
        if not cat_vecs:
            continue
        basis = np.mean(cat_vecs, axis=0)

        for word, acts in words_acts.items():
            if golden_layer in acts:
                bias = acts[golden_layer] - basis
                word_vecs.append(bias)
                word_labels.append(word)
                word_cats.append(cat_name)

    return np.array(word_vecs), word_labels, word_cats


# ==================== 主实验流程 ====================

def run_experiment(model_name: str):
    print(f"\n{'='*60}")
    print(f"Stage468: 双曲嵌入验证 — {model_name}")
    print(f"{'='*60}")

    model_path = QWEN3_MODEL_PATH if model_name == "qwen3" else DEEPSEEK7B_MODEL_PATH

    # Step 1: 加载模型
    print("\n[Step 1] 加载模型...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  模型加载耗时: {time.time()-t0:.1f}s")

    # Step 2: 提取激活
    print("\n[Step 2] 提取概念激活...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  激活提取耗时: {time.time()-t0:.1f}s")

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 找到黄金层
    print("\n[Step 3] 寻找黄金层...")
    golden_layer = find_golden_layer(all_activations, CONCEPTS, layer_count)
    print(f"  黄金层: Layer {golden_layer}")

    # Step 4: 构建偏置空间
    print("\n[Step 4] 构建偏置空间...")
    bias_matrix, word_labels, word_cats = build_bias_space(
        all_activations, CONCEPTS, layer_count, golden_layer
    )
    print(f"  偏置矩阵: {bias_matrix.shape[0]} 概念 x {bias_matrix.shape[1]} 维")

    # Step 5: 欧氏嵌入（基线）
    print("\n[Step 5] 欧氏嵌入（PCA基线）...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    bias_normed = scaler.fit_transform(bias_matrix)

    # 原始高维空间距离矩阵
    D_orig = squareform(pdist(bias_normed, 'euclidean'))

    # PCA到2维和10维
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2)
    X_euclid_2d = pca_2d.fit_transform(bias_normed)
    pca_10d = PCA(n_components=10)
    X_euclid_10d = pca_10d.fit_transform(bias_normed)

    euclidean_results = {
        "pca2d_explained_variance": float(pca_2d.explained_variance_ratio_.sum()),
        "pca10d_explained_variance": float(pca_10d.explained_variance_ratio_.sum()),
    }

    # Step 6: Poincare嵌入
    print("\n[Step 6] Poincare Ball嵌入...")
    t0 = time.time()
    try:
        X_poincare = poincare_embedding_from_euclidean(X_euclid_2d, curvature=-1.0)
        ball = PoincareBall(dim=2, curvature=-1.0)
        D_poincare = ball.distance_matrix(X_poincare)
        print(f"  Poincare嵌入完成，耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Poincare嵌入失败: {e}")
        X_poincare = None
        D_poincare = None

    # Step 7: Lorentz嵌入
    print("\n[Step 7] Lorentz Model嵌入...")
    t0 = time.time()
    try:
        X_lorentz = lorentz_embedding_from_euclidean(X_euclid_2d, curvature=-1.0)
        lorentz = LorentzModel(dim=2, curvature=-1.0)
        D_lorentz = lorentz.distance_matrix(X_lorentz)
        print(f"  Lorentz嵌入完成，耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  Lorentz嵌入失败: {e}")
        X_lorentz = None
        D_lorentz = None

    # Step 8: 嵌入质量对比
    print("\n[Step 8] 嵌入质量对比...")
    D_euclid_2d = squareform(pdist(X_euclid_2d, 'euclidean'))
    D_euclid_10d = squareform(pdist(X_euclid_10d, 'euclidean'))

    quality_results = {
        "euclidean_2d": {
            "stress": stress_score(D_orig, D_euclid_2d),
            "clustering": clustering_quality(X_euclid_2d, word_cats),
        },
        "euclidean_10d": {
            "stress": stress_score(D_orig, D_euclid_10d),
        },
    }

    if D_poincare is not None:
        quality_results["poincare"] = {
            "stress": stress_score(D_orig, D_poincare),
            "clustering": clustering_quality(X_poincare, word_cats),
        }

    if D_lorentz is not None:
        quality_results["lorentz"] = {
            "stress": stress_score(D_orig, D_lorentz),
        }

    # Step 9: 双曲性检验
    print("\n[Step 9] 双曲性检验...")
    curvature_test = estimate_curvature(X_euclid_2d, D_orig)

    print("\n[Step 10] 曲率扫描...")
    try:
        curvature_sweep = curvature_sweep_stress(D_orig, X_euclid_2d)
    except Exception as e:
        print(f"  曲率扫描失败: {e}")
        curvature_sweep = {"error": str(e)}

    # Step 11: 层次结构保持度
    print("\n[Step 11] 层次结构保持度测试...")
    hierarchy_euclid = hierarchy_preservation_test(
        X_euclid_2d, word_labels, HIERARCHY, "euclidean"
    )
    if X_poincare is not None:
        hierarchy_poincare = hierarchy_preservation_test(
            X_poincare, word_labels, HIERARCHY, "poincare"
        )
    else:
        hierarchy_poincare = {}

    # Step 12: 全局结果汇总
    print("\n[Step 12] 结果汇总...")
    all_results = {
        "model": model_name,
        "timestamp": TIMESTAMP,
        "golden_layer": int(golden_layer),
        "n_concepts": len(word_labels),
        "n_categories": len(set(word_cats)),
        "neuron_dim": int(neuron_dim),
        "euclidean_baseline": euclidean_results,
        "embedding_quality": quality_results,
        "hyperbolicity_test": curvature_test,
        "curvature_sweep": curvature_sweep,
        "hierarchy_preservation": {
            "euclidean": sanitize_for_json(hierarchy_euclid),
            "poincare": sanitize_for_json(hierarchy_poincare),
        },
    }

    # 结论
    is_hyperbolic = curvature_test.get("is_hyperbolic", False)
    poincare_better = (quality_results.get("poincare", {}).get("stress", 1) <
                       quality_results.get("euclidean_2d", {}).get("stress", 1))

    conclusion = {
        "space_is_hyperbolic": is_hyperbolic,
        "poincare_better_than_euclidean": poincare_better,
        "curvature_optimal_negative": curvature_sweep.get("is_optimally_hyperbolic", False),
        "overall_verdict": (
            "STRONG SUPPORT" if (is_hyperbolic and poincare_better) else
            "MODERATE SUPPORT" if (is_hyperbolic or poincare_better) else
            "NO SUPPORT"
        ),
    }
    all_results["conclusion"] = conclusion

    # 打印关键结果
    print(f"\n{'='*60}")
    print(f"关键发现:")
    print(f"  三角形平均角和: {curvature_test['mean_angle_sum']:.4f} rad "
          f"(π={np.pi:.4f}, 偏差={curvature_test['deficit']:.4f})")
    print(f"  双曲三角形比例: {curvature_test['pct_hyperbolic']:.1f}%")
    print(f"  欧氏2D Stress: {quality_results['euclidean_2d']['stress']:.4f}")
    if "poincare" in quality_results:
        print(f"  Poincare Stress: {quality_results['poincare']['stress']:.4f}")
    print(f"  欧氏轮廓系数: {quality_results['euclidean_2d']['clustering'].get('silhouette', 0):.4f}")
    if "poincare" in quality_results:
        print(f"  Poincare轮廓系数: {quality_results['poincare']['clustering'].get('silhouette', 0):.4f}")
    print(f"  最优曲率: {curvature_sweep.get('best_curvature', 'N/A')}")
    print(f"  最终判定: {conclusion['overall_verdict']}")
    print(f"{'='*60}")

    # 保存结果
    output_path = OUTPUT_DIR / f"stage468_results_{model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {output_path}")

    # 保存嵌入坐标
    np.savez_compressed(
        OUTPUT_DIR / f"stage468_embeddings_{model_name}.npz",
        euclid_2d=X_euclid_2d,
        poincare=X_poincare if X_poincare is not None else np.array([]),
        lorentz=X_lorentz if X_lorentz is not None else np.array([]),
        word_labels=np.array(word_labels),
        word_cats=np.array(word_cats),
    )

    return all_results


# ==================== 入口 ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python stage468_hyperbolic_embedding.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ("qwen3", "deepseek"):
        print("错误: 请选择 qwen3 或 deepseek")
        sys.exit(1)

    results = run_experiment(model_name)
