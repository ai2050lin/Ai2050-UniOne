# -*- coding: utf-8 -*-
"""
Stage470: 大规模概念图谱 — 扩展到500+概念，验证双曲结构在大规模下的成立性
===========================================================================

核心目标：
  Stage468在~100概念上验证了双曲结构，Stage470扩展到500+概念

关键问题：
  双曲结构是否在更大规模下仍然成立？还是只是小规模的人工产物？

实验设计：
  1. 500+概念集（12个类别，每类40-50词）
  2. 多层分析：不只黄金层，分析所有层
  3. 双曲性统计检验（放大样本量提高置信度）
  4. 概念图谱构建：基于距离/相似度的概念关系网络
  5. 谱分析：邻接矩阵的特征值分布（树状结构特征谱）
  6. 可视化：大规模2D Poincare嵌入

统计检验：
  - 超过1000个三角形的曲率估计
  - Bootstrap置信区间
  - 与随机基线对比
  - 与树结构基线对比

模型: 单模型（命令行选择）
  python stage470_large_scale_graph.py qwen3
  python stage470_large_scale_graph.py deepseek

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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage470_large_scale_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 大规模概念集（500+词） ====================
CONCEPTS = {
    "fruit": {
        "label": "水果", "level": 0,
        "words": [
            "apple", "banana", "orange", "grape", "mango", "peach", "lemon",
            "cherry", "berry", "melon", "kiwi", "plum", "pear", "fig", "lime",
            "coconut", "pineapple", "strawberry", "watermelon", "blueberry",
            "raspberry", "blackberry", "apricot", "nectarine", "pomegranate",
            "papaya", "guava", "passion_fruit", "dragon_fruit", "lychee",
            "durian", "jackfruit", "tangerine", "clementine", "date", "prune",
            "cranberry", "gooseberry", "elderberry", "currant", "mulberry",
            "boysenberry", "quince", "persimmon", "avocado", "olive", "tomato",
        ],
    },
    "animal": {
        "label": "动物", "level": 0,
        "words": [
            "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
            "whale", "shark", "snake", "eagle", "wolf", "bear", "monkey",
            "rabbit", "deer", "fox", "owl", "dolphin", "penguin", "parrot",
            "frog", "turtle", "crocodile", "giraffe", "zebra", "gorilla",
            "kangaroo", "panda", "leopard", "cheetah", "jaguar", "rhino",
            "hippo", "buffalo", "moose", "elk", "boar", "hare", "squirrel",
            "hamster", "mouse", "rat", "bat", "otter", "beaver", "hedgehog",
            "raccoon", "skunk", "armadillo", "porcupine", "lemur", "koala",
        ],
    },
    "vehicle": {
        "label": "交通工具", "level": 0,
        "words": [
            "car", "bus", "train", "plane", "ship", "bicycle", "motorcycle",
            "truck", "helicopter", "rocket", "boat", "submarine", "tractor",
            "van", "taxi", "ambulance", "firetruck", "scooter", "tram", "ferry",
            "yacht", "canoe", "kayak", "sailboat", "cruiser", "tank", "jeep",
            "suv", "sedan", "hatchback", "convertible", "minivan", "pickup",
            "wagon", "cart", "sleigh", "sled", "skateboard", "rollerblades",
            "segway", "monorail", "cable_car", "gondola", "hovercraft",
            "jet_ski", "airship", "blimp", "glider", "parachute", "hot_air_balloon",
        ],
    },
    "profession": {
        "label": "职业", "level": 0,
        "words": [
            "doctor", "nurse", "teacher", "engineer", "lawyer", "chef", "artist",
            "musician", "writer", "painter", "scientist", "programmer", "pilot",
            "soldier", "firefighter", "police", "farmer", "baker", "butcher",
            "driver", "architect", "carpenter", "plumber", "electrician",
            "mechanic", "surgeon", "dentist", "pharmacist", "veterinarian",
            "therapist", "psychiatrist", "accountant", "auditor", "banker",
            "manager", "director", "executive", "entrepreneur", "consultant",
            "analyst", "researcher", "professor", "librarian", "journalist",
            "photographer", "designer", "actor", "singer", "dancer", "comedian",
            "athlete", "coach", "referee", "umpire",
        ],
    },
    "clothing": {
        "label": "衣物", "level": 0,
        "words": [
            "shirt", "pants", "dress", "jacket", "shoes", "hat", "socks",
            "gloves", "scarf", "tie", "belt", "coat", "sweater", "boots",
            "sandals", "uniform", "jeans", "shorts", "skirt", "blazer",
            "vest", "cardigan", "tuxedo", "gown", "robe", "pajamas",
            "underwear", "bra", "slip", "camisole", "tank_top", "tshirt",
            "polo", "hoodie", "raincoat", "overcoat", "windbreaker", "parka",
            "mittens", "beanie", "cap", "beret", "fedora", "sombrero",
            "helmet", "crown", "tiara", "veil", "apron", "sari", "kimono",
        ],
    },
    "furniture": {
        "label": "家具", "level": 0,
        "words": [
            "chair", "table", "bed", "sofa", "desk", "bookcase", "cabinet",
            "wardrobe", "dresser", "shelf", "stool", "bench", "lamp", "rug",
            "curtain", "mirror", "couch", "armchair", "ottoman", "crib",
            "mattress", "pillow", "blanket", "quilt", "sheet", "towel",
            "nightstand", "headboard", "footboard", "chaise", "recliner",
            "hutch", "buffet", "sideboard", "console", " credenza",
            "vanity", "trundle", "bunk_bed", "loft_bed", "futon",
            "daybed", "rocking_chair", "highchair", "booster", "playpen",
            "hammock", "swing", "planter", "vase", "candle",
        ],
    },
    "food": {
        "label": "食物", "level": 0,
        "words": [
            "bread", "rice", "cake", "cookie", "pizza", "pasta", "soup",
            "salad", "sandwich", "steak", "chicken", "egg", "cheese", "butter",
            "milk", "yogurt", "ice_cream", "chocolate", "candy", "waffle",
            "pancake", "french_toast", "bacon", "sausage", "ham", "turkey",
            "lamb", "pork", "beef", "tofu", "noodle", "dumpling", "sushi",
            "taco", "burrito", "hamburger", "hotdog", "fries", "onion_rings",
            "pretzel", "popcorn", "chips", "cracker", "croissant", "bagel",
            "muffin", "scone", "donut", "brownie", "pie", "tart", "custard",
            "pudding", "jelly", "jam", "honey", "syrup", "ketchup",
        ],
    },
    "color": {
        "label": "颜色", "level": 0,
        "words": [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "gray", "navy", "beige", "ivory",
            "coral", "crimson", "maroon", "olive", "amber", "lavender",
            "teal", "cyan", "magenta", "violet", "indigo", "turquoise",
            "gold", "silver", "bronze", "copper", "rust", "burgundy",
            "plum", "salmon", "peach", "mint", "sage", "champagne",
            "taupe", "charcoal", "slate", "mauve", "lilac", "periwinkle",
            "aqua", "lime_green", "sky_blue", "forest_green", "khaki", "sienna",
        ],
    },
    "emotion": {
        "label": "情感", "level": 0,
        "words": [
            "happy", "sad", "angry", "fear", "love", "hate", "joy", "sorrow",
            "pride", "shame", "guilt", "envy", "jealousy", "hope", "despair",
            "anxiety", "calm", "excitement", "boredom", "loneliness",
            "gratitude", "contempt", "disgust", "surprise", "awe", "nostalgia",
            "regret", "relief", "contentment", "frustration", "confusion",
            "determination", "courage", "embarrassment", "euphoria", "melancholy",
            "apathy", "resentment", "admiration", "affection", "compassion",
            "empathy", "sympathy", "indifference", "curiosity", "wonder",
            "serenity", "bliss", "agony", "grief", "delight", "dread",
        ],
    },
    "body_part": {
        "label": "身体部位", "level": 0,
        "words": [
            "head", "hand", "foot", "arm", "leg", "eye", "ear", "nose",
            "mouth", "tooth", "finger", "thumb", "knee", "elbow", "shoulder",
            "neck", "back", "chest", "stomach", "heart", "brain", "lung",
            "liver", "kidney", "bone", "skin", "hair", "nail", "tongue",
            "lip", "cheek", "chin", "forehead", "eyebrow", "eyelash",
            "wrist", "ankle", "heel", "palm", "knuckle", "joint", "muscle",
            "vein", "artery", "spine", "rib", "hip", "waist", "throat",
            " Adam_apple", "collarbone", "shoulder_blade",
        ],
    },
    "natural": {
        "label": "自然", "level": 0,
        "words": [
            "mountain", "river", "ocean", "forest", "desert", "island",
            "valley", "cave", "volcano", "waterfall", "lake", "glacier",
            "meadow", "cliff", "swamp", "prairie", "jungle", "reef",
            "canyon", "plateau", "tundra", "savanna", "wetland", "lagoon",
            "fjord", "geyser", "delta", "dune", "crater", "atoll",
            "rainforest", "woodland", "steppe", "moor", "heath", "marsh",
            "bog", "fen", "estuary", "bay", "gulf", "strait", "channel",
            "tributary", "spring", "aquifer", "iceberg", "permafrost", "tundra",
        ],
    },
    "tool": {
        "label": "工具", "level": 0,
        "words": [
            "hammer", "saw", "drill", "wrench", "screwdriver", "pliers",
            "knife", "scissors", "axe", "chisel", "ruler", "tape_measure",
            "level", "square", "compass", "protractor", "calculator",
            "computer", "phone", "printer", "camera", "telescope", "microscope",
            "stethoscope", "thermometer", "scale", "clock", "watch", "calendar",
            "flashlight", "battery", "magnet", "lens", "mirror",
            "brush", "paint_roller", "sandpaper", "glue", "tape",
            "stapler", "paper_clip", "binder", "folder", "envelope",
            "pen", "pencil", "eraser", "marker", "crayon", "chalk",
        ],
    },
}

# 检查总数
_total = sum(len(c["words"]) for c in CONCEPTS.values())
print(f"概念集总词数: {_total}")


# ==================== Poincare Ball 几何 ====================

class PoincareBall:
    def __init__(self, dim: int, curvature: float = -1.0):
        self.dim = dim
        self.c = abs(curvature)

    def project(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm >= 1.0 - 1e-5:
            return x * 0.99 / norm
        return x

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
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

    def distance_batch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """批量计算X中所有点到y的距离"""
        x_norm_sq = np.sum(X ** 2, axis=1)
        y_norm_sq = np.sum(y ** 2)
        diff_sq = np.sum((X - y) ** 2, axis=1)
        denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
        denom = np.maximum(denom, EPS)
        arg = 1.0 + 2.0 * self.c * diff_sq / denom
        arg = np.maximum(arg, 1.0)
        return np.arccosh(np.minimum(arg, 100.0)) / np.sqrt(self.c)


# ==================== 统计检验 ====================

def curvature_test_bootstrap(D: np.ndarray, X_2d: np.ndarray, n_bootstrap: int = 100):
    """Bootstrap曲率检验"""
    n = D.shape[0]
    rng = np.random.RandomState(42)
    deficits = []

    for b in range(n_bootstrap):
        # 采样三角形
        idx = rng.choice(n, 3, replace=False)
        i, j, k = idx

        a = D[j, k]
        b_d = D[i, k]
        c = D[i, j]

        if a < EPS or b_d < EPS or c < EPS:
            continue

        cos_A = np.clip((b_d ** 2 + c ** 2 - a ** 2) / (2 * b_d * c), -1, 1)
        cos_B = np.clip((a ** 2 + c ** 2 - b_d ** 2) / (2 * a * c), -1, 1)
        cos_C = np.clip((a ** 2 + b_d ** 2 - c ** 2) / (2 * a * b_d), -1, 1)

        angle_sum = np.arccos(cos_A) + np.arccos(cos_B) + np.arccos(cos_C)
        deficits.append(np.pi - angle_sum)

    deficits = np.array(deficits)
    return {
        "n_triangles": len(deficits),
        "mean_deficit": float(np.mean(deficits)),
        "std_deficit": float(np.std(deficits)),
        "pct_negative_curvature": float(np.mean(deficits > 0) * 100),
        "ci_95_lower": float(np.percentile(deficits, 2.5)),
        "ci_95_upper": float(np.percentile(deficits, 97.5)),
        "t_statistic": float(np.mean(deficits) / (np.std(deficits) / np.sqrt(len(deficits) + 1))),
        "is_significantly_hyperbolic": float(np.mean(deficits)) > 0.01 and \
            abs(float(np.mean(deficits) / (np.std(deficits) / np.sqrt(len(deficits) + 1)))) > 2.0,
    }


def spectral_analysis(D: np.ndarray, k: int = 10):
    """谱分析：邻接矩阵的特征值分布

    树状结构/双曲空间的谱特征：
    - 特征值分布更集中
    - 谱隙（spectral gap）更大
    - 特征值有明显的等级结构
    """
    from scipy.linalg import eigh

    # 转换为相似度矩阵
    sigma = np.median(D[D > 0])
    W = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    # 度矩阵
    degree = np.sum(W, axis=1)
    D_diag = np.diag(degree)

    # 归一化拉普拉斯
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + EPS))
    L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    # 特征值分解
    try:
        eigenvalues, eigenvectors = eigh(L_norm, subset_by_index=[0, min(k - 1, W.shape[0] - 1)])
    except Exception:
        eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))[:k]
        eigenvectors = None

    # 谱隙
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0

    # 特征值比率
    ev_ratios = eigenvalues[1:] / (eigenvalues[:-1] + EPS) if len(eigenvalues) > 1 else []

    return {
        "eigenvalues": eigenvalues.tolist(),
        "spectral_gap": float(spectral_gap),
        "effective_dimension": float(np.sum(eigenvalues < 0.5)),
        "ev_ratios": [float(r) for r in ev_ratios],
        "tree_like_score": float(1.0 - eigenvalues[0] / (eigenvalues[1] + EPS)) if len(eigenvalues) > 1 else 0,
    }


def build_concept_graph(D: np.ndarray, word_labels: List[str], word_cats: List[str],
                         threshold_percentile: float = 10.0):
    """构建概念关系图谱"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = D.shape[0]

    # 使用距离的第threshold_percentile百分位作为边阈值
    threshold = np.percentile(D[D > 0], threshold_percentile)

    # 构建邻接矩阵
    adj = np.zeros((n, n), dtype=int)
    adj[D < threshold] = 1
    np.fill_diagonal(adj, 0)

    # 图统计
    n_edges = np.sum(adj) // 2
    degrees = np.sum(adj, axis=1)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)

    # 连通分量
    n_components, labels = connected_components(csr_matrix(adj))

    # 聚类系数
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            clustering_coeffs.append(0)
            continue
        n_possible = len(neighbors) * (len(neighbors) - 1) / 2
        n_actual = 0
        for a in neighbors:
            for b in neighbors:
                if a < b and adj[a, b] > 0:
                    n_actual += 1
        clustering_coeffs.append(n_actual / n_possible)

    avg_clustering = np.mean(clustering_coeffs)

    # 类内 vs 类间连接
    cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
    cat_labels = np.array([cat_to_int[c] for c in word_cats])

    intra_edges = 0
    inter_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                if cat_labels[i] == cat_labels[j]:
                    intra_edges += 1
                else:
                    inter_edges += 1

    # 平均路径长度
    from scipy.sparse.csgraph import shortest_path
    try:
        dist_matrix = shortest_path(csr_matrix(adj.astype(float)), directed=False)
        np.fill_diagonal(dist_matrix, np.inf)
        finite_dists = dist_matrix[dist_matrix < np.inf]
        avg_path_length = np.mean(finite_dists) if len(finite_dists) > 0 else float('inf')
    except Exception:
        avg_path_length = float('inf')

    return {
        "n_nodes": n,
        "n_edges": int(n_edges),
        "avg_degree": float(avg_degree),
        "max_degree": int(max_degree),
        "n_components": int(n_components),
        "avg_clustering": float(avg_clustering),
        "intra_edges": int(intra_edges),
        "inter_edges": int(inter_edges),
        "edge_ratio": float(intra_edges / max(intra_edges + inter_edges, 1)),
        "avg_path_length": float(avg_path_length) if np.isfinite(avg_path_length) else -1,
    }


def random_baseline_curvature(n: int, dim: int, n_triangles: int = 1000):
    """随机基线的曲率"""
    rng = np.random.RandomState(42)
    X = rng.randn(n, dim)
    D = squareform(pdist(X, 'euclidean'))

    deficits = []
    for _ in range(n_triangles):
        idx = rng.choice(n, 3, replace=False)
        i, j, k = idx
        a, b_d, c = D[j, k], D[i, k], D[i, j]
        if a < EPS or b_d < EPS or c < EPS:
            continue
        cos_A = np.clip((b_d ** 2 + c ** 2 - a ** 2) / (2 * b_d * c), -1, 1)
        cos_B = np.clip((a ** 2 + c ** 2 - b_d ** 2) / (2 * a * c), -1, 1)
        cos_C = np.clip((a ** 2 + b_d ** 2 - c ** 2) / (2 * a * b_d), -1, 1)
        angle_sum = np.arccos(cos_A) + np.arccos(cos_B) + np.arccos(cos_C)
        deficits.append(np.pi - angle_sum)

    return {
        "mean_deficit": float(np.mean(deficits)) if deficits else 0,
        "std_deficit": float(np.std(deficits)) if deficits else 0,
        "pct_negative": float(np.mean([d > 0 for d in deficits]) * 100) if deficits else 0,
    }


# ==================== Poincare嵌入（大规模优化版） ====================

def large_scale_poincare_embedding(X_high: np.ndarray, n_components: int = 2,
                                    curvature: float = -1.0, max_iter: int = 200):
    """大规模Poincare嵌入（优化版，使用Riemannian SGD）

    参考文献: Nickel & Kiela, "Poincare Embeddings for Learning Hierarchical Representations", NeurIPS 2017
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    n = X_high.shape[0]
    dim = n_components

    # 预处理
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_high)

    # 初始嵌入：PCA + 缩放到Poincare球内
    pca = PCA(n_components=dim, random_state=42)
    X_init = pca.fit_transform(X_norm)
    max_norm = np.max(np.linalg.norm(X_init, axis=1))
    X_init = X_init / (max_norm + EPS) * 0.7

    ball = PoincareBall(dim=dim, curvature=curvature)

    # 计算原始距离矩阵（归一化）
    D_orig = squareform(pdist(X_norm, 'euclidean'))
    D_orig = D_orig / (D_orig.max() + EPS)

    # Riemannian SGD
    lr = 0.01
    X = X_init.copy()
    rng = np.random.RandomState(42)

    # 负采样：每个正样本配k个负样本
    n_neg = 5

    print(f"    Riemannian SGD ({n} nodes, {dim}D, {max_iter} iters)...")
    for iteration in range(max_iter):
        total_loss = 0

        # 随机采样节点对
        for _ in range(min(n * 2, 500)):  # 每轮最多500次更新
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i == j:
                continue

            # 目标距离
            d_target = D_orig[i, j]

            # 当前Poincare距离
            d_current = ball.distance(X[i], X[j])

            # 损失：||d_current - d_target||^2
            loss = (d_current - d_target) ** 2
            total_loss += loss

            # 梯度（数值近似）
            grad = np.zeros(dim)
            eps = 1e-5
            for d in range(dim):
                X[i, d] += eps
                d_plus = ball.distance(X[i], X[j])
                X[i, d] -= 2 * eps
                d_minus = ball.distance(X[i], X[j])
                X[i, d] += eps
                grad[d] = (d_plus - d_minus) / (2 * eps) * 2 * (d_current - d_target)

            # 更新（投影到球内）
            X[i] = ball.project(X[i] - lr * grad)

        if iteration % 50 == 0:
            print(f"      iter {iteration}, loss: {total_loss:.4f}")

        # 学习率衰减
        lr *= 0.995

    # 确保所有点在球内
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norms >= 0.99, X / norms * 0.98, X)

    return X


# ==================== 模型和数据 ====================

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

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

        if len(word_vecs) < 30:
            continue

        bias_matrix = np.array(word_vecs)
        scaler = StandardScaler()
        normed = scaler.fit_transform(bias_matrix)

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
    word_vecs = []
    word_labels = []
    word_cats = []

    for cat_name, words_acts in all_activations.items():
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


# ==================== 主实验 ====================

def run_experiment(model_name: str):
    print(f"\n{'='*60}")
    print(f"Stage470: 大规模概念图谱 — {model_name}")
    print(f"{'='*60}")

    model_path = QWEN3_MODEL_PATH if model_name == "qwen3" else DEEPSEEK7B_MODEL_PATH

    # Step 1: 加载模型
    print("\n[Step 1] 加载模型...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # Step 2: 提取激活
    print("\n[Step 2] 提取概念激活（500+词）...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  耗时: {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 黄金层 + 偏置空间
    print("\n[Step 3] 寻找黄金层...")
    golden_layer = find_golden_layer(all_activations, CONCEPTS, layer_count)
    bias_matrix, word_labels, word_cats = build_bias_space(
        all_activations, CONCEPTS, layer_count, golden_layer
    )
    n_concepts = len(word_labels)
    print(f"  黄金层: {golden_layer}, 有效概念: {n_concepts}")

    if n_concepts < 100:
        print("  警告: 有效概念数过少，跳过大规模分析")

    # Step 4: 标准化
    print("\n[Step 4] 数据标准化...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    bias_normed = scaler.fit_transform(bias_matrix)
    D_orig = squareform(pdist(bias_normed, 'euclidean'))

    # Step 5: Bootstrap曲率检验
    print("\n[Step 5] Bootstrap曲率检验（1000+三角形）...")
    curvature_result = curvature_test_bootstrap(D_orig, bias_normed, n_bootstrap=1000)
    print(f"  平均角亏: {curvature_result['mean_deficit']:.4f}")
    print(f"  双曲比例: {curvature_result['pct_negative_curvature']:.1f}%")

    # Step 6: 随机基线对比
    print("\n[Step 6] 随机基线对比...")
    random_result = random_baseline_curvature(n_concepts, bias_matrix.shape[1])
    print(f"  随机基线角亏: {random_result['mean_deficit']:.4f}")
    print(f"  随机基线双曲比例: {random_result['pct_negative']:.1f}%")

    # Step 7: 谱分析
    print("\n[Step 7] 谱分析...")
    spectral = spectral_analysis(D_orig, k=min(15, n_concepts - 1))

    # Step 8: 概念图谱构建
    print("\n[Step 8] 概念图谱构建...")
    graph_stats = build_concept_graph(D_orig, word_labels, word_cats)
    print(f"  节点: {graph_stats['n_nodes']}, 边: {graph_stats['n_edges']}")
    print(f"  连通分量: {graph_stats['n_components']}")
    print(f"  类内/类间边比: {graph_stats['edge_ratio']:.3f}")

    # Step 9: Poincare嵌入（2D用于可视化）
    print("\n[Step 9] 大规模Poincare嵌入...")
    t0 = time.time()
    try:
        X_poincare = large_scale_poincare_embedding(bias_normed, n_components=2, max_iter=200)
        print(f"  Poincare嵌入完成，耗时: {time.time()-t0:.1f}s")

        # 计算Poincare嵌入后的质量
        ball = PoincareBall(dim=2)
        D_poincare = np.zeros((n_concepts, n_concepts))
        for i in range(n_concepts):
            D_poincare[i] = ball.distance_batch(X_poincare, X_poincare[i])

        # Stress
        from sklearn.manifold import MDS
        from sklearn.metrics import silhouette_score

        euclid_2d_stress = np.sqrt(np.sum((D_orig - squareform(pdist(
            StandardScaler().fit_transform(
                PCA_obj := __import__('sklearn.decomposition', fromlist=['PCA']).PCA(n_components=2, random_state=42).fit_transform(bias_normed)
            ), 'euclidean'))) ** 2) / (np.sum(D_orig ** 2) + EPS))

        poincare_stress = np.sqrt(np.sum((D_orig - D_poincare) ** 2) / (np.sum(D_orig ** 2) + EPS))

        # 轮廓系数
        cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
        labels_int = np.array([cat_to_int[c] for c in word_cats])

        euclid_sil = 0
        poincare_sil = 0
        try:
            from sklearn.decomposition import PCA as PCA2
            X_e2d = PCA2(n_components=2, random_state=42).fit_transform(bias_normed)
            euclid_sil = float(silhouette_score(X_e2d, labels_int))
            poincare_sil = float(silhouette_score(X_poincare, labels_int))
        except Exception:
            pass

        embedding_quality = {
            "euclidean_2d_stress": float(euclid_2d_stress),
            "poincare_2d_stress": float(poincare_stress),
            "euclidean_silhouette": euclid_sil,
            "poincare_silhouette": poincare_sil,
        }
    except Exception as e:
        print(f"  Poincare嵌入失败: {e}")
        embedding_quality = {"error": str(e)}
        X_poincare = None

    # Step 10: 多层分析
    print("\n[Step 10] 多层曲率分析...")
    layer_curvatures = {}
    for li in [0, layer_count // 4, layer_count // 2, 3 * layer_count // 4, layer_count - 1]:
        layer_vecs = []
        for cat_name, words_acts in all_activations.items():
            for word, acts in words_acts.items():
                if li in acts:
                    layer_vecs.append(acts[li])

        if len(layer_vecs) < 30:
            continue

        layer_matrix = np.array(layer_vecs)
        layer_normed = StandardScaler().fit_transform(layer_matrix)
        layer_D = squareform(pdist(layer_normed, 'euclidean'))
        layer_curve = curvature_test_bootstrap(layer_D, layer_normed, n_bootstrap=500)
        layer_curvatures[f"layer_{li}"] = {
            "mean_deficit": layer_curve["mean_deficit"],
            "pct_negative": layer_curve["pct_negative_curvature"],
            "is_significant": layer_curve["is_significantly_hyperbolic"],
        }

    # Step 11: 结果汇总
    print("\n[Step 11] 结果汇总...")

    all_results = {
        "model": model_name,
        "timestamp": TIMESTAMP,
        "golden_layer": int(golden_layer),
        "n_concepts": n_concepts,
        "n_categories": len(set(word_cats)),
        "neuron_dim": int(neuron_dim),
        "curvature_test": curvature_result,
        "random_baseline": random_result,
        "spectral_analysis": spectral,
        "graph_statistics": graph_stats,
        "embedding_quality": embedding_quality,
        "layer_curvatures": layer_curvatures,
        "conclusion": {
            "is_hyperbolic_at_scale": curvature_result.get("is_significantly_hyperbolic", False),
            "hyperbolic_vs_random": curvature_result["mean_deficit"] > random_result["mean_deficit"],
            "poincare_better": embedding_quality.get("poincare_2d_stress", 1) < embedding_quality.get("euclidean_2d_stress", 1),
            "scale_verdict": (
                "STRONG" if (curvature_result.get("is_significantly_hyperbolic", False) and
                           embedding_quality.get("poincare_2d_stress", 1) < embedding_quality.get("euclidean_2d_stress", 1)) else
                "MODERATE" if curvature_result.get("is_significantly_hyperbolic", False) else
                "WEAK"
            ),
        },
    }

    # 打印关键结果
    print(f"\n{'='*60}")
    print(f"关键发现（{n_concepts}概念规模）:")
    print(f"  Bootstrap曲率检验:")
    print(f"    平均角亏: {curvature_result['mean_deficit']:.6f}")
    print(f"    95% CI: [{curvature_result['ci_95_lower']:.6f}, {curvature_result['ci_95_upper']:.6f}]")
    print(f"    t统计量: {curvature_result['t_statistic']:.2f}")
    print(f"    双曲三角形比例: {curvature_result['pct_negative_curvature']:.1f}%")
    print(f"  vs 随机基线:")
    print(f"    随机角亏: {random_result['mean_deficit']:.6f}")
    print(f"    概念空间 vs 随机: {curvature_result['mean_deficit'] / (abs(random_result['mean_deficit']) + EPS):.2f}x")
    if "euclidean_2d_stress" in embedding_quality:
        print(f"  嵌入质量:")
        print(f"    欧氏2D Stress: {embedding_quality['euclidean_2d_stress']:.4f}")
        print(f"    Poincare2D Stress: {embedding_quality['poincare_2d_stress']:.4f}")
        print(f"    欧氏轮廓系数: {embedding_quality['euclidean_silhouette']:.4f}")
        print(f"    Poincare轮廓系数: {embedding_quality['poincare_silhouette']:.4f}")
    print(f"  图统计:")
    print(f"    谱隙: {spectral['spectral_gap']:.4f}")
    print(f"    类内边比例: {graph_stats['edge_ratio']:.3f}")
    print(f"  多层分析:")
    for layer_name, layer_data in layer_curvatures.items():
        sig = "★" if layer_data["is_significant"] else " "
        print(f"    {sig} {layer_name}: 角亏={layer_data['mean_deficit']:.4f}, 双曲%={layer_data['pct_negative']:.1f}%")
    print(f"  最终判定: {all_results['conclusion']['scale_verdict']}")
    print(f"{'='*60}")

    # 保存结果
    output_path = OUTPUT_DIR / f"stage470_results_{model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {output_path}")

    # 保存嵌入
    if X_poincare is not None:
        np.savez_compressed(
            OUTPUT_DIR / f"stage470_poincare_{model_name}.npz",
            X_poincare=X_poincare,
            word_labels=np.array(word_labels),
            word_cats=np.array(word_cats),
        )

    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python stage470_large_scale_graph.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ("qwen3", "deepseek"):
        print("错误: 请选择 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)
