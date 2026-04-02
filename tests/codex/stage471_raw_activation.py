# -*- coding: utf-8 -*-
"""
Stage471: 原始激活空间分析 — 跳出偏置空间，在完整激活空间中寻找概念编码的数学结构
================================================================================

核心动机：
  Stage461-470一直聚焦于偏置空间（概念-类别均值的差），
  这可能丢失了概念编码中重要的全局结构信息。
  Stage471回到完整的原始激活空间，直接分析其几何和拓扑结构。

关键问题：
  1. 原始激活空间的内在维度(intrinsic dimensionality)是多少？
  2. 空间是线性的还是弯曲的？曲率如何随空间位置变化？
  3. 概念在原始空间中的分布是否有特定的几何模式？
  4. 类别边界是超平面（线性可分）还是流形边界（弯曲）？
  5. 原始空间和偏置空间的关系是什么？（投影几何）

实验设计：
  1. 内在维度估计（MLE, TwoNN, 参与比）
  2. 流形学习分析（Isomap, t-SNE, UMAP的局部vs全局结构）
  3. 局部曲率估计（Hessian特征值分析）
  4. 线性可分性检验（SVM vs RBF-SVM vs 核方法）
  5. 概念轨迹分析（概念空间中的路径和流形）
  6. 信息瓶颈分析（各层的信息保留vs压缩）
  7. 高斯混合模型拟合（概念是否形成高斯簇）
  8. 残差分析（偏置空间丢失了什么信息）

模型: 单模型（命令行选择）
  python stage471_raw_activation.py qwen3
  python stage471_raw_activation.py deepseek

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage471_raw_activation_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ==================== 概念集 ====================
CONCEPTS = {
    "fruit": {
        "label": "水果", "level": 0,
        "words": ["apple", "banana", "orange", "grape", "mango", "peach",
                  "lemon", "cherry", "watermelon", "strawberry", "pear",
                  "pineapple", "coconut", "kiwi", "blueberry", "melon",
                  "fig", "plum", "lime", "raspberry", "blackberry",
                  "apricot", "nectarine", "pomegranate", "papaya", "guava",
                  "lychee", "tangerine", "date", "cranberry", "persimmon",
                  "avocado", "olive", "tomato", "quince"],
    },
    "animal": {
        "label": "动物", "level": 0,
        "words": ["dog", "cat", "bird", "fish", "horse", "lion", "tiger",
                  "elephant", "whale", "shark", "snake", "eagle", "wolf",
                  "bear", "monkey", "dolphin", "penguin", "rabbit", "deer",
                  "fox", "giraffe", "zebra", "gorilla", "kangaroo", "panda",
                  "leopard", "cheetah", "jaguar", "rhino", "hippo", "buffalo",
                  "moose", "elk", "squirrel", "hamster", "mouse", "bat",
                  "otter", "beaver", "hedgehog", "raccoon", "skunk", "lemur",
                  "koala", "crocodile", "frog", "turtle", "parrot", "owl"],
    },
    "vehicle": {
        "label": "交通工具", "level": 0,
        "words": ["car", "bus", "train", "plane", "ship", "bicycle",
                  "motorcycle", "truck", "helicopter", "rocket", "boat",
                  "submarine", "tractor", "van", "taxi", "ambulance",
                  "scooter", "tram", "ferry", "yacht", "canoe", "kayak",
                  "sailboat", "tank", "jeep", "suv", "sedan", "wagon",
                  "skateboard", "segway", "monorail", "cable_car", "hovercraft",
                  "jet_ski", "airship", "glider", "parachute", "hot_air_balloon"],
    },
    "profession": {
        "label": "职业", "level": 0,
        "words": ["doctor", "nurse", "teacher", "engineer", "lawyer", "chef",
                  "artist", "musician", "writer", "scientist", "programmer",
                  "pilot", "soldier", "firefighter", "police", "farmer",
                  "baker", "architect", "carpenter", "plumber", "electrician",
                  "mechanic", "surgeon", "dentist", "pharmacist", "veterinarian",
                  "therapist", "accountant", "manager", "director", "entrepreneur",
                  "consultant", "analyst", "researcher", "professor", "librarian",
                  "journalist", "photographer", "designer", "actor", "singer",
                  "dancer", "athlete", "coach"],
    },
    "color": {
        "label": "颜色", "level": 0,
        "words": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
                  "brown", "black", "white", "gray", "navy", "beige", "coral",
                  "crimson", "maroon", "olive", "amber", "lavender", "teal",
                  "cyan", "magenta", "violet", "indigo", "turquoise", "gold",
                  "silver", "bronze", "rust", "burgundy", "salmon", "peach",
                  "mint", "sage", "khaki", "charcoal", "mauve", "lilac", "aqua"],
    },
    "emotion": {
        "label": "情感", "level": 0,
        "words": ["happy", "sad", "angry", "fear", "love", "hate", "joy",
                  "sorrow", "pride", "shame", "guilt", "envy", "hope", "despair",
                  "anxiety", "calm", "excitement", "boredom", "loneliness",
                  "gratitude", "disgust", "surprise", "awe", "nostalgia",
                  "regret", "relief", "contentment", "frustration", "confusion",
                  "determination", "courage", "embarrassment", "euphoria",
                  "melancholy", "apathy", "resentment", "admiration", "compassion",
                  "curiosity", "wonder", "serenity", "bliss", "agony", "grief"],
    },
    "food": {
        "label": "食物", "level": 0,
        "words": ["bread", "rice", "cake", "cookie", "pizza", "pasta", "soup",
                  "salad", "sandwich", "steak", "chicken", "egg", "cheese",
                  "butter", "milk", "yogurt", "ice_cream", "chocolate", "candy",
                  "waffle", "pancake", "bacon", "sausage", "ham", "turkey",
                  "tofu", "noodle", "dumpling", "sushi", "taco", "burrito",
                  "hamburger", "fries", "pretzel", "popcorn", "chips", "croissant",
                  "bagel", "muffin", "scone", "donut", "brownie", "pie", "tart",
                  "pudding", "jelly", "honey", "ketchup", "mustard", "vinegar"],
    },
}


# ==================== 内在维度估计 ====================

def estimate_intrinsic_dim_mle(X: np.ndarray, k: int = 10) -> Dict:
    """MLE内在维度估计（Levina & Bickel, 2004）

    原理：假设数据均匀分布在d维流形上，
    到第k近邻的距离的期望与d有关
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # 排除自身（距离=0）
    distances = distances[:, 1:]

    # MLE估计
    dim_estimates = []
    for i in range(n):
        for j in range(k):
            if distances[i, j] < EPS:
                continue
            ratio = distances[i, k - 1] / distances[i, j]
            if ratio > 1:
                dim_estimates.append(1.0 / (np.log(ratio) + EPS))

    if not dim_estimates:
        return {"intrinsic_dim": 0, "method": "MLE"}

    # 取中位数作为估计（更鲁棒）
    est = np.median(dim_estimates)
    return {
        "intrinsic_dim": float(est),
        "method": "MLE",
        "k": k,
        "mean_dim": float(np.mean(dim_estimates)),
        "std_dim": float(np.std(dim_estimates)),
    }


def estimate_intrinsic_dim_twonn(X: np.ndarray, k1: int = 1, k2: int = 2) -> Dict:
    """TwoNN内在维度估计（Facco et al., 2017）

    更简单的方法：使用第1和第2近邻距离比
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k2 + 1, metric='euclidean')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    r1 = distances[:, k1]
    r2 = distances[:, k2]

    # 避免除零
    valid = (r1 > EPS) & (r2 > EPS)
    r1 = r1[valid]
    r2 = r2[valid]

    if len(r1) < 10:
        return {"intrinsic_dim": 0, "method": "TwoNN"}

    mu = r2 / (r1 + EPS)
    mu = np.clip(mu, 1.0 + EPS, 1e6)

    # MLE: d = 1 / (log(mu) - 1)
    log_mu = np.log(mu)
    # 负对数似然
    d_est = (len(mu) - 2) / np.sum(log_mu - 1)

    return {
        "intrinsic_dim": float(d_est),
        "method": "TwoNN",
        "n_valid_points": int(len(r1)),
    }


def participation_ratio(X: np.ndarray) -> Dict:
    """参与比（Participation Ratio）
    
    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    等价于有效自由度
    """
    # 协方差矩阵
    cov = np.cov(X.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    pr = np.sum(eigenvalues) ** 2 / (np.sum(eigenvalues ** 2) + EPS)

    # 累积方差
    total_var = np.sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues[::-1])[::-1] / total_var

    milestones = {}
    for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
        idx = np.searchsorted(-cum_var + 1, threshold)
        if idx < len(cum_var):
            milestones[f"{int(threshold*100)}%"] = int(idx + 1)

    return {
        "participation_ratio": float(pr),
        "effective_dim": float(pr),
        "variance_milestones": milestones,
        "total_eigenvalues": len(eigenvalues),
    }


# ==================== 线性可分性检验 ====================

def linear_separability_test(X: np.ndarray, labels: np.ndarray, test_size: float = 0.3) -> Dict:
    """线性可分性检验

    对比线性SVM和RBF-SVM的性能差异
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier

    if len(set(labels)) < 2 or len(labels) < 20:
        return {"error": "insufficient data"}

    cv = StratifiedKFold(n_splits=min(5, len(set(labels))), shuffle=True, random_state=42)

    results = {}

    # 线性SVM
    try:
        linear_svc = SVC(kernel='linear', C=1.0, random_state=42)
        linear_scores = cross_val_score(linear_svc, X, labels, cv=cv, scoring='accuracy')
        results["linear_svm"] = {
            "mean_acc": float(np.mean(linear_scores)),
            "std_acc": float(np.std(linear_scores)),
        }
    except Exception as e:
        results["linear_svm"] = {"error": str(e)}

    # RBF SVM
    try:
        rbf_svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        rbf_scores = cross_val_score(rbf_svc, X, labels, cv=cv, scoring='accuracy')
        results["rbf_svm"] = {
            "mean_acc": float(np.mean(rbf_scores)),
            "std_acc": float(np.std(rbf_scores)),
        }
    except Exception as e:
        results["rbf_svm"] = {"error": str(e)}

    # LDA（线性判别）
    try:
        lda = LinearDiscriminantAnalysis()
        lda_scores = cross_val_score(lda, X, labels, cv=cv, scoring='accuracy')
        results["lda"] = {
            "mean_acc": float(np.mean(lda_scores)),
            "std_acc": float(np.std(lda_scores)),
        }
    except Exception as e:
        results["lda"] = {"error": str(e)}

    # KNN（非线性）
    try:
        knn = KNeighborsClassifier(n_neighbors=min(5, len(labels) // len(set(labels))))
        knn_scores = cross_val_score(knn, X, labels, cv=cv, scoring='accuracy')
        results["knn"] = {
            "mean_acc": float(np.mean(knn_scores)),
            "std_acc": float(np.std(knn_scores)),
        }
    except Exception as e:
        results["knn"] = {"error": str(e)}

    # 线性性判断
    linear_acc = results.get("linear_svm", {}).get("mean_acc", 0)
    rbf_acc = results.get("rbf_svm", {}).get("mean_acc", 0)
    lda_acc = results.get("lda", {}).get("mean_acc", 0)

    results["linearity_score"] = float(
        (linear_acc + lda_acc) / 2 - rbf_acc
    )
    results["is_primarily_linear"] = linear_acc >= rbf_acc - 0.05

    return results


# ==================== 局部曲率估计 ====================

def local_curvature_analysis(X: np.ndarray, n_samples: int = 50) -> Dict:
    """局部曲率估计（通过局部PCA/Hessian分析）

    在每个点的邻域内做PCA，分析局部维度和曲率
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k = min(20, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    local_dims = []
    local_eigenvalue_gaps = []

    for i in range(min(n_samples, n)):
        neighbors = X[indices[i, 1:]]  # 排除自身
        local_center = np.mean(neighbors, axis=0)
        local_data = neighbors - local_center

        # 局部PCA
        cov = np.cov(local_data.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[::-1]  # 降序

        # 有效局部维度（eigenvalue > 1% max）
        max_ev = eigenvalues[0] + EPS
        effective_dim = np.sum(eigenvalues > 0.01 * max_ev)
        local_dims.append(effective_dim)

        # 特征值间隙（衡量曲率的代理指标）
        if len(eigenvalues) > 1:
            gap = eigenvalues[0] / (eigenvalues[1] + EPS)
            local_eigenvalue_gaps.append(gap)

    return {
        "mean_local_dim": float(np.mean(local_dims)),
        "std_local_dim": float(np.std(local_dims)),
        "mean_eigenvalue_gap": float(np.mean(local_eigenvalue_gaps)) if local_eigenvalue_gaps else 0,
        "local_dim_variance": float(np.var(local_dims)),
        "is_uniform_dim": float(np.std(local_dims)) < 1.0,
    }


# ==================== 高斯混合模型拟合 ====================

def gmm_analysis(X: np.ndarray, labels: np.ndarray, n_components_range: List[int] = None) -> Dict:
    """高斯混合模型分析

    验证概念是否形成高斯簇
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    if n_components_range is None:
        n_components_range = [2, 3, 4, 5, 6, 7, 8, 10, 12]

    n_true = len(set(labels))
    results = {"true_n_categories": n_true, "models": {}}

    # 只在低维空间做GMM（避免维度灾难）
    pca = PCA(n_components=min(30, X.shape[1], X.shape[0] - 1), random_state=42)
    X_pca = pca.fit_transform(X)

    for n_comp in n_components_range:
        try:
            gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                                  random_state=42, max_iter=200)
            pred = gmm.fit_predict(X_pca)

            # BIC（越低越好）
            bic = gmm.bic(X_pca)
            aic = gmm.aic(X_pca)

            # ARI（与真实标签的一致性）
            if n_comp == n_true:
                ari = adjusted_rand_score(labels, pred)
            else:
                ari = -1

            try:
                sil = silhouette_score(X_pca, pred)
            except Exception:
                sil = -1

            results["models"][n_comp] = {
                "bic": float(bic),
                "aic": float(aic),
                "ari": float(ari) if ari >= 0 else None,
                "silhouette": float(sil),
            }
        except Exception:
            pass

    # 找BIC最优的n_components
    model_results = {k: v for k, v in results["models"].items() if v is not None}
    if model_results:
        best_n = min(model_results, key=lambda k: model_results[k]["bic"])
        results["optimal_n_components"] = best_n
        results["gmm_matches_categories"] = (best_n == n_true)

    return results


# ==================== 信息瓶颈分析 ====================

def information_bottleneck_analysis(all_activations, concepts, layer_count):
    """信息瓶颈分析：各层的信息保留vs压缩

    通过各层的可分性和方差解释率来量化信息流
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    layer_info = {}

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

        X = np.array(word_vecs)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        # 方差解释（前10个PC）
        pca = PCA(n_components=min(10, X_norm.shape[0] - 1, X_norm.shape[1]))
        pca.fit(X_norm)
        var_explained = float(pca.explained_variance_ratio_.sum())

        # 可分性（LDA）
        labels = np.array([c for c in word_cats])
        cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
        int_labels = np.array([cat_to_int[c] for c in word_cats])

        separability = 0
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_norm, int_labels)
            separability = float(lda.score(X_norm, int_labels))
        except Exception:
            pass

        # 总方差
        total_var = float(np.mean(np.var(X_norm, axis=0)))

        layer_info[f"layer_{li}"] = {
            "n_words": len(word_vecs),
            "var_explained_10pc": var_explained,
            "lda_separability": separability,
            "total_variance": total_var,
            "dim": X_norm.shape[1],
        }

    # 找信息瓶颈层（可分性最高的层）
    best_layer = max(layer_info, key=lambda k: layer_info[k]["lda_separability"]) if layer_info else None

    return {
        "per_layer": layer_info,
        "bottleneck_layer": best_layer,
        "bottleneck_separability": layer_info[best_layer]["lda_separability"] if best_layer else 0,
    }


# ==================== 残差分析（偏置空间丢失了什么） ====================

def residual_analysis(bias_matrix: np.ndarray, raw_matrix: np.ndarray,
                      word_cats: List[str]) -> Dict:
    """残差分析：偏置空间 = raw - category_mean

    残差 = raw - (bias + category_mean) = 0（理论上）
    但实际上投影可能丢失信息，分析丢失了什么
    """
    # 类别均值
    cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
    int_cats = np.array([cat_to_int[c] for c in word_cats])

    # 重建：bias + cat_mean
    cat_means = {}
    for cat in set(word_cats):
        mask = int_cats == cat_to_int[cat]
        cat_means[cat] = np.mean(bias_matrix[mask], axis=0) if np.any(mask) else np.zeros(bias_matrix.shape[1])

    reconstructed = np.zeros_like(bias_matrix)
    for i, cat in enumerate(word_cats):
        # raw = cat_mean + bias → reconstructed = cat_mean + bias
        reconstructed[i] = cat_means[cat] + bias_matrix[i]

    # 残差
    residual = raw_matrix - reconstructed

    # 残差能量
    residual_energy = float(np.mean(np.sum(residual ** 2, axis=1)))
    raw_energy = float(np.mean(np.sum(raw_matrix ** 2, axis=1)))
    relative_residual = residual_energy / (raw_energy + EPS)

    # 残差是否有结构（SVD分析）
    pca = PCA(n_components=min(5, residual.shape[0] - 1))
    pca.fit(residual)
    residual_var_explained = float(pca.explained_variance_ratio_.sum())

    # 残差与类别的相关性
    residual_cat_corr = 0
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis()
        lda.fit(residual, int_cats)
        residual_cat_corr = float(lda.score(residual, int_cats))
    except Exception:
        pass

    return {
        "residual_energy": residual_energy,
        "relative_residual": relative_residual,
        "residual_var_explained_5pc": residual_var_explained,
        "residual_category_correlation": residual_cat_corr,
        "info_loss_fraction": float(relative_residual * 100),
        "interpretation": (
            "偏置空间保留了大部分信息" if relative_residual < 0.1 else
            "偏置空间丢失了中等量信息" if relative_residual < 0.3 else
            "偏置空间丢失了大量信息！原始空间结构更丰富"
        ),
    }


# ==================== 概念轨迹分析 ====================

def concept_trajectory_analysis(X: np.ndarray, word_labels: List[str],
                                 word_cats: List[str]) -> Dict:
    """概念轨迹分析

    在概念空间中寻找"路径"和"流形结构"
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import Isomap

    cat_to_int = {c: i for i, c in enumerate(sorted(set(word_cats)))}
    int_cats = np.array([cat_to_int[c] for c in word_cats])

    results = {}

    # 1. Isomap嵌入（保持测地距离）
    try:
        iso = Isomap(n_components=2, n_neighbors=min(10, len(word_labels) - 1))
        X_iso = iso.fit_transform(X)
        results["isomap_stress"] = float(iso.reconstruction_error()) if hasattr(iso, 'reconstruction_error') else 0
    except Exception as e:
        results["isomap_error"] = str(e)

    # 2. 流形假设检验：Isomap vs PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # 两者的最近邻保持度
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X)

    # 原始空间最近邻
    _, orig_nn = nn.kneighbors(X)

    # PCA空间最近邻
    nn_pca = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_pca.fit(X_pca)
    _, pca_nn = nn_pca.kneighbors(X)

    # Isomap空间最近邻
    iso_nn = None
    if 'X_iso' in dir():
        nn_iso = NearestNeighbors(n_neighbors=5, metric='euclidean')
        nn_iso.fit(X_iso)
        _, iso_nn = nn_iso.kneighbors(X)

    # 计算最近邻保持度
    def nn_overlap(nn1, nn2):
        """最近邻集合重叠度"""
        n = nn1.shape[0]
        overlap = 0
        total = 0
        for i in range(n):
            set1 = set(nn1[i])
            set2 = set(nn2[i])
            overlap += len(set1 & set2)
            total += len(set1)
        return overlap / (total + EPS)

    results["pca_nn_preservation"] = float(nn_overlap(orig_nn, pca_nn))
    if iso_nn is not None:
        results["isomap_nn_preservation"] = float(nn_overlap(orig_nn, iso_nn))
        results["isomap_vs_pca"] = float(
            results.get("isomap_nn_preservation", 0) - results["pca_nn_preservation"]
        )

    # 3. 类内流形维度
    from sklearn.decomposition import PCA
    for cat in set(word_cats):
        mask = int_cats == cat_to_int[cat]
        if np.sum(mask) < 5:
            continue
        cat_X = X[mask]

        # 类内PCA
        cat_pca = PCA(n_components=min(10, cat_X.shape[0] - 1))
        cat_pca.fit(StandardScaler().fit_transform(cat_X))

        cum_var = np.cumsum(cat_pca.explained_variance_ratio_)
        eff_dim_90 = int(np.searchsorted(cum_var, 0.90) + 1)

        results[f"cat_{cat}_eff_dim_90"] = eff_dim_90
        results[f"cat_{cat}_var_first3"] = float(cum_var[min(2, len(cum_var)-1)])

    return results


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


def build_raw_space(all_activations, concepts, layer_count, target_layer):
    """构建原始激活空间（非偏置空间）"""
    word_vecs = []
    word_labels = []
    word_cats = []

    for cat_name, words_acts in all_activations.items():
        for word, acts in words_acts.items():
            if target_layer in acts:
                word_vecs.append(acts[target_layer])
                word_labels.append(word)
                word_cats.append(cat_name)

    return np.array(word_vecs), word_labels, word_cats


def build_bias_space(all_activations, concepts, layer_count, target_layer):
    """构建偏置空间（概念-类别均值差）"""
    word_vecs = []
    word_labels = []
    word_cats = []
    cat_means = {}

    for cat_name, words_acts in all_activations.items():
        cat_vecs = []
        for word, acts in words_acts.items():
            if target_layer in acts:
                cat_vecs.append(acts[target_layer])
        if not cat_vecs:
            continue
        cat_means[cat_name] = np.mean(cat_vecs, axis=0)

        for word, acts in words_acts.items():
            if target_layer in acts:
                bias = acts[target_layer] - cat_means[cat_name]
                word_vecs.append(bias)
                word_labels.append(word)
                word_cats.append(cat_name)

    return np.array(word_vecs), word_labels, word_cats, cat_means


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

        if len(word_vecs) < 20:
            continue

        X = np.array(word_vecs)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        svd = TruncatedSVD(n_components=min(10, len(word_vecs) - 1), random_state=42)
        comp = svd.fit_transform(X_norm)

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


# ==================== 主实验 ====================

def run_experiment(model_name: str):
    print(f"\n{'='*60}")
    print(f"Stage471: 原始激活空间分析 — {model_name}")
    print(f"{'='*60}")

    model_path = QWEN3_MODEL_PATH if model_name == "qwen3" else DEEPSEEK7B_MODEL_PATH

    # Step 1: 加载模型
    print("\n[Step 1] 加载模型...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # Step 2: 提取激活
    print("\n[Step 2] 提取概念激活...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  耗时: {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 黄金层
    print("\n[Step 3] 寻找黄金层...")
    golden_layer = find_golden_layer(all_activations, CONCEPTS, layer_count)
    print(f"  黄金层: {golden_layer}")

    # Step 4: 构建原始空间和偏置空间
    print("\n[Step 4] 构建空间...")
    raw_matrix, raw_labels, raw_cats = build_raw_space(
        all_activations, CONCEPTS, layer_count, golden_layer
    )
    bias_matrix, bias_labels, bias_cats, cat_means = build_bias_space(
        all_activations, CONCEPTS, layer_count, golden_layer
    )
    print(f"  原始空间: {raw_matrix.shape}")
    print(f"  偏置空间: {bias_matrix.shape}")

    # 标准化
    scaler_raw = StandardScaler()
    raw_normed = scaler_raw.fit_transform(raw_matrix)
    scaler_bias = StandardScaler()
    bias_normed = scaler_bias.fit_transform(bias_matrix)

    cat_to_int = {c: i for i, c in enumerate(sorted(set(raw_cats)))}
    int_labels = np.array([cat_to_int[c] for c in raw_cats])

    # ===== 核心分析 =====

    # Step 5: 内在维度估计
    print("\n[Step 5] 内在维度估计...")
    dim_mle = estimate_intrinsic_dim_mle(raw_normed, k=10)
    dim_twonn = estimate_intrinsic_dim_twonn(raw_normed)
    dim_pr = participation_ratio(raw_normed)

    print(f"  MLE内在维度: {dim_mle['intrinsic_dim']:.1f}")
    print(f"  TwoNN内在维度: {dim_twonn['intrinsic_dim']:.1f}")
    print(f"  参与比: {dim_pr['participation_ratio']:.1f}")
    print(f"  方差里程碑: {dim_pr['variance_milestones']}")

    # Step 6: 线性可分性检验（原始空间）
    print("\n[Step 6] 线性可分性检验（原始空间）...")
    separability_raw = linear_separability_test(raw_normed, int_labels)
    print(f"  线性SVM: {separability_raw.get('linear_svm', {}).get('mean_acc', 0):.3f}")
    print(f"  RBF SVM: {separability_raw.get('rbf_svm', {}).get('mean_acc', 0):.3f}")
    print(f"  LDA: {separability_raw.get('lda', {}).get('mean_acc', 0):.3f}")
    print(f"  KNN: {separability_raw.get('knn', {}).get('mean_acc', 0):.3f}")
    print(f"  线性性分数: {separability_raw.get('linearity_score', 0):.3f}")

    # Step 7: 线性可分性检验（偏置空间）
    print("\n[Step 7] 线性可分性检验（偏置空间）...")
    separability_bias = linear_separability_test(bias_normed, int_labels)
    print(f"  线性SVM: {separability_bias.get('linear_svm', {}).get('mean_acc', 0):.3f}")
    print(f"  RBF SVM: {separability_bias.get('rbf_svm', {}).get('mean_acc', 0):.3f}")

    # Step 8: 局部曲率分析
    print("\n[Step 8] 局部曲率分析...")
    curvature_raw = local_curvature_analysis(raw_normed)
    curvature_bias = local_curvature_analysis(bias_normed)
    print(f"  原始空间 平均局部维度: {curvature_raw['mean_local_dim']:.1f}")
    print(f"  偏置空间 平均局部维度: {curvature_bias['mean_local_dim']:.1f}")

    # Step 9: GMM分析
    print("\n[Step 9] 高斯混合模型分析...")
    gmm_raw = gmm_analysis(raw_normed, int_labels)
    print(f"  真实类别数: {gmm_raw['true_n_categories']}")
    print(f"  GMM最优类别数: {gmm_raw.get('optimal_n_components', 'N/A')}")

    # Step 10: 信息瓶颈分析
    print("\n[Step 10] 信息瓶颈分析（多层）...")
    ib_analysis = information_bottleneck_analysis(all_activations, CONCEPTS, layer_count)
    print(f"  瓶颈层: {ib_analysis['bottleneck_layer']}")
    print(f"  瓶颈层可分性: {ib_analysis['bottleneck_separability']:.3f}")

    # Step 11: 残差分析
    print("\n[Step 11] 残差分析（偏置空间丢失了什么）...")
    residual = residual_analysis(bias_matrix, raw_matrix, raw_cats)
    print(f"  相对残差: {residual['relative_residual']:.4f}")
    print(f"  信息丢失: {residual['info_loss_fraction']:.1f}%")
    print(f"  解释: {residual['interpretation']}")

    # Step 12: 概念轨迹分析
    print("\n[Step 12] 概念轨迹分析...")
    trajectory = concept_trajectory_analysis(raw_normed, raw_labels, raw_cats)
    print(f"  PCA最近邻保持度: {trajectory.get('pca_nn_preservation', 0):.3f}")
    print(f"  Isomap最近邻保持度: {trajectory.get('isomap_nn_preservation', 0):.3f}")

    # Step 13: 曲率检验（原始空间 vs 偏置空间）
    print("\n[Step 13] 空间曲率对比...")
    D_raw = squareform(pdist(raw_normed, 'euclidean'))
    D_bias = squareform(pdist(bias_normed, 'euclidean'))

    # 原始空间曲率
    rng = np.random.RandomState(42)
    raw_deficits = []
    bias_deficits = []
    for _ in range(500):
        idx = rng.choice(len(raw_normed), 3, replace=False)
        i, j, k = idx

        for D, deficits_list in [(D_raw, raw_deficits), (D_bias, bias_deficits)]:
            a, b, c = D[j, k], D[i, k], D[i, j]
            if a < EPS or b < EPS or c < EPS:
                continue
            cos_A = np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1)
            cos_B = np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1)
            cos_C = np.clip((a**2 + b**2 - c**2) / (2*a*b), -1, 1)
            angle_sum = np.arccos(cos_A) + np.arccos(cos_B) + np.arccos(cos_C)
            deficits_list.append(np.pi - angle_sum)

    raw_deficits = np.array(raw_deficits)
    bias_deficits = np.array(bias_deficits)

    curvature_comparison = {
        "raw_space": {
            "mean_deficit": float(np.mean(raw_deficits)),
            "pct_hyperbolic": float(np.mean(raw_deficits > 0) * 100),
        },
        "bias_space": {
            "mean_deficit": float(np.mean(bias_deficits)),
            "pct_hyperbolic": float(np.mean(bias_deficits > 0) * 100),
        },
    }

    # Step 14: 结果汇总
    print("\n[Step 14] 结果汇总...")

    all_results = {
        "model": model_name,
        "timestamp": TIMESTAMP,
        "golden_layer": int(golden_layer),
        "n_concepts": len(raw_labels),
        "n_categories": len(set(raw_cats)),
        "neuron_dim": int(neuron_dim),
        "intrinsic_dimension": {
            "mle": dim_mle,
            "twonn": dim_twonn,
            "participation_ratio": dim_pr,
        },
        "separability": {
            "raw_space": separability_raw,
            "bias_space": separability_bias,
        },
        "local_curvature": {
            "raw_space": curvature_raw,
            "bias_space": curvature_bias,
        },
        "gmm": gmm_raw,
        "information_bottleneck": sanitize_for_json(ib_analysis),
        "residual_analysis": residual,
        "trajectory_analysis": sanitize_for_json(trajectory),
        "curvature_comparison": curvature_comparison,
        "conclusion": {
            "intrinsic_dim": dim_mle["intrinsic_dim"],
            "is_low_dimensional": dim_mle["intrinsic_dim"] < neuron_dim * 0.3,
            "is_primarily_linear": separability_raw.get("is_primarily_linear", False),
            "space_is_curved": curvature_comparison["raw_space"]["mean_deficit"] > 0.01,
            "bias_loses_info": residual["relative_residual"] > 0.1,
            "raw_more_informative_than_bias": (
                separability_raw.get("linear_svm", {}).get("mean_acc", 0) >
                separability_bias.get("linear_svm", {}).get("mean_acc", 0)
            ),
            "overall_verdict": self_describe(
                dim_mle, separability_raw, curvature_raw, residual, curvature_comparison
            ),
        },
    }

    # 打印关键结果
    print(f"\n{'='*60}")
    print(f"关键发现:")
    print(f"  内在维度: {dim_mle['intrinsic_dim']:.1f} (外维: {neuron_dim})")
    print(f"  压缩比: {neuron_dim / (dim_mle['intrinsic_dim'] + 1):.1f}x")
    print(f"  线性可分性(原始): 线性SVM={separability_raw.get('linear_svm',{}).get('mean_acc',0):.3f}, "
          f"RBF={separability_raw.get('rbf_svm',{}).get('mean_acc',0):.3f}")
    print(f"  线性可分性(偏置): 线性SVM={separability_bias.get('linear_svm',{}).get('mean_acc',0):.3f}, "
          f"RBF={separability_bias.get('rbf_svm',{}).get('mean_acc',0):.3f}")
    print(f"  空间曲率:")
    print(f"    原始空间: 角亏={curvature_comparison['raw_space']['mean_deficit']:.4f}, "
          f"双曲%={curvature_comparison['raw_space']['pct_hyperbolic']:.1f}%")
    print(f"    偏置空间: 角亏={curvature_comparison['bias_space']['mean_deficit']:.4f}, "
          f"双曲%={curvature_comparison['bias_space']['pct_hyperbolic']:.1f}%")
    print(f"  残差分析: 偏置空间丢失{residual['info_loss_fraction']:.1f}%信息")
    print(f"  GMM最优类别数: {gmm_raw.get('optimal_n_components', 'N/A')} (真实: {gmm_raw['true_n_categories']})")
    print(f"  最终判定: {all_results['conclusion']['overall_verdict']}")
    print(f"{'='*60}")

    # 保存结果
    output_path = OUTPUT_DIR / f"stage471_results_{model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {output_path}")

    return all_results


def self_describe(dim_mle, sep_raw, curv_raw, residual, curv_comp):
    """生成总体判定"""
    parts = []
    if dim_mle["intrinsic_dim"] < 100:
        parts.append(f"低维流形(d={dim_mle['intrinsic_dim']:.0f})")
    if sep_raw.get("is_primarily_linear", False):
        parts.append("近似线性")
    elif sep_raw.get("linearity_score", 0) < -0.05:
        parts.append("非线性结构")
    if curv_comp["raw_space"]["mean_deficit"] > 0.01:
        parts.append("负曲率(双曲)")
    elif curv_comp["raw_space"]["mean_deficit"] < -0.01:
        parts.append("正曲率(球面)")
    else:
        parts.append("近零曲率(欧氏)")
    if residual["relative_residual"] > 0.1:
        parts.append("偏置空间丢失信息")
    return " | ".join(parts) if parts else "无明显特征"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python stage471_raw_activation.py [qwen3|deepseek]")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ("qwen3", "deepseek"):
        print("错误: 请选择 qwen3 或 deepseek")
        sys.exit(1)

    run_experiment(model_name)
