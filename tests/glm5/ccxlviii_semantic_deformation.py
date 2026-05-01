"""
CCXLVIII(348): 变形单纯形的语义结构分析
==============================================
★★★★★ CCXLVII核心发现:
  1. fit_r2无统计意义(高维平凡)
  2. edge_cv高于随机 → 有语义结构, 但不均匀
  3. 面内方向有语义规律(happy→径向, sad→angry等)

★★★★★ 本实验核心问题:
  edge_cv=0.07-0.11说明边长不均匀, 但:
  Q1: 哪些类别对最近/最远?
  Q2: 这种不均匀有语义含义吗? (如: 负面情绪更近?)
  Q3: 跨模型/跨领域是否一致?

★★★★★ 方法论:
  实验1: Pairwise距离矩阵分析
    - 计算所有类别对的距离
    - 归一化(除以均值)
    - 排序, 识别最近/最远对
    - 与正则单纯形基线比较(所有距离=1.0)
  
  实验2: 语义相似性预测
    - 情感环形模型(circumplex): valence × arousal
      happy = (+val, +aro), sad = (-val, -aro)
      angry = (-val, +aro), scared = (-val, ±aro)
    - 假设: 同valence的情感更近
    - 测试: 负面情绪间距离 < 正面-负面间距离
  
  实验3: 跨领域验证
    - 动物类: mammal, bird, fish, insect
    - 颜色类: red, blue, green, yellow
    - 检查: 变形模式是否领域特异还是通用?

  实验4: 变形向量分析
    - 变形 = actual_simplex - regular_simplex
    - PCA分析变形向量 → 变形的主要方向
    - 这些方向是否有语义解释?

用法:
  python ccxlviii_semantic_deformation.py --model qwen3
  python ccxlviii_semantic_deformation.py --model glm4
  python ccxlviii_semantic_deformation.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxlviii_semantic_deformation_log.txt"

# ============================================================
# 语义类别定义
# ============================================================

EMOTION_4 = {
    "classes": {
        "happy": ["joy", "delight", "bliss", "glee", "cheer", "elation",
                  "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
        "sad":   ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                  "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
        "angry": ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                  "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
        "scared":["fear", "terror", "dread", "panic", "fright", "horror",
                  "anxiety", "apprehension", "trepidation", "phobia", "alarm", "consternation"],
    },
    "order": ["happy", "sad", "angry", "scared"],
    "prompt": "The person felt {word} about the",
    "domain": "emotion",
    # 情感环形模型: (valence, arousal)
    # happy=+,+ ; sad=-,- ; angry=-,+ ; scared=-,0
    "semantics": {
        "happy":  {"valence": +1, "arousal": +1},
        "sad":    {"valence": -1, "arousal": -1},
        "angry":  {"valence": -1, "arousal": +1},
        "scared": {"valence": -1, "arousal":  0},
    },
}

EMOTION_6 = {
    "classes": {
        "happy":   ["joy", "delight", "bliss", "glee", "cheer", "elation",
                    "contentment", "pleasure", "gladness", "merriment", "euphoria", "jubilation"],
        "sad":     ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning", "dejection", "despondency"],
        "angry":   ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                    "indignation", "animosity", "vexation", "exasperation", "irritation", "anger"],
        "scared":  ["fear", "terror", "dread", "panic", "fright", "horror",
                    "anxiety", "apprehension", "trepidation", "phobia", "alarm", "consternation"],
        "surprise":["astonishment", "amazement", "wonder", "shock", "stunned", "staggered",
                    "bewilderment", "awe", "disbelief", "startlement", "astonished", "dumbfounded"],
        "disgust": ["revulsion", "repugnance", "loathing", "abhorrence", "nausea", "aversion",
                    "distaste", "repulsion", "contempt", "dislike", "antipathy", "abomination"],
    },
    "order": ["happy", "sad", "angry", "scared", "surprise", "disgust"],
    "prompt": "The person felt {word} about the",
    "domain": "emotion",
    "semantics": {
        "happy":   {"valence": +1, "arousal": +1},
        "sad":     {"valence": -1, "arousal": -1},
        "angry":   {"valence": -1, "arousal": +1},
        "scared":  {"valence": -1, "arousal":  0},
        "surprise":{"valence":  0, "arousal": +1},
        "disgust": {"valence": -1, "arousal":  0},
    },
}

ANIMAL_4 = {
    "classes": {
        "mammal": ["dog", "cat", "horse", "elephant", "whale", "lion",
                   "tiger", "bear", "wolf", "deer", "rabbit", "monkey"],
        "bird":   ["eagle", "hawk", "sparrow", "robin", "owl", "parrot",
                   "crow", "pigeon", "falcon", "stork", "heron", "swan"],
        "fish":   ["salmon", "trout", "shark", "tuna", "bass", "cod",
                   "carp", "perch", "eel", "herring", "pike", "flounder"],
        "insect": ["ant", "bee", "butterfly", "spider", "fly", "mosquito",
                   "beetle", "wasp", "moth", "grasshopper", "cricket", "ladybug"],
    },
    "order": ["mammal", "bird", "fish", "insect"],
    "prompt": "The {word} was in the",
    "domain": "animal",
    # 语义维度: (warm_blooded, aquatic)
    # mammal=+,0 ; bird=+,0 ; fish=-,+ ; insect=-,0
    "semantics": {
        "mammal": {"warm_blooded": +1, "aquatic":  0},
        "bird":   {"warm_blooded": +1, "aquatic":  0},
        "fish":   {"warm_blooded": -1, "aquatic": +1},
        "insect": {"warm_blooded": -1, "aquatic":  0},
    },
}

COLOR_4 = {
    "classes": {
        "red":   ["crimson", "scarlet", "maroon", "ruby", "cherry", "vermilion",
                  "garnet", "burgundy", "cerise", "carmine", "rust", "rose"],
        "blue":  ["azure", "cobalt", "navy", "sapphire", "indigo", "cerulean",
                  "turquoise", "cyan", "teal", "aquamarine", "periwinkle", "lapis"],
        "green": ["emerald", "jade", "olive", "mint", "lime", "forest",
                  "sage", "moss", "chartreuse", "verdant", "viridian", "celadon"],
        "yellow":["golden", "amber", "lemon", "mustard", "saffron", "canary",
                  "honey", "sunflower", "gold", "ochre", "maize", "buff"],
    },
    "order": ["red", "blue", "green", "yellow"],
    "prompt": "The color was {word} and",
    "domain": "color",
    # 语义维度: (warm, light)
    # red=+,- ; blue=-,- ; green=-,+ ; yellow=+,+
    "semantics": {
        "red":    {"warm": +1, "light": -1},
        "blue":   {"warm": -1, "light": -1},
        "green":  {"warm": -1, "light": +1},
        "yellow": {"warm": +1, "light": +1},
    },
}

CATEGORY_SETS = {
    "emotion4": EMOTION_4,
    "emotion6": EMOTION_6,
    "animal4": ANIMAL_4,
    "color4": COLOR_4,
}

# ============================================================
# 核心算法
# ============================================================

def construct_regular_simplex(N, scale=1.0):
    """构造N个点在R^{N-1}中的正则单纯形"""
    if N <= 1:
        return np.array([[0.0]])
    if N == 2:
        v = np.array([[-1.0], [1.0]]) * scale
        return v - np.mean(v, axis=0)
    D = N - 1
    r = 1.0
    G = np.full((N, N), -r**2 / N)
    np.fill_diagonal(G, r**2 * (N - 1) / N)
    L = np.linalg.cholesky(G)
    vertices = L[:, :D]
    vertices = vertices - np.mean(vertices, axis=0)
    current_scale = np.linalg.norm(vertices[0])
    if current_scale > 1e-10:
        vertices = vertices * scale / current_scale
    return vertices


def collect_class_centers(model, tokenizer, device, cat_set, layer_idx, n_words=8):
    """
    收集指定层的类别中心
    
    Returns:
        dict: {class_name: center_vector [d_model]}
    """
    layers = get_layers(model)
    d_model = model.get_input_embeddings().weight.shape[1]
    
    centers = {}
    for cls_name in cat_set["order"]:
        words = cat_set["classes"][cls_name][:n_words]
        residuals = []
        
        for word in words:
            prompt = cat_set["prompt"].format(word=word)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            # Hook收集指定层输出
            captured = {}
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            h = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids)
                except Exception as e:
                    print(f"    Forward failed for '{word}': {e}")
            
            h.remove()
            
            if f"L{layer_idx}" in captured:
                # 取最后一个token的残差
                res = captured[f"L{layer_idx}"][0, -1, :].numpy()
                residuals.append(res)
        
        if len(residuals) > 0:
            centers[cls_name] = np.mean(residuals, axis=0)
    
    return centers


def compute_pairwise_distance_analysis(centers, class_order):
    """
    实验1: Pairwise距离矩阵分析
    
    Returns:
        dict with distance_matrix, normalized_distances, rankings, etc.
    """
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    
    # 去均值
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    # SVD投影到N-1维
    U, S, Vt = svd(centered, full_matrices=False)
    D = N - 1
    proj = centered @ Vt[:D].T  # [N, D]
    
    # 计算所有pairwise距离
    dists = pdist(proj)
    dist_matrix = squareform(dists)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    
    # 归一化距离(除以均值)
    norm_dists = dists / mean_dist
    norm_matrix = dist_matrix / mean_dist
    
    # 排序(从小到大)
    pair_labels = []
    for i in range(N):
        for j in range(i+1, N):
            pair_labels.append((class_order[i], class_order[j]))
    
    sorted_indices = np.argsort(dists)
    sorted_pairs = [(pair_labels[idx], float(dists[idx]), float(norm_dists[idx])) 
                    for idx in sorted_indices]
    
    # edge_cv
    cv = std_dist / (mean_dist + 1e-10)
    
    # 等距比
    iso_ratio = np.min(dists) / (np.max(dists) + 1e-10)
    
    return {
        "distance_matrix": dist_matrix,
        "normalized_matrix": norm_matrix,
        "mean_dist": float(mean_dist),
        "std_dist": float(std_dist),
        "edge_cv": float(cv),
        "isoperimetric_ratio": float(iso_ratio),
        "sorted_pairs": sorted_pairs,
        "closest_pair": sorted_pairs[0],
        "farthest_pair": sorted_pairs[-1],
        "proj_centers": proj,
        "singular_values": S[:D].tolist(),
    }


def compute_semantic_prediction(cat_set, class_order):
    """
    实验2: 基于语义相似性的距离预测
    
    使用valence-arousal等语义维度预测哪些类别对应该更近
    
    Returns:
        dict with predicted_ranking, semantic_distances
    """
    semantics = cat_set.get("semantics", {})
    if not semantics:
        return None
    
    N = len(class_order)
    
    # 获取语义维度
    sem_keys = list(semantics[class_order[0]].keys())
    
    # 计算语义距离(欧氏距离, 在语义空间中)
    sem_dists = {}
    for i in range(N):
        for j in range(i+1, N):
            ci, cj = class_order[i], class_order[j]
            vi = np.array([semantics[ci][k] for k in sem_keys], dtype=float)
            vj = np.array([semantics[cj][k] for k in sem_keys], dtype=float)
            sem_dists[(ci, cj)] = float(np.linalg.norm(vi - vj))
    
    # 排序
    sorted_sem = sorted(sem_dists.items(), key=lambda x: x[1])
    
    # 特别分析: valence同号 vs 异号的距离差异
    valence_same = []
    valence_diff = []
    for (ci, cj), d in sem_dists.items():
        vi = semantics[ci].get("valence", 0)
        vj = semantics[cj].get("valence", 0)
        if vi * vj > 0:  # 同号
            valence_same.append((ci, cj, d))
        elif vi * vj < 0:  # 异号
            valence_diff.append((ci, cj, d))
    
    return {
        "semantic_dimensions": sem_keys,
        "semantic_distances": sem_dists,
        "predicted_ranking": sorted_sem,
        "valence_same_pairs": valence_same,
        "valence_diff_pairs": valence_diff,
        "valence_same_mean": float(np.mean([d for _, _, d in valence_same])) if valence_same else None,
        "valence_diff_mean": float(np.mean([d for _, _, d in valence_diff])) if valence_diff else None,
    }


def compute_geometric_semantic_correlation(geo_result, sem_result, class_order):
    """
    实验2续: 几何距离与语义距离的相关性
    
    Returns:
        dict with pearson_r, spearman_r, p_values
    """
    if sem_result is None:
        return None
    
    N = len(class_order)
    
    # 构造距离向量
    geo_dists = []
    sem_dists = []
    for i in range(N):
        for j in range(i+1, N):
            ci, cj = class_order[i], class_order[j]
            geo_dists.append(geo_result["normalized_matrix"][i, j])
            key = (ci, cj) if (ci, cj) in sem_result["semantic_distances"] else (cj, ci)
            sem_dists.append(sem_result["semantic_distances"].get(key, 0))
    
    geo_dists = np.array(geo_dists)
    sem_dists = np.array(sem_dists)
    
    if len(geo_dists) < 3:
        return None
    
    pr, pp = pearsonr(geo_dists, sem_dists)
    sr, sp = spearmanr(geo_dists, sem_dists)
    
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n_pairs": len(geo_dists),
        "geo_dists": geo_dists.tolist(),
        "sem_dists": sem_dists.tolist(),
    }


def compute_deformation_analysis(centers, class_order):
    """
    实验4: 变形向量分析
    
    计算 actual_simplex - regular_simplex, 分析变形的主要方向
    
    Returns:
        dict with deformation_vectors, pca_of_deformation, etc.
    """
    N = len(class_order)
    D = N - 1
    
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    # SVD投影
    U, S, Vt = svd(centered, full_matrices=False)
    proj = centered @ Vt[:D].T  # [N, D]
    
    # 正则单纯形(同尺度)
    data_scale = np.linalg.norm(proj[0])
    regular = construct_regular_simplex(N, scale=data_scale)
    
    # Procrustes对齐
    R, scale_p = orthogonal_procrustes(proj, regular)
    aligned = proj @ R
    
    # 变形向量
    deformation = aligned - regular  # [N, D]
    
    # 变形的Frobenius范数
    deform_norm = np.linalg.norm(deformation)
    regular_norm = np.linalg.norm(regular)
    deform_ratio = deform_norm / (regular_norm + 1e-10)
    
    # PCA分析变形
    # 去均值
    deform_centered = deformation - np.mean(deformation, axis=0)
    U_d, S_d, Vt_d = svd(deform_centered, full_matrices=False)
    
    # 每个顶点的变形大小
    vertex_deform_norms = [float(np.linalg.norm(deformation[i])) for i in range(N)]
    
    # 变形方向的主成分
    # Vt_d[0] 是第一主成分方向
    # 各顶点在这个方向上的投影
    proj_on_pc1 = deformation @ Vt_d[0]
    
    # 检查PC1是否与语义维度对齐
    # (在后续分析中与valence/arousal等比较)
    
    return {
        "deformation": deformation,
        "deformation_norm": float(deform_norm),
        "regular_norm": float(regular_norm),
        "deformation_ratio": float(deform_ratio),
        "vertex_deform_norms": vertex_deform_norms,
        "deformation_singular_values": S_d.tolist(),
        "deformation_pc1": Vt_d[0].tolist(),
        "proj_on_pc1": proj_on_pc1.tolist(),
        "aligned_vertices": aligned,
        "regular_vertices": regular,
    }


def compute_valence_clustering_test(centers, class_order, cat_set):
    """
    实验2核心: 正面情绪vs负面情绪的聚类测试
    
    假设: 负面情绪(sad, angry, scared)比正面情绪(happy)更近
    
    Returns:
        dict with intra_negative_dist, positive_negative_dist, ratio
    """
    semantics = cat_set.get("semantics", {})
    if not semantics:
        return None
    
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    U, S, Vt = svd(centered, full_matrices=False)
    D = N - 1
    proj = centered @ Vt[:D].T
    
    dist_matrix = squareform(pdist(proj))
    
    # 识别正面/负面类别
    positive = [c for c in class_order if semantics.get(c, {}).get("valence", 0) > 0]
    negative = [c for c in class_order if semantics.get(c, {}).get("valence", 0) < 0]
    
    if len(positive) < 1 or len(negative) < 2:
        return None
    
    # 负面情绪内部距离
    neg_dists = []
    for i, ci in enumerate(class_order):
        for j, cj in enumerate(class_order):
            if i >= j:
                continue
            if ci in negative and cj in negative:
                neg_dists.append(dist_matrix[i, j])
    
    # 正面-负面距离
    pos_neg_dists = []
    for i, ci in enumerate(class_order):
        for j, cj in enumerate(class_order):
            if i >= j:
                continue
            if (ci in positive and cj in negative) or (ci in negative and cj in positive):
                pos_neg_dists.append(dist_matrix[i, j])
    
    neg_mean = np.mean(neg_dists) if neg_dists else 0
    pos_neg_mean = np.mean(pos_neg_dists) if pos_neg_dists else 0
    
    ratio = neg_mean / (pos_neg_mean + 1e-10) if pos_neg_mean > 0 else 0
    
    return {
        "positive_classes": positive,
        "negative_classes": negative,
        "intra_negative_dists": [float(d) for d in neg_dists],
        "positive_negative_dists": [float(d) for d in pos_neg_dists],
        "intra_negative_mean": float(neg_mean),
        "positive_negative_mean": float(pos_neg_mean),
        "ratio": float(ratio),
        "prediction": "negative_closer" if ratio < 1.0 else "negative_farther",
    }


def run_experiment(model_name, cat_names=None):
    """运行完整实验"""
    if cat_names is None:
        cat_names = ["emotion4", "emotion6", "animal4", "color4"]
    
    print(f"\n{'='*70}")
    print(f"CCXLVIII: 变形单纯形的语义结构分析 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print(f"  模型: {model_info.model_class}, {n_layers}层, d={d_model}")
    
    # 选择最佳层(基于CCXLVII经验: 中高层语义最强)
    # Qwen3(36L)->L25, GLM4(40L)->L36, DS7B(30L)->L2
    best_layers = {
        "qwen3": min(25, n_layers - 1),
        "glm4": min(36, n_layers - 1),
        "deepseek7b": min(2, n_layers - 1),
    }
    best_layer = best_layers.get(model_name, n_layers * 2 // 3)
    
    # 也测试多层
    sample_layers = sorted(set([
        0, n_layers // 4, n_layers // 2, best_layer, n_layers - 1
    ]))
    
    all_results = {}
    
    for cat_name in cat_names:
        cat_set = CATEGORY_SETS[cat_name]
        domain = cat_set["domain"]
        class_order = cat_set["order"]
        N = len(class_order)
        
        print(f"\n--- {cat_name} (N={N}, domain={domain}) ---")
        
        # 在最佳层收集中心
        print(f"  收集L{best_layer}的类别中心...")
        centers = collect_class_centers(model, tokenizer, device, cat_set, best_layer, n_words=8)
        
        if len(centers) < N:
            print(f"  ✗ 只收集到{len(centers)}/{N}个类别, 跳过")
            continue
        
        # 实验1: Pairwise距离分析
        print(f"  实验1: Pairwise距离矩阵分析...")
        geo_result = compute_pairwise_distance_analysis(centers, class_order)
        
        print(f"    edge_cv = {geo_result['edge_cv']:.4f}")
        print(f"    iso_ratio = {geo_result['isoperimetric_ratio']:.4f}")
        print(f"    最近对: {geo_result['closest_pair'][0]} = {geo_result['closest_pair'][2]:.3f}×均值")
        print(f"    最远对: {geo_result['farthest_pair'][0]} = {geo_result['farthest_pair'][2]:.3f}×均值")
        
        print(f"    距离排序(归一化):")
        for (pair, dist, norm_dist) in geo_result['sorted_pairs']:
            print(f"      {pair[0]:>8}-{pair[1]:<8}: {norm_dist:.3f}×")
        
        # 实验2: 语义相似性预测
        print(f"  实验2: 语义相似性预测...")
        sem_result = compute_semantic_prediction(cat_set, class_order)
        
        if sem_result:
            print(f"    语义维度: {sem_result['semantic_dimensions']}")
            print(f"    语义距离排序:")
            for (pair, d) in sem_result['predicted_ranking']:
                print(f"      {pair[0]:>8}-{pair[1]:<8}: sem_dist={d:.2f}")
            
            # 几何-语义相关性
            corr_result = compute_geometric_semantic_correlation(geo_result, sem_result, class_order)
            if corr_result:
                print(f"    几何-语义相关性:")
                print(f"      Pearson  r={corr_result['pearson_r']:.3f} p={corr_result['pearson_p']:.3f}")
                print(f"      Spearman r={corr_result['spearman_r']:.3f} p={corr_result['spearman_p']:.3f}")
        
        # 实验2续: Valence聚类测试
        valence_result = compute_valence_clustering_test(centers, class_order, cat_set)
        if valence_result:
            print(f"  Valence聚类测试:")
            print(f"    正面类: {valence_result['positive_classes']}")
            print(f"    负面类: {valence_result['negative_classes']}")
            print(f"    负面内部距离: {valence_result['intra_negative_mean']:.3f}")
            print(f"    正面-负面距离: {valence_result['positive_negative_mean']:.3f}")
            print(f"    比值: {valence_result['ratio']:.3f}")
            print(f"    结论: {valence_result['prediction']}")
        
        # 实验4: 变形向量分析
        print(f"  实验4: 变形向量分析...")
        deform_result = compute_deformation_analysis(centers, class_order)
        
        print(f"    变形比: {deform_result['deformation_ratio']:.3f}")
        print(f"    各顶点变形量:")
        for i, cls in enumerate(class_order):
            print(f"      {cls:>8}: {deform_result['vertex_deform_norms'][i]:.4f}")
        n_deform_sv = min(N-1, len(deform_result['deformation_singular_values']))
        print(f"    变形奇异值: {[f'{s:.3f}' for s in deform_result['deformation_singular_values'][:n_deform_sv]]}")
        print(f"    PC1投影: {[f'{p:.3f}' for p in deform_result['proj_on_pc1']]}")
        
        # 存储结果
        cat_results = {
            "geo": geo_result,
            "sem": sem_result,
            "valence": valence_result,
            "deform": deform_result,
            "corr": corr_result if sem_result else None,
            "layer": best_layer,
        }
        
        # 去掉numpy数组和tuple keys(不可JSON序列化)
        def make_serializable(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                new_dict = {}
                for kk, vv in obj.items():
                    # 把tuple key转为字符串
                    key = f"{kk[0]}-{kk[1]}" if isinstance(kk, tuple) else str(kk)
                    new_dict[key] = make_serializable(vv)
                return new_dict
            if isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            return obj
        
        cat_results_save = make_serializable(cat_results)
        
        all_results[cat_name] = cat_results_save
    
    # 跨类别比较
    print(f"\n{'='*70}")
    print(f"跨类别比较")
    print(f"{'='*70}")
    
    cross_comparison = {}
    for cat_name, res in all_results.items():
        if res.get("geo"):
            cross_comparison[cat_name] = {
                "edge_cv": res["geo"]["edge_cv"],
                "iso_ratio": res["geo"]["isoperimetric_ratio"],
                "closest": res["geo"]["closest_pair"],
                "farthest": res["geo"]["farthest_pair"],
                "deform_ratio": res["deform"]["deformation_ratio"] if res.get("deform") else None,
                "valence_ratio": res["valence"]["ratio"] if res.get("valence") else None,
                "pearson_r": res["corr"]["pearson_r"] if res.get("corr") else None,
                "spearman_r": res["corr"]["spearman_r"] if res.get("corr") else None,
            }
            cc = cross_comparison[cat_name]
            print(f"  {cat_name}:")
            print(f"    edge_cv={cc['edge_cv']:.4f}, iso_ratio={cc['iso_ratio']:.3f}")
            print(f"    最近={cc['closest'][0]}, 最远={cc['farthest'][0]}")
            if cc['deform_ratio'] is not None:
                print(f"    变形比={cc['deform_ratio']:.3f}")
            if cc['valence_ratio'] is not None:
                print(f"    valence聚类比={cc['valence_ratio']:.3f}")
            if cc['pearson_r'] is not None:
                print(f"    几何-语义: Pearson={cc['pearson_r']:.3f}, Spearman={cc['spearman_r']:.3f}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    output_path = TEMP / f"ccxlviii_semantic_{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "best_layer": best_layer,
            "results": all_results,
            "cross_comparison": cross_comparison,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_path}")
    
    return all_results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    results = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
