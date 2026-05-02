"""
CCIV(354): 子空间对齐分析 — Embedding空间与残差空间的轴对齐
==================================================================
★★★★★ CCIII核心发现:
  - Embedding距离主导71%, 但vehicle领域β_emb为负
  - profession领域β_sem为负, 手定义语义维度错配
  - 86%方差无法解释, 需要理解两个空间的轴结构

★★★★★ 本实验目标:
  1. 直接比较embedding空间和残差空间的主成分方向
  2. 计算主角度(principal angles)量化子空间对齐
  3. PC-PC相关矩阵: 哪些emb PCs对应哪些res PCs
  4. 方差传递: emb子空间解释了多少res方差
  5. 解释vehicle的β_emb负: 轴交叉/反转
  6. 层间演变: 对齐如何随深度变化

★★★★★ 核心方法:
  - PCA: 获取两个空间的PC方向和得分
  - 主角度: SVD(V_E^T V_R) → cos(θ_k) = σ_k
  - PC相关矩阵: corr(emb_score_i, res_score_j)
  - 方差传递: ||P_{emb} res_PC_j||^2 / ||res_PC_j||^2

用法:
  python cciv_subspace_alignment.py --model qwen3
  python cciv_subspace_alignment.py --model glm4
  python cciv_subspace_alignment.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 领域定义 — 与CCIII完全相同
# ============================================================

DOMAINS = {
    "animal10": {
        "categories": {
            "dog":      ["dog", "puppy", "hound", "canine", "pooch", "mutt"],
            "cat":      ["cat", "kitten", "feline", "tomcat", "pussy", "kitty"],
            "wolf":     ["wolf", "werewolf", "lupine", "coyote", "jackal", "husky"],
            "lion":     ["lion", "tiger", "leopard", "cheetah", "panther", "cougar"],
            "bird":     ["bird", "sparrow", "robin", "finch", "wren", "swallow"],
            "eagle":    ["eagle", "hawk", "falcon", "vulture", "osprey", "condor"],
            "fish":     ["fish", "trout", "salmon", "bass", "perch", "cod"],
            "shark":    ["shark", "whale", "dolphin", "porpoise", "orca", "narwhal"],
            "snake":    ["snake", "serpent", "viper", "cobra", "python", "adder"],
            "lizard":   ["lizard", "gecko", "iguana", "chameleon", "salamander", "newt"],
        },
        "semantic_dims": {
            "dog":    [3, 1.0, 0.5, 3],
            "cat":    [2, 1.0, 0.7, 4],
            "wolf":   [4, 0.0, 1.0, 5],
            "lion":   [5, 0.0, 1.0, 4],
            "bird":   [1, 0.0, 0.1, 3],
            "eagle":  [3, 0.0, 1.0, 5],
            "fish":   [2, 0.0, 0.3, 3],
            "shark":  [5, 0.0, 1.0, 5],
            "snake":  [2, 0.0, 0.8, 2],
            "lizard": [1, 0.0, 0.3, 3],
        },
        "dim_names": ["size", "domestic", "predator", "speed"],
    },
    "emotion10": {
        "categories": {
            "happy":    ["happy", "joyful", "cheerful", "glad", "pleased", "delighted"],
            "sad":      ["sad", "sorrowful", "unhappy", "gloomy", "miserable", "depressed"],
            "angry":    ["angry", "furious", "enraged", "irate", "hostile", "livid"],
            "scared":   ["scared", "afraid", "fearful", "terrified", "anxious", "frightened"],
            "calm":     ["calm", "peaceful", "serene", "tranquil", "relaxed", "composed"],
            "excited":  ["excited", "thrilled", "elated", "eager", "enthusiastic", "energetic"],
            "proud":    ["proud", "honored", "dignified", "triumphant", "boastful", "arrogant"],
            "ashamed":  ["ashamed", "embarrassed", "guilty", "humiliated", "remorseful", "contrite"],
            "surprised":["surprised", "amazed", "astonished", "shocked", "stunned", "startled"],
            "disgusted":["disgusted", "revolted", "repulsed", "nauseated", "appalled", "sickened"],
        },
        "semantic_dims": {
            "happy":     [ 0.81,  0.51,  0.66],
            "sad":       [-0.63, -0.27, -0.33],
            "angry":     [-0.64,  0.83,  0.60],
            "scared":    [-0.63,  0.83, -0.36],
            "calm":      [ 0.56, -0.49,  0.32],
            "excited":   [ 0.68,  0.83,  0.40],
            "proud":     [ 0.68,  0.50,  0.78],
            "ashamed":   [-0.61,  0.20, -0.50],
            "surprised": [ 0.32,  0.83, -0.15],
            "disgusted": [-0.60,  0.35,  0.12],
        },
        "dim_names": ["valence", "arousal", "dominance"],
    },
    "profession10": {
        "categories": {
            "doctor":   ["doctor", "physician", "surgeon", "medic", "clinician", "healer"],
            "nurse":    ["nurse", "caregiver", "medic", "attendant", "practitioner", "orderly"],
            "teacher":  ["teacher", "instructor", "educator", "professor", "tutor", "lecturer"],
            "student":  ["student", "pupil", "scholar", "learner", "undergraduate", "trainee"],
            "chef":     ["chef", "cook", "culinary", "baker", "caterer", "pastry"],
            "waiter":   ["waiter", "server", "bartender", "barista", "attendant", "host"],
            "artist":   ["artist", "painter", "sculptor", "illustrator", "designer", "creator"],
            "musician": ["musician", "singer", "guitarist", "pianist", "drummer", "vocalist"],
            "lawyer":   ["lawyer", "attorney", "barrister", "counsel", "advocate", "solicitor"],
            "judge":    ["judge", "magistrate", "justice", "arbitrator", "referee", "adjudicator"],
        },
        "semantic_dims": {
            "doctor":   [5, 5, 5, 4],
            "nurse":    [3, 3, 4, 5],
            "teacher":  [4, 2, 5, 4],
            "student":  [1, 1, 2, 1],
            "chef":     [3, 3, 3, 1],
            "waiter":   [1, 1, 1, 3],
            "artist":   [3, 2, 3, 1],
            "musician": [3, 3, 3, 1],
            "lawyer":   [5, 5, 5, 1],
            "judge":    [5, 4, 5, 2],
        },
        "dim_names": ["prestige", "income", "education", "caring"],
    },
    "color10": {
        "categories": {
            "red":      ["red", "crimson", "scarlet", "ruby", "maroon", "cherry"],
            "blue":     ["blue", "azure", "cobalt", "navy", "sapphire", "indigo"],
            "green":    ["green", "emerald", "olive", "lime", "jade", "forest"],
            "yellow":   ["yellow", "gold", "amber", "lemon", "mustard", "canary"],
            "purple":   ["purple", "violet", "lavender", "plum", "magenta", "lilac"],
            "orange":   ["orange", "tangerine", "apricot", "peach", "coral", "salmon"],
            "pink":     ["pink", "rose", "blush", "fuchsia", "magenta", "carnation"],
            "brown":    ["brown", "tan", "beige", "bronze", "mahogany", "chestnut"],
            "gray":     ["gray", "silver", "slate", "ash", "charcoal", "pewter"],
            "white":    ["white", "ivory", "cream", "pearl", "snow", "alabaster"],
        },
        "semantic_dims": {
            "red":    [0.90, 0.05, 0.05],
            "blue":   [0.10, 0.10, 0.90],
            "green":  [0.05, 0.55, 0.05],
            "yellow": [0.95, 0.95, 0.10],
            "purple": [0.50, 0.05, 0.55],
            "orange": [0.95, 0.55, 0.05],
            "pink":   [0.95, 0.55, 0.60],
            "brown":  [0.55, 0.30, 0.05],
            "gray":   [0.50, 0.50, 0.50],
            "white":  [0.95, 0.95, 0.95],
        },
        "dim_names": ["R", "G", "B"],
    },
    "vehicle10": {
        "categories": {
            "car":      ["car", "automobile", "sedan", "vehicle", "coupe", "convertible"],
            "truck":    ["truck", "lorry", "pickup", "freight", "haul", "trailer"],
            "bus":      ["bus", "coach", "shuttle", "transit", "omnibus", "minibus"],
            "train":    ["train", "locomotive", "railway", "express", "metro", "subway"],
            "plane":    ["plane", "aircraft", "jet", "airplane", "liner", "airbus"],
            "boat":     ["boat", "ship", "vessel", "yacht", "ferry", "canoe"],
            "bike":     ["bike", "bicycle", "cycle", "motorcycle", "scooter", "moped"],
            "helicopter":["helicopter", "chopper", "rotorcraft", "copter", "whirlybird", "autogyro"],
            "tank":     ["tank", "armor", "panzer", "fighting", "military", "armored"],
            "rocket":   ["rocket", "spaceship", "shuttle", "missile", "spacecraft", "capsule"],
        },
        "semantic_dims": {
            "car":        [2, 3, 2, 1.0],
            "truck":      [4, 2, 4, 1.0],
            "bus":        [4, 2, 5, 1.0],
            "train":      [5, 4, 5, 1.0],
            "plane":      [5, 5, 4, 1.0],
            "boat":       [4, 2, 3, 1.0],
            "bike":       [1, 2, 1, 1.0],
            "helicopter": [3, 4, 2, 0.5],
            "tank":       [5, 3, 3, 0.0],
            "rocket":     [5, 5, 2, 0.0],
        },
        "dim_names": ["size", "speed", "capacity", "civilian"],
    },
}


# ============================================================
# 核心函数
# ============================================================

def get_category_centers_residual(model, tokenizer, device, categories, layer_idx):
    """在指定层收集残差中心"""
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        residuals = []
        for word in words:
            prompt = f"The word is {word}"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            
            with torch.no_grad():
                inputs_embeds = embed_layer(input_ids)
                
                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu().numpy()
                        else:
                            captured[key] = output.detach().float().cpu().numpy()
                    return hook
                
                hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
                _ = model(inputs_embeds=inputs_embeds)
                hook.remove()
                
                if f"L{layer_idx}" in captured:
                    res = captured[f"L{layer_idx}"][0, -1, :]
                    residuals.append(res)
        
        if len(residuals) > 0:
            cat_centers[cat_name] = np.mean(residuals, axis=0)
    
    return cat_centers


def get_category_centers_embedding(model, tokenizer, categories):
    """从token embedding层获取类别中心"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    cat_centers = {}
    
    for cat_name, words in categories.items():
        embeddings = []
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        
        if len(embeddings) > 0:
            cat_centers[cat_name] = np.mean(embeddings, axis=0)
    
    return cat_centers


def compute_pca(points):
    """
    PCA分析, 返回主成分方向和得分
    
    Args:
        points: [N, d] numpy数组
    
    Returns:
        dict with:
            scores: [N, K] PC得分 (K=min(N-1, d))
            directions: [K, d] PC方向(行向量)
            variance_explained: [K] 各PC解释的方差
            cumvar: [K] 累积方差解释比
            mean: [d] 均值
    """
    N, d = points.shape
    K = min(N - 1, d)
    
    mean = points.mean(axis=0)
    centered = points - mean
    
    # SVD: centered = U @ diag(S) @ Vt
    # PC方向 = Vt的行, PC得分 = U * S
    U, S, Vt = svd(centered, full_matrices=False)
    
    scores = U[:, :K] * S[:K]  # [N, K]
    directions = Vt[:K, :]     # [K, d]
    
    total_var = np.sum(S**2)
    variance_explained = S[:K]**2
    cumvar = np.cumsum(variance_explained) / total_var
    
    return {
        "scores": scores,
        "directions": directions,
        "singular_values": S[:K],
        "variance_explained": variance_explained,
        "cumvar": cumvar,
        "mean": mean,
        "K": K,
    }


def compute_principal_angles(emb_dirs, res_dirs, K=None):
    """
    计算两个子空间之间的主角度(principal angles)
    
    Args:
        emb_dirs: [K, d] embedding PC方向
        res_dirs: [K, d] residual PC方向
        K: 使用前K个PC (默认全部)
    
    Returns:
        cos_angles: [min(K1,K2)] 各主角度的余弦值
        angles_deg: [min(K1,K2)] 各主角度(度)
    """
    if K is not None:
        emb_dirs = emb_dirs[:K]
        res_dirs = res_dirs[:K]
    
    # M = V_E^T @ V_R → SVD(M)的奇异值 = cos(主角度)
    M = emb_dirs @ res_dirs.T  # [K, K]
    _, s, _ = svd(M, full_matrices=False)
    
    # 确保在[0,1]范围内(数值误差可能导致略超)
    cos_angles = np.clip(s, 0, 1)
    angles_deg = np.degrees(np.arccos(cos_angles))
    
    return cos_angles, angles_deg


def compute_pc_correlation_matrix(emb_scores, res_scores, K=None):
    """
    计算embedding PC得分和residual PC得分之间的相关矩阵
    
    Args:
        emb_scores: [N, K1] embedding PC得分
        res_scores: [N, K2] residual PC得分
        K: 使用前K个PC
    
    Returns:
        corr_mat: [K, K] 相关系数矩阵
        p_mat: [K, K] p值矩阵
    """
    if K is not None:
        emb_scores = emb_scores[:, :K]
        res_scores = res_scores[:, :K]
    
    K1 = emb_scores.shape[1]
    K2 = res_scores.shape[1]
    N = emb_scores.shape[0]
    
    corr_mat = np.zeros((K1, K2))
    p_mat = np.zeros((K1, K2))
    
    for i in range(K1):
        for j in range(K2):
            if N > 2:
                r, p = pearsonr(emb_scores[:, i], res_scores[:, j])
                corr_mat[i, j] = r
                p_mat[i, j] = p
    
    return corr_mat, p_mat


def compute_variance_transfer(emb_dirs, res_dirs, res_variance, K=None):
    """
    计算embedding子空间对residual各PC的方差传递
    
    对每个residual PC j:
        transfer[j] = ||P_{emb} res_PC_j||^2 / ||res_PC_j||^2
        = ||emb_dirs^T @ (emb_dirs @ res_PC_j)||^2
        = sum_i (emb_dir_i · res_PC_j)^2
    
    Args:
        emb_dirs: [K1, d] embedding PC方向(已归一化)
        res_dirs: [K2, d] residual PC方向(已归一化)
        res_variance: [K2] 各residual PC的方差
        K: 使用前K个PC
    
    Returns:
        transfer: [K2] 各residual PC被embedding子空间解释的方差比
    """
    if K is not None:
        emb_dirs = emb_dirs[:K]
        res_dirs = res_dirs[:K]
        res_variance = res_variance[:K]
    
    # 投影: emb_dirs @ res_dirs^T → [K1, K2]
    proj = emb_dirs @ res_dirs.T
    
    # 每个residual PC被embedding子空间解释的方差比
    transfer = np.sum(proj**2, axis=0)  # [K2]
    
    # 总体: embedding子空间解释的residual方差比例
    weighted_transfer = np.sum(transfer * res_variance) / np.sum(res_variance)
    
    return transfer, weighted_transfer


def compute_cosine_dist_matrix(centers, cat_names):
    """计算cosine距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    cos_sim = points_norm @ points_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist


def compute_euclidean_dist_matrix(centers, cat_names):
    """计算Euclidean距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    dists = squareform(pdist(points, metric='euclidean'))
    return dists


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCIV: 子空间对齐分析 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"  模型: {info.model_class}, d_model={d_model}, n_layers={n_layers}")
    
    # 选择测试层
    layer_candidates = sorted(set([
        1,
        max(2, n_layers // 6),
        n_layers // 4,
        n_layers // 3,
        5 * n_layers // 12,
        n_layers // 2,
        7 * n_layers // 12,
        2 * n_layers // 3,
        3 * n_layers // 4,
        min(n_layers - 2, 5 * n_layers // 6),
    ]))
    print(f"  测试层: {layer_candidates}")
    
    all_cell_results = []
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n--- 领域: {domain_name} (N={N}) ---")
        
        # 1. Embedding中心 (固定, 只算一次)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个embedding中心, 跳过")
            continue
        
        emb_points = np.array([emb_centers[name] for name in cat_names])
        emb_pca = compute_pca(emb_points)
        
        # Embedding距离矩阵(cosine)
        emb_dist = compute_cosine_dist_matrix(emb_centers, cat_names)
        
        print(f"  Embedding PCA: 前5PC方差={emb_pca['cumvar'][min(4,N-2)]:.3f}")
        
        for layer_idx in layer_candidates:
            print(f"  L{layer_idx}...", end=" ", flush=True)
            
            # 2. Residual中心
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            if len(res_centers) != N:
                print(f"跳过(只有{len(res_centers)}个中心)")
                continue
            
            res_points = np.array([res_centers[name] for name in cat_names])
            res_pca = compute_pca(res_points)
            
            # Residual距离矩阵(Euclidean, SVD投影后)
            K = min(N - 1, d_model)
            res_proj = res_pca["scores"]  # [N, K]
            res_dist = squareform(pdist(res_proj, metric='euclidean'))
            
            # =============================================
            # 核心: 子空间对齐分析
            # =============================================
            
            # A. 主角度
            cos_angles, angles_deg = compute_principal_angles(
                emb_pca["directions"], res_pca["directions"], K=K
            )
            
            # B. PC-PC相关矩阵
            corr_mat, p_mat = compute_pc_correlation_matrix(
                emb_pca["scores"], res_pca["scores"], K=K
            )
            
            # C. 方差传递
            transfer, weighted_transfer = compute_variance_transfer(
                emb_pca["directions"], res_pca["directions"],
                res_pca["variance_explained"], K=K
            )
            
            # D. Emb→Geo距离的β (复现CCIII的关键指标)
            upper = np.triu_indices(N, k=1)
            emb_flat = emb_dist[upper]
            geo_flat = res_dist[upper]
            
            # Z-score归一化
            emb_z = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-10)
            geo_z = (geo_flat - geo_flat.mean()) / (geo_flat.std() + 1e-10)
            
            r_emb_geo, p_emb_geo = pearsonr(emb_z, geo_z)
            beta_emb = r_emb_geo  # 简单回归中β = r
            
            # =============================================
            # 打印关键结果
            # =============================================
            
            # 主角度: 前5个
            n_show = min(5, len(cos_angles))
            angles_str = ", ".join([f"{angles_deg[i]:.1f}°" for i in range(n_show)])
            mean_cos = np.mean(cos_angles)
            
            # PC相关: 最大相关(对角线平均)
            diag_corr = np.mean(np.abs(np.diag(corr_mat[:n_show, :n_show])))
            max_corr_per_res = [np.max(np.abs(corr_mat[:, j])) for j in range(min(5, corr_mat.shape[1]))]
            max_corr_str = ", ".join([f"{v:.3f}" for v in max_corr_per_res[:3]])
            
            # 方差传递
            transfer_str = ", ".join([f"{transfer[i]:.3f}" for i in range(min(3, len(transfer)))])
            
            # 找最强PC对齐
            abs_corr = np.abs(corr_mat)
            best_i, best_j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
            best_corr = corr_mat[best_i, best_j]
            
            print(f"mean_cos={mean_cos:.3f}, "
                  f"angles=[{angles_str}], "
                  f"β_emb={beta_emb:+.3f}, "
                  f"transfer={weighted_transfer:.3f}")
            
            cell_result = {
                "model": model_name,
                "domain": domain_name,
                "layer": layer_idx,
                "d_model": d_model,
                "N": N,
                "K": K,
                # 主角度
                "mean_cos_angle": float(mean_cos),
                "cos_angles": cos_angles.tolist(),
                "angles_deg": angles_deg.tolist(),
                # 方差传递
                "weighted_transfer": float(weighted_transfer),
                "per_pc_transfer": transfer.tolist(),
                # PC相关
                "diag_corr_mean": float(diag_corr),
                "best_pc_alignment": {"emb_pc": int(best_i), "res_pc": int(best_j), "corr": float(best_corr)},
                # Emb→Geo
                "beta_emb_geo": float(beta_emb),
                "r_emb_geo": float(r_emb_geo),
                "p_emb_geo": float(p_emb_geo),
                # PCA方差解释
                "emb_cumvar_top5": float(emb_pca["cumvar"][min(4, K-1)]),
                "res_cumvar_top5": float(res_pca["cumvar"][min(4, K-1)]),
                # PC相关矩阵(前5×5)
                "pc_corr_top5": corr_mat[:min(5,K), :min(5,K)].tolist(),
            }
            
            all_cell_results.append(cell_result)
    
    # ============================================================
    # 汇总分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CCIV 汇总分析 — {model_name}")
    print(f"{'='*70}")
    
    # === 1. 各领域平均对齐度 ===
    print("\n--- 1. 各领域平均对齐度 ---")
    print(f"  {'领域':14s} {'mean_cos':>8s} {'transfer':>8s} {'diag_corr':>10s} "
          f"{'β_emb':>7s} {'angle1':>7s} {'angle2':>7s} {'angle3':>7s}")
    
    for domain_name in DOMAINS.keys():
        domain_data = [d for d in all_cell_results if d["domain"] == domain_name]
        if not domain_data:
            continue
        
        avg_cos = np.mean([d["mean_cos_angle"] for d in domain_data])
        avg_transfer = np.mean([d["weighted_transfer"] for d in domain_data])
        avg_diag = np.mean([d["diag_corr_mean"] for d in domain_data])
        avg_beta = np.mean([d["beta_emb_geo"] for d in domain_data])
        
        # 前3个主角度
        angles_all = np.array([d["angles_deg"] for d in domain_data])
        avg_angles = angles_all.mean(axis=0)
        a1 = avg_angles[0] if len(avg_angles) > 0 else 0
        a2 = avg_angles[1] if len(avg_angles) > 1 else 0
        a3 = avg_angles[2] if len(avg_angles) > 2 else 0
        
        print(f"  {domain_name:14s} {avg_cos:8.3f} {avg_transfer:8.3f} {avg_diag:10.3f} "
              f"{avg_beta:+7.3f} {a1:7.1f}° {a2:7.1f}° {a3:7.1f}°")
    
    # === 2. β_emb正/负分组的对齐度差异 ===
    print("\n--- 2. β_emb正/负分组的对齐度差异 ---")
    pos_cells = [d for d in all_cell_results if d["beta_emb_geo"] > 0]
    neg_cells = [d for d in all_cell_results if d["beta_emb_geo"] <= 0]
    
    print(f"  β_emb>0: {len(pos_cells)} cells")
    if pos_cells:
        print(f"    mean_cos={np.mean([d['mean_cos_angle'] for d in pos_cells]):.3f}, "
              f"transfer={np.mean([d['weighted_transfer'] for d in pos_cells]):.3f}, "
              f"diag_corr={np.mean([d['diag_corr_mean'] for d in pos_cells]):.3f}")
    
    print(f"  β_emb≤0: {len(neg_cells)} cells")
    if neg_cells:
        print(f"    mean_cos={np.mean([d['mean_cos_angle'] for d in neg_cells]):.3f}, "
              f"transfer={np.mean([d['weighted_transfer'] for d in neg_cells]):.3f}, "
              f"diag_corr={np.mean([d['diag_corr_mean'] for d in neg_cells]):.3f}")
        # 哪些领域贡献了负β?
        neg_domains = {}
        for d in neg_cells:
            neg_domains[d["domain"]] = neg_domains.get(d["domain"], 0) + 1
        print(f"    负β领域分布: {neg_domains}")
    
    # === 3. 层间演变 ===
    print("\n--- 3. 层间演变 ---")
    for layer_idx in layer_candidates:
        layer_data = [d for d in all_cell_results if d["layer"] == layer_idx]
        if not layer_data:
            continue
        avg_cos = np.mean([d["mean_cos_angle"] for d in layer_data])
        avg_transfer = np.mean([d["weighted_transfer"] for d in layer_data])
        avg_beta = np.mean([d["beta_emb_geo"] for d in layer_data])
        print(f"  L{layer_idx:2d}: mean_cos={avg_cos:.3f}, transfer={avg_transfer:.3f}, β_emb={avg_beta:+.3f}")
    
    # === 4. Vehicle领域的详细分析 ===
    print("\n--- 4. Vehicle领域详细分析 (β_emb可能为负) ---")
    vehicle_data = [d for d in all_cell_results if d["domain"] == "vehicle10"]
    if vehicle_data:
        for d in vehicle_data:
            # PC相关矩阵: 找反转的PC
            corr = np.array(d["pc_corr_top5"])
            print(f"  L{d['layer']:2d}: β_emb={d['beta_emb_geo']:+.3f}, "
                  f"mean_cos={d['mean_cos_angle']:.3f}, transfer={d['weighted_transfer']:.3f}")
            if corr.size > 0:
                # 找负相关的PC对
                neg_pairs = []
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        if corr[i, j] < -0.3:
                            neg_pairs.append(f"embPC{i+1}↔resPC{j+1}={corr[i,j]:+.2f}")
                if neg_pairs:
                    print(f"        负相关PC对: {', '.join(neg_pairs[:5])}")
                # 对角线相关
                diag = [corr[i,i] for i in range(min(corr.shape[0], corr.shape[1]))]
                diag_str = ", ".join([f"{v:+.2f}" for v in diag[:5]])
                print(f"        对角线相关: [{diag_str}]")
    
    # === 5. 全局统计 ===
    print("\n--- 5. 全局统计 ---")
    n_total = len(all_cell_results)
    avg_cos = np.mean([d["mean_cos_angle"] for d in all_cell_results])
    avg_transfer = np.mean([d["weighted_transfer"] for d in all_cell_results])
    avg_diag = np.mean([d["diag_corr_mean"] for d in all_cell_results])
    avg_beta = np.mean([d["beta_emb_geo"] for d in all_cell_results])
    
    # PC1对齐: 第一个主角度
    avg_angle1 = np.mean([d["angles_deg"][0] for d in all_cell_results if len(d["angles_deg"]) > 0])
    
    print(f"  总cell数: {n_total}")
    print(f"  平均主角度余弦: {avg_cos:.3f}")
    print(f"  平均第一个主角度: {avg_angle1:.1f}°")
    print(f"  平均方差传递: {avg_transfer:.3f}")
    print(f"  平均对角线相关: {avg_diag:.3f}")
    print(f"  平均β_emb: {avg_beta:+.3f}")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_tested": layer_candidates,
        "cell_results": all_cell_results,
    }
    
    out_path = TEMP / f"cciv_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    release_model(model)
    gc.collect()
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    result = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
