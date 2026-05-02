"""
CCV(355): Procrustes层间旋转分析 — 量化Transformer每层如何旋转空间几何
=====================================================================
★★★★★ CCIV核心发现:
  - Embedding和Residual子空间几乎正交(82°)
  - >99%几何结构来自Transformer变换
  - Vehicle领域β_emb系统性为负——轴旋转/反转
  - 深层重新引入embedding结构

★★★★★ 本实验目标:
  1. Procrustes分析: 量化相邻层之间的旋转矩阵R_{l→l+1}
  2. 旋转角度谱: 每层旋转了多少度?
  3. 累积旋转: 从embedding到各层的总旋转
  4. 旋转方向: 是否有层间一致的旋转模式?
  5. Domain特异性: Vehicle是否有不同的旋转模式?
  6. 与β_emb的关系: 旋转模式如何影响β_emb?

★★★★★ 核心方法:
  - Orthogonal Procrustes: min ||Y - XR||_F s.t. R^T R = I
  - SVD(X^T Y) = U Σ V^T → R = V U^T
  - 旋转角度: arccos((trace(R) - 1) / (K - 1)) (K维空间)
  - 层间旋转率: ||R_{l→l+1} - I||_F / sqrt(K)

用法:
  python ccv_procrustes_rotation.py --model qwen3
  python ccv_procrustes_rotation.py --model glm4
  python ccv_procrustes_rotation.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 领域定义 — 与CCIV完全相同
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
    """PCA分析"""
    N, d = points.shape
    K = min(N - 1, d)
    
    mean = points.mean(axis=0)
    centered = points - mean
    
    U, S, Vt = svd(centered, full_matrices=False)
    
    scores = U[:, :K] * S[:K]  # [N, K]
    directions = Vt[:K, :]     # [K, d]
    
    total_var = np.sum(S**2)
    variance_explained = S[:K]**2
    cumvar = np.cumsum(variance_explained) / total_var if total_var > 0 else np.zeros(K)
    
    return {
        "scores": scores,
        "directions": directions,
        "singular_values": S[:K],
        "variance_explained": variance_explained,
        "cumvar": cumvar,
        "mean": mean,
        "K": K,
    }


def procrustes_align(X, Y):
    """
    Orthogonal Procrustes: 找R使得||Y - XR||_F最小, R^T R = I
    
    Args:
        X: [N, K] 源矩阵(居中)
        Y: [N, K] 目标矩阵(居中)
    
    Returns:
        R: [K, K] 最优正交旋转矩阵
        scale: float 缩放因子
        error: float 对齐误差
        singular_values: [K] M = X^T Y的奇异值
    """
    # M = X^T Y
    M = X.T @ Y  # [K, K]
    
    # SVD: M = U Σ V^T
    U, sigma, Vt = svd(M, full_matrices=False)
    
    # 最优旋转: R = V U^T
    R = Vt.T @ U.T
    
    # 确保R是proper rotation (det(R)=+1), 不是reflection
    if np.linalg.det(R) < 0:
        # 翻转Vt的最后一行
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 缩放因子
    scale = np.sum(sigma) / np.sum(X**2) if np.sum(X**2) > 0 else 0
    
    # 对齐误差
    Y_pred = X @ R
    error = np.sum((Y - Y_pred)**2) / np.sum(Y**2) if np.sum(Y**2) > 0 else 1.0
    
    return R, scale, error, sigma


def compute_rotation_angle(R):
    """
    计算旋转矩阵的旋转角度
    
    对于K维旋转矩阵R:
    - 旋转角度θ = arccos((trace(R) - 1) / (K - 1))
    - 这是"平均旋转角度"
    
    Returns:
        angle_deg: 旋转角度(度)
        trace_val: 迹
    """
    K = R.shape[0]
    trace_val = np.trace(R)
    # arccos参数需要clip到[-1, 1]
    cos_angle = np.clip((trace_val - 1) / max(K - 1, 1), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg, float(trace_val)


def compute_rotation_decomposition(R):
    """
    分解旋转矩阵R的特征值结构
    
    旋转矩阵的特征值是共轭复数对e^{±iθ_k}或实数±1
    分析旋转的"维度分布": 各个2D旋转平面上的角度
    
    Returns:
        dict: {rotation_angles, n_identity, n_reflection, n_rotations, total_rotation_energy}
    """
    eigenvalues = np.linalg.eigvals(R)
    
    rotation_angles = []
    n_identity = 0  # +1特征值
    n_reflection = 0  # -1特征值
    
    # 按对处理
    used = set()
    for i, ev in enumerate(eigenvalues):
        if i in used:
            continue
        if np.isreal(ev):
            real_val = np.real(ev)
            if real_val > 0.99:
                n_identity += 1
            elif real_val < -0.99:
                n_reflection += 1
            else:
                # 实数但不是±1 → 与另一个特征值配对
                rotation_angles.append(np.degrees(np.arccos(np.clip(real_val, -1, 1))))
            used.add(i)
        else:
            # 复数特征值 → e^{iθ}
            theta = np.degrees(np.arccos(np.clip(np.real(ev), -1, 1)))
            rotation_angles.append(theta)
            # 找共轭对
            for j in range(i+1, len(eigenvalues)):
                if j not in used and np.abs(np.real(eigenvalues[j]) - np.real(ev)) < 0.01:
                    used.add(j)
                    break
            used.add(i)
    
    return {
        "rotation_angles": sorted(rotation_angles, reverse=True),
        "n_identity": n_identity,
        "n_reflection": n_reflection,
        "n_rotations": len(rotation_angles),
        "total_rotation_energy": sum(r**2 for r in rotation_angles),
    }


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


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCV: Procrustes层间旋转分析 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"  模型: {info.model_class}, d_model={d_model}, n_layers={n_layers}")
    
    # 选择测试层 — 更密的采样以捕获旋转模式
    layer_candidates = sorted(set([
        0, 1,
        max(2, n_layers // 8),
        n_layers // 6,
        n_layers // 4,
        n_layers // 3,
        5 * n_layers // 12,
        n_layers // 2,
        7 * n_layers // 12,
        2 * n_layers // 3,
        3 * n_layers // 4,
        5 * n_layers // 6,
        min(n_layers - 2, 7 * n_layers // 8),
        n_layers - 1,
    ]))
    print(f"  测试层({len(layer_candidates)}个): {layer_candidates}")
    
    all_results = {}
    all_procrustes = {}
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        K = N - 1  # PC数量
        
        print(f"\n--- 领域: {domain_name} (N={N}, K={K}) ---")
        
        # 1. 收集所有层的中心
        print(f"  收集各层中心...")
        
        # Embedding层 (layer=-1表示embedding)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个embedding中心, 跳过")
            continue
        
        # 各Residual层
        layer_centers = {"emb": emb_centers}
        for layer_idx in layer_candidates:
            print(f"    L{layer_idx}...", end=" ", flush=True)
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            if len(res_centers) != N:
                print(f"跳过(只有{len(res_centers)}个中心)")
                continue
            layer_centers[f"L{layer_idx}"] = res_centers
            print("OK")
        
        # 2. PCA at each layer
        print(f"  PCA分析...")
        layer_pcas = {}
        for layer_key, centers in layer_centers.items():
            points = np.array([centers[name] for name in cat_names])
            pca = compute_pca(points)
            layer_pcas[layer_key] = pca
        
        # 3. Procrustes between consecutive layers
        print(f"  Procrustes分析...")
        layer_keys = list(layer_pcas.keys())  # ["emb", "L0", "L1", ...]
        
        procrustes_results = []
        
        for i in range(len(layer_keys) - 1):
            key_from = layer_keys[i]
            key_to = layer_keys[i + 1]
            
            pca_from = layer_pcas[key_from]
            pca_to = layer_pcas[key_to]
            
            # 使用PCA scores进行Procrustes (N×K)
            X = pca_from["scores"]  # [N, K]
            Y = pca_to["scores"]    # [N, K]
            
            # 居中(虽然PCA scores已经居中, 但确保)
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            # Procrustes
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            
            # 旋转角度
            angle_deg, trace_val = compute_rotation_angle(R)
            
            # 旋转分解
            rot_decomp = compute_rotation_decomposition(R)
            
            # 距离保持: 相邻层的距离矩阵相关性
            dist_from = squareform(pdist(X_c, metric='euclidean'))
            dist_to = squareform(pdist(Y_c, metric='euclidean'))
            upper = np.triu_indices(N, k=1)
            r_dist, p_dist = pearsonr(dist_from[upper], dist_to[upper])
            
            # β_emb (与CCIV一致): embedding距离 → 当前层距离
            emb_dist = compute_cosine_dist_matrix(emb_centers, cat_names)
            current_centers = layer_centers[key_to]
            current_points = np.array([current_centers[name] for name in cat_names])
            current_proj = pca_to["scores"]
            current_dist = squareform(pdist(current_proj, metric='euclidean'))
            
            emb_flat = emb_dist[upper]
            cur_flat = current_dist[upper]
            emb_z = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-10)
            cur_z = (cur_flat - cur_flat.mean()) / (cur_flat.std() + 1e-10)
            r_emb, p_emb = pearsonr(emb_z, cur_z)
            
            result = {
                "from": key_from,
                "to": key_to,
                "from_layer": -1 if key_from == "emb" else int(key_from[1:]),
                "to_layer": -1 if key_to == "emb" else int(key_to[1:]),
                # Procrustes
                "rotation_angle_deg": float(angle_deg),
                "trace": float(trace_val),
                "scale": float(scale),
                "alignment_error": float(error),
                "singular_values": sigma.tolist(),
                # 旋转分解
                "rotation_angles_top3": rot_decomp["rotation_angles"][:3],
                "n_identity": rot_decomp["n_identity"],
                "n_reflection": rot_decomp["n_reflection"],
                "n_rotations": rot_decomp["n_rotations"],
                "total_rotation_energy": rot_decomp["total_rotation_energy"],
                # 距离保持
                "r_dist_preservation": float(r_dist),
                "p_dist_preservation": float(p_dist),
                # β_emb
                "beta_emb": float(r_emb),
                "p_emb": float(p_emb),
                # PCA信息
                "cumvar_from_top5": float(pca_from["cumvar"][min(4, K-1)]),
                "cumvar_to_top5": float(pca_to["cumvar"][min(4, K-1)]),
            }
            
            procrustes_results.append(result)
            
            # 打印
            rot_str = ", ".join([f"{a:.1f}°" for a in rot_decomp["rotation_angles"][:3]])
            print(f"    {key_from}→{key_to}: θ={angle_deg:.1f}°, "
                  f"error={error:.4f}, r_dist={r_dist:.3f}, "
                  f"β_emb={r_emb:+.3f}, top_rot=[{rot_str}]")
        
        # 4. 累积旋转: emb → each layer
        print(f"  累积旋转分析...")
        emb_pca = layer_pcas["emb"]
        
        cumulative_results = []
        R_cumulative = np.eye(K)  # 累积旋转矩阵
        
        for i in range(len(procrustes_results)):
            # 从第i步的R更新累积旋转
            p = procrustes_results[i]
            
            # 重新计算R(因为之前没有保存矩阵)
            key_from = p["from"]
            key_to = p["to"]
            X = layer_pcas[key_from]["scores"]
            Y = layer_pcas[key_to]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            R, _, _, _ = procrustes_align(X_c, Y_c)
            
            R_cumulative = R_cumulative @ R
            
            # 累积旋转角度
            cum_angle, cum_trace = compute_rotation_angle(R_cumulative)
            
            cumulative_results.append({
                "to_layer": p["to_layer"],
                "cumulative_angle_deg": float(cum_angle),
                "cumulative_trace": float(cum_trace),
                "step_angle_deg": p["rotation_angle_deg"],
                "beta_emb": p["beta_emb"],
            })
            
            print(f"    →{key_to}: cum_θ={cum_angle:.1f}°, "
                  f"step_θ={p['rotation_angle_deg']:.1f}°, "
                  f"β_emb={p['beta_emb']:+.3f}")
        
        # 5. 残差连接效应: 直接测量 emb→L_l 的旋转
        #    (不经过中间层, 而是直接Procrustes)
        print(f"  直接emb→各层旋转...")
        direct_results = []
        
        for layer_key in layer_keys[1:]:  # skip emb
            X = emb_pca["scores"]
            Y = layer_pcas[layer_key]["scores"]
            X_c = X - X.mean(axis=0)
            Y_c = Y - Y.mean(axis=0)
            
            R, scale, error, sigma = procrustes_align(X_c, Y_c)
            angle_deg, trace_val = compute_rotation_angle(R)
            rot_decomp = compute_rotation_decomposition(R)
            
            direct_results.append({
                "layer_key": layer_key,
                "layer_idx": -1 if layer_key == "emb" else int(layer_key[1:]),
                "direct_angle_deg": float(angle_deg),
                "direct_trace": float(trace_val),
                "direct_error": float(error),
                "rotation_angles_top3": rot_decomp["rotation_angles"][:3],
                "n_identity": rot_decomp["n_identity"],
                "n_reflection": rot_decomp["n_reflection"],
            })
            
            print(f"    emb→{layer_key}: θ={angle_deg:.1f}°, "
                  f"error={error:.4f}, "
                  f"identity={rot_decomp['n_identity']}/{K}, "
                  f"reflection={rot_decomp['n_reflection']}/{K}")
        
        all_results[domain_name] = {
            "procrustes": procrustes_results,
            "cumulative": cumulative_results,
            "direct": direct_results,
            "N": N,
            "K": K,
            "layer_keys": layer_keys,
        }
    
    # ============================================================
    # 汇总分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CCV 汇总分析 — {model_name}")
    print(f"{'='*70}")
    
    # === 1. 各领域平均旋转角度 ===
    print("\n--- 1. 各领域旋转特征 ---")
    print(f"  {'领域':14s} {'mean_step_θ':>12s} {'mean_direct_θ':>14s} "
          f"{'mean_error':>11s} {'mean_r_dist':>12s} {'mean_β':>7s}")
    
    for domain_name in DOMAINS.keys():
        if domain_name not in all_results:
            continue
        
        data = all_results[domain_name]
        proc = data["procrustes"]
        direct = data["direct"]
        
        mean_step = np.mean([p["rotation_angle_deg"] for p in proc]) if proc else 0
        mean_direct = np.mean([d["direct_angle_deg"] for d in direct]) if direct else 0
        mean_error = np.mean([p["alignment_error"] for p in proc]) if proc else 0
        mean_rdist = np.mean([p["r_dist_preservation"] for p in proc]) if proc else 0
        mean_beta = np.mean([p["beta_emb"] for p in proc]) if proc else 0
        
        print(f"  {domain_name:14s} {mean_step:12.1f}° {mean_direct:14.1f}° "
              f"{mean_error:11.4f} {mean_rdist:12.3f} {mean_beta:+7.3f}")
    
    # === 2. Vehicle vs 非Vehicle的旋转差异 ===
    print("\n--- 2. Vehicle vs 非Vehicle旋转差异 ---")
    
    vehicle_proc = []
    other_proc = []
    for domain_name, data in all_results.items():
        if domain_name == "vehicle10":
            vehicle_proc.extend(data["procrustes"])
        else:
            other_proc.extend(data["procrustes"])
    
    if vehicle_proc and other_proc:
        v_angles = [p["rotation_angle_deg"] for p in vehicle_proc]
        o_angles = [p["rotation_angle_deg"] for p in other_proc]
        v_errors = [p["alignment_error"] for p in vehicle_proc]
        o_errors = [p["alignment_error"] for p in other_proc]
        v_betas = [p["beta_emb"] for p in vehicle_proc]
        o_betas = [p["beta_emb"] for p in other_proc]
        v_rdist = [p["r_dist_preservation"] for p in vehicle_proc]
        o_rdist = [p["r_dist_preservation"] for p in other_proc]
        
        print(f"  Vehicle:  step_θ={np.mean(v_angles):.1f}°, "
              f"error={np.mean(v_errors):.4f}, "
              f"r_dist={np.mean(v_rdist):.3f}, "
              f"β_emb={np.mean(v_betas):+.3f}")
        print(f"  其他:     step_θ={np.mean(o_angles):.1f}°, "
              f"error={np.mean(o_errors):.4f}, "
              f"r_dist={np.mean(o_rdist):.3f}, "
              f"β_emb={np.mean(o_betas):+.3f}")
        
        # 统计检验
        if len(v_angles) >= 3 and len(o_angles) >= 3:
            from scipy.stats import mannwhitneyu
            try:
                u_angle, p_angle = mannwhitneyu(v_angles, o_angles, alternative='two-sided')
                u_beta, p_beta = mannwhitneyu(v_betas, o_betas, alternative='two-sided')
                print(f"  Mann-Whitney: angle p={p_angle:.3f}, β_emb p={p_beta:.3f}")
            except:
                pass
    
    # === 3. 层间旋转率演变 ===
    print("\n--- 3. 层间旋转率演变 ---")
    
    # 按层分组
    layer_step_angles = {}
    layer_direct_angles = {}
    layer_betas = {}
    
    for domain_name, data in all_results.items():
        for p in data["procrustes"]:
            to_layer = p["to_layer"]
            if to_layer not in layer_step_angles:
                layer_step_angles[to_layer] = []
                layer_direct_angles[to_layer] = []
                layer_betas[to_layer] = []
            layer_step_angles[to_layer].append(p["rotation_angle_deg"])
            layer_betas[to_layer].append(p["beta_emb"])
        
        for d in data["direct"]:
            layer_idx = d["layer_idx"]
            if layer_idx not in layer_direct_angles:
                layer_direct_angles[layer_idx] = []
            layer_direct_angles[layer_idx].append(d["direct_angle_deg"])
    
    print(f"  {'层':6s} {'step_θ':>8s} {'direct_θ':>10s} {'β_emb':>7s}")
    for layer_idx in sorted(layer_step_angles.keys()):
        step_avg = np.mean(layer_step_angles[layer_idx])
        beta_avg = np.mean(layer_betas[layer_idx])
        direct_avg = np.mean(layer_direct_angles.get(layer_idx, [0]))
        print(f"  L{layer_idx:3d}  {step_avg:8.1f}° {direct_avg:10.1f}° {beta_avg:+7.3f}")
    
    # === 4. 旋转与β_emb的关系 ===
    print("\n--- 4. 旋转特征与β_emb的关系 ---")
    
    all_angles = []
    all_errors = []
    all_betas = []
    all_rdist = []
    
    for domain_name, data in all_results.items():
        for p in data["procrustes"]:
            all_angles.append(p["rotation_angle_deg"])
            all_errors.append(p["alignment_error"])
            all_betas.append(p["beta_emb"])
            all_rdist.append(p["r_dist_preservation"])
    
    if len(all_angles) > 5:
        r1, p1 = pearsonr(all_angles, all_betas)
        r2, p2 = pearsonr(all_errors, all_betas)
        r3, p3 = pearsonr(all_rdist, all_betas)
        
        print(f"  step_θ vs β_emb:    r={r1:+.3f}, p={p1:.3f}")
        print(f"  error vs β_emb:     r={r2:+.3f}, p={p2:.3f}")
        print(f"  r_dist vs β_emb:    r={r3:+.3f}, p={p3:.3f}")
    
    # === 5. 累积旋转: 深层是否"回到"embedding? ===
    print("\n--- 5. 累积旋转: emb→各层的直接旋转角度 ---")
    
    for domain_name, data in all_results.items():
        direct = data["direct"]
        if not direct:
            continue
        
        # 浅层 vs 深层 vs 最深层
        shallow = [d for d in direct if d["layer_idx"] < n_layers // 3]
        middle = [d for d in direct if n_layers // 3 <= d["layer_idx"] < 2 * n_layers // 3]
        deep = [d for d in direct if d["layer_idx"] >= 2 * n_layers // 3]
        
        shallow_avg = np.mean([d["direct_angle_deg"] for d in shallow]) if shallow else 0
        middle_avg = np.mean([d["direct_angle_deg"] for d in middle]) if middle else 0
        deep_avg = np.mean([d["direct_angle_deg"] for d in deep]) if deep else 0
        
        print(f"  {domain_name:14s}: "
              f"浅层={shallow_avg:.1f}°, 中层={middle_avg:.1f}°, 深层={deep_avg:.1f}°")
    
    # === 6. 反射成分分析 ===
    print("\n--- 6. 反射成分分析 (det(R)符号与β_emb) ---")
    
    for domain_name, data in all_results.items():
        direct = data["direct"]
        for d in direct:
            if d["n_reflection"] > 0:
                # 找对应的β_emb
                layer_idx = d["layer_idx"]
                matching_proc = [p for p in data["procrustes"] if p["to_layer"] == layer_idx]
                beta = matching_proc[0]["beta_emb"] if matching_proc else 0
                print(f"  {domain_name} L{layer_idx}: "
                      f"reflections={d['n_reflection']}/{data['K']}, "
                      f"β_emb={beta:+.3f}, "
                      f"direct_θ={d['direct_angle_deg']:.1f}°")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_tested": layer_candidates,
        "domain_results": all_results,
    }
    
    out_path = TEMP / f"ccv_{model_name}_results.json"
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
