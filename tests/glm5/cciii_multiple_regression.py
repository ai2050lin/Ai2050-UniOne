"""
CCIII(353): 多元回归模型 — 量化各因素独立贡献
==========================================
★★★★★ CCII核心发现:
  d_emb→d_geo的R²≈0.1, 幂律假说被推翻
  但CCI的"增强对比"残差模式仍然有效

★★★★★ 本实验目标:
  1. 在统一回归框架中: d_geo ~ d_emb + d_sem
  2. 量化embedding距离和语义距离的独立贡献
  3. 分析回归系数的跨层/跨领域/跨模型模式
  4. 交互效应: d_emb * d_sem
  5. 增强对比的定量刻画

设计:
  - 5领域, N=10类别, 45对/领域
  - 10采样层, 3模型
  - 每cell: geo ~ emb + sem + emb:sem
  - 系数β_emb, β_sem, β_inter的跨域/跨层/跨模型分析

用法:
  python cciii_multiple_regression.py --model qwen3
  python cciii_multiple_regression.py --model glm4
  python cciii_multiple_regression.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr
from numpy.linalg import lstsq

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 领域定义 — 5领域, 含语义维度
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
        # [size(1-5), domestic(0-1), predator(0-1), speed(1-5)]
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
        # [valence, arousal, dominance] (VAD model)
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
        # [prestige(1-5), income(1-5), education(1-5), caring(1-5)]
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
        # [R, G, B] normalized 0-1
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
        # [size(1-5), speed(1-5), capacity(1-5), civilian(0-1)]
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


def compute_pairwise_distances(centers, cat_names):
    """计算Euclidean距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    dists = squareform(pdist(points, metric='euclidean'))
    return dists


def compute_pairwise_cosine_dist(centers, cat_names):
    """计算cosine距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    cos_sim = points_norm @ points_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist


def compute_semantic_distances(semantic_dims, cat_names):
    """计算语义维度空间中的Euclidean距离矩阵"""
    points = np.array([semantic_dims[name] for name in cat_names], dtype=float)
    # Z-score归一化每个维度
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    stds = np.maximum(stds, 1e-10)
    points_norm = (points - means) / stds
    dists = squareform(pdist(points_norm, metric='euclidean'))
    return dists


def run_regression(geo_flat, emb_flat, sem_flat):
    """多元回归: geo ~ emb + sem + emb:sem"""
    N = len(geo_flat)
    
    # 标准化变量
    geo_z = (geo_flat - geo_flat.mean()) / (geo_flat.std() + 1e-10)
    emb_z = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-10)
    sem_z = (sem_flat - sem_flat.mean()) / (sem_flat.std() + 1e-10)
    inter_z = emb_z * sem_z  # 交互项
    
    # 模型1: geo ~ emb (基线)
    X1 = np.column_stack([emb_z, np.ones(N)])
    coeffs1, _, _, _ = lstsq(X1, geo_z, rcond=None)
    pred1 = X1 @ coeffs1
    ss_res1 = np.sum((geo_z - pred1)**2)
    ss_tot = np.sum((geo_z - np.mean(geo_z))**2)
    R2_emb = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0
    r_emb, p_emb = pearsonr(emb_z, geo_z)
    
    # 模型2: geo ~ sem
    X2 = np.column_stack([sem_z, np.ones(N)])
    coeffs2, _, _, _ = lstsq(X2, geo_z, rcond=None)
    pred2 = X2 @ coeffs2
    ss_res2 = np.sum((geo_z - pred2)**2)
    R2_sem = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
    r_sem, p_sem = pearsonr(sem_z, geo_z)
    
    # 模型3: geo ~ emb + sem (主效应)
    X3 = np.column_stack([emb_z, sem_z, np.ones(N)])
    coeffs3, _, _, _ = lstsq(X3, geo_z, rcond=None)
    pred3 = X3 @ coeffs3
    ss_res3 = np.sum((geo_z - pred3)**2)
    R2_both = 1 - ss_res3 / ss_tot if ss_tot > 0 else 0
    beta_emb = coeffs3[0]
    beta_sem = coeffs3[1]
    
    # 模型4: geo ~ emb + sem + emb:sem (含交互)
    X4 = np.column_stack([emb_z, sem_z, inter_z, np.ones(N)])
    coeffs4, _, _, _ = lstsq(X4, geo_z, rcond=None)
    pred4 = X4 @ coeffs4
    ss_res4 = np.sum((geo_z - pred4)**2)
    R2_full = 1 - ss_res4 / ss_tot if ss_tot > 0 else 0
    beta_emb_full = coeffs4[0]
    beta_sem_full = coeffs4[1]
    beta_inter = coeffs4[2]
    
    # 增量R²
    delta_R2_sem = R2_both - R2_emb  # sem的增量贡献
    delta_R2_emb = R2_both - R2_sem  # emb的增量贡献
    delta_R2_inter = R2_full - R2_both  # 交互的增量贡献
    
    # 半偏相关系数 (使用标准化变量后的系数即等于半偏相关)
    # pr(geo,emb|sem) = beta_emb in model3
    # pr(geo,sem|emb) = beta_sem in model3
    
    # 增强对比指标
    # 分成近对(d_emb小)和远对(d_emb大)
    q50_emb = np.median(emb_flat)
    near_mask = emb_flat <= q50_emb
    far_mask = emb_flat > q50_emb
    
    if near_mask.sum() > 2 and far_mask.sum() > 2:
        # 近对的残差(从线性模型)
        residual_near = (geo_z - pred1)[near_mask].mean()
        residual_far = (geo_z - pred1)[far_mask].mean()
        contrast_index = residual_far - residual_near
    else:
        contrast_index = 0.0
    
    return {
        "R2_emb": R2_emb,
        "R2_sem": R2_sem,
        "R2_both": R2_both,
        "R2_full": R2_full,
        "beta_emb": beta_emb,
        "beta_sem": beta_sem,
        "beta_inter": beta_inter,
        "r_emb": r_emb,
        "p_emb": p_emb,
        "r_sem": r_sem,
        "p_sem": p_sem,
        "delta_R2_sem": delta_R2_sem,
        "delta_R2_emb": delta_R2_emb,
        "delta_R2_inter": delta_R2_inter,
        "contrast_index": contrast_index,
        "n_pairs": N,
    }


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCIII: 多元回归模型 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"  模型: {info.model_class}, d_model={info.d_model}, n_layers={n_layers}")
    
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
    
    all_cell_results = {}
    summary_data = []  # 用于跨模型分析
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        semantic_dims = domain_def["semantic_dims"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n--- 领域: {domain_name} (N={N}, dims={domain_def['dim_names']}) ---")
        
        # 1. 计算embedding距离(固定)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个类别, 跳过")
            continue
        emb_dist_mat = compute_pairwise_cosine_dist(emb_centers, cat_names)
        
        # 2. 计算语义距离(固定)
        sem_dist_mat = compute_semantic_distances(semantic_dims, cat_names)
        
        upper_idx = np.triu_indices(N, k=1)
        emb_flat = emb_dist_mat[upper_idx]
        sem_flat = sem_dist_mat[upper_idx]
        
        domain_results = {}
        
        for layer_idx in layer_candidates:
            print(f"  L{layer_idx}...", end=" ", flush=True)
            
            # 收集残差中心
            res_centers = get_category_centers_residual(model, tokenizer, device, categories, layer_idx)
            if len(res_centers) != N:
                print(f"跳过(只有{len(res_centers)}个类别)")
                continue
            
            # SVD投影
            points = np.array([res_centers[name] for name in cat_names])
            U, S, Vt = svd(points, full_matrices=False)
            D = min(N - 1, points.shape[1])
            points_proj = U[:, :D] @ np.diag(S[:D])
            
            # 几何距离矩阵
            geo_dist_mat = compute_pairwise_distances(
                {name: points_proj[i] for i, name in enumerate(cat_names)},
                cat_names
            )
            geo_flat = geo_dist_mat[upper_idx]
            
            # === 多元回归 ===
            reg_result = run_regression(geo_flat, emb_flat, sem_flat)
            
            domain_results[f"L{layer_idx}"] = reg_result
            
            # 打印关键指标
            winner = "emb" if reg_result["delta_R2_emb"] > reg_result["delta_R2_sem"] else "sem"
            sig_emb = "*" if reg_result["p_emb"] < 0.05 else ""
            sig_sem = "*" if reg_result["p_sem"] < 0.05 else ""
            
            print(f"R²_both={reg_result['R2_both']:.3f}, "
                  f"β_emb={reg_result['beta_emb']:+.3f}{sig_emb}, "
                  f"β_sem={reg_result['beta_sem']:+.3f}{sig_sem}, "
                  f"winner={winner}")
            
            summary_data.append({
                "model": model_name,
                "domain": domain_name,
                "layer": layer_idx,
                **reg_result,
            })
        
        all_cell_results[domain_name] = domain_results
    
    # ============================================================
    # 汇总分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CCIII 汇总分析 — {model_name}")
    print(f"{'='*70}")
    
    # === 1. 各领域的平均回归系数 ===
    print("\n--- 1. 各领域平均回归系数 ---")
    for domain_name in DOMAINS.keys():
        domain_data = [d for d in summary_data if d["domain"] == domain_name]
        if not domain_data:
            continue
        
        avg_R2_both = np.mean([d["R2_both"] for d in domain_data])
        avg_beta_emb = np.mean([d["beta_emb"] for d in domain_data])
        avg_beta_sem = np.mean([d["beta_sem"] for d in domain_data])
        avg_beta_inter = np.mean([d["beta_inter"] for d in domain_data])
        avg_delta_emb = np.mean([d["delta_R2_emb"] for d in domain_data])
        avg_delta_sem = np.mean([d["delta_R2_sem"] for d in domain_data])
        avg_contrast = np.mean([d["contrast_index"] for d in domain_data])
        
        winner_emb = sum(1 for d in domain_data if d["delta_R2_emb"] > d["delta_R2_sem"])
        winner_sem = len(domain_data) - winner_emb
        
        print(f"  {domain_name:14s}: R²_both={avg_R2_both:.3f}, "
              f"β_emb={avg_beta_emb:+.3f}, β_sem={avg_beta_sem:+.3f}, "
              f"β_inter={avg_beta_inter:+.3f}")
        print(f"  {'':14s}  ΔR²_emb={avg_delta_emb:.3f}, ΔR²_sem={avg_delta_sem:.3f}, "
              f"winner: emb={winner_emb}, sem={winner_sem}, "
              f"contrast={avg_contrast:+.3f}")
    
    # === 2. 层间演变 ===
    print("\n--- 2. 层间演变 ---")
    for li, layer_idx in enumerate(layer_candidates):
        layer_data = [d for d in summary_data if d["layer"] == layer_idx]
        if not layer_data:
            continue
        
        avg_R2_both = np.mean([d["R2_both"] for d in layer_data])
        avg_beta_emb = np.mean([d["beta_emb"] for d in layer_data])
        avg_beta_sem = np.mean([d["beta_sem"] for d in layer_data])
        avg_contrast = np.mean([d["contrast_index"] for d in layer_data])
        
        print(f"  L{layer_idx:2d}: R²_both={avg_R2_both:.3f}, "
              f"β_emb={avg_beta_emb:+.3f}, β_sem={avg_beta_sem:+.3f}, "
              f"contrast={avg_contrast:+.3f}")
    
    # === 3. 全局统计 ===
    print("\n--- 3. 全局统计 ---")
    avg_R2_both = np.mean([d["R2_both"] for d in summary_data])
    avg_R2_full = np.mean([d["R2_full"] for d in summary_data])
    avg_beta_emb = np.mean([d["beta_emb"] for d in summary_data])
    avg_beta_sem = np.mean([d["beta_sem"] for d in summary_data])
    avg_beta_inter = np.mean([d["beta_inter"] for d in summary_data])
    avg_delta_emb = np.mean([d["delta_R2_emb"] for d in summary_data])
    avg_delta_sem = np.mean([d["delta_R2_sem"] for d in summary_data])
    avg_delta_inter = np.mean([d["delta_R2_inter"] for d in summary_data])
    avg_contrast = np.mean([d["contrast_index"] for d in summary_data])
    
    # β_emb和β_sem的符号统计
    beta_emb_pos = sum(1 for d in summary_data if d["beta_emb"] > 0)
    beta_sem_pos = sum(1 for d in summary_data if d["beta_sem"] > 0)
    n_total = len(summary_data)
    
    print(f"  R²_both均值: {avg_R2_both:.3f}")
    print(f"  R²_full均值: {avg_R2_full:.3f}")
    print(f"  β_emb均值: {avg_beta_emb:+.3f} (正: {beta_emb_pos}/{n_total}={100*beta_emb_pos/n_total:.0f}%)")
    print(f"  β_sem均值: {avg_beta_sem:+.3f} (正: {beta_sem_pos}/{n_total}={100*beta_sem_pos/n_total:.0f}%)")
    print(f"  β_inter均值: {avg_beta_inter:+.3f}")
    print(f"  ΔR²_emb均值: {avg_delta_emb:.3f}")
    print(f"  ΔR²_sem均值: {avg_delta_sem:.3f}")
    print(f"  ΔR²_inter均值: {avg_delta_inter:.3f}")
    print(f"  对比指标均值: {avg_contrast:+.3f}")
    
    # === 4. embedding胜出 vs semantic胜出 ===
    print("\n--- 4. Embedding vs Semantic 胜出统计 ---")
    emb_wins = sum(1 for d in summary_data if d["delta_R2_emb"] > d["delta_R2_sem"])
    sem_wins = n_total - emb_wins
    print(f"  ΔR²_emb > ΔR²_sem: {emb_wins}/{n_total} ({100*emb_wins/n_total:.0f}%)")
    print(f"  ΔR²_sem ≥ ΔR²_emb: {sem_wins}/{n_total} ({100*sem_wins/n_total:.0f}%)")
    
    # === 5. 对比指数分析 ===
    print("\n--- 5. 对比指数分析 ---")
    contrast_pos = sum(1 for d in summary_data if d["contrast_index"] > 0)
    contrast_neg = sum(1 for d in summary_data if d["contrast_index"] < 0)
    print(f"  对比指数>0(增强对比): {contrast_pos}/{n_total} ({100*contrast_pos/n_total:.0f}%)")
    print(f"  对比指数<0(压缩对比): {contrast_neg}/{n_total} ({100*contrast_neg/n_total:.0f}%)")
    
    # 按领域分组
    for domain_name in DOMAINS.keys():
        domain_data = [d for d in summary_data if d["domain"] == domain_name]
        if not domain_data:
            continue
        domain_contrast = np.mean([d["contrast_index"] for d in domain_data])
        domain_pos = sum(1 for d in domain_data if d["contrast_index"] > 0)
        print(f"  {domain_name:14s}: mean={domain_contrast:+.3f}, positive={domain_pos}/{len(domain_data)}")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "layers_tested": layer_candidates,
        "summary_data": summary_data,
    }
    
    out_path = TEMP / f"cciii_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    # 释放模型
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
