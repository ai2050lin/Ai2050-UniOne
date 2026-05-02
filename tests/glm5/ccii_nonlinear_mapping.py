"""
CCII(352): 非线性映射函数的拟合与验证
=========================================
★★★★★ CCI核心发现:
  增强对比效应: 近的更近, 远的更远
  → d_geo ∝ f(d_emb), f是非线性扩展函数
  → 近对 d_geo < linear_pred (聚合)
  → 远对 d_geo > linear_pred (分离)

★★★★★ 本实验目标:
  1. 收集大量(d_emb, d_geo)对, 绘制散点图
  2. 拟合5种候选函数形式:
     a. Linear:    d_geo = a * d_emb + b
     b. Power:     d_geo = a * d_emb^α + b
     c. Exponential: d_geo = a * exp(β * d_emb) + b
     d. Quadratic: d_geo = a * d_emb² + c * d_emb + d
     e. Log:       d_geo = a * log(d_emb) + b
  3. 用R², AIC, BIC比较拟合优度
  4. 分析α(幂律指数)跨层/跨领域/跨模型的演变
  5. 层间演变: 对比增强从哪层开始出现?

设计要点:
  - 5个领域, 每领域10-12类别 → 每领域45-66对
  - 总对数: 5领域 × ~55对 × 10层 = ~2750数据点/模型
  - 大样本确保拟合稳健

用法:
  python ccii_nonlinear_mapping.py --model qwen3
  python ccii_nonlinear_mapping.py --model glm4
  python ccii_nonlinear_mapping.py --model deepseek7b
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
from scipy.optimize import curve_fit

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 领域定义 — 5个领域, N=10-12类别
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


# ============================================================
# 非线性拟合函数
# ============================================================

def fit_linear(x, y):
    """线性: y = a*x + b"""
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(x, y, 1)  # [b, a]
    y_pred = coeffs[0] + coeffs[1] * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(x)
    k = 2  # parameters
    aic = n * np.log(ss_res / n + 1e-30) + 2 * k
    bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
    return {"model": "linear", "a": coeffs[1], "b": coeffs[0], 
            "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}


def fit_power(x, y):
    """幂律: y = a * x^alpha + b"""
    x_pos = np.maximum(x, 1e-10)
    
    def func(x, a, alpha, b):
        return a * np.power(np.maximum(x, 1e-10), alpha) + b
    
    try:
        popt, _ = curve_fit(func, x_pos, y, p0=[1.0, 1.5, 0.0], maxfev=10000)
        y_pred = func(x_pos, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(x)
        k = 3
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "power", "a": popt[0], "alpha": popt[1], "b": popt[2],
                "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}
    except Exception as e:
        return {"model": "power", "a": 0, "alpha": 0, "b": 0,
                "R2": -999, "AIC": 999, "BIC": 999, "y_pred": np.zeros_like(y), "error": str(e)}


def fit_exponential(x, y):
    """指数: y = a * exp(beta * x) + b"""
    def func(x, a, beta, b):
        return a * np.exp(beta * np.maximum(x, 1e-10)) + b
    
    try:
        popt, _ = curve_fit(func, x, y, p0=[1.0, 1.0, 0.0], maxfev=10000)
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(x)
        k = 3
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "exponential", "a": popt[0], "beta": popt[1], "b": popt[2],
                "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}
    except Exception as e:
        return {"model": "exponential", "a": 0, "beta": 0, "b": 0,
                "R2": -999, "AIC": 999, "BIC": 999, "y_pred": np.zeros_like(y), "error": str(e)}


def fit_quadratic(x, y):
    """二次: y = a * x^2 + c * x + d"""
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(x, y, 2)  # [d, c, a]
    y_pred = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(x)
    k = 3
    aic = n * np.log(ss_res / n + 1e-30) + 2 * k
    bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
    return {"model": "quadratic", "a": coeffs[2], "c": coeffs[1], "d": coeffs[0],
            "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}


def fit_log(x, y):
    """对数: y = a * log(x) + b"""
    x_pos = np.maximum(x, 1e-10)
    
    def func(x, a, b):
        return a * np.log(np.maximum(x, 1e-10)) + b
    
    try:
        popt, _ = curve_fit(func, x_pos, y, p0=[1.0, 0.0], maxfev=10000)
        y_pred = func(x_pos, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(x)
        k = 2
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "log", "a": popt[0], "b": popt[1],
                "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}
    except Exception as e:
        return {"model": "log", "a": 0, "b": 0,
                "R2": -999, "AIC": 999, "BIC": 999, "y_pred": np.zeros_like(y), "error": str(e)}


def fit_sigmoid(x, y):
    """Sigmoid: y = L / (1 + exp(-k*(x - x0))) + b"""
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    
    def func(x, L, k, x0, b):
        return L / (1.0 + np.exp(-k * (x - x0))) + b
    
    try:
        popt, _ = curve_fit(func, x_norm, y, p0=[np.ptp(y), 5.0, 0.5, y.min()], maxfev=10000)
        y_pred = func(x_norm, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(x)
        k_params = 4
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k_params
        bic = n * np.log(ss_res / n + 1e-30) + k_params * np.log(n)
        return {"model": "sigmoid", "L": popt[0], "k": popt[1], "x0": popt[2], "b": popt[3],
                "R2": r2, "AIC": aic, "BIC": bic, "y_pred": y_pred}
    except Exception as e:
        return {"model": "sigmoid", "L": 0, "k": 0, "x0": 0, "b": 0,
                "R2": -999, "AIC": 999, "BIC": 999, "y_pred": np.zeros_like(y), "error": str(e)}


ALL_FIT_FUNCS = {
    "linear": fit_linear,
    "power": fit_power,
    "exponential": fit_exponential,
    "quadratic": fit_quadratic,
    "log": fit_log,
    "sigmoid": fit_sigmoid,
}


# ============================================================
# 主实验
# ============================================================

def run_experiment(model_name):
    """运行单个模型的完整实验"""
    print(f"\n{'='*70}")
    print(f"CCII: 非线性映射函数的拟合与验证 - {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"  模型: {info.model_class}, d_model={info.d_model}, n_layers={n_layers}")
    
    # 选择测试层: 10层密集采样
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
    
    # === 全局数据收集 ===
    all_emb_dists = []  # 所有(d_emb, d_geo)对
    all_geo_dists = []
    all_domain_labels = []
    all_layer_labels = []
    
    # === 每领域×层的结果 ===
    per_domain_layer = {}
    
    for domain_name, domain_def in DOMAINS.items():
        categories = domain_def["categories"]
        cat_names = list(categories.keys())
        N = len(cat_names)
        
        print(f"\n--- 领域: {domain_name} (N={N}) ---")
        
        # 1. 计算embedding距离(固定, 只算一次)
        emb_centers = get_category_centers_embedding(model, tokenizer, categories)
        if len(emb_centers) != N:
            print(f"  WARNING: 只得到{len(emb_centers)}/{N}个类别, 跳过")
            continue
        emb_dist_mat = compute_pairwise_cosine_dist(emb_centers, cat_names)
        upper_idx = np.triu_indices(N, k=1)
        emb_flat = emb_dist_mat[upper_idx]
        
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
            
            # 全局收集
            all_emb_dists.extend(emb_flat.tolist())
            all_geo_dists.extend(geo_flat.tolist())
            all_domain_labels.extend([domain_name] * len(emb_flat))
            all_layer_labels.extend([layer_idx] * len(emb_flat))
            
            # === 对该领域×层拟合所有函数 ===
            fit_results = {}
            for fname, ffunc in ALL_FIT_FUNCS.items():
                result = ffunc(emb_flat, geo_flat)
                fit_results[fname] = result
            
            # 找最优模型(按AIC)
            best_model = min(fit_results.keys(), key=lambda k: fit_results[k]["AIC"])
            best_r2 = fit_results[best_model]["R2"]
            
            domain_results[f"L{layer_idx}"] = {
                "fits": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"} 
                         for k, v in fit_results.items()},
                "best_model": best_model,
                "best_r2": best_r2,
                "n_pairs": len(emb_flat),
            }
            
            # 打印该层结果
            print(f"best={best_model}(R2={best_r2:.3f})", end="")
            if "power" in fit_results and fit_results["power"]["R2"] > -100:
                print(f" α={fit_results['power'].get('alpha', 'N/A')}", end="")
            print()
        
        per_domain_layer[domain_name] = domain_results
    
    # ============================================================
    # 全局拟合: 合并所有数据
    # ============================================================
    print(f"\n{'='*70}")
    print(f"全局拟合 (N={len(all_emb_dists)}对)")
    print(f"{'='*70}")
    
    all_emb = np.array(all_emb_dists)
    all_geo = np.array(all_geo_dists)
    
    global_fits = {}
    for fname, ffunc in ALL_FIT_FUNCS.items():
        result = ffunc(all_emb, all_geo)
        global_fits[fname] = result
        r2 = result["R2"]
        aic = result["AIC"]
        print(f"  {fname:12s}: R2={r2:.4f}, AIC={aic:.1f}", end="")
        if fname == "power" and "alpha" in result:
            print(f", α={result['alpha']:.4f}", end="")
        if fname == "quadratic" and "a" in result:
            print(f", a(quad)={result['a']:.6f}", end="")
        print()
    
    best_global = min(global_fits.keys(), key=lambda k: global_fits[k]["AIC"])
    print(f"\n  → 全局最优模型: {best_global} (R2={global_fits[best_global]['R2']:.4f})")
    
    # ============================================================
    # 每领域单独拟合
    # ============================================================
    print(f"\n{'='*70}")
    print(f"每领域拟合")
    print(f"{'='*70}")
    
    per_domain_fits = {}
    for domain_name in DOMAINS.keys():
        mask = np.array([d == domain_name for d in all_domain_labels])
        if mask.sum() < 10:
            continue
        emb_d = all_emb[mask]
        geo_d = all_geo[mask]
        
        domain_fit_results = {}
        for fname, ffunc in ALL_FIT_FUNCS.items():
            result = ffunc(emb_d, geo_d)
            domain_fit_results[fname] = result
        
        best_dom = min(domain_fit_results.keys(), key=lambda k: domain_fit_results[k]["AIC"])
        per_domain_fits[domain_name] = {
            "best_model": best_dom,
            "best_r2": domain_fit_results[best_dom]["R2"],
            "power_alpha": domain_fit_results.get("power", {}).get("alpha", None),
            "quadratic_a": domain_fit_results.get("quadratic", {}).get("a", None),
            "linear_r2": domain_fit_results.get("linear", {}).get("R2", None),
            "power_r2": domain_fit_results.get("power", {}).get("R2", None),
            "n_pairs": mask.sum(),
        }
        
        palpha = domain_fit_results.get("power", {}).get("alpha", "N/A")
        if isinstance(palpha, float):
            palpha = f"{palpha:.3f}"
        print(f"  {domain_name:14s}: best={best_dom:12s}, R2={domain_fit_results[best_dom]['R2']:.4f}, "
              f"power_α={palpha}, n={mask.sum()}")
    
    # ============================================================
    # 每层单独拟合 — α的层间演变
    # ============================================================
    print(f"\n{'='*70}")
    print(f"每层拟合 — α的层间演变")
    print(f"{'='*70}")
    
    per_layer_fits = {}
    for li, layer_idx in enumerate(layer_candidates):
        mask = np.array([l == layer_idx for l in all_layer_labels])
        if mask.sum() < 10:
            continue
        emb_l = all_emb[mask]
        geo_l = all_geo[mask]
        
        layer_fit_results = {}
        for fname, ffunc in ALL_FIT_FUNCS.items():
            result = ffunc(emb_l, geo_l)
            layer_fit_results[fname] = result
        
        best_lay = min(layer_fit_results.keys(), key=lambda k: layer_fit_results[k]["AIC"])
        palpha = layer_fit_results.get("power", {}).get("alpha", None)
        qa = layer_fit_results.get("quadratic", {}).get("a", None)
        
        per_layer_fits[f"L{layer_idx}"] = {
            "best_model": best_lay,
            "best_r2": layer_fit_results[best_lay]["R2"],
            "power_alpha": palpha,
            "quadratic_a": qa,
            "power_r2": layer_fit_results.get("power", {}).get("R2", None),
            "linear_r2": layer_fit_results.get("linear", {}).get("R2", None),
            "n_pairs": mask.sum(),
        }
        
        pa_str = f"{palpha:.3f}" if isinstance(palpha, float) else "N/A"
        qa_str = f"{qa:.6f}" if isinstance(qa, float) else "N/A"
        print(f"  L{layer_idx:2d}: best={best_lay:12s}, R2={layer_fit_results[best_lay]['R2']:.4f}, "
              f"α={pa_str}, quad_a={qa_str}, n={mask.sum()}")
    
    # ============================================================
    # 残差分析: 非线性是否显著优于线性?
    # ============================================================
    print(f"\n{'='*70}")
    print(f"非线性 vs 线性: 残差分析")
    print(f"{'='*70}")
    
    # 全局
    lin_res = global_fits.get("linear", {})
    pow_res = global_fits.get("power", {})
    quad_res = global_fits.get("quadratic", {})
    
    if lin_res.get("R2", -999) > -100 and pow_res.get("R2", -999) > -100:
        delta_r2 = pow_res["R2"] - lin_res["R2"]
        print(f"  全局: Power R2 - Linear R2 = {delta_r2:.4f}")
        if delta_r2 > 0.01:
            print(f"  → Power显著优于Linear! (ΔR2={delta_r2:.4f})")
        elif delta_r2 > 0.005:
            print(f"  → Power略优于Linear (ΔR2={delta_r2:.4f})")
        else:
            print(f"  → Power与Linear无显著差异 (ΔR2={delta_r2:.4f})")
    
    if lin_res.get("R2", -999) > -100 and quad_res.get("R2", -999) > -100:
        delta_r2 = quad_res["R2"] - lin_res["R2"]
        print(f"  全局: Quadratic R2 - Linear R2 = {delta_r2:.4f}")
        if delta_r2 > 0.01:
            print(f"  → Quadratic显著优于Linear! (ΔR2={delta_r2:.4f})")
    
    # ============================================================
    # 增强/压缩分析: 小d_emb vs 大d_emb
    # ============================================================
    print(f"\n{'='*70}")
    print(f"增强/压缩分析: 小d_emb vs 大d_emb")
    print(f"{'='*70}")
    
    if len(all_emb) > 0:
        q25 = np.percentile(all_emb, 25)
        q75 = np.percentile(all_emb, 75)
        
        small_mask = all_emb <= q25
        large_mask = all_emb >= q75
        
        # 线性预测
        lin_pred = lin_res.get("y_pred", None) if lin_res.get("R2", -999) > -100 else None
        
        if lin_pred is not None:
            residual = all_geo - lin_pred
            small_res = residual[small_mask].mean() if small_mask.sum() > 0 else 0
            large_res = residual[large_mask].mean() if large_mask.sum() > 0 else 0
            
            print(f"  d_emb ≤ Q25({q25:.4f}): 平均残差 = {small_res:.4f} ({'聚合' if small_res < 0 else '扩展'})")
            print(f"  d_emb ≥ Q75({q75:.4f}): 平均残差 = {large_res:.4f} ({'聚合' if large_res < 0 else '扩展'})")
            print(f"  差值 = {large_res - small_res:.4f} ({'增强对比' if large_res > small_res else '压缩对比'})")
    
    # ============================================================
    # 每领域×层的α汇总
    # ============================================================
    print(f"\n{'='*70}")
    print(f"每领域×层的幂律指数α汇总")
    print(f"{'='*70}")
    
    alpha_table = {}
    for domain_name, domain_results in per_domain_layer.items():
        alpha_row = {}
        for layer_key, layer_data in domain_results.items():
            pfit = layer_data["fits"].get("power", {})
            alpha_val = pfit.get("alpha", None)
            alpha_row[layer_key] = alpha_val
        alpha_table[domain_name] = alpha_row
        
        # 打印
        alpha_strs = []
        for lk, av in alpha_row.items():
            if isinstance(av, float):
                alpha_strs.append(f"{lk}={av:.2f}")
            else:
                alpha_strs.append(f"{lk}=N/A")
        print(f"  {domain_name:14s}: {', '.join(alpha_strs)}")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "layers_tested": layer_candidates,
        "domains_tested": list(DOMAINS.keys()),
        "global_fits": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"} 
                        for k, v in global_fits.items()},
        "best_global_model": best_global,
        "per_domain_fits": per_domain_fits,
        "per_layer_fits": per_layer_fits,
        "per_domain_layer": per_domain_layer,
        "alpha_table": {d: {k: v for k, v in row.items() if isinstance(v, (int, float, type(None)))} 
                        for d, row in alpha_table.items()},
    }
    
    out_path = TEMP / f"ccii_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    
    return output


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    result = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
