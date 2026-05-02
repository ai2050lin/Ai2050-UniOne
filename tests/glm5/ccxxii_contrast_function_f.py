"""
CCXXII(370): 增强对比函数f的精确形式
=========================================

★★★★★ 核心问题:
  MEMO中的P0: 破解增强对比函数f
  已知: d_geo ∝ f(d_emb), f是非线性扩展
  未知: f的具体形式

★★★★★ CCII失败原因:
  CCII全局拟合R²≈0.006 — 因为不同层的距离尺度完全不同
  层0: d_geo~1, 层24: d_geo~100, 混合后无规律
  → 必须归一化!

★★★★★ 本实验核心改进:
  1. 每层独立归一化: d_norm = d / median(d)
  2. 研究对比增强比: CER = (d_geo_norm/d_emb_norm)
     CER>1: 对比增强(距离被放大)
     CER<1: 对比压缩(距离被缩小)
  3. 将CER作为d_emb_norm的函数拟合
  4. 跨模型验证: 如果三模型的CER函数相同→语言普遍定律

★★★★★ 实验设计:
  Exp1: 大规模词对距离采集 + 归一化 + CER函数拟合
  Exp2: 层间CER演变 — 对比增强从哪层开始?到哪层最强?
  Exp3: 跨模型CER函数比较 — 是否存在统一的语言数学定律?

★★★★★ 候选CER函数:
  A. CER = 1 (无增强, 线性)
  B. CER = a * d_norm^α + 1 (幂律增强)
  C. CER = a * exp(b * d_norm) + 1 (指数增强)
  D. CER = a * tanh(b * d_norm) + 1 (双曲增强)
  E. CER = a * log(1 + b * d_norm) + 1 (对数增强)

用法:
  python ccxxii_contrast_function_f.py --model qwen3 --exp 1
  python ccxxii_contrast_function_f.py --model qwen3 --exp 2
  python ccxxii_contrast_function_f.py --model qwen3 --exp 3
  python ccxxii_contrast_function_f.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ================================================================
# 大规模词对定义 — 覆盖不同语义距离
# ================================================================
# 每个类别有6个近义词, 共12个类别, 产生66个对
# 5个领域 × ~66对 = ~330个对

WORD_DOMAINS = {
    "animal": {
        "dog":      ["dog", "puppy", "hound", "canine", "pooch", "mutt"],
        "cat":      ["cat", "kitten", "feline", "tomcat", "pussy", "kitty"],
        "bird":     ["bird", "sparrow", "robin", "finch", "wren", "swallow"],
        "fish":     ["fish", "trout", "salmon", "bass", "perch", "cod"],
        "snake":    ["snake", "serpent", "viper", "cobra", "python", "adder"],
        "horse":    ["horse", "stallion", "mare", "pony", "colt", "foal"],
        "cow":      ["cow", "bull", "cattle", "ox", "heifer", "calf"],
        "pig":      ["pig", "swine", "hog", "boar", "sow", "piglet"],
        "sheep":    ["sheep", "lamb", "ewe", "ram", "mutton", "fleece"],
        "monkey":   ["monkey", "ape", "chimpanzee", "gorilla", "baboon", "lemur"],
        "bear":     ["bear", "grizzly", "polar", "cub", "ursa", "bruin"],
        "rabbit":   ["rabbit", "bunny", "hare", "buck", "doe", "leveret"],
    },
    "emotion": {
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
        "bored":    ["bored", "uninterested", "apathetic", "listless", "weary", "indifferent"],
        "confused": ["confused", "perplexed", "bewildered", "puzzled", "baffled", "mystified"],
    },
    "profession": {
        "doctor":   ["doctor", "physician", "surgeon", "medic", "clinician", "healer"],
        "teacher":  ["teacher", "instructor", "educator", "professor", "tutor", "lecturer"],
        "chef":     ["chef", "cook", "culinary", "baker", "caterer", "pastry"],
        "artist":   ["artist", "painter", "sculptor", "illustrator", "designer", "creator"],
        "lawyer":   ["lawyer", "attorney", "barrister", "counsel", "advocate", "solicitor"],
        "farmer":   ["farmer", "agriculturist", "rancher", "grower", "planter", "cultivator"],
        "engineer": ["engineer", "technician", "architect", "builder", "constructor", "designer"],
        "soldier":  ["soldier", "warrior", "trooper", "infantry", "fighter", "military"],
        "scientist":["scientist", "researcher", "scholar", "academic", "experimenter", "investigator"],
        "writer":   ["writer", "author", "novelist", "poet", "scribe", "journalist"],
        "musician": ["musician", "singer", "guitarist", "pianist", "drummer", "vocalist"],
        "driver":   ["driver", "chauffeur", "motorist", "operator", "pilot", "steersman"],
    },
    "nature": {
        "mountain": ["mountain", "peak", "summit", "ridge", "cliff", "bluff"],
        "ocean":    ["ocean", "sea", "water", "wave", "tide", "current"],
        "forest":   ["forest", "woods", "grove", "jungle", "timber", "woodland"],
        "desert":   ["desert", "wasteland", "dunes", "sahara", "arid", "barren"],
        "river":    ["river", "stream", "creek", "brook", "rapids", "waterway"],
        "lake":     ["lake", "pond", "pool", "lagoon", "reservoir", "tarn"],
        "island":   ["island", "isle", "atoll", "archipelago", "cay", "key"],
        "valley":   ["valley", "dale", "glen", "hollow", "basin", "depression"],
        "cave":     ["cave", "cavern", "grotto", "den", "hollow", "cavity"],
        "volcano":  ["volcano", "crater", "lava", "magma", "eruption", "vent"],
        "glacier":  ["glacier", "ice", "frost", "snow", "permafrost", "iceberg"],
        "meadow":   ["meadow", "field", "pasture", "prairie", "grassland", "plain"],
    },
    "object": {
        "car":      ["car", "automobile", "sedan", "vehicle", "coupe", "convertible"],
        "house":    ["house", "home", "dwelling", "residence", "mansion", "cottage"],
        "book":     ["book", "novel", "tome", "volume", "manuscript", "publication"],
        "phone":    ["phone", "telephone", "mobile", "smartphone", "cell", "device"],
        "chair":    ["chair", "seat", "stool", "bench", "sofa", "recliner"],
        "table":    ["table", "desk", "counter", "surface", "board", "stand"],
        "door":     ["door", "portal", "gateway", "entrance", "entry", "hatch"],
        "window":   ["window", "pane", "glass", "skylight", "casement", "aperture"],
        "knife":    ["knife", "blade", "dagger", "scalpel", "cleaver", "cutter"],
        "lamp":     ["lamp", "light", "lantern", "fixture", "bulb", "chandelier"],
        "clock":    ["clock", "watch", "timer", "chronometer", "timepiece", "dial"],
        "mirror":   ["mirror", "glass", "reflector", "looking", "speculum", "polish"],
    },
}


def get_category_representations(model, tokenizer, device, categories, layer_indices, prompt_template="The word is {word}"):
    """在多个层同时收集类别表示, 减少forward次数"""
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    n_layers = len(layers)
    
    cat_reps = {}  # {cat_name: {layer_idx: center_vector}}
    
    for cat_name, words in categories.items():
        all_layer_res = {li: [] for li in layer_indices}
        
        for word in words:
            prompt = prompt_template.format(word=word)
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
                
                hooks = []
                for li in layer_indices:
                    if li < n_layers:
                        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
                
                _ = model(inputs_embeds=inputs_embeds)
                
                for h in hooks:
                    h.remove()
                
                for li in layer_indices:
                    key = f"L{li}"
                    if key in captured:
                        all_layer_res[li].append(captured[key][0, -1, :])
        
        cat_reps[cat_name] = {}
        for li in layer_indices:
            if len(all_layer_res[li]) > 0:
                cat_reps[cat_name][li] = np.mean(all_layer_res[li], axis=0)
    
    return cat_reps


def get_embedding_representations(model, tokenizer, categories):
    """从token embedding层获取类别表示"""
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().float().cpu().numpy()
    
    cat_reps = {}
    for cat_name, words in categories.items():
        embeddings = []
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            word_emb = np.mean(W_E[token_ids], axis=0)
            embeddings.append(word_emb)
        if len(embeddings) > 0:
            cat_reps[cat_name] = np.mean(embeddings, axis=0)
    
    return cat_reps


def compute_cosine_distances(centers, cat_names):
    """计算cosine距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points_norm = points / norms
    cos_sim = points_norm @ points_norm.T
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist


def compute_l2_distances(centers, cat_names):
    """计算L2距离矩阵"""
    points = np.array([centers[name] for name in cat_names])
    dists = squareform(pdist(points, metric='euclidean'))
    return dists


def get_upper_tri_pairs(mat, cat_names):
    """提取上三角元素(不含对角线)"""
    pairs = []
    n = len(cat_names)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((cat_names[i], cat_names[j], mat[i, j]))
    return pairs


# ================================================================
# CER函数拟合
# ================================================================

def fit_cer_power(d_norm, cer):
    """CER = a * d_norm^alpha + 1"""
    def func(x, a, alpha):
        return a * np.power(np.maximum(x, 1e-10), alpha) + 1.0
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.1, 0.5], maxfev=10000,
                           bounds=([-10, -5], [10, 5]))
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 2
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "power", "a": float(popt[0]), "alpha": float(popt[1]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "power", "a": 0, "alpha": 0, "R2": -999, "AIC": 999, "BIC": 999}


def fit_cer_exp(d_norm, cer):
    """CER = a * exp(b * d_norm) + 1"""
    def func(x, a, b):
        return a * np.exp(b * np.maximum(x, 1e-10)) + 1.0
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.01, 1.0], maxfev=10000,
                           bounds=([-10, -5], [10, 5]))
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 2
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "exponential", "a": float(popt[0]), "beta": float(popt[1]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "exponential", "a": 0, "beta": 0, "R2": -999, "AIC": 999, "BIC": 999}


def fit_cer_tanh(d_norm, cer):
    """CER = a * tanh(b * d_norm) + 1"""
    def func(x, a, b):
        return a * np.tanh(b * np.maximum(x, 1e-10)) + 1.0
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.5, 1.0], maxfev=10000,
                           bounds=([-10, -5], [10, 5]))
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 2
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "tanh", "a": float(popt[0]), "b": float(popt[1]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "tanh", "a": 0, "b": 0, "R2": -999, "AIC": 999, "BIC": 999}


def fit_cer_log(d_norm, cer):
    """CER = a * log(1 + b * d_norm) + 1"""
    def func(x, a, b):
        return a * np.log(1 + b * np.maximum(x, 1e-10)) + 1.0
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.1, 1.0], maxfev=10000,
                           bounds=([-10, -5], [10, 5]))
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 2
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "log", "a": float(popt[0]), "b": float(popt[1]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "log", "a": 0, "b": 0, "R2": -999, "AIC": 999, "BIC": 999}


def fit_cer_linear(d_norm, cer):
    """CER = a * d_norm + 1"""
    def func(x, a):
        return a * x + 1.0
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.1], maxfev=10000)
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 1
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "linear", "a": float(popt[0]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "linear", "a": 0, "R2": -999, "AIC": 999, "BIC": 999}


def fit_cer_sigmoid(d_norm, cer):
    """CER = L / (1 + exp(-k*(d_norm - x0))) + b"""
    def func(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b
    try:
        popt, _ = curve_fit(func, d_norm, cer, p0=[0.5, 2.0, 1.0, 0.5], maxfev=10000,
                           bounds=([-5, -10, -5, -5], [5, 10, 5, 5]))
        y_pred = func(d_norm, *popt)
        ss_res = np.sum((cer - y_pred)**2)
        ss_tot = np.sum((cer - np.mean(cer))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
        n = len(d_norm)
        k = 4
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        bic = n * np.log(ss_res / n + 1e-30) + k * np.log(n)
        return {"model": "sigmoid", "L": float(popt[0]), "k": float(popt[1]),
                "x0": float(popt[2]), "b": float(popt[3]),
                "R2": float(r2), "AIC": float(aic), "BIC": float(bic)}
    except:
        return {"model": "sigmoid", "L": 0, "k": 0, "x0": 0, "b": 0,
                "R2": -999, "AIC": 999, "BIC": 999}


ALL_FIT_FUNCS = {
    "linear": fit_cer_linear,
    "power": fit_cer_power,
    "exponential": fit_cer_exp,
    "tanh": fit_cer_tanh,
    "log": fit_cer_log,
    "sigmoid": fit_cer_sigmoid,
}


# ================================================================
# Exp1: 大规模词对距离采集 + CER函数拟合
# ================================================================

def run_exp1(model_name):
    """收集大规模(d_emb, d_geo)数据, 归一化后拟合CER函数"""
    print(f"\n{'='*70}")
    print(f"Exp1: 大规模词对距离采集 + CER函数拟合 ({model_name})")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 选择测试层
    layer_indices = list(range(0, n_layers, max(1, n_layers // 10)))
    if 0 not in layer_indices:
        layer_indices = [0] + layer_indices
    if n_layers - 1 not in layer_indices:
        layer_indices.append(n_layers - 1)
    layer_indices = sorted(set(layer_indices))
    print(f"  Testing layers: {layer_indices}")
    
    results = {"model": model_name, "n_layers": n_layers, "layers_tested": layer_indices}
    
    all_domain_data = {}
    
    for domain_name, categories in WORD_DOMAINS.items():
        print(f"\n  Domain: {domain_name} ({len(categories)} categories)")
        cat_names = list(categories.keys())
        n_cats = len(cat_names)
        n_pairs = n_cats * (n_cats - 1) // 2
        print(f"    {n_cats} categories → {n_pairs} pairs")
        
        # 获取embedding层距离
        emb_centers = get_embedding_representations(model, tokenizer, categories)
        if len(emb_centers) < 2:
            print(f"    Skipping {domain_name}: too few valid categories")
            continue
        valid_cats_emb = [c for c in cat_names if c in emb_centers]
        emb_cos_dist = compute_cosine_distances(emb_centers, valid_cats_emb)
        emb_l2_dist = compute_l2_distances(emb_centers, valid_cats_emb)
        
        # 获取各层残差距离
        layer_centers = {}
        for li in layer_indices:
            reps = get_category_representations(model, tokenizer, device, categories, [li])
            layer_centers[li] = reps
        
        # 收集所有层的距离数据
        domain_data = {
            "categories": valid_cats_emb,
            "n_pairs": len(valid_cats_emb) * (len(valid_cats_emb) - 1) // 2,
            "emb_distances": {},
            "layer_distances": {},
        }
        
        # Embedding距离
        emb_pairs = get_upper_tri_pairs(emb_cos_dist, valid_cats_emb)
        for c1, c2, d in emb_pairs:
            domain_data["emb_distances"][(c1, c2)] = {"cos": float(d)}
        
        emb_l2_pairs = get_upper_tri_pairs(emb_l2_dist, valid_cats_emb)
        for c1, c2, d in emb_l2_pairs:
            if (c1, c2) in domain_data["emb_distances"]:
                domain_data["emb_distances"][(c1, c2)]["l2"] = float(d)
        
        # 各层距离
        for li in layer_indices:
            if li not in layer_centers:
                continue
            lc = layer_centers[li]
            valid_cats_l = [c for c in valid_cats_emb if c in lc and li in lc[c]]
            if len(valid_cats_l) < 2:
                continue
            
            cos_d = compute_cosine_distances({c: lc[c][li] for c in valid_cats_l}, valid_cats_l)
            l2_d = compute_l2_distances({c: lc[c][li] for c in valid_cats_l}, valid_cats_l)
            
            domain_data["layer_distances"][li] = {
                "categories": valid_cats_l,
                "cos_pairs": {},
                "l2_pairs": {},
            }
            
            cos_pairs = get_upper_tri_pairs(cos_d, valid_cats_l)
            l2_pairs = get_upper_tri_pairs(l2_d, valid_cats_l)
            
            for c1, c2, d in cos_pairs:
                domain_data["layer_distances"][li]["cos_pairs"][(c1, c2)] = float(d)
            for c1, c2, d in l2_pairs:
                domain_data["layer_distances"][li]["l2_pairs"][(c1, c2)] = float(d)
        
        all_domain_data[domain_name] = domain_data
    
    # ================================================================
    # 核心分析: 计算CER并拟合
    # ================================================================
    
    print(f"\n{'='*70}")
    print("  Computing CER (Contrast Enhancement Ratio) and fitting functions")
    print(f"{'='*70}")
    
    # 收集所有域的(d_emb_norm, CER)数据, 按层分组
    cer_by_layer_cos = {}  # {layer_idx: [(d_emb_norm, CER), ...]}
    cer_by_layer_l2 = {}
    
    for domain_name, ddata in all_domain_data.items():
        emb_dists = ddata["emb_distances"]
        if len(emb_dists) == 0:
            continue
        
        # Embedding距离归一化
        emb_cos_vals = [v["cos"] for v in emb_dists.values() if "cos" in v]
        emb_l2_vals = [v["l2"] for v in emb_dists.values() if "l2" in v]
        
        if len(emb_cos_vals) < 3:
            continue
        
        emb_cos_median = np.median(emb_cos_vals)
        emb_l2_median = np.median(emb_l2_vals) if len(emb_l2_vals) > 0 else 1.0
        
        for li, ld in ddata["layer_distances"].items():
            cos_pairs = ld["cos_pairs"]
            l2_pairs = ld["l2_pairs"]
            
            if len(cos_pairs) < 3:
                continue
            
            # Layer距离归一化
            cos_vals = list(cos_pairs.values())
            l2_vals = list(l2_pairs.values())
            
            cos_median = np.median(cos_vals)
            l2_median = np.median(l2_vals) if len(l2_vals) > 0 else 1.0
            
            # 计算CER = (d_geo / d_geo_median) / (d_emb / d_emb_median)
            for pair_key, d_geo_cos in cos_pairs.items():
                if pair_key in emb_dists and "cos" in emb_dists[pair_key]:
                    d_emb_cos = emb_dists[pair_key]["cos"]
                    d_emb_norm = d_emb_cos / emb_cos_median if emb_cos_median > 1e-10 else 0
                    d_geo_norm = d_geo_cos / cos_median if cos_median > 1e-10 else 0
                    
                    if d_emb_norm > 1e-6:  # 避免除零
                        cer = d_geo_norm / d_emb_norm
                        if li not in cer_by_layer_cos:
                            cer_by_layer_cos[li] = []
                        cer_by_layer_cos[li].append((d_emb_norm, cer))
            
            for pair_key, d_geo_l2 in l2_pairs.items():
                if pair_key in emb_dists and "l2" in emb_dists[pair_key]:
                    d_emb_l2 = emb_dists[pair_key]["l2"]
                    d_emb_norm = d_emb_l2 / emb_l2_median if emb_l2_median > 1e-10 else 0
                    d_geo_norm = d_geo_l2 / l2_median if l2_median > 1e-10 else 0
                    
                    if d_emb_norm > 1e-6:
                        cer = d_geo_norm / d_emb_norm
                        if li not in cer_by_layer_l2:
                            cer_by_layer_l2[li] = []
                        cer_by_layer_l2[li].append((d_emb_norm, cer))
    
    # ================================================================
    # 拟合CER函数
    # ================================================================
    
    fit_results = {}
    best_models = {}
    
    for dist_type, cer_data in [("cosine", cer_by_layer_cos), ("l2", cer_by_layer_l2)]:
        fit_results[dist_type] = {}
        best_models[dist_type] = {}
        
        for li in sorted(cer_data.keys()):
            data = cer_data[li]
            if len(data) < 10:
                continue
            
            d_norm = np.array([d[0] for d in data])
            cer = np.array([d[1] for d in data])
            
            # 限制CER范围避免异常值
            valid = (cer > 0.1) & (cer < 10) & (d_norm > 0.01) & (d_norm < 10)
            d_norm_v = d_norm[valid]
            cer_v = cer[valid]
            
            if len(d_norm_v) < 10:
                continue
            
            layer_fits = {}
            best_r2 = -999
            best_model_name = "none"
            
            for fname, ffunc in ALL_FIT_FUNCS.items():
                fit = ffunc(d_norm_v, cer_v)
                layer_fits[fname] = fit
                if fit["R2"] > best_r2:
                    best_r2 = fit["R2"]
                    best_model_name = fname
            
            fit_results[dist_type][li] = layer_fits
            best_models[dist_type][li] = best_model_name
            
            # 打印关键结果
            n_data = len(d_norm_v)
            mean_cer = float(np.mean(cer_v))
            median_cer = float(np.median(cer_v))
            cer_near = float(np.median(cer_v[d_norm_v < 1.0])) if np.any(d_norm_v < 1.0) else None
            cer_far = float(np.median(cer_v[d_norm_v > 1.0])) if np.any(d_norm_v > 1.0) else None
            
            print(f"  L{li:2d} ({dist_type}): n={n_data}, mean_CER={mean_cer:.3f}, "
                  f"median_CER={median_cer:.3f}, "
                  f"CER_near={cer_near:.3f}, CER_far={cer_far:.3f}, "
                  f"best={best_model_name}(R²={best_r2:.4f})")
    
    # ================================================================
    # 汇总: 层间CER演变
    # ================================================================
    
    print(f"\n{'='*70}")
    print("  Layer-by-layer CER evolution summary")
    print(f"{'='*70}")
    
    evolution = {"cosine": {}, "l2": {}}
    for dist_type in ["cosine", "l2"]:
        for li in sorted(fit_results[dist_type].keys()):
            fits = fit_results[dist_type][li]
            best_name = best_models[dist_type][li]
            best_fit = fits[best_name]
            
            # 提取关键参数
            cer_data_list = cer_by_layer_cos[li] if dist_type == "cosine" else cer_by_layer_l2[li]
            d_norm = np.array([d[0] for d in cer_data_list])
            cer = np.array([d[1] for d in cer_data_list])
            valid = (cer > 0.1) & (cer < 10) & (d_norm > 0.01) & (d_norm < 10)
            d_norm_v = d_norm[valid]
            cer_v = cer[valid]
            
            cer_near = float(np.median(cer_v[d_norm_v < 1.0])) if np.any(d_norm_v < 1.0) else 0
            cer_far = float(np.median(cer_v[d_norm_v > 1.0])) if np.any(d_norm_v > 1.0) else 0
            
            evolution[dist_type][li] = {
                "n_data": int(len(d_norm_v)),
                "mean_cer": float(np.mean(cer_v)),
                "cer_near": cer_near,
                "cer_far": cer_far,
                "contrast_ratio": cer_far / cer_near if cer_near > 0 else 0,
                "best_model": best_name,
                "best_r2": best_fit["R2"],
                "best_params": {k: v for k, v in best_fit.items() if k not in ["R2", "AIC", "BIC", "model", "y_pred"]},
            }
            
            ev = evolution[dist_type][li]
            print(f"  L{li:2d} ({dist_type}): CER_near={ev['cer_near']:.3f}, "
                  f"CER_far={ev['cer_far']:.3f}, ratio={ev['contrast_ratio']:.3f}, "
                  f"best={best_name}(R²={ev['best_r2']:.4f})")
    
    # 保存结果
    # 将tuple keys转为string
    serializable_results = {
        "model": model_name,
        "n_layers": n_layers,
        "layers_tested": layer_indices,
        "evolution_cosine": {str(k): v for k, v in evolution["cosine"].items()},
        "evolution_l2": {str(k): v for k, v in evolution["l2"].items()},
        "fit_details_cosine": {},
        "fit_details_l2": {},
    }
    
    for dist_type, fit_res in [("cosine", fit_results["cosine"]), ("l2", fit_results["l2"])]:
        key = f"fit_details_{dist_type}"
        for li, layer_fits in fit_res.items():
            serializable_results[key][str(li)] = {}
            for fname, fit in layer_fits.items():
                serializable_results[key][str(li)][fname] = {
                    k: v for k, v in fit.items() if k != "y_pred"
                }
    
    outpath = TEMP / f"ccxxii_{model_name}_exp1.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    release_model(model)
    return serializable_results


# ================================================================
# Exp2: 层间CER演变细粒度分析
# ================================================================

def run_exp2(model_name):
    """细粒度层间CER分析 + 跨域一致性"""
    print(f"\n{'='*70}")
    print(f"Exp2: 层间CER演变细粒度分析 ({model_name})")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 每层都测试(细粒度)
    layer_indices = list(range(n_layers))
    print(f"  Testing ALL {n_layers} layers")
    
    results = {"model": model_name, "n_layers": n_layers}
    
    # 只用cosine距离(更稳定)
    # 对每个域独立分析
    domain_evolution = {}
    
    for domain_name, categories in WORD_DOMAINS.items():
        print(f"\n  Domain: {domain_name}")
        cat_names = list(categories.keys())
        
        # Embedding距离
        emb_centers = get_embedding_representations(model, tokenizer, categories)
        valid_cats = [c for c in cat_names if c in emb_centers]
        if len(valid_cats) < 3:
            continue
        
        emb_cos_dist = compute_cosine_distances(emb_centers, valid_cats)
        emb_cos_median = np.median(emb_cos_dist[emb_cos_dist > 0])
        
        # 各层距离
        domain_evolution[domain_name] = {"layers": {}, "n_categories": len(valid_cats)}
        
        for li in layer_indices:
            reps = get_category_representations(model, tokenizer, device, categories, [li])
            valid_cats_l = [c for c in valid_cats if c in reps and li in reps[c]]
            if len(valid_cats_l) < 3:
                continue
            
            cos_dist = compute_cosine_distances({c: reps[c][li] for c in valid_cats_l}, valid_cats_l)
            cos_median = np.median(cos_dist[cos_dist > 0])
            
            # CER for each pair
            cer_near_list = []
            cer_far_list = []
            
            for i in range(len(valid_cats_l)):
                for j in range(i+1, len(valid_cats_l)):
                    c1, c2 = valid_cats_l[i], valid_cats_l[j]
                    d_emb = emb_cos_dist[valid_cats.index(c1), valid_cats.index(c2)] if c1 in valid_cats and c2 in valid_cats else 0
                    d_geo = cos_dist[i, j]
                    
                    d_emb_norm = d_emb / emb_cos_median if emb_cos_median > 1e-10 else 0
                    d_geo_norm = d_geo / cos_median if cos_median > 1e-10 else 0
                    
                    if d_emb_norm > 1e-6:
                        cer_val = d_geo_norm / d_emb_norm
                        if d_emb_norm < 1.0:
                            cer_near_list.append(cer_val)
                        else:
                            cer_far_list.append(cer_val)
            
            cer_near = float(np.median(cer_near_list)) if cer_near_list else 0
            cer_far = float(np.median(cer_far_list)) if cer_far_list else 0
            
            domain_evolution[domain_name]["layers"][li] = {
                "cer_near": cer_near,
                "cer_far": cer_far,
                "contrast_ratio": cer_far / cer_near if cer_near > 0.01 else 0,
                "n_near": len(cer_near_list),
                "n_far": len(cer_far_list),
            }
        
        # 打印该域的关键层
        last_li = layer_indices[-1]
        if last_li in domain_evolution[domain_name]["layers"]:
            dl = domain_evolution[domain_name]["layers"][last_li]
            first_li = layer_indices[0]
            fl = domain_evolution[domain_name]["layers"].get(first_li, {})
            print(f"    L{first_li}: near={fl.get('cer_near',0):.3f}, far={fl.get('cer_far',0):.3f}")
            print(f"    L{last_li}: near={dl['cer_near']:.3f}, far={dl['cer_far']:.3f}, ratio={dl['contrast_ratio']:.3f}")
    
    # 跨域平均
    avg_evolution = {}
    for li in layer_indices:
        near_vals = []
        far_vals = []
        for domain_name in WORD_DOMAINS:
            if li in domain_evolution.get(domain_name, {}).get("layers", {}):
                dl = domain_evolution[domain_name]["layers"][li]
                if dl["n_near"] > 0:
                    near_vals.append(dl["cer_near"])
                if dl["n_far"] > 0:
                    far_vals.append(dl["cer_far"])
        
        if near_vals and far_vals:
            avg_near = float(np.mean(near_vals))
            avg_far = float(np.mean(far_vals))
            avg_evolution[li] = {
                "avg_cer_near": avg_near,
                "avg_cer_far": avg_far,
                "avg_contrast_ratio": avg_far / avg_near if avg_near > 0.01 else 0,
                "n_domains": len(near_vals),
            }
    
    print(f"\n{'='*70}")
    print("  Cross-domain average CER evolution:")
    print(f"{'='*70}")
    for li in sorted(avg_evolution.keys()):
        ev = avg_evolution[li]
        print(f"  L{li:2d}: near={ev['avg_cer_near']:.3f}, far={ev['avg_cer_far']:.3f}, "
              f"ratio={ev['avg_contrast_ratio']:.3f} ({ev['n_domains']} domains)")
    
    # 保存
    serializable = {
        "model": model_name,
        "n_layers": n_layers,
        "domain_evolution": {
            domain: {
                "n_categories": data["n_categories"],
                "layers": {str(k): v for k, v in data["layers"].items()}
            } for domain, data in domain_evolution.items()
        },
        "avg_evolution": {str(k): v for k, v in avg_evolution.items()},
    }
    
    outpath = TEMP / f"ccxxii_{model_name}_exp2.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    release_model(model)
    return serializable


# ================================================================
# Exp3: 跨模型CER函数比较
# ================================================================

def run_exp3():
    """比较三模型的CER函数, 检验是否存在统一的语言数学定律"""
    print(f"\n{'='*70}")
    print(f"Exp3: 跨模型CER函数比较")
    print(f"{'='*70}")
    
    # 加载三模型的Exp2结果
    all_results = {}
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        path = TEMP / f"ccxxii_{model_name}_exp2.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_results[model_name] = json.load(f)
        else:
            print(f"  WARNING: {model_name} exp2 results not found, running now...")
            all_results[model_name] = run_exp2(model_name)
    
    # 比较跨模型的avg_evolution
    print(f"\n{'='*70}")
    print("  Cross-model CER comparison (average across domains):")
    print(f"{'='*70}")
    
    # 找到共同的层深比例
    comparison = {}
    for model_name, res in all_results.items():
        n_layers = res["n_layers"]
        avg_ev = res.get("avg_evolution", {})
        
        # 归一化层深度: depth = layer / (n_layers - 1)
        for li_str, ev in avg_ev.items():
            li = int(li_str)
            depth = li / (n_layers - 1) if n_layers > 1 else 0
            depth_key = f"{depth:.2f}"
            
            if depth_key not in comparison:
                comparison[depth_key] = {}
            comparison[depth_key][model_name] = ev
    
    # 打印比较表
    print(f"  {'Depth':>6s}", end="")
    for m in all_results:
        print(f"  {m:>12s}_near {m:>12s}_far {m:>12s}_ratio", end="")
    print()
    
    for depth_key in sorted(comparison.keys()):
        d = float(depth_key)
        if d < 0.05 or d > 0.95:
            continue  # 跳过极浅/深层(数据可能不准)
        print(f"  {depth_key:>6s}", end="")
        for m in all_results:
            if m in comparison[depth_key]:
                ev = comparison[depth_key][m]
                print(f"  {ev['avg_cer_near']:>12.3f} {ev['avg_cer_far']:>12.3f} {ev['avg_contrast_ratio']:>12.3f}", end="")
            else:
                print(f"  {'N/A':>12s} {'N/A':>12s} {'N/A':>12s}", end="")
        print()
    
    # 计算跨模型CER相关性
    print(f"\n{'='*70}")
    print("  Cross-model CER correlation:")
    print(f"{'='*70}")
    
    model_names = list(all_results.keys())
    correlations = {}
    
    for mi in range(len(model_names)):
        for mj in range(mi+1, len(model_names)):
            m1, m2 = model_names[mi], model_names[mj]
            n1 = all_results[m1]["n_layers"]
            n2 = all_results[m2]["n_layers"]
            
            # 对齐到相同的归一化深度
            depths = np.arange(0, 1.01, 0.1)
            near1, near2, far1, far2 = [], [], [], []
            
            for d in depths:
                li1 = int(d * (n1 - 1))
                li2 = int(d * (n2 - 1))
                
                ev1 = all_results[m1].get("avg_evolution", {}).get(str(li1))
                ev2 = all_results[m2].get("avg_evolution", {}).get(str(li2))
                
                if ev1 and ev2:
                    near1.append(ev1["avg_cer_near"])
                    near2.append(ev2["avg_cer_near"])
                    far1.append(ev1["avg_cer_far"])
                    far2.append(ev2["avg_cer_far"])
            
            if len(near1) >= 3:
                r_near, p_near = pearsonr(near1, near2)
                r_far, p_far = pearsonr(far1, far2)
                
                pair_key = f"{m1}_vs_{m2}"
                correlations[pair_key] = {
                    "r_near": float(r_near), "p_near": float(p_near),
                    "r_far": float(r_far), "p_far": float(p_far),
                    "n_points": len(near1),
                }
                
                sig_near = "***" if p_near < 0.001 else "**" if p_near < 0.01 else "*" if p_near < 0.05 else ""
                sig_far = "***" if p_far < 0.001 else "**" if p_far < 0.01 else "*" if p_far < 0.05 else ""
                
                print(f"  {m1} vs {m2}:")
                print(f"    CER_near: r={r_near:.4f}, p={p_near:.4f} {sig_near}")
                print(f"    CER_far:  r={r_far:.4f}, p={p_far:.4f} {sig_far}")
    
    # 关键判断: 三模型CER是否一致
    print(f"\n{'='*70}")
    print("  VERDICT: Is CER a universal language law?")
    print(f"{'='*70}")
    
    all_significant = all(
        c["p_near"] < 0.05 and c["p_far"] < 0.05
        for c in correlations.values()
    )
    all_high_r = all(
        c["r_near"] > 0.7 and c["r_far"] > 0.7
        for c in correlations.values()
    )
    
    if all_significant and all_high_r:
        print("  ★★★★★ YES! CER is highly correlated across models!")
        print("  This is strong evidence for a universal language mathematical law.")
    elif all_significant:
        print("  ★★★★ Partial: CER is significantly correlated but R varies.")
        print("  Language structure constrains the direction but not the magnitude.")
    else:
        print("  ★★ NO: CER is NOT consistently correlated across models.")
        print("  The contrast enhancement may be model-specific.")
    
    # 保存
    comparison_results = {
        "correlations": correlations,
        "all_significant": all_significant,
        "all_high_r": all_high_r,
        "verdict": "universal" if (all_significant and all_high_r) else
                   "partial" if all_significant else "model_specific",
    }
    
    outpath = TEMP / "ccxxii_cross_model_comparison.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    return comparison_results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1",
                       choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    
    if args.exp == "1":
        run_exp1(args.model)
    elif args.exp == "2":
        run_exp2(args.model)
    elif args.exp == "3":
        run_exp3()
    elif args.exp == "all":
        # Run exp1 and exp2 on the specified model, then exp3 (cross-model)
        run_exp1(args.model)
        run_exp2(args.model)
        # Check if other models have results
        for other_model in ["qwen3", "glm4", "deepseek7b"]:
            if other_model != args.model:
                path = TEMP / f"ccxxii_{other_model}_exp2.json"
                if not path.exists():
                    print(f"\n*** Running exp2 for {other_model} for cross-model comparison ***")
                    run_exp2(other_model)
        run_exp3()


if __name__ == "__main__":
    main()
