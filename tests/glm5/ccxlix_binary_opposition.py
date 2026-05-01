"""
CCXLIX(349): 二元对立与对比组织假说的系统验证
==============================================
★★★★★ CCXLVIII核心发现:
  情感领域: 几何-语义负相关 → 语义相反的类别几何更近(对比组织)
  动物/颜色领域: 几何-语义正相关 → 语义相似的类别几何更近(相似组织)

★★★★★ 本实验核心假设:
  "对比组织"不是情感领域特有的, 而是所有含二元对立的领域的共性!
  
  预测:
  - 含二元对立的领域(emotion/evaluation/temperature) → 负相关
  - 不含二元对立的领域(animal/food/material) → 正相关

★★★★★ 方法论:
  实验1: 6个领域的几何-语义相关性
    - 3个对立领域 + 3个非对立领域
    - 每个领域N=6类别, 10词/类
    - 计算Pearson/Spearman相关

  实验2: 反义词接近测试(Antonym Proximity Test)
    - 对每个领域, 识别"语义最远对"(即反义词对)
    - 检查该对的几何距离是否低于中位数
    - 对立领域: 反义词对应该几何最近
    - 非对立领域: 语义最远对也应该几何最远

  实验3: 跨模型一致性
    - Qwen3/GLM4/DS7B三个模型
    - 检查相关性符号是否跨模型一致

  实验4: 统计显著性增强
    - 汇总所有领域的所有pair对
    - 按领域类型(对立/非对立)分组
    - 计算组内平均相关性

用法:
  python ccxlix_binary_opposition.py --model qwen3
  python ccxlix_binary_opposition.py --model glm4
  python ccxlix_binary_opposition.py --model deepseek7b
"""

import argparse, os, sys, json, gc, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# ============================================================
# 语义类别定义 — 6个领域
# ============================================================

# --- 对立领域 (predicted: negative geo-sem correlation) ---

EMOTION_6 = {
    "classes": {
        "happy":   ["joy", "delight", "bliss", "glee", "cheer", "elation",
                    "contentment", "pleasure", "gladness", "merriment"],
        "sad":     ["sorrow", "grief", "melancholy", "despair", "gloom", "dismay",
                    "woe", "anguish", "heartache", "mourning"],
        "angry":   ["fury", "rage", "wrath", "ire", "outrage", "hostility",
                    "indignation", "animosity", "vexation", "exasperation"],
        "scared":  ["fear", "terror", "dread", "panic", "fright", "horror",
                    "anxiety", "apprehension", "trepidation", "phobia"],
        "surprise":["astonishment", "amazement", "wonder", "shock", "stunned",
                    "bewilderment", "awe", "disbelief", "startlement", "dumbfounded"],
        "disgust": ["revulsion", "repugnance", "loathing", "abhorrence", "nausea",
                    "aversion", "distaste", "repulsion", "contempt", "antipathy"],
    },
    "order": ["happy", "sad", "angry", "scared", "surprise", "disgust"],
    "prompt": "The person felt {word} about the",
    "domain": "emotion",
    "domain_type": "oppositional",
    "semantics": {
        "happy":   {"valence": +1, "arousal": +1},
        "sad":     {"valence": -1, "arousal": -1},
        "angry":   {"valence": -1, "arousal": +1},
        "scared":  {"valence": -1, "arousal":  0},
        "surprise":{"valence":  0, "arousal": +1},
        "disgust": {"valence": -1, "arousal":  0},
    },
    "antonym_pairs": [("happy", "sad"), ("angry", "scared")],
}

EVALUATION_6 = {
    "classes": {
        "excellent":  ["superb", "outstanding", "magnificent", "splendid", "brilliant",
                       "remarkable", "exceptional", "marvelous", "extraordinary", "phenomenal"],
        "terrible":   ["awful", "horrible", "dreadful", "atrocious", "appalling",
                       "abysmal", "pathetic", "wretched", "vile", "horrendous"],
        "good":       ["fine", "nice", "pleasant", "decent", "worthy",
                       "admirable", "favorable", "positive", "satisfying", "commendable"],
        "bad":        ["poor", "inferior", "inadequate", "deficient", "subpar",
                       "lacking", "mediocre", "unsatisfactory", "shoddy", "defective"],
        "amazing":    ["incredible", "fantastic", "wonderful", "astonishing", "astounding",
                       "spectacular", "miraculous", "fabulous", "stupendous", "breathtaking"],
        "horrific":   ["ghastly", "nightmarish", "hideous", "monstrous", "repulsive",
                       "nauseating", "abominable", "heinous", "gruesome", "macabre"],
    },
    "order": ["excellent", "terrible", "good", "bad", "amazing", "horrific"],
    "prompt": "The result was {word} and",
    "domain": "evaluation",
    "domain_type": "oppositional",
    "semantics": {
        "excellent": {"valence": +1, "intensity": +2},
        "terrible":  {"valence": -1, "intensity": +2},
        "good":      {"valence": +1, "intensity":  0},
        "bad":       {"valence": -1, "intensity":  0},
        "amazing":   {"valence": +1, "intensity": +1},
        "horrific":  {"valence": -1, "intensity": +1},
    },
    "antonym_pairs": [("excellent", "terrible"), ("good", "bad"), ("amazing", "horrific")],
}

TEMPERATURE_6 = {
    "classes": {
        "hot":      ["scorching", "sweltering", "blazing", "searing", "burning",
                     "torrid", "sizzling", "fiery", "roasting", "boiling"],
        "cold":     ["freezing", "frigid", "icy", "frosty", "arctic",
                     "glacial", "wintry", "subzero", "gelid", "polar"],
        "warm":     ["balmy", "cozy", "temperate", "genial", "mild",
                     "pleasant", "comfortable", "agreeable", "sunlit", "snug"],
        "cool":     ["refreshing", "brisk", "crisp", "breezy", "fresh",
                     "invigorating", "airy", "moderate", "soothing", "calm"],
        "boiling":  ["ebullient", "simmering", "steaming", "volcanic", "incandescent",
                     "heated", "molten", "bubbling", "seething", "fermenting"],
        "freezing": ["cryogenic", "permafrost", "frozen", "icebound", "frostbitten",
                     "numbing", "bone_chilling", "siberian", "hypothermic", "frosty"],
    },
    "order": ["hot", "cold", "warm", "cool", "boiling", "freezing"],
    "prompt": "The weather was {word} today",
    "domain": "temperature",
    "domain_type": "oppositional",
    "semantics": {
        "hot":      {"temperature": +1, "intensity": +1},
        "cold":     {"temperature": -1, "intensity": +1},
        "warm":     {"temperature": +1, "intensity":  0},
        "cool":     {"temperature": -1, "intensity":  0},
        "boiling":  {"temperature": +1, "intensity": +2},
        "freezing": {"temperature": -1, "intensity": +2},
    },
    "antonym_pairs": [("hot", "cold"), ("warm", "cool"), ("boiling", "freezing")],
}

# --- 非对立领域 (predicted: positive geo-sem correlation) ---

ANIMAL_6 = {
    "classes": {
        "mammal":    ["dog", "cat", "horse", "elephant", "whale", "lion",
                      "tiger", "bear", "wolf", "deer"],
        "bird":      ["eagle", "hawk", "sparrow", "robin", "owl", "parrot",
                      "crow", "pigeon", "falcon", "stork"],
        "fish":      ["salmon", "trout", "shark", "tuna", "bass", "cod",
                      "carp", "perch", "eel", "herring"],
        "insect":    ["ant", "bee", "butterfly", "spider", "fly", "mosquito",
                      "beetle", "wasp", "moth", "grasshopper"],
        "reptile":   ["snake", "lizard", "crocodile", "turtle", "tortoise", "gecko",
                      "iguana", "chameleon", "cobra", "alligator"],
        "amphibian": ["frog", "toad", "salamander", "newt", "tadpole", "caecilian",
                      "axolotl", "treefrog", "bullfrog", "toadlet"],
    },
    "order": ["mammal", "bird", "fish", "insect", "reptile", "amphibian"],
    "prompt": "The {word} was in the",
    "domain": "animal",
    "domain_type": "non_oppositional",
    "semantics": {
        "mammal":    {"vertebrate": +1, "aquatic":  0},
        "bird":      {"vertebrate": +1, "aquatic":  0},
        "fish":      {"vertebrate": +1, "aquatic": +1},
        "insect":    {"vertebrate": -1, "aquatic":  0},
        "reptile":   {"vertebrate": +1, "aquatic":  0},
        "amphibian": {"vertebrate": +1, "aquatic": +1},
    },
    "antonym_pairs": [],  # no clear binary oppositions
}

FOOD_6 = {
    "classes": {
        "fruit":    ["apple", "orange", "banana", "grape", "mango", "peach",
                     "pear", "cherry", "plum", "berry"],
        "vegetable":["carrot", "broccoli", "spinach", "potato", "onion",
                     "pepper", "celery", "lettuce", "tomato", "cucumber"],
        "meat":     ["beef", "pork", "chicken", "lamb", "veal",
                     "bacon", "ham", "steak", "poultry", "mutton"],
        "grain":    ["rice", "wheat", "corn", "oats", "barley",
                     "rye", "millet", "quinoa", "cereal", "flour"],
        "dairy":    ["milk", "cheese", "butter", "cream", "yogurt",
                     "whey", "curd", "ghee", "custard", "brie"],
        "seafood":  ["salmon", "shrimp", "tuna", "crab", "lobster",
                     "oyster", "cod", "clam", "squid", "anchovy"],
    },
    "order": ["fruit", "vegetable", "meat", "grain", "dairy", "seafood"],
    "prompt": "The {word} was fresh and",
    "domain": "food",
    "domain_type": "non_oppositional",
    "semantics": {
        "fruit":    {"plant_based": +1, "sweet": +1},
        "vegetable":{"plant_based": +1, "sweet": -1},
        "meat":     {"plant_based": -1, "sweet": -1},
        "grain":    {"plant_based": +1, "sweet":  0},
        "dairy":    {"plant_based": -1, "sweet":  0},
        "seafood":  {"plant_based": -1, "sweet": -1},
    },
    "antonym_pairs": [],
}

MATERIAL_6 = {
    "classes": {
        "wood":   ["oak", "pine", "cedar", "maple", "birch",
                   "walnut", "mahogany", "teak", "ash", "elm"],
        "metal":  ["iron", "steel", "copper", "aluminum", "brass",
                   "bronze", "titanium", "zinc", "silver", "nickel"],
        "stone":  ["granite", "marble", "limestone", "sandstone", "slate",
                   "quartz", "basalt", "obsidian", "jade", "onyx"],
        "fabric": ["cotton", "silk", "wool", "linen", "velvet",
                   "denim", "satin", "canvas", "tweed", "lace"],
        "glass":  ["crystal", "porcelain", "ceramic", "frosted", "stained",
                   "mirrored", "glazed", "translucent", "transparent", "opaque"],
        "plastic":["nylon", "acrylic", "vinyl", "polymer", "resin",
                   "silicone", "polystyrene", "fiberglass", "polyester", "composite"],
    },
    "order": ["wood", "metal", "stone", "fabric", "glass", "plastic"],
    "prompt": "The object was made of {word} and",
    "domain": "material",
    "domain_type": "non_oppositional",
    "semantics": {
        "wood":   {"natural": +1, "flexible":  0},
        "metal":  {"natural": +1, "flexible": -1},
        "stone":  {"natural": +1, "flexible": -1},
        "fabric": {"natural": +1, "flexible": +1},
        "glass":  {"natural": +1, "flexible": -1},
        "plastic":{"natural": -1, "flexible":  0},
    },
    "antonym_pairs": [],
}

CATEGORY_SETS = {
    "emotion": EMOTION_6,
    "evaluation": EVALUATION_6,
    "temperature": TEMPERATURE_6,
    "animal": ANIMAL_6,
    "food": FOOD_6,
    "material": MATERIAL_6,
}

# ============================================================
# 核心算法
# ============================================================

def construct_regular_simplex(N, scale=1.0):
    if N <= 1: return np.array([[0.0]])
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


def collect_class_centers(model, tokenizer, device, cat_set, layer_idx, n_words=10):
    """收集指定层的类别中心"""
    layers = get_layers(model)
    centers = {}
    for cls_name in cat_set["order"]:
        words = cat_set["classes"][cls_name][:n_words]
        residuals = []
        for word in words:
            prompt = cat_set["prompt"].format(word=word)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
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
                    _ = model(toks.input_ids)
                except Exception:
                    pass
            h.remove()
            if f"L{layer_idx}" in captured:
                res = captured[f"L{layer_idx}"][0, -1, :].numpy()
                residuals.append(res)
        if len(residuals) > 0:
            centers[cls_name] = np.mean(residuals, axis=0)
    return centers


def compute_geo_sem_correlation(centers, class_order, semantics):
    """
    计算几何距离与语义距离的相关性
    
    Returns:
        dict with pearson_r, spearman_r, geo_dists, sem_dists, pairs
    """
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    U, S, Vt = svd(centered, full_matrices=False)
    D = N - 1
    proj = centered @ Vt[:D].T
    
    # 几何距离
    geo_dists = []
    sem_dists = []
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            ci, cj = class_order[i], class_order[j]
            gd = float(np.linalg.norm(proj[i] - proj[j]))
            vi = np.array(list(semantics[ci].values()), dtype=float)
            vj = np.array(list(semantics[cj].values()), dtype=float)
            sd = float(np.linalg.norm(vi - vj))
            geo_dists.append(gd)
            sem_dists.append(sd)
            pairs.append((ci, cj))
    
    geo_dists = np.array(geo_dists)
    sem_dists = np.array(sem_dists)
    
    # 归一化
    geo_norm = geo_dists / (np.mean(geo_dists) + 1e-10)
    sem_norm = sem_dists / (np.mean(sem_dists) + 1e-10)
    
    pr, pp = pearsonr(geo_norm, sem_norm)
    sr, sp = spearmanr(geo_norm, sem_norm)
    
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "geo_dists": geo_dists.tolist(),
        "sem_dists": sem_dists.tolist(),
        "geo_norm": geo_norm.tolist(),
        "sem_norm": sem_norm.tolist(),
        "pairs": pairs,
        "proj_centers": proj,
    }


def compute_antonym_proximity_test(centers, class_order, semantics, antonym_pairs):
    """
    反义词接近测试: 反义词对是否几何更近?
    
    Returns:
        dict with antonym_geo_dists, non_antonym_geo_dists, ratio, etc.
    """
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    
    U, S, Vt = svd(centered, full_matrices=False)
    D = N - 1
    proj = centered @ Vt[:D].T
    
    # 所有pairwise距离
    all_dists = {}
    for i in range(N):
        for j in range(i+1, N):
            ci, cj = class_order[i], class_order[j]
            gd = float(np.linalg.norm(proj[i] - proj[j]))
            all_dists[(ci, cj)] = gd
    
    mean_dist = np.mean(list(all_dists.values()))
    
    # 反义词对的距离
    antonym_dists = []
    for (a, b) in antonym_pairs:
        key = (a, b) if (a, b) in all_dists else (b, a)
        if key in all_dists:
            antonym_dists.append(all_dists[key] / mean_dist)
    
    # 非反义词对的距离
    antonym_set = set()
    for (a, b) in antonym_pairs:
        antonym_set.add((a, b))
        antonym_set.add((b, a))
    
    non_antonym_dists = []
    for (a, b), d in all_dists.items():
        if (a, b) not in antonym_set:
            non_antonym_dists.append(d / mean_dist)
    
    # 语义最远对
    sem_dists_all = {}
    for i in range(N):
        for j in range(i+1, N):
            ci, cj = class_order[i], class_order[j]
            vi = np.array(list(semantics[ci].values()), dtype=float)
            vj = np.array(list(semantics[cj].values()), dtype=float)
            sd = float(np.linalg.norm(vi - vj))
            sem_dists_all[(ci, cj)] = sd
    
    max_sem_pair = max(sem_dists_all, key=sem_dists_all.get)
    max_sem_geo = all_dists.get(max_sem_pair, all_dists.get((max_sem_pair[1], max_sem_pair[0]), 0)) / mean_dist
    
    # 语义最近对
    min_sem_pair = min(sem_dists_all, key=sem_dists_all.get)
    min_sem_geo = all_dists.get(min_sem_pair, all_dists.get((min_sem_pair[1], min_sem_pair[0]), 0)) / mean_dist
    
    # 中位数距离
    median_dist = np.median(list(all_dists.values())) / mean_dist
    
    return {
        "antonym_dists": antonym_dists,
        "antonym_mean": float(np.mean(antonym_dists)) if antonym_dists else None,
        "non_antonym_dists": non_antonym_dists,
        "non_antonym_mean": float(np.mean(non_antonym_dists)) if non_antonym_dists else None,
        "ratio": float(np.mean(antonym_dists) / np.mean(non_antonym_dists)) if antonym_dists and non_antonym_dists else None,
        "max_sem_pair": max_sem_pair,
        "max_sem_geo": float(max_sem_geo),
        "min_sem_pair": min_sem_pair,
        "min_sem_geo": float(min_sem_geo),
        "median_geo_dist": float(median_dist),
        "max_sem_below_median": max_sem_geo < median_dist,
    }


def compute_edge_cv(centers, class_order):
    """计算edge_cv"""
    N = len(class_order)
    center_mat = np.array([centers[c] for c in class_order])
    global_center = np.mean(center_mat, axis=0)
    centered = center_mat - global_center
    U, S, Vt = svd(centered, full_matrices=False)
    D = N - 1
    proj = centered @ Vt[:D].T
    dists = pdist(proj)
    cv = float(np.std(dists) / (np.mean(dists) + 1e-10))
    return cv


def run_experiment(model_name):
    """运行完整实验"""
    print(f"\n{'='*70}")
    print(f"CCXLIX: 二元对立与对比组织假说的系统验证 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    print(f"  模型: {model_info.model_class}, {n_layers}层, d={model_info.d_model}")
    
    best_layers = {
        "qwen3": min(25, n_layers - 1),
        "glm4": min(36, n_layers - 1),
        "deepseek7b": min(2, n_layers - 1),
    }
    best_layer = best_layers.get(model_name, n_layers * 2 // 3)
    
    all_results = {}
    
    for cat_name, cat_set in CATEGORY_SETS.items():
        class_order = cat_set["order"]
        N = len(class_order)
        domain = cat_set["domain"]
        domain_type = cat_set["domain_type"]
        semantics = cat_set["semantics"]
        antonym_pairs = cat_set.get("antonym_pairs", [])
        
        print(f"\n--- {cat_name} (N={N}, type={domain_type}) ---")
        
        centers = collect_class_centers(model, tokenizer, device, cat_set, best_layer, n_words=10)
        
        if len(centers) < N:
            print(f"  ✗ 只收集到{len(centers)}/{N}个类别, 跳过")
            continue
        
        # 实验1: 几何-语义相关性
        corr_result = compute_geo_sem_correlation(centers, class_order, semantics)
        print(f"  几何-语义相关性:")
        print(f"    Pearson  r={corr_result['pearson_r']:.3f} p={corr_result['pearson_p']:.3f}")
        print(f"    Spearman r={corr_result['spearman_r']:.3f} p={corr_result['spearman_p']:.3f}")
        direction = "对比组织(负相关)" if corr_result['pearson_r'] < 0 else "相似组织(正相关)"
        print(f"    方向: {direction}")
        
        # edge_cv
        cv = compute_edge_cv(centers, class_order)
        print(f"    edge_cv = {cv:.4f}")
        
        # 距离排序
        print(f"    距离排序(几何归一化):")
        paired = list(zip(corr_result['pairs'], corr_result['geo_norm'], corr_result['sem_norm']))
        paired.sort(key=lambda x: x[1])
        for (ci, cj), gn, sn in paired:
            marker = " ← 反义" if (ci, cj) in [(a,b) for a,b in antonym_pairs] or (cj, ci) in [(a,b) for a,b in antonym_pairs] else ""
            print(f"      {ci:>10}-{cj:<10}: geo={gn:.3f}×  sem={sn:.3f}×{marker}")
        
        # 实验2: 反义词接近测试
        if antonym_pairs:
            apt = compute_antonym_proximity_test(centers, class_order, semantics, antonym_pairs)
            print(f"  反义词接近测试:")
            print(f"    反义词对平均距离: {apt['antonym_mean']:.3f}×")
            print(f"    非反义词对平均距离: {apt['non_antonym_mean']:.3f}×")
            print(f"    比值: {apt['ratio']:.3f}")
            print(f"    语义最远对: {apt['max_sem_pair']} → 几何={apt['max_sem_geo']:.3f}× {'★低于中位数!' if apt['max_sem_below_median'] else '高于中位数'}")
            print(f"    语义最近对: {apt['min_sem_pair']} → 几何={apt['min_sem_geo']:.3f}×")
        else:
            apt = compute_antonym_proximity_test(centers, class_order, semantics, [])
            print(f"  无反义词对 — 跳过反义词测试")
            print(f"    语义最远对: {apt['max_sem_pair']} → 几何={apt['max_sem_geo']:.3f}× {'★低于中位数!' if apt.get('max_sem_below_median') else '高于中位数'}")
            print(f"    语义最近对: {apt['min_sem_pair']} → 几何={apt['min_sem_geo']:.3f}×")
        
        # 存储
        all_results[cat_name] = {
            "domain_type": domain_type,
            "pearson_r": corr_result['pearson_r'],
            "spearman_r": corr_result['spearman_r'],
            "pearson_p": corr_result['pearson_p'],
            "spearman_p": corr_result['spearman_p'],
            "edge_cv": cv,
            "antonym_ratio": apt.get('ratio'),
            "max_sem_pair": apt.get('max_sem_pair'),
            "max_sem_geo": apt.get('max_sem_geo'),
            "max_sem_below_median": apt.get('max_sem_below_median'),
            "layer": best_layer,
        }
    
    # ============================================================
    # 实验4: 汇总分析
    # ============================================================
    print(f"\n{'='*70}")
    print(f"汇总分析")
    print(f"{'='*70}")
    
    # 按领域类型分组
    opp_pearson = []
    opp_spearman = []
    non_opp_pearson = []
    non_opp_spearman = []
    
    for cat_name, res in all_results.items():
        if res["domain_type"] == "oppositional":
            opp_pearson.append(res["pearson_r"])
            opp_spearman.append(res["spearman_r"])
        else:
            non_opp_pearson.append(res["pearson_r"])
            non_opp_spearman.append(res["spearman_r"])
    
    print(f"\n  对立领域 (N={len(opp_pearson)}个):")
    print(f"    Pearson  均值={np.mean(opp_pearson):.3f} 范围=[{np.min(opp_pearson):.3f}, {np.max(opp_pearson):.3f}]")
    print(f"    Spearman 均值={np.mean(opp_spearman):.3f} 范围=[{np.min(opp_spearman):.3f}, {np.max(opp_spearman):.3f}]")
    
    print(f"\n  非对立领域 (N={len(non_opp_pearson)}个):")
    print(f"    Pearson  均值={np.mean(non_opp_pearson):.3f} 范围=[{np.min(non_opp_pearson):.3f}, {np.max(non_opp_pearson):.3f}]")
    print(f"    Spearman 均值={np.mean(non_opp_spearman):.3f} 范围=[{np.min(non_opp_spearman):.3f}, {np.max(non_opp_spearman):.3f}]")
    
    # 组间比较
    if len(opp_pearson) >= 2 and len(non_opp_pearson) >= 2:
        # Mann-Whitney U test (小样本更稳健)
        try:
            u_stat, u_p = mannwhitneyu(opp_pearson, non_opp_pearson, alternative='less')
            print(f"\n  组间比较 (Pearson):")
            print(f"    Mann-Whitney U={u_stat:.1f} p={u_p:.3f}")
            print(f"    → 对立领域的相关性显著更{'低' if u_p < 0.05 else '低(不显著)'}")
        except Exception as e:
            print(f"\n  组间比较失败: {e}")
    
    # 逐领域汇总表
    print(f"\n  逐领域汇总:")
    print(f"  {'领域':<14} {'类型':<10} {'Pearson':>8} {'Spearman':>10} {'反义比值':>8} {'最远对<中位':>10}")
    for cat_name, res in all_results.items():
        ant_ratio = f"{res['antonym_ratio']:.3f}" if res['antonym_ratio'] is not None else "N/A"
        below_med = "★是" if res.get('max_sem_below_median') else "否"
        print(f"  {cat_name:<14} {res['domain_type']:<10} {res['pearson_r']:>8.3f} "
              f"{res['spearman_r']:>10.3f} {ant_ratio:>8} {below_med:>10}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    output_path = TEMP / f"ccxlix_opposition_{model_name}.json"
    
    def make_serializable(obj):
        if obj is None: return None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.bool_)): return float(obj) if not isinstance(obj, np.bool_) else bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, bool): return obj
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(x) for x in obj]
        if isinstance(obj, tuple): return str(obj)
        return obj
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable({
            "model": model_name,
            "best_layer": best_layer,
            "results": all_results,
            "summary": {
                "opp_pearson_mean": float(np.mean(opp_pearson)) if opp_pearson else None,
                "non_opp_pearson_mean": float(np.mean(non_opp_pearson)) if non_opp_pearson else None,
                "opp_spearman_mean": float(np.mean(opp_spearman)) if opp_spearman else None,
                "non_opp_spearman_mean": float(np.mean(non_opp_spearman)) if non_opp_spearman else None,
            }
        }), f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    t0 = time.time()
    results = run_experiment(args.model)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
