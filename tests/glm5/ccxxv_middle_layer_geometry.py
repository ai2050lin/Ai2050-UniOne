"""
CCXXV(373): 中间层几何学 — 大规模概念扫描
目标: 系统验证 eff_dim = f(n_concepts) 的精确形式

关键改进: 使用句子上下文 "The word is X" 代替单独的词X
  - 单词表示被token embedding主导, 余弦相似度≈1.0
  - 句子上下文让注意力机制工作, 产生有意义的区分度
"""

import sys, os, argparse, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# 246概念, 10个语义类别
CONCEPTS_BY_CATEGORY = {
    "animals": [
        "cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", 
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "mouse",
        "snake", "eagle", "whale", "dolphin", "elephant", "giraffe", "monkey", "penguin",
        "shark", "octopus", "butterfly", "spider", "ant", "bee"
    ],
    "colors": [
        "red", "blue", "green", "yellow", "black", "white", "purple", "orange",
        "pink", "brown", "gray", "gold", "silver", "violet", "crimson", "scarlet",
        "azure", "turquoise", "emerald", "ivory", "coral", "amber", "indigo", "magenta"
    ],
    "emotions": [
        "happy", "sad", "angry", "fear", "love", "hate", "joy", "grief",
        "hope", "despair", "pride", "shame", "guilt", "envy", "jealousy", "curiosity",
        "surprise", "disgust", "contempt", "anxiety", "excitement", "boredom", "nostalgia", "awe"
    ],
    "materials": [
        "wood", "stone", "metal", "glass", "paper", "cloth", "leather", "rubber",
        "plastic", "ceramic", "concrete", "brick", "copper", "iron", "steel", "aluminum",
        "diamond", "marble", "granite", "sand", "clay", "silk", "cotton", "wool"
    ],
    "weather": [
        "rain", "snow", "wind", "storm", "fog", "cloud", "sun", "frost",
        "hail", "thunder", "lightning", "hurricane", "tornado", "drought", "flood", "blizzard",
        "drizzle", "breeze", "monsoon", "sleet", "rainbow", "dew", "mist", "ice"
    ],
    "body_parts": [
        "head", "hand", "foot", "eye", "ear", "nose", "mouth", "heart",
        "brain", "lung", "liver", "bone", "blood", "skin", "muscle", "finger",
        "toe", "knee", "elbow", "shoulder", "wrist", "ankle", "spine", "rib"
    ],
    "foods": [
        "bread", "rice", "meat", "fruit", "vegetable", "cheese", "egg",
        "milk", "water", "wine", "beer", "cake", "cookie", "soup", "salad",
        "pasta", "noodle", "steak", "chicken", "potato", "tomato", "onion", "garlic", "pepper"
    ],
    "tools": [
        "hammer", "saw", "drill", "nail", "screw", "wrench", "pliers", "knife",
        "scissors", "axe", "shovel", "rake", "hoe", "chisel", "file", "clamp",
        "ruler", "compass", "level", "wedge", "lever", "pulley", "gear", "spring"
    ],
    "abstract": [
        "time", "space", "energy", "matter", "force", "motion", "speed", "weight",
        "distance", "direction", "angle", "area", "volume", "density", "pressure", "temperature",
        "frequency", "wavelength", "amplitude", "phase", "field", "wave", "particle", "quantum"
    ],
    "social": [
        "family", "friend", "enemy", "king", "queen", "priest", "soldier", "merchant",
        "teacher", "student", "doctor", "lawyer", "artist", "farmer", "worker", "leader",
        "nation", "city", "village", "army", "church", "school", "market", "court"
    ]
}

ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
for cat, concepts in CONCEPTS_BY_CATEGORY.items():
    for c in concepts:
        ALL_CONCEPTS.append(c)
        CONCEPT_CATEGORIES[c] = cat

print(f"Total concepts: {len(ALL_CONCEPTS)}, Categories: {len(CONCEPTS_BY_CATEGORY)}")


def extract_concept_reps(model_name, concepts, target_layers, model=None, tokenizer=None, device=None):
    """从指定层提取概念表示 (使用句子上下文)"""
    if model is None:
        model, tokenizer, device = load_model(model_name)
        own_model = True
    else:
        own_model = False
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    embed_layer = model.get_input_embeddings()
    
    print(f"Model: {model_name}, layers={n_layers}, d_model={d_model}")
    print(f"Target layers: {target_layers}")
    print(f"Extracting {len(concepts)} concepts with sentence context...")
    
    reps_by_layer = {l: {} for l in target_layers}
    
    for i, concept in enumerate(concepts):
        if i % 50 == 0:
            print(f"  Concept {i}/{len(concepts)}...")
        
        # 关键改进: 使用句子上下文!
        prompt = f"The word is {concept}"
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
            for li in target_layers:
                if li < len(layers_list):
                    hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))
            
            _ = model(inputs_embeds=inputs_embeds)
            
            for h in hooks:
                h.remove()
            
            # 取最后一个token的表示
            for li in target_layers:
                key = f"L{li}"
                if key in captured:
                    reps_by_layer[li][concept] = captured[key][0, -1, :]
    
    if own_model:
        release_model(model)
    print("Model released." if own_model else "Done extracting.")
    return reps_by_layer, n_layers, d_model


def compute_pca_stats(X):
    """计算PCA统计信息"""
    pca = PCA()
    pca.fit(X)
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    eff_dim_90 = int(np.argmax(cum_var >= 0.90) + 1)
    eff_dim_95 = int(np.argmax(cum_var >= 0.95) + 1)
    eff_dim_99 = int(np.argmax(cum_var >= 0.99) + 1)
    
    top1 = pca.explained_variance_ratio_[0]
    top5 = np.sum(pca.explained_variance_ratio_[:5])
    top10 = np.sum(pca.explained_variance_ratio_[:10])
    
    # 幂律指数
    s = pca.singular_values_
    log_s = np.log10(s[s > 0])
    log_k = np.log10(np.arange(1, len(log_s) + 1))
    n_fit = min(len(log_s) // 2, 25)
    if n_fit > 2:
        coeffs = np.polyfit(log_k[:n_fit], log_s[:n_fit], 1)
        alpha = -coeffs[0]
    else:
        alpha = 0
    
    return {
        'eff_dim_90': eff_dim_90,
        'eff_dim_95': eff_dim_95,
        'eff_dim_99': eff_dim_99,
        'top1': float(top1),
        'top5': float(top5),
        'top10': float(top10),
        'alpha': float(alpha),
        'ratio_95': float(eff_dim_95 / X.shape[0]),
        'ratio_90': float(eff_dim_90 / X.shape[0]),
    }


# ================================================================
# Exp1: 大规模概念扫描 — eff_dim = f(n_concepts)
# ================================================================

def run_exp1(model_name):
    print(f'\n{"="*60}')
    print(f'CCXXV Exp1: Large-scale concept scan ({model_name})')
    print(f'{"="*60}')
    
    # 加载模型一次
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    mid_layer = info.n_layers // 2
    
    # 提取所有概念的中间层表示
    reps_by_layer, n_layers, d_model = extract_concept_reps(
        model_name, ALL_CONCEPTS, target_layers=[mid_layer],
        model=model, tokenizer=tokenizer, device=device
    )
    
    # 释放模型
    release_model(model)
    
    concept_reps = reps_by_layer[mid_layer]
    concept_names = list(concept_reps.keys())
    X_all = np.array([concept_reps[c] for c in concept_names])
    print(f"Total extracted: {X_all.shape}")
    
    # 对不同概念数计算eff_dim
    n_list = [10, 20, 30, 50, 75, 100, 150, 200, len(concept_names)]
    results = {}
    
    for n_target in n_list:
        if n_target > len(concept_names):
            continue
        
        # 多次采样取平均
        n_trials = 5 if n_target < 100 else 3
        trial_stats = []
        
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial)
            indices = rng.choice(len(concept_names), n_target, replace=False)
            X_sub = X_all[indices]
            stats = compute_pca_stats(X_sub)
            trial_stats.append(stats)
        
        # 平均
        avg = {}
        std = {}
        for key in trial_stats[0]:
            vals = [s[key] for s in trial_stats]
            if isinstance(vals[0], int):
                avg[key] = int(round(np.mean(vals)))
            else:
                avg[key] = float(np.mean(vals))
            std[key + '_std'] = float(np.std(vals))
        
        results[str(n_target)] = {**avg, **std}
        
        print(f"  n={n_target}: rank95={avg['eff_dim_95']}+-{std['eff_dim_95_std']:.1f}, "
              f"ratio={avg['ratio_95']:.3f}, top1={avg['top1']:.3f}, a={avg['alpha']:.3f}")
    
    # 线性拟合
    n_vals = np.array([int(k) for k in results.keys()])
    rank95_vals = np.array([results[k]['eff_dim_95'] for k in results.keys()])
    
    if len(n_vals) >= 3:
        coeffs = np.polyfit(n_vals, rank95_vals, 1)
        slope, intercept = coeffs
        predicted = np.polyval(coeffs, n_vals)
        ss_res = np.sum((rank95_vals - predicted)**2)
        ss_tot = np.sum((rank95_vals - rank95_vals.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        print(f"\n  Linear fit: rank95 = {slope:.3f} * n + {intercept:.3f}")
        print(f"  R-squared = {r2:.4f}")
        print(f"  Slope (dim/concept at 95%) = {slope:.3f}")
        
        # 也做rank90的拟合
        rank90_vals = np.array([results[k]['eff_dim_90'] for k in results.keys()])
        coeffs90 = np.polyfit(n_vals, rank90_vals, 1)
        print(f"  Linear fit (rank90): rank90 = {coeffs90[0]:.3f} * n + {coeffs90[1]:.3f}")
        
        linear_fit = {
            'slope_95': float(slope), 'intercept_95': float(intercept), 'r2_95': float(r2),
            'slope_90': float(coeffs90[0]), 'intercept_90': float(coeffs90[1]),
        }
    else:
        linear_fit = {}
    
    # 保存
    output = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'mid_layer': mid_layer,
        'total_concepts': len(ALL_CONCEPTS),
        'n_concepts_extracted': len(concept_names),
        'results_by_n': results,
        'linear_fit': linear_fit,
    }
    
    outpath = TEMP / f"ccxxv_exp1_{model_name}_results.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to {outpath}")
    
    return output


# ================================================================
# Exp2: 语义分类对eff_dim的影响
# ================================================================

def run_exp2(model_name):
    print(f'\n{"="*60}')
    print(f'CCXXV Exp2: Category effect ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    mid_layer = info.n_layers // 2
    
    reps_by_layer, n_layers, d_model = extract_concept_reps(
        model_name, ALL_CONCEPTS, target_layers=[mid_layer],
        model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    concept_reps = reps_by_layer[mid_layer]
    
    results = {}
    
    # 每个类别单独
    for cat, concepts in CONCEPTS_BY_CATEGORY.items():
        valid = [c for c in concepts if c in concept_reps]
        if len(valid) < 5:
            continue
        X_cat = np.array([concept_reps[c] for c in valid])
        stats = compute_pca_stats(X_cat)
        results[cat] = {'n_concepts': len(valid), **stats}
        print(f"  {cat}: n={len(valid)}, rank95={stats['eff_dim_95']}, ratio={stats['ratio_95']:.3f}, top1={stats['top1']:.3f}")
    
    # 混合类别
    rng = np.random.RandomState(42)
    n_per_cat = 10
    mixed = []
    for cat, concepts in CONCEPTS_BY_CATEGORY.items():
        selected = rng.choice(concepts, min(n_per_cat, len(concepts)), replace=False).tolist()
        mixed.extend(selected)
    valid_mixed = [c for c in mixed if c in concept_reps]
    X_mixed = np.array([concept_reps[c] for c in valid_mixed])
    stats_mixed = compute_pca_stats(X_mixed)
    results['mixed_10percat'] = {'n_concepts': len(valid_mixed), **stats_mixed}
    print(f"  mixed_10percat: n={len(valid_mixed)}, rank95={stats_mixed['eff_dim_95']}, ratio={stats_mixed['ratio_95']:.3f}")
    
    # 全部
    X_all = np.array([concept_reps[c] for c in concept_reps.keys()])
    stats_all = compute_pca_stats(X_all)
    results['all'] = {'n_concepts': len(concept_reps), **stats_all}
    print(f"  all: n={len(concept_reps)}, rank95={stats_all['eff_dim_95']}, ratio={stats_all['ratio_95']:.3f}")
    
    output = {
        'model': model_name,
        'mid_layer': mid_layer,
        'category_results': results,
    }
    
    outpath = TEMP / f"ccxxv_exp2_{model_name}_results.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to {outpath}")
    
    return output


# ================================================================
# Exp3: 角度分布和正交分组
# ================================================================

def run_exp3(model_name):
    print(f'\n{"="*60}')
    print(f'CCXXV Exp3: Angle distribution ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    mid_layer = info.n_layers // 2
    
    reps_by_layer, _, _ = extract_concept_reps(
        model_name, ALL_CONCEPTS, target_layers=[mid_layer],
        model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    concept_reps = reps_by_layer[mid_layer]
    
    concept_names = list(concept_reps.keys())
    X = np.array([concept_reps[c] for c in concept_names])
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # 角度矩阵
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X_centered / norms
    cos_sim = X_norm @ X_norm.T
    angles = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    
    n = len(concept_names)
    upper_tri = angles[np.triu_indices(n, k=1)]
    
    angle_stats = {
        'mean': float(np.mean(upper_tri)),
        'median': float(np.median(upper_tri)),
        'std': float(np.std(upper_tri)),
        'pct_near_orthogonal': float(np.mean(upper_tri > 80)),
        'pct_near_parallel': float(np.mean(upper_tri < 10)),
        'pct_45_90': float(np.mean((upper_tri > 45) & (upper_tri < 90))),
    }
    
    print(f"  Angle stats (degrees): mean={angle_stats['mean']:.1f}, median={angle_stats['median']:.1f}")
    print(f"    Near-orthogonal (>80): {angle_stats['pct_near_orthogonal']:.3f}")
    print(f"    Near-parallel (<10): {angle_stats['pct_near_parallel']:.3f}")
    
    # 类内vs类间
    within = []
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            cat_i = CONCEPT_CATEGORIES.get(concept_names[i], '?')
            cat_j = CONCEPT_CATEGORIES.get(concept_names[j], '?')
            if cat_i == cat_j:
                within.append(angles[i, j])
            else:
                between.append(angles[i, j])
    
    cat_stats = {
        'within_mean': float(np.mean(within)),
        'within_std': float(np.std(within)),
        'between_mean': float(np.mean(between)),
        'between_std': float(np.std(between)),
        'separation_ratio': float(np.mean(between) / max(np.mean(within), 1e-10)),
    }
    
    print(f"  Within-cat angle: {cat_stats['within_mean']:.1f} +- {cat_stats['within_std']:.1f}")
    print(f"  Between-cat angle: {cat_stats['between_mean']:.1f} +- {cat_stats['between_std']:.1f}")
    print(f"  Separation ratio: {cat_stats['separation_ratio']:.3f}")
    
    # 每个类别内角
    cat_internal = {}
    for cat in CONCEPTS_BY_CATEGORY:
        cat_concepts = [c for c in concept_names if CONCEPT_CATEGORIES.get(c) == cat]
        if len(cat_concepts) < 2:
            continue
        cat_idx = [concept_names.index(c) for c in cat_concepts]
        internal = []
        for idx_i, i in enumerate(cat_idx):
            for j in cat_idx[idx_i + 1:]:
                internal.append(angles[i, j])
        cat_internal[cat] = {
            'mean': float(np.mean(internal)),
            'std': float(np.std(internal)),
            'n': len(cat_concepts),
        }
        print(f"    {cat}: mean={cat_internal[cat]['mean']:.1f} deg (n={len(cat_concepts)})")
    
    output = {
        'model': model_name,
        'mid_layer': mid_layer,
        'n_concepts': n,
        'angle_stats': angle_stats,
        'category_angle_stats': cat_stats,
        'category_internal_angles': cat_internal,
    }
    
    outpath = TEMP / f"ccxxv_exp3_{model_name}_results.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to {outpath}")
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--exp', type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    if args.exp == 1:
        run_exp1(args.model)
    elif args.exp == 2:
        run_exp2(args.model)
    elif args.exp == 3:
        run_exp3(args.model)
