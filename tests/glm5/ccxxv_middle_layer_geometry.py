"""
CCXXV(373): 中间层几何学 — 大规模概念扫描
目标: 系统验证 eff_dim = f(n_concepts) 的精确形式

核心假设:
  - 中间层 eff_dim_95 ≈ c × n_concepts (线性关系)
  - c ≈ 0.84 (之前50概念的结果)
  
实验设计:
  Exp1: 200+概念的大规模扫描, 测量不同概念数下的eff_dim
  Exp2: 概念语义分类对eff_dim的影响 (同类vs混合)
  Exp3: 中间层概念向量之间的角度分布和正交分组
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 200+概念的系统列表, 按语义类别分组
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
        "gold_metal", "silver_metal", "diamond", "marble", "granite", "sand", "clay", "silk"
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
        "bread", "rice", "meat", "fish_food", "fruit", "vegetable", "cheese", "egg",
        "milk", "water", "wine", "beer", "cake", "cookie", "soup", "salad",
        "pasta", "noodle", "steak", "chicken", "potato", "tomato", "onion", "garlic"
    ],
    "tools": [
        "hammer", "saw", "drill", "nail", "screw", "wrench", "pliers", "knife",
        "scissors", "axe", "shovel", "rake", "hoe", "chisel", "file", "clamp",
        "ruler", "compass_tool", "level", "wedge", "lever", "pulley", "gear", "spring"
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

# 展平的概念列表
ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
for cat, concepts in CONCEPTS_BY_CATEGORY.items():
    for c in concepts:
        ALL_CONCEPTS.append(c)
        CONCEPT_CATEGORIES[c] = cat

print(f"Total concepts: {len(ALL_CONCEPTS)}")
print(f"Categories: {len(CONCEPTS_BY_CATEGORY)}")
print(f"Concepts per category: {[f'{k}={len(v)}' for k, v in CONCEPTS_BY_CATEGORY.items()]}")


def run_exp1(model_name, n_concepts_list=None):
    """Exp1: 大规模概念扫描 — 测量 eff_dim = f(n_concepts)"""
    from tests.glm5.model_utils import load_model, get_model_info, release_model
    import torch
    
    if n_concepts_list is None:
        n_concepts_list = [10, 20, 30, 50, 75, 100, 150, 200, len(ALL_CONCEPTS)]
    
    print(f"\n{'='*60}")
    print(f"CCXXV Exp1: Large-scale concept scan for {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model)
    n_layers = info['n_layers']
    d_model = info['d_model']
    
    # 选择中间层
    mid_layer = n_layers // 2
    print(f"Model: {model_name}, layers={n_layers}, d_model={d_model}, mid_layer={mid_layer}")
    
    # 获取所有概念的中间层表示
    print(f"Extracting representations for {len(ALL_CONCEPTS)} concepts...")
    concept_reps = {}
    
    for i, concept in enumerate(ALL_CONCEPTS):
        if i % 50 == 0:
            print(f"  Processing concept {i}/{len(ALL_CONCEPTS)}...")
        
        # 使用概念词作为输入
        inputs = tokenizer(concept.replace('_', ' '), return_tensors='pt').to(device)
        
        with torch.no_grad():
            # 获取中间层的残差流
            _, cache = model.run_with_cache(inputs['input_ids'])
            
            if hasattr(model, 'blocks'):
                rep = cache['resid_mid', mid_layer]
            else:
                # 尝试不同的缓存键
                possible_keys = [k for k in cache.keys() if 'mid' in str(k) or f'layer_{mid_layer}' in str(k)]
                if possible_keys:
                    rep = cache[possible_keys[0]]
                else:
                    # 回退: 使用resid_post
                    rep = cache['resid_post', mid_layer]
            
            # 取最后一个token的表示
            concept_reps[concept] = rep[0, -1, :].detach().cpu().float().numpy()
    
    # 释放模型
    release_model(model)
    print("Model released. Computing PCA...")
    
    # 构建表示矩阵
    concept_names = list(concept_reps.keys())
    X = np.array([concept_reps[c] for c in concept_names])  # shape: (n_concepts, d_model)
    print(f"Representation matrix shape: {X.shape}")
    
    # 对不同概念数计算eff_dim
    results = {}
    
    for n_target in n_concepts_list:
        if n_target > len(concept_names):
            continue
        
        # 随机选择n_target个概念 (用固定种子保证可复现)
        rng = np.random.RandomState(42)
        indices = rng.choice(len(concept_names), n_target, replace=False)
        X_sub = X[indices]
        
        # PCA
        from sklearn.decomposition import PCA
        
        # 中心化
        X_centered = X_sub - X_sub.mean(axis=0, keepdims=True)
        
        # SVD计算
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_energy = np.sum(s**2)
        cum = np.cumsum(s**2) / total_energy
        
        # 有效维度
        eff_dims = {}
        for threshold_pct in [50, 80, 90, 95, 99]:
            idx = int(np.argmax(cum >= threshold_pct / 100.0) + 1)
            eff_dims[f'rank_{threshold_pct}'] = int(idx)
        
        # top-k方差占比
        top_k_energy = {}
        for k in [1, 5, 10, 20, 50]:
            if k <= len(s):
                top_k_energy[f'top{k}'] = float(np.sum(s[:k]**2) / total_energy)
        
        # 幂律拟合 (前50个奇异值)
        n_fit = min(50, len(s))
        s_fit = s[:n_fit]
        s_fit = s_fit[s_fit > 0]
        if len(s_fit) > 5:
            log_r = np.log(np.arange(1, len(s_fit) + 1))
            log_s = np.log(s_fit)
            alpha = -np.polyfit(log_r, log_s, 1)[0]
        else:
            alpha = 0
        
        result = {
            'n_concepts': int(n_target),
            'eff_dims': eff_dims,
            'top_k_energy': top_k_energy,
            'alpha': float(alpha),
            'ratio_95': float(eff_dims['rank_95'] / n_target),
            'ratio_90': float(eff_dims['rank_90'] / n_target),
            'singular_values_top20': [float(x) for x in s[:20]],
        }
        results[str(n_target)] = result
        
        print(f"  n={n_target}: rank95={eff_dims['rank_95']}, ratio={eff_dims['rank_95']/n_target:.3f}, "
              f"rank90={eff_dims['rank_90']}, α={alpha:.3f}")
    
    # 线性回归: rank95 = a * n + b
    n_values = np.array([int(k) for k in results.keys()])
    rank95_values = np.array([results[k]['rank_95'] for k in results.keys()])
    
    if len(n_values) >= 3:
        coeffs = np.polyfit(n_values, rank95_values, 1)
        slope, intercept = coeffs
        print(f"\n  Linear fit: rank95 = {slope:.3f} * n + {intercept:.3f}")
        print(f"  Slope (dim/concept): {slope:.3f}")
        print(f"  R²: {1 - np.sum((rank95_values - np.polyval(coeffs, n_values))**2) / np.sum((rank95_values - rank95_values.mean())**2):.4f}")
    
    # 保存结果
    output = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'mid_layer': mid_layer,
        'total_concepts': len(ALL_CONCEPTS),
        'results_by_n': results,
        'linear_fit': {
            'slope': float(slope) if len(n_values) >= 3 else None,
            'intercept': float(intercept) if len(n_values) >= 3 else None,
        },
        'concept_categories': {k: len(v) for k, v in CONCEPTS_BY_CATEGORY.items()},
    }
    
    out_path = f'tests/glm5_temp/ccxxv_exp1_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")
    
    return output


def run_exp2(model_name):
    """Exp2: 语义分类对eff_dim的影响 — 同类vs混合"""
    from tests.glm5.model_utils import load_model, get_model_info, release_model
    import torch
    from sklearn.decomposition import PCA
    
    print(f"\n{'='*60}")
    print(f"CCXXV Exp2: Category effect for {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model)
    n_layers = info['n_layers']
    d_model = info['d_model']
    mid_layer = n_layers // 2
    
    print(f"Model: {model_name}, mid_layer={mid_layer}")
    
    # 获取所有概念的表示
    print(f"Extracting representations...")
    concept_reps = {}
    
    for i, concept in enumerate(ALL_CONCEPTS):
        if i % 50 == 0:
            print(f"  Processing concept {i}/{len(ALL_CONCEPTS)}...")
        inputs = tokenizer(concept.replace('_', ' '), return_tensors='pt').to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inputs['input_ids'])
            if hasattr(model, 'blocks'):
                rep = cache['resid_mid', mid_layer]
            else:
                rep = cache['resid_post', mid_layer]
            concept_reps[concept] = rep[0, -1, :].detach().cpu().float().numpy()
    
    release_model(model)
    print("Model released. Computing category effects...")
    
    results = {}
    
    # 对每个类别单独计算eff_dim
    for cat, concepts in CONCEPTS_BY_CATEGORY.items():
        valid = [c for c in concepts if c in concept_reps]
        if len(valid) < 5:
            continue
        
        X_cat = np.array([concept_reps[c] for c in valid])
        X_centered = X_cat - X_cat.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_energy = np.sum(s**2)
        cum = np.cumsum(s**2) / total_energy
        
        rank_95 = int(np.argmax(cum >= 0.95) + 1)
        rank_90 = int(np.argmax(cum >= 0.90) + 1)
        
        results[cat] = {
            'n_concepts': len(valid),
            'rank_95': rank_95,
            'rank_90': rank_90,
            'ratio_95': rank_95 / len(valid),
            'ratio_90': rank_90 / len(valid),
            'top1_energy': float(s[0]**2 / total_energy),
            'top5_energy': float(np.sum(s[:5]**2) / total_energy) if len(s) >= 5 else float(np.sum(s**2) / total_energy),
        }
        
        print(f"  {cat}: n={len(valid)}, rank95={rank_95}, ratio={rank_95/len(valid):.3f}, top1={results[cat]['top1_energy']:.3f}")
    
    # 混合类别: 随机从每个类别选相同数量
    rng = np.random.RandomState(42)
    n_per_cat = 10
    mixed_concepts = []
    for cat, concepts in CONCEPTS_BY_CATEGORY.items():
        selected = rng.choice(concepts, min(n_per_cat, len(concepts)), replace=False).tolist()
        mixed_concepts.extend(selected)
    
    if len(mixed_concepts) >= 10:
        X_mixed = np.array([concept_reps[c] for c in mixed_concepts if c in concept_reps])
        X_centered = X_mixed - X_mixed.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_energy = np.sum(s**2)
        cum = np.cumsum(s**2) / total_energy
        
        rank_95 = int(np.argmax(cum >= 0.95) + 1)
        results['mixed_10percat'] = {
            'n_concepts': len(X_mixed),
            'rank_95': rank_95,
            'rank_90': int(np.argmax(cum >= 0.90) + 1),
            'ratio_95': rank_95 / len(X_mixed),
            'top1_energy': float(s[0]**2 / total_energy),
            'top5_energy': float(np.sum(s[:5]**2) / total_energy) if len(s) >= 5 else float(np.sum(s**2) / total_energy),
        }
        print(f"  mixed_10percat: n={len(X_mixed)}, rank95={rank_95}, ratio={rank_95/len(X_mixed):.3f}")
    
    # 保存结果
    output = {
        'model': model_name,
        'mid_layer': mid_layer,
        'category_results': results,
    }
    
    out_path = f'tests/glm5_temp/ccxxv_exp2_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")
    
    return output


def run_exp3(model_name):
    """Exp3: 中间层概念向量之间的角度分布和正交分组"""
    from tests.glm5.model_utils import load_model, get_model_info, release_model
    import torch
    
    print(f"\n{'='*60}")
    print(f"CCXXV Exp3: Angle distribution and orthogonal grouping for {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model)
    n_layers = info['n_layers']
    d_model = info['d_model']
    mid_layer = n_layers // 2
    
    print(f"Model: {model_name}, mid_layer={mid_layer}")
    
    # 获取所有概念的表示
    print(f"Extracting representations...")
    concept_reps = {}
    
    for i, concept in enumerate(ALL_CONCEPTS):
        if i % 50 == 0:
            print(f"  Processing concept {i}/{len(ALL_CONCEPTS)}...")
        inputs = tokenizer(concept.replace('_', ' '), return_tensors='pt').to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(inputs['input_ids'])
            if hasattr(model, 'blocks'):
                rep = cache['resid_mid', mid_layer]
            else:
                rep = cache['resid_post', mid_layer]
            concept_reps[concept] = rep[0, -1, :].detach().cpu().float().numpy()
    
    release_model(model)
    print("Model released. Computing angles...")
    
    # 中心化
    concept_names = list(concept_reps.keys())
    X = np.array([concept_reps[c] for c in concept_names])
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # 计算角度矩阵
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_normalized = X_centered / norms
    
    cosine_sim = X_normalized @ X_normalized.T
    
    # 角度 (弧度)
    angles = np.arccos(np.clip(cosine_sim, -1, 1))
    angle_degrees = np.degrees(angles)
    
    # 统计
    n = len(concept_names)
    upper_tri = angle_degrees[np.triu_indices(n, k=1)]
    
    angle_stats = {
        'mean': float(np.mean(upper_tri)),
        'median': float(np.median(upper_tri)),
        'std': float(np.std(upper_tri)),
        'min': float(np.min(upper_tri)),
        'max': float(np.max(upper_tri)),
        'pct_near_orthogonal': float(np.mean(upper_tri > 80)),  # 角度 > 80度
        'pct_near_parallel': float(np.mean(upper_tri < 10)),    # 角度 < 10度
        'pct_45_90': float(np.mean((upper_tri > 45) & (upper_tri < 90))),
    }
    
    print(f"  Angle statistics (degrees):")
    print(f"    Mean: {angle_stats['mean']:.1f}, Median: {angle_stats['median']:.1f}")
    print(f"    Near-orthogonal (>80°): {angle_stats['pct_near_orthogonal']:.3f}")
    print(f"    Near-parallel (<10°): {angle_stats['pct_near_parallel']:.3f}")
    print(f"    45-90°: {angle_stats['pct_45_90']:.3f}")
    
    # 类内和类间角度比较
    within_cat_angles = []
    between_cat_angles = []
    
    for i in range(n):
        for j in range(i + 1, n):
            cat_i = CONCEPT_CATEGORIES.get(concept_names[i], 'unknown')
            cat_j = CONCEPT_CATEGORIES.get(concept_names[j], 'unknown')
            if cat_i == cat_j:
                within_cat_angles.append(angle_degrees[i, j])
            else:
                between_cat_angles.append(angle_degrees[i, j])
    
    cat_angle_stats = {
        'within_mean': float(np.mean(within_cat_angles)),
        'within_std': float(np.std(within_cat_angles)),
        'between_mean': float(np.mean(between_cat_angles)),
        'between_std': float(np.std(between_cat_angles)),
        'separation_ratio': float(np.mean(between_cat_angles) / max(np.mean(within_cat_angles), 1e-10)),
    }
    
    print(f"  Within-category angle: {cat_angle_stats['within_mean']:.1f} ± {cat_angle_stats['within_std']:.1f}")
    print(f"  Between-category angle: {cat_angle_stats['between_mean']:.1f} ± {cat_angle_stats['between_std']:.1f}")
    print(f"  Separation ratio: {cat_angle_stats['separation_ratio']:.3f}")
    
    # 每个类别的平均内角
    cat_internal_angles = {}
    for cat in CONCEPTS_BY_CATEGORY:
        cat_concepts = [c for c in concept_names if CONCEPT_CATEGORIES.get(c) == cat]
        if len(cat_concepts) < 2:
            continue
        cat_indices = [concept_names.index(c) for c in cat_concepts]
        internal = []
        for idx_i, i in enumerate(cat_indices):
            for j in cat_indices[idx_i + 1:]:
                internal.append(angle_degrees[i, j])
        cat_internal_angles[cat] = {
            'mean': float(np.mean(internal)),
            'std': float(np.std(internal)),
            'n': len(cat_concepts),
        }
        print(f"    {cat}: mean_angle={cat_internal_angles[cat]['mean']:.1f}° (n={len(cat_concepts)})")
    
    # 正交分组分析: 找出高度正交的概念对
    near_orthogonal_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if angle_degrees[i, j] > 85:
                near_orthogonal_pairs.append({
                    'c1': concept_names[i],
                    'c2': concept_names[j],
                    'angle': float(angle_degrees[i, j]),
                    'cat1': CONCEPT_CATEGORIES.get(concept_names[i], 'unknown'),
                    'cat2': CONCEPT_CATEGORIES.get(concept_names[j], 'unknown'),
                })
    
    # 统计正交对中同类vs异类的比例
    same_cat_orth = sum(1 for p in near_orthogonal_pairs if p['cat1'] == p['cat2'])
    diff_cat_orth = len(near_orthogonal_pairs) - same_cat_orth
    
    print(f"\n  Near-orthogonal pairs (>85°): {len(near_orthogonal_pairs)}")
    print(f"    Same category: {same_cat_orth}")
    print(f"    Different category: {diff_cat_orth}")
    
    # 保存结果
    output = {
        'model': model_name,
        'mid_layer': mid_layer,
        'n_concepts': n,
        'angle_stats': angle_stats,
        'category_angle_stats': cat_angle_stats,
        'category_internal_angles': cat_internal_angles,
        'near_orthogonal_pairs_count': len(near_orthogonal_pairs),
        'same_cat_orthogonal': same_cat_orth,
        'diff_cat_orthogonal': diff_cat_orth,
        'near_orthogonal_pairs_sample': near_orthogonal_pairs[:20],  # 只保存前20个
    }
    
    out_path = f'tests/glm5_temp/ccxxv_exp3_{model_name}_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")
    
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
