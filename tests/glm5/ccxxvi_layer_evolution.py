"""
CCXXVI(374): 层间几何演化 & V_shared语义解码

核心问题: 
  1. 近正交性是中间层特有还是所有层都存在?
  2. V_shared(共享维度)编码了什么语义信息?
  3. 近正交是语言结构还是高维统计效应? (关键对照实验!)

实验设计:
  Exp1: 层间演化 — 在每个层测量slope, 角度分布
  Exp2: V_shared解码 — 提取top-10 PCA分量, 分析其语义含义
  Exp3: 随机对照 — 用随机向量做同样分析, 区分统计效应vs语言结构
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

# 80个代表性概念(从246中精选, 覆盖10类别, 每类8个)
REPRESENTATIVE_CONCEPTS = [
    # animals (8)
    "cat", "dog", "bird", "fish", "lion", "tiger", "eagle", "whale",
    # colors (8)
    "red", "blue", "green", "yellow", "white", "black", "purple", "orange",
    # emotions (8)
    "happy", "sad", "angry", "fear", "love", "hate", "hope", "joy",
    # materials (8)
    "wood", "stone", "metal", "glass", "paper", "cloth", "plastic", "rubber",
    # weather (8)
    "rain", "snow", "wind", "storm", "sun", "cloud", "fog", "ice",
    # body_parts (8)
    "hand", "foot", "head", "eye", "heart", "brain", "blood", "bone",
    # foods (8)
    "bread", "rice", "meat", "fruit", "water", "milk", "salt", "sugar",
    # tools (8)
    "hammer", "knife", "sword", "wheel", "rope", "nail", "axe", "saw",
    # abstract (8)
    "time", "space", "truth", "beauty", "justice", "freedom", "power", "knowledge",
    # social (8)
    "king", "queen", "child", "mother", "father", "friend", "enemy", "teacher",
]

CATEGORIES = {
    "animals": REPRESENTATIVE_CONCEPTS[0:8],
    "colors": REPRESENTATIVE_CONCEPTS[8:16],
    "emotions": REPRESENTATIVE_CONCEPTS[16:24],
    "materials": REPRESENTATIVE_CONCEPTS[24:32],
    "weather": REPRESENTATIVE_CONCEPTS[32:40],
    "body_parts": REPRESENTATIVE_CONCEPTS[40:48],
    "foods": REPRESENTATIVE_CONCEPTS[48:56],
    "tools": REPRESENTATIVE_CONCEPTS[56:64],
    "abstract": REPRESENTATIVE_CONCEPTS[64:72],
    "social": REPRESENTATIVE_CONCEPTS[72:80],
}

# 验证CATEGORIES正确
for cat, words in CATEGORIES.items():
    idx = list(CATEGORIES.keys()).index(cat)
    assert words == REPRESENTATIVE_CONCEPTS[idx*8:(idx+1)*8], f"Category {cat} mismatch"


def extract_reps_all_layers(model_name, concepts, model=None, tokenizer=None, device=None):
    """在所有层提取概念表示"""
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
    print(f"Extracting {len(concepts)} concepts across ALL layers...")
    
    # 存储每层的表示: {layer_idx: {concept: vector}}
    reps_by_layer = {i: {} for i in range(n_layers + 1)}  # +1 for embedding layer
    
    for ci, concept in enumerate(concepts):
        if ci % 20 == 0:
            print(f"  Concept {ci}/{len(concepts)}: {concept}")
        
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        
        # 找到目标词的token位置
        word_tokens = tokenizer(concept, add_special_tokens=False)['input_ids']
        n_word_tokens = len(word_tokens)
        # 目标词在prompt中的最后一个token位置
        target_pos = input_ids.shape[1] - 1  # 最后一个token
        
        # 提取每层的残差流
        layer_outputs = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output可能是tuple, 取第一个
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach()
                else:
                    layer_outputs[layer_idx] = output.detach()
            return hook_fn
        
        handles = []
        for i, layer in enumerate(layers_list):
            handles.append(layer.register_forward_hook(make_hook(i)))
        
        with torch.no_grad():
            # 先获取embedding
            embeds = embed_layer(input_ids)
            # 存储embedding层的表示
            reps_by_layer[0][concept] = embeds[0, target_pos, :].detach().cpu().float().numpy()
            
            # 前向传播
            try:
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # hidden_states[0] = embedding, hidden_states[1] = layer 0 output, etc.
                for i in range(n_layers):
                    reps_by_layer[i+1][concept] = hidden_states[i+1][0, target_pos, :].detach().cpu().float().numpy()
            except Exception as e:
                print(f"  Hidden states failed, using hooks: {e}")
                model(input_ids)
                for i in range(n_layers):
                    if i in layer_outputs:
                        reps_by_layer[i+1][concept] = layer_outputs[i][0, target_pos, :].detach().cpu().float().numpy()
        
        for h in handles:
            h.remove()
    
    if own_model:
        release_model(model)
        print("Model released.")
    
    return reps_by_layer, n_layers, d_model


def compute_geometry_stats(vectors_dict):
    """计算一组向量的几何统计"""
    names = list(vectors_dict.keys())
    vecs = np.array([vectors_dict[n] for n in names])
    n = len(names)
    
    # 中心化
    mean_vec = vecs.mean(axis=0)
    centered = vecs - mean_vec
    
    # PCA
    pca = PCA()
    pca.fit(centered)
    
    # 有效秩(95%方差)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    rank95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rank99 = int(np.searchsorted(cumvar, 0.99) + 1)
    
    # 角度分布
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = centered / norms
    
    angles = []
    for i in range(n):
        for j in range(i+1, n):
            cos_sim = np.clip(np.dot(normalized[i], normalized[j]), -1, 1)
            angle = np.degrees(np.arccos(cos_sim))
            angles.append(angle)
    
    angles = np.array(angles)
    
    return {
        'rank95': rank95,
        'rank99': rank99,
        'mean_angle': float(np.mean(angles)),
        'median_angle': float(np.median(angles)),
        'pct_gt80': float(np.mean(angles > 80)),
        'pct_lt10': float(np.mean(angles < 10)),
        'pct_70_110': float(np.mean((angles > 70) & (angles < 110))),
        'top10_variance': pca.explained_variance_ratio_[:10].tolist(),
        'top10_components': pca.components_[:10].tolist(),  # 用于语义解码
        'mean_vec': mean_vec.tolist(),
    }


def run_exp1(model_name):
    """Exp1: 层间几何演化"""
    print(f'\n{"="*60}')
    print(f'CCXXVI Exp1: Layer-by-layer geometry evolution ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # 使用80概念
    reps_by_layer, n_layers, d_model = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    
    # 在每层计算几何统计
    print("\nComputing geometry for each layer...")
    layer_stats = {}
    
    for layer_idx in range(n_layers + 1):
        layer_reps = reps_by_layer[layer_idx]
        if len(layer_reps) < 2:
            continue
        
        stats = compute_geometry_stats(layer_reps)
        stats.pop('top10_components', None)  # 太大, 只在exp2中保存
        stats.pop('mean_vec', None)
        layer_stats[layer_idx] = stats
        
        print(f"  Layer {layer_idx:2d}: rank95={stats['rank95']:3d}, "
              f"angle={stats['mean_angle']:.1f}°, "
              f">80°={stats['pct_gt80']:.3f}, "
              f"top1_var={stats['top10_variance'][0]:.4f}")
    
    # 保存结果
    results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'n_concepts': len(REPRESENTATIVE_CONCEPTS),
        'layer_stats': {str(k): v for k, v in layer_stats.items()},
    }
    
    out_path = TEMP / f"ccxxvi_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    
    # 分析关键趋势
    print("\n" + "="*50)
    print("KEY TRENDS:")
    print("="*50)
    
    rank95s = [layer_stats[i]['rank95'] for i in sorted(layer_stats.keys())]
    angles = [layer_stats[i]['mean_angle'] for i in sorted(layer_stats.keys())]
    gt80s = [layer_stats[i]['pct_gt80'] for i in sorted(layer_stats.keys())]
    top1vars = [layer_stats[i]['top10_variance'][0] for i in sorted(layer_stats.keys())]
    
    print(f"  rank95 range: {min(rank95s)} - {max(rank95s)}")
    print(f"  rank95 min at layer: {rank95s.index(min(rank95s))}")
    print(f"  angle range: {min(angles):.1f}° - {max(angles):.1f}°")
    print(f"  angle > 88° layers: {sum(1 for a in angles if a > 88)}/{len(angles)}")
    print(f"  top1_variance range: {min(top1vars):.4f} - {max(top1vars):.4f}")
    print(f"  top1_variance min at layer: {top1vars.index(min(top1vars))}")
    
    # 识别"压缩期"和"扩展期"
    print("\n  Layer transitions (rank95 changes):")
    for i in range(1, len(rank95s)):
        delta = rank95s[i] - rank95s[i-1]
        if abs(delta) >= 3:
            direction = "↑" if delta > 0 else "↓"
            print(f"    Layer {i-1}→{i}: rank95 {rank95s[i-1]}→{rank95s[i]} ({direction}{abs(delta)})")
    
    return results


def run_exp2(model_name):
    """Exp2: V_shared语义解码 — 顶层PCA分量编码了什么?"""
    print(f'\n{"="*60}')
    print(f'CCXXVI Exp2: V_shared semantic decoding ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    mid_layer = n_layers // 2
    
    # 只在中间层做详细分析
    reps_by_layer, n_layers, d_model = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    
    # 中间层的表示
    mid_reps = reps_by_layer[mid_layer]
    names = list(mid_reps.keys())
    vecs = np.array([mid_reps[n] for n in names])
    
    # 中心化
    mean_vec = vecs.mean(axis=0)
    centered = vecs - mean_vec
    
    # PCA
    pca = PCA()
    pca.fit(centered)
    
    print(f"\nMid-layer ({mid_layer}) PCA analysis:")
    print(f"  Top-10 variance ratios: {[f'{v:.4f}' for v in pca.explained_variance_ratio_[:10]]}")
    print(f"  Cumulative: {[f'{v:.4f}' for v in np.cumsum(pca.explained_variance_ratio_[:10])]}")
    
    # 分析每个PCA分量上各概念的投影
    print("\n  Top PCA components - which concepts load highest?")
    
    components = pca.components_  # (n_components, d_model)
    projections = centered @ components.T  # (n_concepts, n_components)
    
    component_analysis = []
    for pc_idx in range(min(10, components.shape[0])):
        proj = projections[:, pc_idx]
        
        # 按投影值排序
        sorted_indices = np.argsort(proj)
        top5_pos = [(names[i], float(proj[i])) for i in sorted_indices[-5:]][::-1]
        top5_neg = [(names[i], float(proj[i])) for i in sorted_indices[:5]]
        
        # 按类别分析
        cat_means = {}
        for cat, cat_words in CATEGORIES.items():
            cat_indices = [i for i, n in enumerate(names) if n in cat_words]
            if cat_indices:
                cat_means[cat] = float(np.mean(proj[cat_indices]))
        
        # 排序类别
        sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)
        
        info = {
            'pc': pc_idx,
            'variance_ratio': float(pca.explained_variance_ratio_[pc_idx]),
            'top5_positive': top5_pos,
            'top5_negative': top5_neg,
            'category_ranking': sorted_cats,
        }
        component_analysis.append(info)
        
        print(f"\n  PC{pc_idx} (var={pca.explained_variance_ratio_[pc_idx]:.4f}):")
        print(f"    Top +: {top5_pos}")
        print(f"    Top -: {top5_neg}")
        print(f"    Cat order: {[(c, f'{v:.3f}') for c, v in sorted_cats]}")
    
    # 分析: 是否存在"语义轴"?
    print("\n" + "="*50)
    print("SEMANTIC AXIS ANALYSIS:")
    print("="*50)
    
    # 检查是否有PC能区分大的语义类别
    for pc_idx, info in enumerate(component_analysis[:5]):
        cat_ranking = info['category_ranking']
        # 计算类别间差异
        cat_values = [v for _, v in cat_ranking]
        cat_spread = max(cat_values) - min(cat_values)
        print(f"  PC{pc_idx}: category spread = {cat_spread:.3f}, "
              f"top_cat={cat_ranking[0][0]}, bottom_cat={cat_ranking[-1][0]}")
    
    # 保存完整结果
    results = {
        'model': model_name,
        'mid_layer': mid_layer,
        'd_model': d_model,
        'n_concepts': len(REPRESENTATIVE_CONCEPTS),
        'top10_variance': pca.explained_variance_ratio_[:10].tolist(),
        'component_analysis': component_analysis,
    }
    
    out_path = TEMP / f"ccxxvi_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


def run_exp3(model_name):
    """Exp3: 随机对照实验 — 区分统计效应vs语言结构"""
    print(f'\n{"="*60}')
    print(f'CCXXVI Exp3: Random control experiment ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2
    
    # 获取真实概念表示
    reps_by_layer, _, _ = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    
    # 中间层真实表示
    real_reps = reps_by_layer[mid_layer]
    real_names = list(real_reps.keys())
    real_vecs = np.array([real_reps[n] for n in real_names])
    real_centered = real_vecs - real_vecs.mean(axis=0)
    
    # 中间层embedding层表示
    embed_reps = reps_by_layer[0]
    embed_vecs = np.array([embed_reps[n] for n in real_names])
    embed_centered = embed_vecs - embed_vecs.mean(axis=0)
    
    # 最后一层表示
    last_reps = reps_by_layer[n_layers]
    last_vecs = np.array([last_reps[n] for n in real_names])
    last_centered = last_vecs - last_vecs.mean(axis=0)
    
    # 对照3: 随机词 — 在释放模型之前获取
    print("\nGenerating random-token embeddings...")
    n_concepts = len(REPRESENTATIVE_CONCEPTS)
    embed_layer = model.get_input_embeddings()
    random_word_reps = {}
    for i in range(n_concepts):
        random_ids = torch.randint(100, 5000, (1, 5)).to(device)
        with torch.no_grad():
            embeds = embed_layer(random_ids)
        random_word_reps[f"random_{i}"] = embeds[0, -1, :].detach().cpu().float().numpy()
    
    release_model(model)
    print("Model released.")
    
    # 对照1: 同分布随机向量 (从真实分布采样)
    rng = np.random.RandomState(42)
    random_same_dist = real_vecs + rng.randn(*real_vecs.shape) * 0.1
    random_same_dist_centered = random_same_dist - random_same_dist.mean(axis=0)
    
    # 对照2: 纯随机向量 (与真实数据同维度, 但独立高斯)
    rng2 = np.random.RandomState(123)
    real_norms = np.linalg.norm(real_centered, axis=1)
    mean_norm = np.mean(real_norms)
    random_pure = rng2.randn(n_concepts, d_model)
    random_pure_norms = np.linalg.norm(random_pure, axis=1, keepdims=True)
    random_pure = random_pure / random_pure_norms * mean_norm
    random_pure_centered = random_pure - random_pure.mean(axis=0)
    
    random_word_vecs = np.array([random_word_reps[f"random_{i}"] for i in range(n_concepts)])
    random_word_centered = random_word_vecs - random_word_vecs.mean(axis=0)
    
    # 在所有数据上计算几何统计
    datasets = {
        'real_mid': ('Real mid-layer', real_centered, real_names),
        'real_embed': ('Real embedding', embed_centered, real_names),
        'real_last': ('Real last layer', last_centered, real_names),
        'random_same_dist': ('Random (noise around real)', random_same_dist_centered, real_names),
        'random_pure': ('Random (pure Gaussian)', random_pure_centered, None),
        'random_words': ('Random word embeddings', random_word_centered, None),
    }
    
    print("\n" + "="*70)
    print(f"{'Dataset':<30} {'rank95':>7} {'angle':>8} {'>80°':>6} {'top1':>8} {'cat_sep':>8}")
    print("="*70)
    
    comparison = {}
    
    for key, (label, centered, names) in datasets.items():
        pca = PCA()
        pca.fit(centered)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        rank95 = int(np.searchsorted(cumvar, 0.95) + 1)
        
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = centered / norms
        
        n = len(centered)
        angles = []
        for i in range(n):
            for j in range(i+1, n):
                cos_sim = np.clip(np.dot(normalized[i], normalized[j]), -1, 1)
                angle = np.degrees(np.arccos(cos_sim))
                angles.append(angle)
        angles = np.array(angles)
        
        # 类别分离度 (仅当有类别信息时)
        cat_separation = None
        if names is not None:
            cat_angles = {'within': [], 'between': []}
            for i in range(n):
                cat_i = None
                for cat, words in CATEGORIES.items():
                    if names[i] in words:
                        cat_i = cat
                        break
                for j in range(i+1, n):
                    cat_j = None
                    for cat, words in CATEGORIES.items():
                        if names[j] in words:
                            cat_j = cat
                            break
                    if cat_i and cat_j:
                        if cat_i == cat_j:
                            cat_angles['within'].append(angles[i * (n-1) - i*(i+1)//2 + j - i - 1] if i*(n-1) - i*(i+1)//2 + j - i - 1 < len(angles) else 90)
                        else:
                            cat_angles['between'].append(angles[i * (n-1) - i*(i+1)//2 + j - i - 1] if i*(n-1) - i*(i+1)//2 + j - i - 1 < len(angles) else 90)
            
            if cat_angles['within'] and cat_angles['between']:
                cat_separation = np.mean(cat_angles['between']) / max(np.mean(cat_angles['within']), 1)
        
        mean_angle = float(np.mean(angles))
        pct_gt80 = float(np.mean(angles > 80))
        top1_var = float(pca.explained_variance_ratio_[0])
        
        cat_sep_str = f"{cat_separation:.3f}" if cat_separation else "N/A"
        print(f"  {label:<28} {rank95:>7} {mean_angle:>7.1f}° {pct_gt80:>6.3f} {top1_var:>8.4f} {cat_sep_str:>8}")
        
        comparison[key] = {
            'rank95': rank95,
            'mean_angle': mean_angle,
            'pct_gt80': pct_gt80,
            'top1_variance': top1_var,
            'cat_separation': cat_separation,
        }
    
    # 重新计算正确的类别分离度
    print("\n\nRecalculating category separation correctly...")
    for key, (label, centered, names) in datasets.items():
        if names is None:
            continue
        
        n = len(centered)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = centered / norms
        
        within_angles = []
        between_angles = []
        
        # 构建name→category映射
        name_to_cat = {}
        for name in names:
            for cat, words in CATEGORIES.items():
                if name in words:
                    name_to_cat[name] = cat
                    break
        
        for i in range(n):
            for j in range(i+1, n):
                cos_sim = np.clip(np.dot(normalized[i], normalized[j]), -1, 1)
                angle = np.degrees(np.arccos(cos_sim))
                
                ci = name_to_cat.get(names[i])
                cj = name_to_cat.get(names[j])
                
                if ci and cj:
                    if ci == cj:
                        within_angles.append(angle)
                    else:
                        between_angles.append(angle)
        
        if within_angles and between_angles:
            mean_within = np.mean(within_angles)
            mean_between = np.mean(between_angles)
            sep = mean_between / mean_within
            print(f"  {label}: within={mean_within:.1f}°, between={mean_between:.1f}°, sep={sep:.3f}")
            comparison[key]['within_angle'] = float(mean_within)
            comparison[key]['between_angle'] = float(mean_between)
            comparison[key]['cat_separation'] = float(sep)
    
    # 关键判断
    print("\n" + "="*70)
    print("CRITICAL ANALYSIS: Language structure vs Statistical effect?")
    print("="*70)
    
    real_rank95 = comparison['real_mid']['rank95']
    random_rank95 = comparison['random_pure']['rank95']
    real_angle = comparison['real_mid']['mean_angle']
    random_angle = comparison['random_pure']['mean_angle']
    
    print(f"\n  Real mid-layer: rank95={real_rank95}, angle={real_angle:.1f}°")
    print(f"  Random Gaussian: rank95={random_rank95}, angle={random_angle:.1f}°")
    print(f"  Random words: rank95={comparison['random_words']['rank95']}, angle={comparison['random_words']['mean_angle']:.1f}°")
    
    if abs(real_angle - 90) < 5 and abs(random_angle - 90) < 5:
        print("\n  ⚠ BOTH real and random have angle≈90°!")
        print("  → Near-orthogonality may be a HIGH-DIMENSIONAL STATISTICAL EFFECT")
        print("  → Need to look at CATEGORY SEPARATION to distinguish!")
    
    if 'cat_separation' in comparison.get('real_mid', {}):
        real_sep = comparison['real_mid']['cat_separation']
        random_word_sep = comparison.get('random_words', {}).get('cat_separation', 'N/A')
        print(f"\n  Category separation:")
        print(f"    Real mid-layer: {real_sep:.3f}")
        print(f"    Random words: {random_word_sep}")
        if isinstance(random_word_sep, (int, float)):
            if real_sep > random_word_sep + 0.02:
                print("  → Real concepts have STRONGER category structure than random!")
                print("  → This supports LANGUAGE STRUCTURE (not just statistics)")
            else:
                print("  → Category separation is similar to random!")
                print("  → May be a statistical artifact")
    
    # 保存结果
    results = {
        'model': model_name,
        'mid_layer': info.n_layers // 2,
        'n_concepts': len(REPRESENTATIVE_CONCEPTS),
        'comparison': comparison,
    }
    
    out_path = TEMP / f"ccxxvi_exp3_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


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
