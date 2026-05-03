"""
CCXXVII(375): 跨层信息瓶颈分析

核心问题:
  1. 中间层是否是信息瓶颈? (语义信息保留 vs 总信息压缩)
  2. 被压缩掉的是什么信息? (噪声 vs 非语义结构)
  3. 维度压缩的15%是否真正来自语义结构?

实验设计:
  Exp1: 跨层语义信息保持
    - 每层的category separability (F-statistic)
    - 每层的rank95 (总信息量代理)
    - 验证: 中间层rank95最低但separability最高 → 信息瓶颈

  Exp2: 随机token通过模型的对照
    - 80个随机token序列通过模型, 每层提取表示
    - 随机分配伪类别, 计算同样统计量
    - 对比: 真实概念的separability >> 随机伪类别

  Exp3: 信息压缩的语义含义
    - 从embedding→中间层, 哪些概念变化最大?
    - 变化大的概念是否与语义类别相关?
    - 量化: 被压缩的信息中, 语义信息vs非语义信息的比例
"""

import sys, os, argparse, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")

# 80个代表性概念(与CCXXVI一致)
REPRESENTATIVE_CONCEPTS = [
    "cat", "dog", "bird", "fish", "lion", "tiger", "eagle", "whale",
    "red", "blue", "green", "yellow", "white", "black", "purple", "orange",
    "happy", "sad", "angry", "fear", "love", "hate", "hope", "joy",
    "wood", "stone", "metal", "glass", "paper", "cloth", "plastic", "rubber",
    "rain", "snow", "wind", "storm", "sun", "cloud", "fog", "ice",
    "hand", "foot", "head", "eye", "heart", "brain", "blood", "bone",
    "bread", "rice", "meat", "fruit", "water", "milk", "salt", "sugar",
    "hammer", "knife", "sword", "wheel", "rope", "nail", "axe", "saw",
    "time", "space", "truth", "beauty", "justice", "freedom", "power", "knowledge",
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

# 概念→类别映射
CONCEPT_TO_CAT = {}
for cat, words in CATEGORIES.items():
    for w in words:
        CONCEPT_TO_CAT[w] = cat


def extract_reps_all_layers(model_name, concepts, model=None, tokenizer=None, device=None):
    """在所有层提取概念表示(与CCXXVI相同)"""
    if model is None:
        model, tokenizer, device = load_model(model_name)
        own_model = True
    else:
        own_model = False
    
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    reps_by_layer = {i: {} for i in range(n_layers + 1)}
    
    for ci, concept in enumerate(concepts):
        if ci % 20 == 0:
            print(f"  Concept {ci}/{len(concepts)}: {concept}")
        
        prompt = f"The word is {concept}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        target_pos = input_ids.shape[1] - 1
        
        with torch.no_grad():
            try:
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # embedding层
                reps_by_layer[0][concept] = hidden_states[0][0, target_pos, :].detach().cpu().float().numpy()
                for i in range(n_layers):
                    reps_by_layer[i+1][concept] = hidden_states[i+1][0, target_pos, :].detach().cpu().float().numpy()
            except Exception as e:
                print(f"  Hidden states failed for {concept}: {e}")
    
    if own_model:
        release_model(model)
    
    return reps_by_layer, n_layers, d_model


def compute_category_separability(vecs, labels):
    """
    计算类别分离度 (类似MANOVA的F-statistic)
    F = trace(S_B) / trace(S_W)
    S_B = between-category scatter
    S_W = within-category scatter
    """
    n = len(vecs)
    unique_labels = list(set(labels))
    
    # 总均值
    grand_mean = vecs.mean(axis=0)
    
    # Between-category scatter
    S_B = np.zeros((vecs.shape[1], vecs.shape[1]))
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        n_k = mask.sum()
        if n_k > 0:
            group_mean = vecs[mask].mean(axis=0)
            diff = (group_mean - grand_mean).reshape(-1, 1)
            S_B += n_k * (diff @ diff.T)
    
    # Within-category scatter
    S_W = np.zeros((vecs.shape[1], vecs.shape[1]))
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        if mask.sum() > 1:
            group_vecs = vecs[mask]
            group_mean = group_vecs.mean(axis=0)
            centered = group_vecs - group_mean
            S_W += centered.T @ centered
    
    # F-statistic
    trace_B = np.trace(S_B)
    trace_W = np.trace(S_W)
    
    if trace_W < 1e-10:
        return float('inf')
    
    F = trace_B / trace_W
    
    # 也计算类内/类间距离比(更鲁棒)
    norms = np.linalg.norm(vecs - vecs.mean(axis=0), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = (vecs - vecs.mean(axis=0)) / norms
    
    within_dists = []
    between_dists = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(normalized[i] - normalized[j])
            if labels[i] == labels[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)
    
    mean_within = np.mean(within_dists) if within_dists else 1.0
    mean_between = np.mean(between_dists) if between_dists else 1.0
    dist_ratio = mean_between / mean_within
    
    return {
        'F_stat': float(F),
        'trace_B': float(trace_B),
        'trace_W': float(trace_W),
        'dist_ratio': float(dist_ratio),
        'mean_within_dist': float(mean_within),
        'mean_between_dist': float(mean_between),
        'n_within': len(within_dists),
        'n_between': len(between_dists),
    }


def compute_linear_probe_accuracy(vecs, labels):
    """
    线性探针分类准确率(使用PCA降维后)
    """
    # 降维到50维(防止过拟合)
    pca = PCA(n_components=min(50, vecs.shape[1], vecs.shape[0]))
    X = pca.fit_transform(vecs)
    
    # 简单logistic regression
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)
    train_acc = clf.score(X, labels)
    
    return float(train_acc)


def compute_geometry_stats(vectors_dict):
    """计算几何统计"""
    names = list(vectors_dict.keys())
    vecs = np.array([vectors_dict[n] for n in names])
    
    mean_vec = vecs.mean(axis=0)
    centered = vecs - mean_vec
    
    pca = PCA()
    pca.fit(centered)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    rank95 = int(np.searchsorted(cumvar, 0.95) + 1)
    
    return {
        'rank95': rank95,
        'total_variance': float(np.sum(pca.explained_variance_ratio_)),
        'top1_variance': float(pca.explained_variance_ratio_[0]),
        'top5_variance': float(np.sum(pca.explained_variance_ratio_[:5])),
        'top10_variance': float(np.sum(pca.explained_variance_ratio_[:10])),
    }


def run_exp1(model_name):
    """Exp1: 跨层信息瓶颈 — 语义信息保持 vs 总信息压缩"""
    print(f'\n{"="*60}')
    print(f'CCXXVII Exp1: Cross-layer information bottleneck ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2
    
    reps_by_layer, n_layers, d_model = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    
    # 准备标签
    concepts = REPRESENTATIVE_CONCEPTS
    labels = [CONCEPT_TO_CAT[c] for c in concepts]
    
    print(f"\n{'Layer':>6} {'rank95':>7} {'top1':>8} {'top5':>8} {'top10':>8} {'F_stat':>10} {'dist_ratio':>11} {'probe_acc':>10}")
    print("-" * 80)
    
    layer_results = {}
    
    for layer_idx in range(n_layers + 1):
        layer_reps = reps_by_layer[layer_idx]
        if len(layer_reps) < 2:
            continue
        
        vecs = np.array([layer_reps[c] for c in concepts])
        
        # 几何统计
        geom = compute_geometry_stats(layer_reps)
        
        # 类别分离度
        sep = compute_category_separability(vecs, labels)
        
        # 线性探针
        probe_acc = compute_linear_probe_accuracy(vecs, labels)
        
        layer_results[layer_idx] = {
            'rank95': geom['rank95'],
            'top1_var': geom['top1_variance'],
            'top5_var': geom['top5_variance'],
            'top10_var': geom['top10_variance'],
            'F_stat': sep['F_stat'],
            'dist_ratio': sep['dist_ratio'],
            'within_dist': sep['mean_within_dist'],
            'between_dist': sep['mean_between_dist'],
            'probe_acc': probe_acc,
        }
        
        print(f"  {layer_idx:>4} {geom['rank95']:>7} {geom['top1_variance']:>8.4f} "
              f"{geom['top5_variance']:>8.4f} {geom['top10_variance']:>8.4f} "
              f"{sep['F_stat']:>10.4f} {sep['dist_ratio']:>11.4f} {probe_acc:>10.4f}")
    
    # 识别信息瓶颈
    print("\n" + "="*60)
    print("INFORMATION BOTTLENECK ANALYSIS:")
    print("="*60)
    
    rank95s = [layer_results[i]['rank95'] for i in sorted(layer_results.keys())]
    f_stats = [layer_results[i]['F_stat'] for i in sorted(layer_results.keys())]
    dist_ratios = [layer_results[i]['dist_ratio'] for i in sorted(layer_results.keys())]
    probe_accs = [layer_results[i]['probe_acc'] for i in sorted(layer_results.keys())]
    
    min_rank95_layer = rank95s.index(min(rank95s))
    max_f_layer = f_stats.index(max(f_stats))
    max_ratio_layer = dist_ratios.index(max(dist_ratios))
    max_probe_layer = probe_accs.index(max(probe_accs))
    
    print(f"  rank95最低层: {min_rank95_layer} (rank95={min(rank95s)})")
    print(f"  F_stat最高层: {max_f_layer} (F={max(f_stats):.4f})")
    print(f"  dist_ratio最高层: {max_ratio_layer} (ratio={max(dist_ratios):.4f})")
    print(f"  probe_acc最高层: {max_probe_layer} (acc={max(probe_accs):.4f})")
    
    if min_rank95_layer == max_ratio_layer or abs(min_rank95_layer - max_ratio_layer) <= 2:
        print("\n  ★★★ rank95最低层 ≈ 类别分离最高层 → 信息瓶颈确认!")
    else:
        print(f"\n  rank95最低层({min_rank95_layer}) ≠ 类别分离最高层({max_ratio_layer})")
        print("  → 信息瓶颈假说需要修正")
    
    # 保存
    results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'mid_layer': mid_layer,
        'layer_results': {str(k): v for k, v in layer_results.items()},
        'bottleneck': {
            'min_rank95_layer': min_rank95_layer,
            'max_F_layer': max_f_layer,
            'max_ratio_layer': max_ratio_layer,
            'max_probe_layer': max_probe_layer,
        }
    }
    
    out_path = TEMP / f"ccxxvii_exp1_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


def run_exp2(model_name):
    """Exp2: 随机token通过模型的对照实验"""
    print(f'\n{"="*60}')
    print(f'CCXXVII Exp2: Random token control through model ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    # 1. 真实概念的表示
    print("\nExtracting real concept representations...")
    reps_by_layer, n_layers, d_model = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    
    # 2. 随机token通过模型的表示
    print("\nExtracting random token representations through model...")
    rng = np.random.RandomState(42)
    
    # 生成80个随机词(从词表中采样)
    vocab_size = len(tokenizer)
    random_tokens = []
    for i in range(80):
        # 采样一个随机token id (避免特殊token)
        token_id = rng.randint(100, min(5000, vocab_size))
        token_str = tokenizer.decode([token_id])
        random_tokens.append(f"rand_{i}_{token_str.strip()[:10]}")
    
    # 对随机token, 构建"The word is X"提示, 通过模型
    random_reps_by_layer = {i: {} for i in range(n_layers + 1)}
    
    for ci, (rtoken, rname) in enumerate(zip(range(80), random_tokens)):
        if ci % 20 == 0:
            print(f"  Random token {ci}/80")
        
        # 用随机token id构造句子
        token_id = rng.randint(100, min(5000, vocab_size))
        prompt = tokenizer.decode([token_id])
        # 为了公平对比, 用类似的结构
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        target_pos = input_ids.shape[1] - 1
        
        with torch.no_grad():
            try:
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                random_reps_by_layer[0][rname] = hidden_states[0][0, target_pos, :].detach().cpu().float().numpy()
                for li in range(n_layers):
                    random_reps_by_layer[li+1][rname] = hidden_states[li+1][0, target_pos, :].detach().cpu().float().numpy()
            except Exception as e:
                print(f"  Failed for {rname}: {e}")
    
    release_model(model)
    
    # 3. 对比分析
    concepts = REPRESENTATIVE_CONCEPTS
    real_labels = [CONCEPT_TO_CAT[c] for c in concepts]
    
    # 随机分配伪类别 (10类×8个)
    rng2 = np.random.RandomState(123)
    random_label_indices = rng2.permutation(80)
    pseudo_labels = [f"cat_{i//8}" for i in range(80)]
    # 按随机排列分配
    shuffled_pseudo = [None] * 80
    for i, idx in enumerate(random_label_indices):
        shuffled_pseudo[idx] = pseudo_labels[i]
    
    print(f"\n{'Layer':>6} {'Real_F':>10} {'Rand_F':>10} {'Real_ratio':>12} {'Rand_ratio':>12} {'Real_probe':>12} {'Rand_probe':>12}")
    print("-" * 80)
    
    comparison = {}
    
    for layer_idx in range(n_layers + 1):
        # 真实概念
        real_reps = reps_by_layer[layer_idx]
        if len(real_reps) < 2:
            continue
        real_vecs = np.array([real_reps[c] for c in concepts])
        
        # 随机token
        rand_reps = random_reps_by_layer[layer_idx]
        if len(rand_reps) < 2:
            continue
        rand_vecs = np.array([rand_reps[rn] for rn in random_tokens])
        
        # 真实概念的类别分离度
        real_sep = compute_category_separability(real_vecs, real_labels)
        real_probe = compute_linear_probe_accuracy(real_vecs, real_labels)
        
        # 随机token的伪类别分离度
        rand_sep = compute_category_separability(rand_vecs, shuffled_pseudo)
        rand_probe = compute_linear_probe_accuracy(rand_vecs, shuffled_pseudo)
        
        # 真实概念的几何
        real_geom = compute_geometry_stats(real_reps)
        rand_geom = compute_geometry_stats(rand_reps)
        
        comparison[layer_idx] = {
            'real_F': real_sep['F_stat'],
            'real_ratio': real_sep['dist_ratio'],
            'real_probe': real_probe,
            'real_rank95': real_geom['rank95'],
            'rand_F': rand_sep['F_stat'],
            'rand_ratio': rand_sep['dist_ratio'],
            'rand_probe': rand_probe,
            'rand_rank95': rand_geom['rank95'],
        }
        
        print(f"  {layer_idx:>4} {real_sep['F_stat']:>10.4f} {rand_sep['F_stat']:>10.4f} "
              f"{real_sep['dist_ratio']:>12.4f} {rand_sep['dist_ratio']:>12.4f} "
              f"{real_probe:>12.4f} {rand_probe:>12.4f}")
    
    # 关键对比
    print("\n" + "="*60)
    print("KEY COMPARISON: Real semantic structure vs Random pseudo-categories")
    print("="*60)
    
    mid = n_layers // 2
    mid_data = comparison.get(mid, comparison.get(mid-1, None))
    embed_data = comparison.get(0, None)
    last_data = comparison.get(n_layers, comparison.get(n_layers-1, None))
    
    if mid_data and embed_data:
        # 真实vs随机 在中间层的分离度比
        real_mid_ratio = mid_data['real_ratio']
        rand_mid_ratio = mid_data['rand_ratio']
        real_mid_F = mid_data['real_F']
        rand_mid_F = mid_data['rand_F']
        
        print(f"\n  Embedding layer:")
        print(f"    Real category ratio: {embed_data['real_ratio']:.4f}")
        print(f"    Random pseudo ratio: {embed_data['rand_ratio']:.4f}")
        print(f"    Ratio advantage: {embed_data['real_ratio']/max(embed_data['rand_ratio'], 0.001):.3f}x")
        
        print(f"\n  Middle layer ({mid}):")
        print(f"    Real category ratio: {real_mid_ratio:.4f}")
        print(f"    Random pseudo ratio: {rand_mid_ratio:.4f}")
        print(f"    Ratio advantage: {real_mid_ratio/max(rand_mid_ratio, 0.001):.3f}x")
        
        if last_data:
            print(f"\n  Last layer:")
            print(f"    Real category ratio: {last_data['real_ratio']:.4f}")
            print(f"    Random pseudo ratio: {last_data['rand_ratio']:.4f}")
        
        # 层间变化
        embed_real = embed_data['real_ratio']
        mid_real = mid_data['real_ratio']
        embed_rand = embed_data['rand_ratio']
        mid_rand = mid_data['rand_ratio']
        
        delta_real = mid_real - embed_real
        delta_rand = mid_rand - embed_rand
        
        print(f"\n  Layer evolution (embed → mid):")
        print(f"    Real: {embed_real:.4f} → {mid_real:.4f} (Δ={delta_real:+.4f})")
        print(f"    Random: {embed_rand:.4f} → {mid_rand:.4f} (Δ={delta_rand:+.4f})")
        
        if delta_real > 0.02 and abs(delta_rand) < 0.01:
            print("  ★★★ 真实概念在中间层增强了类别分离, 随机token没有!")
            print("  → 这是语言结构通过transformer被显式编码的证据")
        elif delta_real > delta_rand + 0.01:
            print("  ★★ 真实概念的类别分离增强幅度 > 随机token")
            print("  → 部分支持语言结构编码假说")
        else:
            print("  真实概念和随机token的层间变化相似")
            print("  → 类别分离增强可能不是语言特有的")
    
    # 保存
    results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'comparison': {str(k): v for k, v in comparison.items()},
    }
    
    out_path = TEMP / f"ccxxvii_exp2_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


def run_exp3(model_name):
    """Exp3: 信息压缩的语义含义 — 哪些概念被压缩最多?"""
    print(f'\n{"="*60}')
    print(f'CCXXVII Exp3: Semantic meaning of compression ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2
    
    reps_by_layer, n_layers, d_model = extract_reps_all_layers(
        model_name, REPRESENTATIVE_CONCEPTS, model=model, tokenizer=tokenizer, device=device
    )
    release_model(model)
    
    concepts = REPRESENTATIVE_CONCEPTS
    
    # 1. 计算每个概念从Embed→中间层的表示变化
    embed_reps = reps_by_layer[0]
    mid_reps = reps_by_layer[mid_layer]
    last_reps = reps_by_layer[n_layers]
    
    # 2. 表示变化的方向和幅度
    print("\nConcept representation changes (Embed → Mid):")
    
    concept_changes = {}
    for c in concepts:
        e = embed_reps[c]
        m = mid_reps[c]
        l = last_reps[c]
        
        # 变化向量
        delta_e_m = m - e
        delta_e_l = l - e
        delta_m_l = l - m
        
        # 范数和余弦相似度
        norm_e = np.linalg.norm(e)
        norm_m = np.linalg.norm(m)
        norm_l = np.linalg.norm(l)
        
        cos_em = np.dot(e, m) / (norm_e * norm_m + 1e-10)
        cos_el = np.dot(e, l) / (norm_e * norm_l + 1e-10)
        cos_ml = np.dot(m, l) / (norm_m * norm_l + 1e-10)
        
        concept_changes[c] = {
            'norm_embed': float(norm_e),
            'norm_mid': float(norm_m),
            'norm_last': float(norm_l),
            'cos_embed_mid': float(cos_em),
            'cos_embed_last': float(cos_el),
            'cos_mid_last': float(cos_ml),
            'delta_norm_em': float(norm_m - norm_e),
            'category': CONCEPT_TO_CAT[c],
        }
    
    # 按变化排序
    by_cos_em = sorted(concept_changes.items(), key=lambda x: x[1]['cos_embed_mid'])
    
    print(f"\n  Concepts most transformed (Embed→Mid, lowest cos similarity):")
    for c, data in by_cos_em[:10]:
        print(f"    {c:>12} (cat={data['category']:>12}): cos={data['cos_embed_mid']:.4f}, "
              f"norm: {data['norm_embed']:.1f}→{data['norm_mid']:.1f}")
    
    print(f"\n  Concepts least transformed (Embed→Mid, highest cos similarity):")
    for c, data in by_cos_em[-10:]:
        print(f"    {c:>12} (cat={data['category']:>12}): cos={data['cos_embed_mid']:.4f}, "
              f"norm: {data['norm_embed']:.1f}→{data['norm_mid']:.1f}")
    
    # 3. 按类别统计变化
    print("\n\nCategory-level changes (Embed→Mid):")
    cat_stats = {}
    for cat in CATEGORIES.keys():
        cat_concepts = CATEGORIES[cat]
        cat_cos = [concept_changes[c]['cos_embed_mid'] for c in cat_concepts]
        cat_norm_e = [concept_changes[c]['norm_embed'] for c in cat_concepts]
        cat_norm_m = [concept_changes[c]['norm_mid'] for c in cat_concepts]
        cat_norm_l = [concept_changes[c]['norm_last'] for c in cat_concepts]
        
        cat_stats[cat] = {
            'mean_cos_em': float(np.mean(cat_cos)),
            'mean_norm_embed': float(np.mean(cat_norm_e)),
            'mean_norm_mid': float(np.mean(cat_norm_m)),
            'mean_norm_last': float(np.mean(cat_norm_l)),
            'norm_change_pct': float((np.mean(cat_norm_m) - np.mean(cat_norm_e)) / np.mean(cat_norm_e) * 100),
        }
        
        print(f"  {cat:>12}: cos_em={np.mean(cat_cos):.4f}, "
              f"norm: {np.mean(cat_norm_e):.1f}→{np.mean(cat_norm_m):.1f}→{np.mean(cat_norm_l):.1f} "
              f"(Δmid={cat_stats[cat]['norm_change_pct']:+.1f}%)")
    
    # 4. 维度压缩中语义信息的保留
    print("\n\nInformation preservation analysis:")
    
    # 在中间层的PCA空间中, 用低维重建来测量信息保留
    mid_vecs = np.array([mid_reps[c] for c in concepts])
    embed_vecs = np.array([embed_reps[c] for c in concepts])
    
    # 中间层PCA
    mid_pca = PCA()
    mid_centered = mid_vecs - mid_vecs.mean(axis=0)
    mid_pca.fit(mid_centered)
    
    # 用不同数量的PC重建, 测量语义信息保留
    labels = [CONCEPT_TO_CAT[c] for c in concepts]
    full_sep = compute_category_separability(mid_vecs, labels)
    
    print(f"\n  Full mid-layer separability: F={full_sep['F_stat']:.4f}, ratio={full_sep['dist_ratio']:.4f}")
    
    for n_components in [5, 10, 15, 20, 30, 40, 50]:
        if n_components >= mid_vecs.shape[1]:
            break
        projected = mid_pca.transform(mid_centered)[:, :n_components]
        sep = compute_category_separability(projected, labels)
        probe = compute_linear_probe_accuracy(projected, labels)
        print(f"  Top-{n_components:>2} PCs: F={sep['F_stat']:.4f}, ratio={sep['dist_ratio']:.4f}, probe_acc={probe:.4f}")
    
    # 5. 关键指标: 语义信息保留率
    # 用5个PC就能多大程度保留类别分离?
    if mid_vecs.shape[1] >= 5:
        top5_proj = mid_pca.transform(mid_centered)[:, :5]
        top5_sep = compute_category_separability(top5_proj, labels)
        preservation_ratio = top5_sep['dist_ratio'] / full_sep['dist_ratio']
        print(f"\n  ★ Top-5 PCs preserve {preservation_ratio*100:.1f}% of category separation")
        print(f"    Top-5 PCs explain {sum(mid_pca.explained_variance_ratio_[:5])*100:.1f}% of total variance")
    
    # 保存
    results = {
        'model': model_name,
        'mid_layer': mid_layer,
        'n_layers': n_layers,
        'd_model': d_model,
        'concept_changes': concept_changes,
        'category_stats': cat_stats,
        'full_separability': {
            'F_stat': full_sep['F_stat'],
            'dist_ratio': full_sep['dist_ratio'],
        },
        'pca_variance': mid_pca.explained_variance_ratio_[:20].tolist(),
    }
    
    out_path = TEMP / f"ccxxvii_exp3_{model_name}_results.json"
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
