"""
CCLXXXXV(295): 断裂层输入分布质变分析
基于CCLXXXXIV发现: 32个头在断裂层全部"翻转"

核心问题: 是什么触发了头翻转? 断裂层的residual stream分布是否发生了质变?

Exp1: 逐层Residual Stream统计量分析
  - Hook每层residual stream输出
  - 计算: mean, var, kurtosis, skewness, L2 norm, per-dimension variance
  - 看断裂层是否有统计量突变

Exp2: 断裂层前后的Residual Stream几何结构
  - 计算per-category residual stream均值
  - 计算类别间距离矩阵
  - PCA分析: 维度贡献的集中度
  - 关键: 断裂层的几何结构是否质变?

Exp3: 断裂层逐维度变化分析
  - 哪些维度在断裂层变化最大?
  - 这些维度是否与特定类别相关?
  - 维度变化是否与head翻转相关?

Exp4: LayerNorm输入输出分析
  - 断裂层的layernorm输入分布
  - layernorm输出分布
  - layernorm是否"放大"了某些维度的变化?
"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXV Script started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

log("Importing torch...")
import torch

log("Importing model_utils...")
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

# ===== 13类别 × 20词 =====
CATEGORIES_13 = {
    "animal": [
        "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "donkey",
        "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit",
        "elephant", "giraffe", "zebra", "monkey", "camel",
    ],
    "bird": [
        "eagle", "hawk", "owl", "crow", "swan", "goose", "duck",
        "penguin", "parrot", "robin", "sparrow", "pigeon", "seagull",
        "falcon", "vulture", "crane", "stork", "heron", "peacock", "flamingo",
    ],
    "fish": [
        "shark", "whale", "dolphin", "salmon", "trout", "tuna",
        "cod", "bass", "carp", "catfish", "perch", "pike", "eel",
        "herring", "sardine", "anchovy", "flounder", "sole", "mackerel", "swordfish",
    ],
    "insect": [
        "ant", "bee", "spider", "butterfly", "mosquito", "fly", "wasp",
        "beetle", "cockroach", "grasshopper", "cricket", "dragonfly",
        "ladybug", "moth", "flea", "tick", "mantis", "caterpillar", "worm", "snail",
    ],
    "plant": [
        "tree", "flower", "grass", "bush", "shrub", "vine", "fern",
        "moss", "algae", "weed", "oak", "pine", "maple", "birch",
        "willow", "cactus", "bamboo", "palm", "rose", "lily",
    ],
    "fruit": [
        "apple", "orange", "banana", "grape", "pear", "peach",
        "cherry", "plum", "mango", "lemon", "lime", "melon",
        "berry", "strawberry", "blueberry", "raspberry", "fig", "date",
        "coconut", "pineapple",
    ],
    "vegetable": [
        "carrot", "potato", "tomato", "onion", "garlic", "cabbage",
        "lettuce", "spinach", "celery", "pea", "bean", "corn",
        "mushroom", "pepper", "cucumber", "pumpkin", "squash",
        "radish", "turnip", "broccoli",
    ],
    "body_part": [
        "hand", "foot", "head", "heart", "brain", "eye", "ear",
        "nose", "mouth", "tooth", "neck", "shoulder", "arm",
        "finger", "knee", "chest", "back", "hip", "ankle", "wrist",
    ],
    "tool": [
        "hammer", "knife", "scissors", "saw", "drill", "wrench",
        "screwdriver", "plier", "axe", "chisel", "ruler", "file",
        "clamp", "level", "shovel", "rake", "hoe", "trowel",
        "spade", "mallet",
    ],
    "vehicle": [
        "car", "bus", "truck", "train", "bicycle", "motorcycle",
        "airplane", "helicopter", "boat", "ship", "submarine",
        "rocket", "tractor", "van", "taxi", "ambulance",
        "sled", "canoe", "wagon", "cart",
    ],
    "clothing": [
        "shirt", "dress", "hat", "coat", "shoe", "belt", "scarf",
        "glove", "jacket", "sweater", "vest", "skirt", "pants",
        "jeans", "sock", "boot", "sandal", "tie", "uniform", "cape",
    ],
    "weapon": [
        "sword", "spear", "bow", "arrow", "shield", "axe_w",
        "dagger", "mace", "pike_w", "lance", "crossbow", "catapult",
        "pistol", "rifle", "cannon", "grenade", "dynamite",
        "knife_w", "club", "whip",
    ],
    "furniture": [
        "chair", "table", "desk", "bed", "sofa", "couch", "shelf",
        "cabinet", "drawer", "wardrobe", "dresser", "bench",
        "stool", "armchair", "bookcase", "mirror", "lamp",
        "rug", "curtain", "pillow",
    ],
}

SUPERCLASS_MAP = {
    "animal": "animate", "bird": "animate", "fish": "animate", "insect": "animate",
    "plant": "plant", "fruit": "plant", "vegetable": "plant",
    "body_part": "body",
    "tool": "artifact", "vehicle": "artifact", "clothing": "artifact",
    "weapon": "artifact", "furniture": "artifact",
}

# 断裂层配置 (来自CCLXXXXIII/CCLXXXXIV)
FRACTURE_LAYERS = {
    "qwen3": 6,
    "glm4": 2,
    "deepseek7b": 7,
}


def get_residual_streams(model, tokenizer, device, model_name, n_layers):
    """
    Hook每层的residual stream, 收集每个token在每层的残差向量
    """
    layers_list = get_layers(model)
    d_model = model.config.hidden_size
    frac_layer = FRACTURE_LAYERS[model_name]
    
    # 选择采样层: 断裂层前后各3层 + 每5层采样
    sample_layers = set()
    for l in range(max(0, frac_layer - 3), min(n_layers, frac_layer + 4)):
        sample_layers.add(l)
    for l in range(0, n_layers, 5):
        sample_layers.add(l)
    sample_layers.add(n_layers - 1)
    sample_layers = sorted(sample_layers)
    log(f"  Sample layers: {sample_layers}")
    
    # 为每个类别收集residual stream
    all_residuals = {}  # layer_idx -> cat -> [vec1, vec2, ...]
    for l in sample_layers:
        all_residuals[l] = defaultdict(list)
    
    # Hook function
    residual_cache = {}
    hook_handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output可能是tuple, 取第一个
            if isinstance(output, tuple):
                residual_cache[layer_idx] = output[0].detach()
            else:
                residual_cache[layer_idx] = output.detach()
        return hook_fn
    
    # 注册hooks - hook在每层的输出
    # 对于Qwen/GLM架构, residual stream在input_layernorm之前
    # 我们hook model.model.layers[i]的forward, 获取residual
    for l in sample_layers:
        h = layers_list[l].register_forward_hook(make_hook(l))
        hook_handles.append(h)
    
    # 逐类别逐token前向传播
    cat_names = sorted(CATEGORIES_13.keys())
    
    for cat in cat_names:
        words = CATEGORIES_13[cat]
        for word in words:
            # Tokenize
            inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            # 前向传播
            with torch.no_grad():
                residual_cache.clear()
                try:
                    model(input_ids)
                except Exception as e:
                    log(f"  ERROR on {cat}/{word}: {e}")
                    continue
            
            # 收集每个采样层的residual stream
            for l in sample_layers:
                if l in residual_cache:
                    rs = residual_cache[l]  # [seq_len, d_model]
                    # 取最后一个token的residual
                    last_rs = rs[0, -1, :].cpu().float().numpy()
                    all_residuals[l][cat].append(last_rs)
    
    # 清理hooks
    for h in hook_handles:
        h.remove()
    
    return all_residuals, sample_layers


def compute_stats(vecs):
    """计算一组向量的统计量"""
    vecs = np.array(vecs)  # [N, D]
    mean = vecs.mean(axis=0)
    var_per_dim = vecs.var(axis=0)
    total_var = var_per_dim.mean()
    
    # L2 norms
    norms = np.linalg.norm(vecs, axis=1)
    avg_norm = norms.mean()
    
    # Kurtosis and skewness per dimension (sample a few dims to save time)
    d = vecs.shape[1]
    sample_dims = np.linspace(0, d-1, min(100, d), dtype=int)
    kurtosis_vals = []
    skew_vals = []
    for dim in sample_dims:
        vals = vecs[:, dim]
        v = np.var(vals)
        if v > 1e-10:
            m = np.mean(vals)
            s = np.std(vals)
            if s > 1e-10:
                skew_vals.append(np.mean(((vals - m) / s) ** 3))
                kurtosis_vals.append(np.mean(((vals - m) / s) ** 4) - 3)
    
    return {
        'mean_norm': avg_norm,
        'total_var': total_var,
        'var_per_dim_mean': var_per_dim.mean(),
        'var_per_dim_max': var_per_dim.max(),
        'var_per_dim_min': var_per_dim.min(),
        'avg_kurtosis': np.mean(kurtosis_vals) if kurtosis_vals else 0,
        'avg_skewness': np.mean(skew_vals) if skew_vals else 0,
    }


def run_exp1(all_residuals, sample_layers, frac_layer, model_name):
    """
    Exp1: 逐层Residual Stream统计量分析
    """
    log(f"\n{'='*60}")
    log(f"Exp1: Per-Layer Residual Stream Statistics ({model_name})")
    log(f"{'='*60}")
    
    # 每层的全局统计量(所有token合在一起)
    log(f"\n--- Global statistics per layer ---")
    log(f"{'Layer':>6} {'Norm':>8} {'TotalVar':>10} {'VarMax':>10} {'VarMean':>10} {'Kurtosis':>10} {'Skewness':>10} {'Marker':>20}")
    
    layer_stats = {}
    for l in sample_layers:
        all_vecs = []
        for cat in all_residuals[l]:
            all_vecs.extend(all_residuals[l][cat])
        
        if len(all_vecs) < 2:
            continue
        
        stats = compute_stats(all_vecs)
        layer_stats[l] = stats
        
        marker = ""
        if l == frac_layer:
            marker = "*** FRACTURE ***"
        elif l == frac_layer - 1:
            marker = "*** Pre-fracture ***"
        elif l == frac_layer + 1:
            marker = "*** Post-fracture ***"
        
        log(f"L{l:>4} {stats['mean_norm']:>8.3f} {stats['total_var']:>10.4f} "
            f"{stats['var_per_dim_max']:>10.4f} {stats['var_per_dim_mean']:>10.5f} "
            f"{stats['avg_kurtosis']:>10.3f} {stats['avg_skewness']:>10.3f} {marker}")
    
    # 检测统计量突变
    log(f"\n--- Statistic jumps (ratio of consecutive layers) ---")
    sorted_layers = sorted(layer_stats.keys())
    for i in range(1, len(sorted_layers)):
        l_prev = sorted_layers[i-1]
        l_curr = sorted_layers[i]
        s_prev = layer_stats[l_prev]
        s_curr = layer_stats[l_curr]
        
        norm_ratio = s_curr['mean_norm'] / max(s_prev['mean_norm'], 1e-10)
        var_ratio = s_curr['total_var'] / max(s_prev['total_var'], 1e-10)
        kurt_diff = s_curr['avg_kurtosis'] - s_prev['avg_kurtosis']
        skew_diff = s_curr['avg_skewness'] - s_prev['avg_skewness']
        
        marker = ""
        if l_curr == frac_layer:
            marker = "*** FRACTURE ***"
        
        log(f"L{l_prev}->L{l_curr}: norm_ratio={norm_ratio:.3f}, var_ratio={var_ratio:.3f}, "
            f"kurt_diff={kurt_diff:+.3f}, skew_diff={skew_diff:+.3f} {marker}")
    
    # Per-category statistics at key layers
    log(f"\n--- Per-category statistics at key layers ---")
    key_layers = [l for l in [max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1, min(sample_layers[-1], frac_layer+2)] 
                  if l in all_residuals]
    
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"\n  --- {marker} ---")
        log(f"  {'Cat':>12} {'Norm':>8} {'TotalVar':>10} {'Kurtosis':>10}")
        
        for cat in sorted(all_residuals[l].keys()):
            vecs = all_residuals[l][cat]
            if len(vecs) < 2:
                continue
            stats = compute_stats(vecs)
            log(f"  {cat:>12} {stats['mean_norm']:>8.3f} {stats['total_var']:>10.4f} {stats['avg_kurtosis']:>10.3f}")
    
    return layer_stats


def run_exp2(all_residuals, sample_layers, frac_layer, model_name):
    """
    Exp2: 断裂层前后的Residual Stream几何结构
    - 类别间距离矩阵
    - PCA维度集中度
    """
    log(f"\n{'='*60}")
    log(f"Exp2: Geometric Structure Analysis ({model_name})")
    log(f"{'='*60}")
    
    key_layers = [l for l in [max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1, min(sample_layers[-1], frac_layer+2)]
                  if l in all_residuals]
    
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        log(f"\n--- Layer {marker}: Geometric structure ---")
        
        # 1. 计算per-category均值
        cat_means = {}
        for cat in all_residuals[l]:
            vecs = np.array(all_residuals[l][cat])
            cat_means[cat] = vecs.mean(axis=0)
        
        # 2. 类别间cosine similarity矩阵
        cat_names = sorted(cat_means.keys())
        log(f"\n  Inter-category cosine similarity (top-5 most similar pairs):")
        cos_pairs = []
        for i, c1 in enumerate(cat_names):
            for j, c2 in enumerate(cat_names):
                if i >= j:
                    continue
                v1 = cat_means[c1]
                v2 = cat_means[c2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_pairs.append((c1, c2, cos))
        
        cos_pairs.sort(key=lambda x: x[2], reverse=True)
        for c1, c2, cos in cos_pairs[:5]:
            log(f"    {c1:>12} - {c2:<12}: cos={cos:.4f}")
        log(f"  ...")
        for c1, c2, cos in cos_pairs[-3:]:
            log(f"    {c1:>12} - {c2:<12}: cos={cos:.4f}")
        
        # 3. PCA维度集中度
        all_vecs = []
        for cat in cat_names:
            all_vecs.extend(all_residuals[l][cat])
        all_vecs = np.array(all_vecs)
        
        # Center
        all_vecs_centered = all_vecs - all_vecs.mean(axis=0, keepdims=True)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(all_vecs_centered, full_matrices=False)
            total_var = (S ** 2).sum()
            cumvar = np.cumsum(S ** 2) / total_var
            
            log(f"\n  PCA concentration:")
            log(f"    Top-1 PC: {S[0]**2/total_var*100:.1f}% variance")
            log(f"    Top-3 PCs: {cumvar[2]*100:.1f}%")
            log(f"    Top-5 PCs: {cumvar[4]*100:.1f}%")
            log(f"    Top-10 PCs: {cumvar[9]*100:.1f}%")
            log(f"    Top-20 PCs: {cumvar[min(19,len(cumvar)-1)]*100:.1f}%")
            log(f"    Effective dim (95% var): {np.searchsorted(cumvar, 0.95) + 1}")
            log(f"    Effective dim (99% var): {np.searchsorted(cumvar, 0.99) + 1}")
            
            # 4. PC0的方向 - 哪些类别在PC0的两端
            pc0 = Vt[0]
            pc0_scores = {}
            for cat in cat_names:
                vecs = np.array(all_residuals[l][cat])
                scores = vecs @ pc0
                pc0_scores[cat] = scores.mean()
            
            sorted_cats = sorted(pc0_scores.items(), key=lambda x: x[1], reverse=True)
            log(f"\n  PC0 category scores (top-5 / bottom-5):")
            for cat, score in sorted_cats[:5]:
                log(f"    {cat:>12}: {score:+.4f}")
            log(f"    ...")
            for cat, score in sorted_cats[-5:]:
                log(f"    {cat:>12}: {score:+.4f}")
            
        except Exception as e:
            log(f"  PCA failed: {e}")
    
    # 5. 断裂层前后的类别间距离变化
    log(f"\n--- Inter-category distance change at fracture ---")
    
    if frac_layer - 1 in all_residuals and frac_layer in all_residuals:
        cat_names = sorted(all_residuals[frac_layer].keys())
        
        log(f"\n  Average inter-category distance:")
        for l in [frac_layer - 1, frac_layer]:
            dists = []
            cat_means = {}
            for cat in cat_names:
                vecs = np.array(all_residuals[l][cat])
                cat_means[cat] = vecs.mean(axis=0)
            
            for c1, c2 in combinations(cat_names, 2):
                d = np.linalg.norm(cat_means[c1] - cat_means[c2])
                dists.append(d)
            
            marker = "FRACTURE" if l == frac_layer else "Pre-fracture"
            log(f"    L{l} ({marker}): mean_dist={np.mean(dists):.4f}, std_dist={np.std(dists):.4f}")
        
        # Per-superclass distance change
        log(f"\n  Per-superclass avg distance to grand mean:")
        for l in [frac_layer - 1, frac_layer]:
            marker = "FRACTURE" if l == frac_layer else "Pre-fracture"
            all_vecs_l = []
            for cat in cat_names:
                all_vecs_l.extend(all_residuals[l][cat])
            grand_mean = np.mean(all_vecs_l, axis=0)
            
            sup_dists = defaultdict(list)
            for cat in cat_names:
                vecs = np.array(all_residuals[l][cat])
                cat_mean = vecs.mean(axis=0)
                dist = np.linalg.norm(cat_mean - grand_mean)
                sup = SUPERCLASS_MAP[cat]
                sup_dists[sup].append(dist)
            
            log(f"    L{l} ({marker}):")
            for sup in ["animate", "plant", "body", "artifact"]:
                if sup in sup_dists:
                    log(f"      {sup:>10}: mean_dist={np.mean(sup_dists[sup]):.4f}")


def run_exp3(all_residuals, sample_layers, frac_layer, model_name):
    """
    Exp3: 断裂层逐维度变化分析
    - 哪些维度变化最大?
    - 变化最大的维度是否与特定类别相关?
    """
    log(f"\n{'='*60}")
    log(f"Exp3: Per-Dimension Change Analysis ({model_name})")
    log(f"{'='*60}")
    
    if frac_layer - 1 not in all_residuals or frac_layer not in all_residuals:
        log("  Missing pre/fracture layers, skipping")
        return
    
    cat_names = sorted(all_residuals[frac_layer].keys())
    d_model = len(all_residuals[frac_layer][cat_names[0]][0])
    
    # 1. 逐维度variance变化
    log(f"\n--- Per-dimension variance change (pre-fracture -> fracture) ---")
    
    pre_var = np.zeros(d_model)
    frac_var = np.zeros(d_model)
    
    for dim in range(d_model):
        pre_vals = []
        frac_vals = []
        for cat in cat_names:
            for vec in all_residuals[frac_layer - 1][cat]:
                pre_vals.append(vec[dim])
            for vec in all_residuals[frac_layer][cat]:
                frac_vals.append(vec[dim])
        pre_var[dim] = np.var(pre_vals)
        frac_var[dim] = np.var(frac_vals)
    
    var_ratio = frac_var / np.maximum(pre_var, 1e-10)
    
    # Top-20 dimensions with largest variance increase
    top_increase_dims = np.argsort(var_ratio)[-20:][::-1]
    top_decrease_dims = np.argsort(var_ratio)[:20]
    
    log(f"\n  Top-20 dims with LARGEST variance increase:")
    for dim in top_increase_dims:
        log(f"    Dim {dim:>5}: pre_var={pre_var[dim]:.6f}, frac_var={frac_var[dim]:.6f}, ratio={var_ratio[dim]:.2f}")
    
    log(f"\n  Top-20 dims with LARGEST variance decrease:")
    for dim in top_decrease_dims:
        log(f"    Dim {dim:>5}: pre_var={pre_var[dim]:.6f}, frac_var={frac_var[dim]:.6f}, ratio={var_ratio[dim]:.2f}")
    
    # 2. 维度变化与类别的关系
    log(f"\n--- Top varying dimensions: category correlation ---")
    
    # 对variance变化最大的维度, 看哪些类别贡献了变化
    for dim in top_increase_dims[:5]:
        log(f"\n  Dim {dim} (var_ratio={var_ratio[dim]:.2f}):")
        for cat in cat_names:
            pre_vals = [vec[dim] for vec in all_residuals[frac_layer - 1][cat]]
            frac_vals = [vec[dim] for vec in all_residuals[frac_layer][cat]]
            pre_mean = np.mean(pre_vals)
            frac_mean = np.mean(frac_vals)
            shift = frac_mean - pre_mean
            log(f"    {cat:>12}: pre_mean={pre_mean:+.4f}, frac_mean={frac_mean:+.4f}, shift={shift:+.4f}")
    
    # 3. 维度变化的全局分布
    log(f"\n--- Variance ratio distribution ---")
    finite_ratios = var_ratio[np.isfinite(var_ratio)]
    log(f"  Mean var_ratio: {np.mean(finite_ratios):.3f}")
    log(f"  Median var_ratio: {np.median(finite_ratios):.3f}")
    log(f"  % dims with ratio > 2: {(finite_ratios > 2).mean()*100:.1f}%")
    log(f"  % dims with ratio > 5: {(finite_ratios > 5).mean()*100:.1f}%")
    log(f"  % dims with ratio < 0.5: {(finite_ratios < 0.5).mean()*100:.1f}%")
    log(f"  % dims with ratio < 0.2: {(finite_ratios < 0.2).mean()*100:.1f}%")
    
    # 4. Per-superclass维度变化
    log(f"\n--- Per-superclass dimension shift magnitude ---")
    for sup in ["animate", "plant", "body", "artifact"]:
        sup_cats = [c for c in cat_names if SUPERCLASS_MAP[c] == sup]
        if not sup_cats:
            continue
        
        pre_vecs = []
        frac_vecs = []
        for cat in sup_cats:
            pre_vecs.extend(all_residuals[frac_layer - 1][cat])
            frac_vecs.extend(all_residuals[frac_layer][cat])
        
        pre_mean = np.mean(pre_vecs, axis=0)
        frac_mean = np.mean(frac_vecs, axis=0)
        shift = frac_mean - pre_mean
        shift_norm = np.linalg.norm(shift)
        
        # 与variance变化最大的维度的相关性
        top_dims = top_increase_dims[:50]
        shift_in_top = np.linalg.norm(shift[top_dims])
        shift_total_ratio = shift_in_top / max(shift_norm, 1e-10)
        
        log(f"  {sup:>10}: shift_norm={shift_norm:.4f}, shift_in_top50_dims={shift_in_top:.4f}, "
            f"ratio={shift_total_ratio:.3f}")


def run_exp4(model, tokenizer, device, model_name, n_layers):
    """
    Exp4: LayerNorm输入输出分析
    - 断裂层的layernorm输入分布
    - layernorm输出分布
    - layernorm是否放大了某些变化?
    """
    log(f"\n{'='*60}")
    log(f"Exp4: LayerNorm Input/Output Analysis ({model_name})")
    log(f"{'='*60}")
    
    layers_list = get_layers(model)
    d_model = model.config.hidden_size
    frac_layer = FRACTURE_LAYERS[model_name]
    
    # 选择关键层
    key_layers = sorted(set([max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1]))
    
    # Hook layernorm input and output
    ln_cache = {}
    hook_handles = []
    
    def make_ln_hook(layer_idx, is_post_attn=False):
        def hook_fn(module, input, output):
            key = f"L{layer_idx}_{'post_attn' if is_post_attn else 'input'}"
            # input[0] is the tensor before layernorm
            if isinstance(input, tuple) and len(input) > 0:
                ln_cache[f"{key}_input"] = input[0].detach().clone()
            # output after layernorm
            if isinstance(output, tuple):
                ln_cache[f"{key}_output"] = output[0].detach().clone()
            else:
                ln_cache[f"{key}_output"] = output.detach().clone()
        return hook_fn
    
    # 注册hooks on input_layernorm and post_attention_layernorm
    for l in key_layers:
        layer = layers_list[l]
        
        # Input layernorm
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(make_ln_hook(l, is_post_attn=False))
                hook_handles.append(h)
                break
        
        # Post-attention layernorm
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                h = ln.register_forward_hook(make_ln_hook(l, is_post_attn=True))
                hook_handles.append(h)
                break
    
    # 前向传播收集数据
    cat_names = sorted(CATEGORIES_13.keys())
    ln_data = defaultdict(lambda: defaultdict(list))  # key -> cat -> [vec, ...]
    
    for cat in cat_names:
        words = CATEGORIES_13[cat]
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            with torch.no_grad():
                ln_cache.clear()
                try:
                    model(input_ids)
                except Exception as e:
                    continue
            
            # 收集layernorm数据
            for key in list(ln_cache.keys()):
                if key.endswith('_input') or key.endswith('_output'):
                    vec = ln_cache[key][0, -1, :].cpu().float().numpy()
                    ln_data[key][cat].append(vec)
    
    # 清理hooks
    for h in hook_handles:
        h.remove()
    
    # 分析layernorm效果
    log(f"\n--- LayerNorm statistics at key layers ---")
    
    for l in key_layers:
        marker = "FRACTURE" if l == frac_layer else f"L{l}"
        
        for ln_type in ["input", "post_attn"]:
            input_key = f"L{l}_{ln_type}_input"
            output_key = f"L{l}_{ln_type}_output"
            
            if input_key not in ln_data or output_key not in ln_data:
                continue
            
            log(f"\n  --- {marker} {ln_type}_layernorm ---")
            
            # 计算输入和输出的统计量
            for label, key in [("LN-input", input_key), ("LN-output", output_key)]:
                all_vecs = []
                for cat in cat_names:
                    all_vecs.extend(ln_data[key][cat])
                all_vecs = np.array(all_vecs)
                
                norms = np.linalg.norm(all_vecs, axis=1)
                per_dim_var = all_vecs.var(axis=0)
                
                log(f"    {label}: norm_mean={norms.mean():.4f}, norm_std={norms.std():.4f}, "
                    f"var_mean={per_dim_var.mean():.6f}, var_max={per_dim_var.max():.6f}")
            
            # LayerNorm的效果: 输入的variance分布 vs 输出的variance分布
            input_vecs = []
            output_vecs = []
            for cat in cat_names:
                input_vecs.extend(ln_data[input_key][cat])
                output_vecs.extend(ln_data[output_key][cat])
            input_vecs = np.array(input_vecs)
            output_vecs = np.array(output_vecs)
            
            # 输入的per-dim variance
            input_var = input_vecs.var(axis=0)
            output_var = output_vecs.var(axis=0)
            
            # LayerNorm的"放大因子": 哪些维度的variance变化最大
            var_change = output_var - input_var
            top_increase = np.argsort(var_change)[-10:][::-1]
            top_decrease = np.argsort(var_change)[:10]
            
            log(f"\n    {ln_type}_LN: Top-10 dims with largest variance INCREASE:")
            for dim in top_increase:
                log(f"      Dim {dim:>5}: input_var={input_var[dim]:.6f}, output_var={output_var[dim]:.6f}, "
                    f"change={var_change[dim]:+.6f}")
            
            log(f"\n    {ln_type}_LN: Top-10 dims with largest variance DECREASE:")
            for dim in top_decrease:
                log(f"      Dim {dim:>5}: input_var={input_var[dim]:.6f}, output_var={output_var[dim]:.6f}, "
                    f"change={var_change[dim]:+.6f}")
            
            # LayerNorm对类别间距离的影响
            log(f"\n    {ln_type}_LN: Effect on inter-category distance:")
            for label, vecs_dict in [("Before LN", ln_data[input_key]), ("After LN", ln_data[output_key])]:
                cat_means = {}
                for cat in cat_names:
                    if cat in vecs_dict and len(vecs_dict[cat]) > 0:
                        cat_means[cat] = np.mean(vecs_dict[cat], axis=0)
                
                if len(cat_means) < 2:
                    continue
                
                dists = []
                for c1, c2 in combinations(sorted(cat_means.keys()), 2):
                    d = np.linalg.norm(cat_means[c1] - cat_means[c2])
                    dists.append(d)
                
                log(f"      {label}: mean_dist={np.mean(dists):.4f}, std_dist={np.std(dists):.4f}")
    
    # 断裂层 vs 前一层的layernorm效果对比
    if frac_layer - 1 in key_layers:
        log(f"\n--- LayerNorm effect comparison: Pre-fracture vs Fracture ---")
        
        for ln_type in ["input", "post_attn"]:
            pre_in_key = f"L{frac_layer-1}_{ln_type}_input"
            pre_out_key = f"L{frac_layer-1}_{ln_type}_output"
            frac_in_key = f"L{frac_layer}_{ln_type}_input"
            frac_out_key = f"L{frac_layer}_{ln_type}_output"
            
            if not all(k in ln_data for k in [pre_in_key, pre_out_key, frac_in_key, frac_out_key]):
                continue
            
            log(f"\n  {ln_type}_layernorm:")
            
            # 计算每层的类别间cosine similarity
            for label, in_key, out_key in [
                ("Pre-fracture", pre_in_key, pre_out_key),
                ("Fracture", frac_in_key, frac_out_key),
            ]:
                in_means = {}
                out_means = {}
                for cat in cat_names:
                    if cat in ln_data[in_key]:
                        in_means[cat] = np.mean(ln_data[in_key][cat], axis=0)
                    if cat in ln_data[out_key]:
                        out_means[cat] = np.mean(ln_data[out_key][cat], axis=0)
                
                # 类别间平均cos
                in_coses = []
                out_coses = []
                for c1, c2 in combinations(sorted(in_means.keys()), 2):
                    v1, v2 = in_means[c1], in_means[c2]
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    in_coses.append(cos)
                for c1, c2 in combinations(sorted(out_means.keys()), 2):
                    v1, v2 = out_means[c1], out_means[c2]
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    out_coses.append(cos)
                
                log(f"    {label}: avg_cos_beforeLN={np.mean(in_coses):.4f}, "
                    f"avg_cos_afterLN={np.mean(out_coses):.4f}, "
                    f"cos_change={np.mean(out_coses)-np.mean(in_coses):+.4f}")


def run_model(model_name):
    log(f"=== Starting {model_name} ===")
    try:
        log("Loading model...")
        model, tokenizer, device = load_model(model_name)
        info = get_model_info(model, model_name)
        log(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
        
        frac_layer = FRACTURE_LAYERS[model_name]
        
        # ===== Exp1-3: 需要residual stream数据 =====
        log("Collecting residual streams...")
        all_residuals, sample_layers = get_residual_streams(
            model, tokenizer, device, model_name, info.n_layers
        )
        
        # 检查数据完整性
        total_tokens = sum(len(v) for v in all_residuals.get(frac_layer, {}).values())
        log(f"  Total tokens at fracture layer: {total_tokens}")
        
        if total_tokens < 100:
            log(f"  WARNING: Too few tokens, results may be unreliable")
        
        # Exp1: 逐层统计量
        layer_stats = run_exp1(all_residuals, sample_layers, frac_layer, model_name)
        
        # Exp2: 几何结构
        run_exp2(all_residuals, sample_layers, frac_layer, model_name)
        
        # Exp3: 逐维度变化
        run_exp3(all_residuals, sample_layers, frac_layer, model_name)
        
        # ===== Exp4: 需要layernorm数据 =====
        run_exp4(model, tokenizer, device, model_name, info.n_layers)
        
        # 释放模型
        log("Releasing model...")
        release_model(model)
        
        # 保存结果
        result_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxxv")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存统计量
        save_stats = {}
        for l, stats in layer_stats.items():
            save_stats[str(l)] = {k: float(v) for k, v in stats.items()}
        
        with open(result_dir / "layer_stats.json", 'w') as f:
            json.dump(save_stats, f, indent=2)
        
        log(f"Results saved to {result_dir}")
        log(f"=== Finished {model_name} ===\n")
        
    except Exception as e:
        log(f"ERROR in {model_name}: {e}")
        traceback.print_exc()
        log(traceback.format_exc())
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 清空日志
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")
    
    log("CCLXXXXV: Input Distribution Analysis at Fracture Layer")
    log("=" * 60)
    
    # 逐个模型运行, 避免GPU内存溢出
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        run_model(model_name)
        log(f"Waiting 10s before next model...")
        time.sleep(10)
    
    log("=== All models completed ===")
