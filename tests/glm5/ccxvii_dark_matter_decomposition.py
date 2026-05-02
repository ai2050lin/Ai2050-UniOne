"""
CCXVII(367): 暗物质分解与概念有效维度
=========================================

CCXVI发现: 77-85%的概念信号不在W_U+W_E空间中("暗物质")。
但steering仍然有效。

本实验回答三个关键问题:
1. 概念的有效维度是多少? 虽然delta在d_model维空间中, 但PCA能压缩到多少维?
2. 暗物质在什么空间中? 是否在各层MLP/Attn的输出空间中?
3. 暗物质是否必要? 只用W_U分量的delta做steering, 效果如何?

三个实验:
  Exp1: 概念有效维度 — 所有概念delta的PCA分析
  Exp2: 暗物质分解 — 暗物质在MLP/Attn输出空间中的投影
  Exp3: W_U-only steering — 只用W_U分量的delta做steering

用法:
  python ccxvii_dark_matter_decomposition.py --model qwen3 --exp 1
  python ccxvii_dark_matter_decomposition.py --model qwen3 --exp 2
  python ccxvii_dark_matter_decomposition.py --model qwen3 --exp 3
  python ccxvii_dark_matter_decomposition.py --model qwen3 --exp all
"""

import argparse, os, sys, json, gc, warnings, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANS_TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)

TEMP = Path("tests/glm5_temp")

CONCEPTS = {
    "apple": {
        "templates": ["The word is apple", "I ate an apple", "A red apple", "The apple fell", "Apple is a fruit"],
        "probe_words": ["fruit", "red", "eat", "sweet", "tree", "banana", "orange", "pear"],
    },
    "dog": {
        "templates": ["The word is dog", "A big dog", "The dog barked", "My pet dog", "Dog is an animal"],
        "probe_words": ["animal", "pet", "bark", "fur", "puppy", "cat", "wolf", "horse"],
    },
    "king": {
        "templates": ["The word is king", "The king ruled", "A wise king", "The king and queen", "King is a ruler"],
        "probe_words": ["queen", "ruler", "royal", "throne", "crown", "prince", "emperor", "lord"],
    },
    "doctor": {
        "templates": ["The word is doctor", "The doctor helped", "A good doctor", "Visit the doctor", "Doctor treats patients"],
        "probe_words": ["hospital", "patient", "medicine", "nurse", "health", "surgeon", "clinic", "cure"],
    },
    "mountain": {
        "templates": ["The word is mountain", "A tall mountain", "The mountain peak", "Climb the mountain", "Mountain is high"],
        "probe_words": ["peak", "high", "climb", "snow", "valley", "hill", "summit", "rock"],
    },
    "ocean": {
        "templates": ["The word is ocean", "The deep ocean", "Ocean waves", "Swim in the ocean", "Ocean is vast"],
        "probe_words": ["sea", "deep", "wave", "water", "fish", "beach", "coast", "blue"],
    },
    "love": {
        "templates": ["The word is love", "Feel the love", "Love is strong", "Show your love", "Love and peace"],
        "probe_words": ["heart", "feel", "care", "passion", "emotion", "hate", "romance", "affection"],
    },
    "science": {
        "templates": ["The word is science", "Study of science", "Science advances", "Modern science", "Science is knowledge"],
        "probe_words": ["research", "study", "theory", "experiment", "physics", "art", "biology", "data"],
    },
}

BASELINE_TEXT = "The word is"


def collect_states_at_layers(model, tokenizer, device, text, capture_layers):
    """用hooks收集指定层的残差流状态"""
    captured = {}
    all_layers = get_layers(model)
    def make_hook(li):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured[li] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    hooks = []
    for li in capture_layers:
        if li < len(all_layers):
            hooks.append(all_layers[li].register_forward_hook(make_hook(li)))
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return {}, None
    for h in hooks: h.remove()
    logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    gc.collect()
    return captured, logits


def collect_attn_mlp_outputs(model, tokenizer, device, text, target_layer):
    """收集目标层的Attn输出和MLP输出"""
    all_layers = get_layers(model)
    if target_layer >= len(all_layers):
        return None, None
    
    layer = all_layers[target_layer]
    captured = {}
    
    def make_attn_hook():
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured['attn_out'] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured['attn_out'] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    
    def make_mlp_hook():
        def hook(module, inp, output):
            if isinstance(output, tuple):
                captured['mlp_out'] = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                captured['mlp_out'] = output[0, -1, :].detach().float().cpu().numpy()
        return hook
    
    hooks = []
    if hasattr(layer, 'self_attn'):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook()))
    if hasattr(layer, 'mlp'):
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook()))
    
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception as e:
            print(f"  Forward failed: {e}")
            for h in hooks: h.remove()
            return None, None
    
    for h in hooks: h.remove()
    attn_out = captured.get('attn_out')
    mlp_out = captured.get('mlp_out')
    gc.collect()
    return attn_out, mlp_out


def compute_steering_effect(model, tokenizer, device, probe_words, 
                            inject_layer, direction, alpha=0.5):
    """
    在指定层注入方向, 测量对probe_words的logit变化
    
    Returns:
        dict: {word: logit_change}
    """
    all_layers = get_layers(model)
    
    # 获取probe token IDs
    probe_ids = {}
    for w in probe_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            probe_ids[w] = ids[0]
    
    # Baseline forward
    baseline_text = BASELINE_TEXT
    input_ids = tokenizer.encode(baseline_text, add_special_tokens=True, return_tensors="pt").to(device)
    
    # Hook to inject and capture logits
    def make_inject_hook(direction_tensor, alpha_val):
        injected = [False]
        def hook(module, inp, output):
            if not injected[0]:
                if isinstance(output, tuple):
                    modified = output[0].clone()
                    modified[0, -1, :] += alpha_val * direction_tensor.to(modified.dtype)
                    return (modified,) + output[1:]
                else:
                    modified = output.clone()
                    modified[0, -1, :] += alpha_val * direction_tensor.to(modified.dtype)
                    return modified
                injected[0] = True
            return output
        return hook
    
    # Baseline logits
    with torch.no_grad():
        try:
            base_out = model(input_ids=input_ids)
            base_logits = base_out.logits[0, -1, :].detach().float().cpu().numpy()
        except:
            return {}
    
    # Injected logits
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
    hook = all_layers[inject_layer].register_forward_hook(
        make_inject_hook(direction_tensor, alpha)
    )
    
    with torch.no_grad():
        try:
            inj_out = model(input_ids=input_ids)
            inj_logits = inj_out.logits[0, -1, :].detach().float().cpu().numpy()
        except:
            hook.remove()
            return {}
    
    hook.remove()
    gc.collect()
    
    # Compute logit changes
    results = {}
    for w, tid in probe_ids.items():
        results[w] = float(inj_logits[tid] - base_logits[tid])
    
    return results


# ================================================================
# Exp1: 概念有效维度 — PCA分析
# ================================================================
def run_exp1(model, tokenizer, device, model_info, concepts):
    """
    PCA分析所有概念的delta, 测量有效维度
    
    方法:
    1. 收集所有概念在关键层的delta (6-8概念 x 5模板 = 30-40个样本)
    2. 对delta矩阵做PCA (每个概念5模板产生5个delta)
    3. 分析累积方差解释率: 前1/5/10/20个PC能解释多少?
    4. 计算Shannon有效秩
    
    这回答: 所有概念是否共享低维子空间? 概念"维度"是多少?
    """
    print(f"\n{'='*60}")
    print(f"  Exp1: Concept Effective Dimension (PCA)")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]
    
    # 收集所有概念在所有层的delta
    all_capture = list(range(n_layers))
    
    print(f"  Collecting baseline states...")
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    # 对每个概念, 收集每个模板的delta
    # delta_matrix[concept][layer] = [delta_template1, delta_template2, ...]
    delta_matrix = {}
    for cname, cdata in concepts.items():
        delta_matrix[cname] = {}
        for l in all_capture:
            delta_matrix[cname][l] = []
        
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    delta_matrix[cname][l].append(states[l] - bl_all[l])
        gc.collect()
    
    # 对每个关键层做PCA
    results = {}
    
    for l in key_layers:
        print(f"\n  --- Layer {l} PCA ---")
        
        # 构建delta矩阵: 每行是一个(概念, 模板)的delta
        all_deltas = []
        labels = []  # (concept_name, template_idx)
        for cname, cdata in concepts.items():
            if l in delta_matrix[cname]:
                for i, d in enumerate(delta_matrix[cname][l]):
                    if np.linalg.norm(d) > 1e-8:
                        all_deltas.append(d)
                        labels.append((cname, i))
        
        if len(all_deltas) < 3:
            print(f"    Not enough samples ({len(all_deltas)})")
            continue
        
        X = np.array(all_deltas)  # [n_samples, d_model]
        n_samples = X.shape[0]
        print(f"    n_samples = {n_samples}, d_model = {d_model}")
        
        # 方法1: 直接SVD (如果n_samples < d_model, 用X@X^T的trick)
        # X = U S V^T, 我们需要V (右奇异向量)
        # 方差解释比 = s_i^2 / sum(s_j^2)
        
        # 中心化
        X_centered = X - X.mean(axis=0, keepdims=True)
        
        # SVD
        # 如果n_samples < d_model, 用 X @ X^T 的特征分解更快
        if n_samples < d_model:
            # X X^T = U S^2 U^T
            XXt = X_centered @ X_centered.T  # [n_samples, n_samples]
            eigenvalues, U_small = np.linalg.eigh(XXt)
            # 排序 (eigh返回升序)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            U_small = U_small[:, idx]
            # 右奇异向量: V = X^T U S^{-1}
            s = np.sqrt(np.maximum(eigenvalues, 0))
            # 计算V: V = X^T @ U @ diag(1/s)
            # 但s可能有0, 需要过滤
            valid = s > 1e-10
            s_valid = s[valid]
            U_valid = U_small[:, valid]
            V = X_centered.T @ U_valid / s_valid  # [d_model, k]
        else:
            # 直接SVD
            U_svd, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            V = Vt.T  # [d_model, min(n, d)]
            s_valid = s
            V = V[:, :len(s_valid)]
        
        # 方差解释比
        total_var = np.sum(s_valid ** 2)
        if total_var < 1e-20:
            print(f"    Total variance too small")
            continue
        
        var_ratio = s_valid ** 2 / total_var
        cum_var = np.cumsum(var_ratio)
        
        # 有效维度 (Shannon entropy)
        p = var_ratio[var_ratio > 1e-10]
        entropy = -np.sum(p * np.log2(p))
        effective_rank = 2 ** entropy
        
        # 关键指标
        n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
        n_99 = int(np.searchsorted(cum_var, 0.99)) + 1
        
        layer_results = {
            "n_samples": n_samples,
            "effective_rank_shannon": float(effective_rank),
            "n_for_90": n_90,
            "n_for_95": n_95,
            "n_for_99": n_99,
            "cum_var_at_1": float(cum_var[0]) if len(cum_var) > 0 else 0,
            "cum_var_at_5": float(cum_var[min(4, len(cum_var)-1)]) if len(cum_var) > 0 else 0,
            "cum_var_at_10": float(cum_var[min(9, len(cum_var)-1)]) if len(cum_var) > 0 else 0,
            "cum_var_at_20": float(cum_var[min(19, len(cum_var)-1)]) if len(cum_var) > 0 else 0,
            "top10_eigenvalues": [float(x) for x in s_valid[:10].tolist()],
            "top10_var_ratio": [float(x) for x in var_ratio[:10].tolist()],
        }
        
        # 概念间的分离度: 不同概念的delta在PCA空间中是否可分?
        # 方法: 在前10个PC中, 同概念的delta是否更近?
        if len(V) > 0 and V.shape[1] >= 10:
            # 投影到前10个PC
            X_proj = X_centered @ V[:, :10]  # [n_samples, 10]
            
            # 同概念内平均距离 vs 不同概念间平均距离
            intra_dists = []
            inter_dists = []
            
            concept_indices = {}
            for i, (cname, _) in enumerate(labels):
                if cname not in concept_indices:
                    concept_indices[cname] = []
                concept_indices[cname].append(i)
            
            for cname, indices in concept_indices.items():
                if len(indices) < 2:
                    continue
                for ii in range(len(indices)):
                    for jj in range(ii+1, len(indices)):
                        d = np.linalg.norm(X_proj[indices[ii]] - X_proj[indices[jj]])
                        intra_dists.append(d)
            
            # 随机采样inter距离
            np.random.seed(42)
            n_inter = min(200, n_samples * n_samples)
            for _ in range(n_inter):
                i, j = np.random.choice(n_samples, 2, replace=False)
                if labels[i][0] != labels[j][0]:
                    d = np.linalg.norm(X_proj[i] - X_proj[j])
                    inter_dists.append(d)
            
            if intra_dists and inter_dists:
                layer_results["intra_dist_mean"] = float(np.mean(intra_dists))
                layer_results["inter_dist_mean"] = float(np.mean(inter_dists))
                layer_results["separation_ratio"] = float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-8))
        
        results[str(l)] = layer_results
        
        print(f"    Effective rank (Shannon): {effective_rank:.1f}")
        print(f"    PCs for 90%/95%/99%: {n_90}/{n_95}/{n_99}")
        print(f"    Cum var at 1/5/10/20: {cum_var[0]:.3f}/{cum_var[min(4,len(cum_var)-1)]:.3f}/{cum_var[min(9,len(cum_var)-1)]:.3f}/{cum_var[min(19,len(cum_var)-1)]:.3f}")
        print(f"    Top-5 var ratio: {[f'{x:.4f}' for x in var_ratio[:5].tolist()]}")
        if "separation_ratio" in layer_results:
            print(f"    Separation (inter/intra): {layer_results['separation_ratio']:.2f}")
    
    return results


# ================================================================
# Exp2: 暗物质分解 — MLP/Attn输出空间分析
# ================================================================
def run_exp2(model, tokenizer, device, model_info, concepts):
    """
    暗物质(不在W_U行空间中的分量)是否在各层MLP/Attn输出空间中?
    
    方法:
    1. 计算delta = h_concept - h_baseline (在关键层)
    2. 分解delta为: delta_WU (W_U投影) + delta_dark (暗物质)
    3. 将delta_dark投影到各层MLP/Attn输出方向上
    4. 测量: delta_dark中有多少在各层MLP/Attn空间中?
    
    关键: 收集所有层MLP/Attn的输出向量, 构建输出空间基
    """
    print(f"\n{'='*60}")
    print(f"  Exp2: Dark Matter Decomposition")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    key_layers = [l for l in [6, 12, 18, 24, 30] if l < n_layers]
    
    # 收集baseline
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    # 收集所有概念delta
    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()
    
    # W_U行空间基
    print(f"  Computing W_U row space basis...")
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    
    # 收集所有层MLP/Attn的输出方向 (baseline)
    print(f"  Collecting Attn/MLP output directions...")
    # 对于每层, 收集MLP输出方向和Attn输出方向
    # 这些是d_model维向量, 可以用来构建子空间
    
    # 收集方式: 对baseline和每个概念的前2个模板, 收集MLP/Attn输出
    mlp_directions = {}  # {layer: [direction1, direction2, ...]}
    attn_directions = {}
    
    for l in range(n_layers):
        mlp_dirs = []
        attn_dirs = []
        
        # Baseline
        attn_bl, mlp_bl = collect_attn_mlp_outputs(model, tokenizer, device, BASELINE_TEXT, l)
        
        # 几个概念模板
        for cname, cdata in concepts.items():
            for template in cdata["templates"][:2]:  # 每概念2个模板
                attn_out, mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, template, l)
                if mlp_out is not None and mlp_bl is not None:
                    mlp_dirs.append(mlp_out - mlp_bl)  # MLP的delta
                if attn_out is not None and attn_bl is not None:
                    attn_dirs.append(attn_out - attn_bl)  # Attn的delta
        
        if mlp_dirs:
            mlp_directions[l] = np.array(mlp_dirs)  # [n_dirs, d_model]
        if attn_dirs:
            attn_directions[l] = np.array(attn_dirs)
        
        if l % 10 == 0:
            print(f"    Processed layer {l}/{n_layers}")
    
    gc.collect()
    
    # 分析暗物质
    results = {}
    
    for l in key_layers:
        print(f"\n  --- Layer {l} ---")
        layer_results = {}
        
        for cname in concepts:
            if l not in all_deltas.get(cname, {}):
                continue
            
            delta = all_deltas[cname][l]
            delta_norm_sq = np.linalg.norm(delta) ** 2
            if delta_norm_sq < 1e-16:
                continue
            
            # W_U投影
            proj_wu = U_wu.T @ delta
            ratio_wu = np.sum(proj_wu ** 2) / delta_norm_sq
            
            # 暗物质 = delta - W_U投影
            delta_wu = U_wu @ proj_wu  # W_U分量的重构
            delta_dark = delta - delta_wu  # 暗物质
            dark_norm_sq = np.linalg.norm(delta_dark) ** 2
            
            # 投影到当前层MLP输出空间
            ratio_mlp_current = 0.0
            cos_mlp_current = 0.0
            if l in mlp_directions:
                mlp_dirs = mlp_directions[l]  # [n_dirs, d_model]
                # 用SVD找子空间基
                mlp_dirs_centered = mlp_dirs - mlp_dirs.mean(axis=0, keepdims=True)
                # 只取前min(20, rank)个方向
                try:
                    U_mlp, s_mlp, _ = np.linalg.svd(mlp_dirs_centered.T, full_matrices=False)
                    k_mlp = min(20, np.sum(s_mlp > 1e-8 * s_mlp[0]))
                    U_mlp_k = U_mlp[:, :k_mlp]  # [d_model, k_mlp]
                    
                    # 暗物质在MLP空间中的投影
                    proj_mlp = U_mlp_k.T @ delta_dark
                    ratio_mlp_current = np.sum(proj_mlp ** 2) / max(dark_norm_sq, 1e-20)
                    cos_mlp_current = float(np.dot(delta_dark, U_mlp_k @ proj_mlp) / 
                                          (np.sqrt(dark_norm_sq) * np.linalg.norm(U_mlp_k @ proj_mlp))) if dark_norm_sq > 1e-16 and np.linalg.norm(U_mlp_k @ proj_mlp) > 1e-8 else 0
                except:
                    pass
            
            # 投影到当前层Attn输出空间
            ratio_attn_current = 0.0
            cos_attn_current = 0.0
            if l in attn_directions:
                attn_dirs = attn_directions[l]
                attn_dirs_centered = attn_dirs - attn_dirs.mean(axis=0, keepdims=True)
                try:
                    U_attn, s_attn, _ = np.linalg.svd(attn_dirs_centered.T, full_matrices=False)
                    k_attn = min(20, np.sum(s_attn > 1e-8 * s_attn[0]))
                    U_attn_k = U_attn[:, :k_attn]
                    
                    proj_attn = U_attn_k.T @ delta_dark
                    ratio_attn_current = np.sum(proj_attn ** 2) / max(dark_norm_sq, 1e-20)
                    cos_attn_current = float(np.dot(delta_dark, U_attn_k @ proj_attn) / 
                                            (np.sqrt(dark_norm_sq) * np.linalg.norm(U_attn_k @ proj_attn))) if dark_norm_sq > 1e-16 and np.linalg.norm(U_attn_k @ proj_attn) > 1e-8 else 0
                except:
                    pass
            
            # 投影到所有层MLP输出空间的并集
            # 选择附近的5层
            nearby_layers = [ll for ll in range(max(0, l-3), min(n_layers, l+4))]
            all_mlp_dirs = []
            all_attn_dirs = []
            for ll in nearby_layers:
                if ll in mlp_directions:
                    all_mlp_dirs.append(mlp_directions[ll])
                if ll in attn_directions:
                    all_attn_dirs.append(attn_directions[ll])
            
            ratio_mlp_nearby = 0.0
            if all_mlp_dirs:
                try:
                    combined_mlp = np.vstack(all_mlp_dirs)
                    combined_mlp_c = combined_mlp - combined_mlp.mean(axis=0, keepdims=True)
                    U_cmlp, s_cmlp, _ = np.linalg.svd(combined_mlp_c.T, full_matrices=False)
                    k_cmlp = min(30, np.sum(s_cmlp > 1e-8 * s_cmlp[0]))
                    U_cmlp_k = U_cmlp[:, :k_cmlp]
                    proj_cmlp = U_cmlp_k.T @ delta_dark
                    ratio_mlp_nearby = np.sum(proj_cmlp ** 2) / max(dark_norm_sq, 1e-20)
                except:
                    pass
            
            layer_results[cname] = {
                "ratio_wu": float(ratio_wu),
                "dark_ratio": float(1.0 - ratio_wu),
                "dark_norm": float(np.sqrt(dark_norm_sq)),
                "ratio_mlp_current": float(ratio_mlp_current),
                "cos_mlp_current": float(cos_mlp_current),
                "ratio_attn_current": float(ratio_attn_current),
                "cos_attn_current": float(cos_attn_current),
                "ratio_mlp_nearby": float(ratio_mlp_nearby),
            }
            
            print(f"    {cname}: WU={ratio_wu:.3f}, dark={1-ratio_wu:.3f}, "
                  f"mlp_current={ratio_mlp_current:.3f}, attn_current={ratio_attn_current:.3f}, "
                  f"mlp_nearby={ratio_mlp_nearby:.3f}")
        
        results[str(l)] = layer_results
    
    return results


# ================================================================
# Exp3: W_U-only steering — 暗物质是否必要?
# ================================================================
def run_exp3(model, tokenizer, device, model_info, concepts):
    """
    只用delta的W_U投影分量做steering, 测试效果
    
    如果W_U-only steering仍然有效:
      → 暗物质是噪声, 8-14%的W_U分量就够了
    如果W_U-only steering失效:
      → 暗物质包含关键信息, 虽然不直接在W_U中
    
    方法:
    1. 计算delta = h_concept - h_baseline
    2. delta_wu = U_wu @ (U_wu^T @ delta)  (投影到W_U行空间)
    3. delta_dark = delta - delta_wu  (暗物质)
    4. 分别用delta, delta_wu, delta_dark做steering
    5. 比较对probe_words的logit变化
    """
    print(f"\n{'='*60}")
    print(f"  Exp3: W_U-Only Steering vs Full Steering")
    print(f"{'='*60}")

    d_model = model_info.d_model
    n_layers = model_info.n_layers
    key_layers = [l for l in [12, 18, 24] if l < n_layers]  # 只测3个关键层
    
    # 收集baseline
    all_capture = list(range(n_layers))
    bl_all, _ = collect_states_at_layers(model, tokenizer, device, BASELINE_TEXT, all_capture)
    
    # 收集所有概念delta
    all_deltas = {}
    for cname, cdata in concepts.items():
        concept_states = {l: [] for l in all_capture}
        for template in cdata["templates"]:
            states, _ = collect_states_at_layers(model, tokenizer, device, template, all_capture)
            for l in all_capture:
                if l in states and l in bl_all:
                    concept_states[l].append(states[l])
        deltas = {}
        for l in all_capture:
            if concept_states[l]:
                deltas[l] = np.mean(concept_states[l], axis=0) - bl_all[l]
        all_deltas[cname] = deltas
        del concept_states
        gc.collect()
    
    # W_U行空间基
    print(f"  Computing W_U row space basis...")
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, s_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    del W_U
    gc.collect()
    
    # 对每个概念和关键层, 测试三种steering
    results = {}
    alpha = 0.5
    
    for l in key_layers:
        print(f"\n  --- Layer {l} ---")
        layer_results = {}
        
        for cname, cdata in concepts.items():
            if l not in all_deltas.get(cname, {}):
                continue
            
            delta = all_deltas[cname][l]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            # 分解
            proj_wu = U_wu.T @ delta
            delta_wu = U_wu @ proj_wu  # W_U分量
            delta_dark = delta - delta_wu  # 暗物质分量
            
            norm_wu = np.linalg.norm(delta_wu)
            norm_dark = np.linalg.norm(delta_dark)
            ratio_wu = norm_wu ** 2 / delta_norm ** 2
            
            print(f"    {cname}: norm_full={delta_norm:.1f}, norm_wu={norm_wu:.1f}({ratio_wu:.3f}), "
                  f"norm_dark={norm_dark:.1f}({1-ratio_wu:.3f})")
            
            # 三种steering
            probe_words = cdata["probe_words"]
            
            # 1. Full delta steering
            full_effects = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta, alpha
            )
            
            # 2. W_U-only steering
            wu_effects = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta_wu, alpha
            )
            
            # 3. Dark-only steering
            dark_effects = compute_steering_effect(
                model, tokenizer, device, probe_words, l, delta_dark, alpha
            )
            
            # 汇总
            full_mean = np.mean(list(full_effects.values())) if full_effects else 0
            wu_mean = np.mean(list(wu_effects.values())) if wu_effects else 0
            dark_mean = np.mean(list(dark_effects.values())) if dark_effects else 0
            
            # 逐词对比
            word_comparison = {}
            for w in probe_words:
                word_comparison[w] = {
                    "full": full_effects.get(w, 0),
                    "wu_only": wu_effects.get(w, 0),
                    "dark_only": dark_effects.get(w, 0),
                }
            
            layer_results[cname] = {
                "ratio_wu": float(ratio_wu),
                "full_mean_logit_change": float(full_mean),
                "wu_mean_logit_change": float(wu_mean),
                "dark_mean_logit_change": float(dark_mean),
                "wu_efficiency": float(wu_mean / full_mean) if abs(full_mean) > 1e-6 else 0,
                "dark_efficiency": float(dark_mean / full_mean) if abs(full_mean) > 1e-6 else 0,
                "word_comparison": word_comparison,
            }
            
            eff_str = f"wu_eff={layer_results[cname]['wu_efficiency']:.3f}"
            dark_str = f"dark_eff={layer_results[cname]['dark_efficiency']:.3f}"
            print(f"      Full={full_mean:.3f}, WU={wu_mean:.3f}({eff_str}), Dark={dark_mean:.3f}({dark_str})")
        
        results[str(l)] = layer_results
    
    # 总结
    print(f"\n  === Summary ===")
    for l, lr in results.items():
        wu_effs = [v["wu_efficiency"] for v in lr.values() if abs(v["full_mean_logit_change"]) > 1e-6]
        dark_effs = [v["dark_efficiency"] for v in lr.values() if abs(v["full_mean_logit_change"]) > 1e-6]
        if wu_effs:
            print(f"  L{l}: WU_eff={np.mean(wu_effs):.3f}±{np.std(wu_effs):.3f}, "
                  f"Dark_eff={np.mean(dark_effs):.3f}±{np.std(dark_effs):.3f}")
    
    return results


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    model_name = args.model

    print(f"\n{'#'*70}")
    print(f"CCXVII: Dark Matter Decomposition & Concept Effective Dimension — {model_name}")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model(model_name)
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    print(f"  d_model={d_model}, n_layers={n_layers}")

    all_results = {}

    if args.exp in ["1", "all"]:
        exp1_results = run_exp1(model, tokenizer, device, model_info, CONCEPTS)
        all_results["exp1"] = exp1_results

    if args.exp in ["2", "all"]:
        exp2_results = run_exp2(model, tokenizer, device, model_info, CONCEPTS)
        all_results["exp2"] = exp2_results

    if args.exp in ["3", "all"]:
        exp3_results = run_exp3(model, tokenizer, device, model_info, CONCEPTS)
        all_results["exp3"] = exp3_results

    # 保存
    output_path = TEMP / f"ccxvii_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    release_model(model)
    print(f"\nCCXVII {model_name} 完成!")


if __name__ == "__main__":
    main()
