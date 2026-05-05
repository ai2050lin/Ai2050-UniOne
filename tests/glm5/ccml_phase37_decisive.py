"""
CCML Phase 37: 两个决定性实验 — 区分 Routing vs Spectral Selection
===================================================================

Phase 36的核心问题: 当前证据能被"spectral alignment + low-rank collapse"解释,
不足以证明"主动信息路由"。

37A: ★★★★★★★★★ 正交方向生存实验 (THE most decisive)
  在中层注入两个等范数扰动:
    δ_aligned: 在W_U列空间内
    δ_orthogonal: 在W_U零空间内
  关键判别:
    如果δ_orthogonal在终层被旋转进W_U → routing (主动变换方向)
    如果δ_orthogonal只是被杀死(范数下降, W_U投影不增加) → spectral selection

37B: ★★★★★★★ 多子空间对齐测试
  提取多个子空间:
    A. W_U (unembedding)
    B. 终层PCA (大量输入的hidden state主方向)
    C. Attention输出子空间 (W_O矩阵的top SVD方向)
    D. 随机子空间 (对照)
  对每个子空间测Jacobian对齐
  如果只有W_U特殊 → routing toward readout
  如果多个子空间都特殊 → general spectral alignment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
import gc
import time

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS)


def get_W_U_np(model):
    """获取lm_head权重矩阵 [vocab_size, d_model]"""
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output_layer'):
        return model.transformer.output_layer.weight.detach().cpu().float().numpy()
    else:
        return model.get_output_embeddings().weight.detach().cpu().float().numpy()


# ============================================================================
# 数据定义
# ============================================================================

CONTEXT_TEMPLATES = [
    "The {word} is here",
    "I see a {word}",
    "The {word} was found",
]

# 更多概念,避免样本太少
TEST_CONCEPTS = ["apple", "dog", "hammer", "water", "car", "tree", "salmon", "knife"]


def find_token_index(tokens, target_word):
    target_lower = target_word.lower().strip()
    for i, t in enumerate(tokens):
        if t.lower().strip() == target_lower:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:3] == target_lower[:3]:
            return i
    for i, t in enumerate(tokens):
        if t.lower().strip()[:2] == target_lower[:2]:
            return i
    return -1


# ============================================================================
# 核心工具函数
# ============================================================================

def collect_all_layer_hs(model, tokenizer, device, word, template, n_layers):
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, dep_idx

    captured = {}
    def make_hook(li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu().numpy()
            else:
                captured[li] = output.detach().float().cpu().numpy()
        return hook

    hooks = []
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        _ = model(**toks)

    for h in hooks:
        h.remove()

    result = {}
    for li in range(n_layers):
        if li in captured:
            result[li] = captured[li][0, dep_idx, :]
    return result, dep_idx


def inject_and_collect(model, tokenizer, device, word, template,
                      source_layer, direction, epsilon, n_layers):
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, None, dep_idx

    captured = {}

    def make_hook(li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu().numpy()
            else:
                captured[li] = output.detach().float().cpu().numpy()
        return hook

    def make_inject_hook(li, dir_np, eps):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            delta = torch.tensor(eps * dir_np, dtype=out.dtype, device=device)
            out[0, dep_idx, :] += delta
            if isinstance(output, tuple):
                captured[li] = out.detach().float().cpu().numpy()
                return (out,) + output[1:]
            captured[li] = out.detach().float().cpu().numpy()
            return out
        return hook

    hooks = []
    for li in range(n_layers):
        if li == source_layer:
            hooks.append(layers[li].register_forward_hook(
                make_inject_hook(li, direction, epsilon)))
        else:
            hooks.append(layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        outputs = model(**toks)
        logits = outputs.logits[0, dep_idx, :].detach().float().cpu().numpy()

    for h in hooks:
        h.remove()

    result = {}
    for li in range(n_layers):
        if li in captured:
            result[li] = captured[li][0, dep_idx, :]
    return result, logits, dep_idx


def get_subspace_basis(W_U, d_model, n_components=200):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_U)
    basis = svd.components_
    S = svd.singular_values_
    return basis, S, n_components


def compute_proj_ratio(vec, basis):
    """计算vec在basis子空间中的投影比 = ||proj||^2 / ||vec||^2"""
    if np.linalg.norm(vec) < 1e-10:
        return 0.0
    proj_coeffs = basis @ vec
    proj_norm_sq = np.sum(proj_coeffs ** 2)
    vec_norm_sq = np.sum(vec ** 2)
    return proj_norm_sq / vec_norm_sq


def project_onto_subspace(vec, basis):
    """将vec投影到basis子空间"""
    coeffs = basis @ vec
    return coeffs @ basis


def project_onto_orthogonal(vec, basis):
    """将vec投影到basis子空间的正交补"""
    proj = project_onto_subspace(vec, basis)
    return vec - proj


def collect_hidden_states_batch(model, tokenizer, device, words, template, n_layers):
    """收集多个词的hidden states,用于PCA"""
    all_hs = {li: [] for li in range(n_layers)}
    for word in words:
        hs, _ = collect_all_layer_hs(model, tokenizer, device, word, template, n_layers)
        if hs is not None:
            for li in range(n_layers):
                if li in hs:
                    all_hs[li].append(hs[li])
    return all_hs


# ============================================================================
# 37A: 正交方向生存实验
# ============================================================================

def expA_orthogonal_survival(model_name, model, tokenizer, device):
    """
    THE decisive experiment: Routing vs Spectral Selection
    
    逻辑:
    1. 在中层L注入δ_aligned (在W_U子空间内) 和 δ_orthogonal (在W_U正交补内)
    2. 两者范数相同
    3. 传播到终层, 测量:
       - δ的存活范数
       - δ在W_U中的投影比
       - 关键: δ_orthogonal是否被旋转进W_U?
    
    判别:
       Routing: δ_orthogonal在终层有显著W_U投影 → 方向被主动变换
       Spectral Selection: δ_orthogonal被杀死, W_U投影不增加 → 只是对齐方向存活
    """
    print(f"\n{'='*70}")
    print(f"37A: 正交方向生存实验 — Routing vs Spectral Selection")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))
    
    print(f"  W_U子空间维度: k={k_wu}, d_model={d_model}, k/d={k_wu/d_model:.4f}")
    
    # 选择注入层: 早期, 中期, 晚期各一个
    inject_layers = []
    if n_layers >= 20:
        inject_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    else:
        inject_layers = [0, n_layers // 2, n_layers - 2]
    
    # 观察层: 从注入层到终层的采样
    eps = 0.01
    concepts = ["apple", "dog", "hammer"]  # 核心概念
    
    results = {
        "model": model_name, "exp": "A",
        "experiment": "orthogonal_survival",
        "n_layers": n_layers, "d_model": d_model,
        "k_wu": k_wu,
        "inject_layers": inject_layers,
        "aligned_trajectories": {},
        "orthogonal_trajectories": {},
        "routing_scores": {},
    }
    
    for concept in concepts:
        print(f"\n--- 概念: {concept} ---")
        results["aligned_trajectories"][concept] = {}
        results["orthogonal_trajectories"][concept] = {}
        results["routing_scores"][concept] = {}
        
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            continue
        
        for src_l in inject_layers:
            if src_l not in baseline_hs:
                continue
            
            h_scale = np.linalg.norm(baseline_hs[src_l])
            actual_eps = eps * h_scale
            
            # 构造δ_aligned和δ_orthogonal
            np.random.seed(42 + src_l)
            rand_vec = np.random.randn(d_model).astype(np.float32)
            rand_vec /= np.linalg.norm(rand_vec)
            
            # δ_aligned: rand_vec在W_U子空间的投影
            delta_aligned = project_onto_subspace(rand_vec, basis_wu)
            aligned_norm = np.linalg.norm(delta_aligned)
            if aligned_norm < 1e-10:
                print(f"  L{src_l}: rand_vec在W_U中无投影, 跳过")
                continue
            delta_aligned /= aligned_norm  # 单位化
            delta_aligned *= actual_eps  # 缩放到目标范数
            
            # δ_orthogonal: rand_vec在W_U正交补的投影
            delta_orthogonal = project_onto_orthogonal(rand_vec, basis_wu)
            orth_norm = np.linalg.norm(delta_orthogonal)
            if orth_norm < 1e-10:
                # 如果rand_vec恰好在W_U中, 换一个
                np.random.seed(123 + src_l)
                rand_vec2 = np.random.randn(d_model).astype(np.float32)
                delta_orthogonal = project_onto_orthogonal(rand_vec2, basis_wu)
                orth_norm = np.linalg.norm(delta_orthogonal)
            if orth_norm < 1e-10:
                print(f"  L{src_l}: 无法构造正交方向, 跳过")
                continue
            delta_orthogonal /= orth_norm  # 单位化
            delta_orthogonal *= actual_eps  # 相同范数
            
            # 验证初始状态
            init_aligned_wu = compute_proj_ratio(delta_aligned, basis_wu)
            init_orth_wu = compute_proj_ratio(delta_orthogonal, basis_wu)
            print(f"  L{src_l}: inject eps={actual_eps:.4f}")
            print(f"    δ_aligned: norm={np.linalg.norm(delta_aligned):.4f}, W_U投影={init_aligned_wu:.4f}")
            print(f"    δ_orthogonal: norm={np.linalg.norm(delta_orthogonal):.4f}, W_U投影={init_orth_wu:.4f}")
            
            # 注入δ_aligned, 收集各层hidden state
            perturbed_aligned, logits_aligned, _ = inject_and_collect(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                src_l, delta_aligned / actual_eps, actual_eps, n_layers)
            
            # 注入δ_orthogonal, 收集各层hidden state
            perturbed_orth, logits_orth, _ = inject_and_collect(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                src_l, delta_orthogonal / actual_eps, actual_eps, n_layers)
            
            if perturbed_aligned is None or perturbed_orth is None:
                continue
            
            # 计算各层的delta和W_U投影比
            obs_layers = sorted([li for li in range(src_l, n_layers) 
                                if li in perturbed_aligned and li in perturbed_orth
                                and li in baseline_hs])
            
            aligned_traj = {}
            orth_traj = {}
            
            for li in obs_layers:
                delta_a = perturbed_aligned[li] - baseline_hs[li]
                delta_o = perturbed_orth[li] - baseline_hs[li]
                
                norm_a = np.linalg.norm(delta_a)
                norm_o = np.linalg.norm(delta_o)
                wu_a = compute_proj_ratio(delta_a, basis_wu)
                wu_o = compute_proj_ratio(delta_o, basis_wu)
                
                # ★关键指标: δ_orthogonal在W_U中的绝对投影能量
                # 如果routing, 这个应该增长; 如果spectral selection, 这个应该不变或减小
                proj_energy_o = wu_o * norm_o**2  # = ||P_{W_U}(δ_o)||^2
                
                aligned_traj[str(li)] = {
                    "norm": float(norm_a),
                    "norm_ratio": float(norm_a / actual_eps),  # 放大/衰减
                    "wu_proj_ratio": float(wu_a),
                    "wu_proj_energy": float(wu_a * norm_a**2),
                }
                orth_traj[str(li)] = {
                    "norm": float(norm_o),
                    "norm_ratio": float(norm_o / actual_eps),
                    "wu_proj_ratio": float(wu_o),
                    "wu_proj_energy": float(proj_energy_o),
                }
            
            results["aligned_trajectories"][concept][str(src_l)] = aligned_traj
            results["orthogonal_trajectories"][concept][str(src_l)] = orth_traj
            
            # ★计算routing score
            # routing_score = δ_orthogonal在终层的W_U绝对投影能量 / 初始W_U绝对投影能量
            # 如果>1: 正交方向被旋转进W_U → routing
            # 如果~0: 正交方向只是被杀死 → spectral selection
            
            final_li = n_layers - 1
            if str(final_li) in orth_traj:
                init_wu_energy_orth = init_orth_wu * actual_eps**2
                final_wu_energy_orth = orth_traj[str(final_li)]["wu_proj_energy"]
                
                if init_wu_energy_orth > 1e-20:
                    routing_score = final_wu_energy_orth / init_wu_energy_orth
                else:
                    routing_score = 0.0
                
                # 也计算aligned方向的W_U能量变化
                init_wu_energy_aligned = init_aligned_wu * actual_eps**2
                final_wu_energy_aligned = aligned_traj[str(final_li)]["wu_proj_energy"]
                if init_wu_energy_aligned > 1e-20:
                    aligned_survival = final_wu_energy_aligned / init_wu_energy_aligned
                else:
                    aligned_survival = 0.0
                
                results["routing_scores"][concept][str(src_l)] = {
                    "orth_routing_score": float(routing_score),
                    "aligned_survival": float(aligned_survival),
                    "orth_final_wu_ratio": float(orth_traj[str(final_li)]["wu_proj_ratio"]),
                    "aligned_final_wu_ratio": float(aligned_traj[str(final_li)]["wu_proj_ratio"]),
                    "orth_final_norm_ratio": float(orth_traj[str(final_li)]["norm_ratio"]),
                    "aligned_final_norm_ratio": float(aligned_traj[str(final_li)]["norm_ratio"]),
                }
                
                print(f"\n  ★ L{src_l}→L{final_li} 判别结果:")
                print(f"    δ_aligned:  norm_ratio={aligned_traj[str(final_li)]['norm_ratio']:.3f}, "
                      f"W_U_ratio={aligned_traj[str(final_li)]['wu_proj_ratio']:.4f}, "
                      f"survival={aligned_survival:.3f}")
                print(f"    δ_orthogonal: norm_ratio={orth_traj[str(final_li)]['norm_ratio']:.3f}, "
                      f"W_U_ratio={orth_traj[str(final_li)]['wu_proj_ratio']:.4f}, "
                      f"routing_score={routing_score:.3f}")
                
                # 判别
                if routing_score > 1.5:
                    print(f"    ★★★ ROUTING: 正交方向被旋转进W_U (score={routing_score:.2f} > 1)")
                elif routing_score < 0.3:
                    print(f"    ✗✗✗ SPECTRAL SELECTION: 正交方向被杀死, 无旋转进W_U (score={routing_score:.2f} << 1)")
                else:
                    print(f"    △ 混合: routing_score={routing_score:.2f}")
    
    return results


# ============================================================================
# 37B: 多子空间对齐测试
# ============================================================================

def expB_multi_subspace_alignment(model_name, model, tokenizer, device):
    """
    多子空间对齐测试: W_U是否是唯一特殊的子空间?
    
    提取4类子空间:
    A. W_U (unembedding top-k)
    B. 终层PCA (大量输入的hidden state主方向)
    C. Attention输出子空间 (最后几层的W_O矩阵top SVD方向)
    D. 随机子空间 (对照)
    
    对每个子空间测: Jacobian的top direction在该子空间的投影倍数
    """
    from sklearn.decomposition import TruncatedSVD
    
    print(f"\n{'='*70}")
    print(f"37B: 多子空间对齐测试 — W_U是否唯一特殊?")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    k_dim = min(200, d_model)  # 子空间维度
    
    # === 提取4类子空间 ===
    print(f"\n--- 提取子空间 ---")
    
    # A. W_U子空间
    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, k_dim)
    print(f"  A. W_U子空间: shape={basis_wu.shape}, top-5 S={S_wu[:5].round(2)}")
    
    # B. 终层PCA: 用所有测试概念收集hidden states
    print(f"  B. 终层PCA: 收集{len(TEST_CONCEPTS)}个概念的终层hidden states...")
    all_final_hs = []
    for concept in TEST_CONCEPTS:
        hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if hs is not None and (n_layers - 1) in hs:
            all_final_hs.append(hs[n_layers - 1])
    
    # 再用不同模板增加多样性
    for concept in TEST_CONCEPTS[:4]:
        for tmpl in CONTEXT_TEMPLATES[1:]:
            hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, tmpl, n_layers)
            if hs is not None and (n_layers - 1) in hs:
                all_final_hs.append(hs[n_layers - 1])
    
    if len(all_final_hs) >= k_dim:
        hs_matrix = np.array(all_final_hs)  # [n_samples, d_model]
        hs_centered = hs_matrix - hs_matrix.mean(axis=0, keepdims=True)
        svd_pca = TruncatedSVD(n_components=k_dim)
        svd_pca.fit(hs_centered)
        basis_pca = svd_pca.components_
        S_pca = svd_pca.singular_values_
    else:
        basis_pca = None
        S_pca = None
    if basis_pca is not None:
        print(f"     终层PCA: shape={basis_pca.shape}, top-5 S={S_pca[:5].round(2)}")
    else:
        print(f"     终层PCA: 样本不足, 跳过")
    
    # C. Attention输出子空间: 最后几层的W_O矩阵
    print(f"  C. Attention W_O子空间: 提取最后4层的W_O...")
    layers = get_layers(model)
    wo_vectors = []
    
    for li in range(max(0, n_layers - 4), n_layers):
        layer = layers[li]
        # 尝试不同的属性路径找到W_O
        w_o = None
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            w_o = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()
        elif hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'o_proj'):
            w_o = layer.self_attention.o_proj.weight.detach().float().cpu().numpy()
        elif hasattr(layer, 'attention') and hasattr(layer.attention, 'wo'):
            w_o = layer.attention.wo.weight.detach().float().cpu().numpy()
        elif hasattr(layer, 'self_attn'):
            sa = layer.self_attn
            if hasattr(sa, 'out_proj'):
                w_o = sa.out_proj.weight.detach().float().cpu().numpy()
            elif hasattr(sa, 'dense'):
                w_o = sa.dense.weight.detach().float().cpu().numpy()
        elif hasattr(layer, 'attn'):
            attn = layer.attn
            if hasattr(attn, 'out_proj'):
                w_o = attn.out_proj.weight.detach().float().cpu().numpy()
            elif hasattr(attn, 'wo'):
                w_o = attn.wo.weight.detach().float().cpu().numpy()
        
        if w_o is not None:
            # W_O可能是 [d_model, d_model] 或 [d_model, n_heads*head_dim]
            if w_o.ndim == 2:
                # 取top SVD左奇异向量 (在d_model空间)
                n_comp = min(k_dim // 4, min(w_o.shape[0], w_o.shape[1]))
                U, s, Vt = np.linalg.svd(w_o, full_matrices=False)
                # U: [d_model, min(d_model, n_heads*head_dim)] - 左奇异向量在d_model空间
                wo_vectors.append(U[:, :n_comp].T)  # [n_comp, d_model]
                print(f"     L{li} W_O: shape={w_o.shape}, top SVD n={n_comp}, top-3 σ={s[:3].round(2)}")
    
    if wo_vectors:
        basis_wo = np.concatenate(wo_vectors, axis=0)  # [n_total, d_model]
        # 正交化
        Q, _ = np.linalg.qr(basis_wo.T)
        basis_wo = Q.T[:min(k_dim, basis_wo.shape[0]), :]
        print(f"     W_O子空间: shape={basis_wo.shape}")
    else:
        basis_wo = None
        print(f"     W_O子空间: 未找到W_O, 跳过")
    
    # D. 随机子空间
    np.random.seed(99999)
    R = np.random.randn(k_dim, d_model).astype(np.float32)
    Q_rand, _ = np.linalg.qr(R.T)
    basis_random = Q_rand.T[:k_dim, :]
    print(f"  D. 随机子空间: shape={basis_random.shape}")
    
    # === 对每个子空间测Jacobian对齐 ===
    subspaces = {"W_U": basis_wu, "Random": basis_random}
    if basis_pca is not None:
        subspaces["FinalPCA"] = basis_pca
    if basis_wo is not None:
        subspaces["AttnWO"] = basis_wo
    
    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(0, n_layers - 1))
    else:
        step = max(1, (n_layers - 1) // 6)
        sample_layers = sorted(set(list(range(0, n_layers - 1, step)) + [n_layers - 2]))
    
    eps = 0.01
    n_random_dirs = 20  # 减少到20次
    concepts_for_jacobian = ["apple", "dog", "hammer"]
    
    results = {
        "model": model_name, "exp": "B",
        "experiment": "multi_subspace_alignment",
        "n_layers": n_layers, "d_model": d_model,
        "k_wu": k_wu,
        "subspace_names": list(subspaces.keys()),
        "alignment_data": {},
    }
    
    for li in sample_layers:
        results["alignment_data"][str(li)] = {}
        print(f"\n  L{li}:")
        
        for concept in concepts_for_jacobian:
            baseline_hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
            if baseline_hs is None or li not in baseline_hs or (li + 1) not in baseline_hs:
                continue
            
            h_l_base = baseline_hs[li]
            h_l1_base = baseline_hs[li + 1]
            h_scale = np.linalg.norm(h_l_base)
            actual_eps = eps * h_scale
            
            # 收集Jacobian的列向量 (通过随机扰动)
            np.random.seed(42 + li)
            jacobian_cols = []
            
            for trial in range(n_random_dirs):
                rand_dir = np.random.randn(d_model).astype(np.float32)
                rand_dir /= np.linalg.norm(rand_dir)
                
                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    li, rand_dir, actual_eps, n_layers)
                
                if perturbed_hs is not None and (li + 1) in perturbed_hs:
                    col = (perturbed_hs[li + 1] - h_l1_base) / actual_eps
                    jacobian_cols.append(col)
            
            if len(jacobian_cols) < 5:
                continue
            
            J_cols = np.array(jacobian_cols)  # [n_trials, d_model]
            
            # 对每个子空间, 计算: top-1 PC在子空间的投影倍数
            # 也计算: 平均列向量在子空间的投影倍数
            for ss_name, ss_basis in subspaces.items():
                if ss_basis is None:
                    continue
                
                # Method 1: 随机方向的平均投影比
                random_proj_ratios = []
                for col in jacobian_cols:
                    ratio = compute_proj_ratio(col, ss_basis)
                    random_proj_ratios.append(ratio)
                mean_random_ratio = float(np.mean(random_proj_ratios))
                
                # Method 2: top-1 PC的投影倍数
                # PCA on Jacobian columns
                from sklearn.decomposition import TruncatedSVD
                n_comp = min(10, len(jacobian_cols) - 1, J_cols.shape[1])
                svd_j = TruncatedSVD(n_components=n_comp)
                svd_j.fit(J_cols)
                top1_pc = svd_j.components_[0]
                top1_ratio = compute_proj_ratio(top1_pc, ss_basis)
                top1_mult = top1_ratio / (ss_basis.shape[0] / d_model) if (ss_basis.shape[0] / d_model) > 0 else 0
                
                # Method 3: top-1右奇异向量 (输入方向) 的投影倍数
                # (不太相关, 跳过)
                
                results["alignment_data"][str(li)][f"{concept}_{ss_name}"] = {
                    "mean_random_proj_ratio": mean_random_ratio,
                    "top1_proj_ratio": float(top1_ratio),
                    "top1_multiplier": float(top1_mult),
                    "expected_random_ratio": float(ss_basis.shape[0] / d_model),
                }
            
            # 打印该层的比较
            print(f"    {concept}:", end="")
            for ss_name in subspaces.keys():
                key = f"{concept}_{ss_name}"
                if key in results["alignment_data"][str(li)]:
                    d = results["alignment_data"][str(li)][key]
                    print(f"  {ss_name}(top1×={d['top1_multiplier']:.1f},avg={d['mean_random_proj_ratio']:.3f})", end="")
            print()
    
    # === 判别 ===
    print(f"\n{'='*60}")
    print(f"37B 判别: W_U是否是唯一特殊的子空间?")
    print(f"{'='*60}")
    
    # 汇总各层的top1_multiplier
    for ss_name in subspaces.keys():
        multipliers = []
        for li_str, li_data in results["alignment_data"].items():
            for key, val in li_data.items():
                if key.endswith(f"_{ss_name}"):
                    multipliers.append(val["top1_multiplier"])
        
        if multipliers:
            mean_mult = float(np.mean(multipliers))
            max_mult = float(np.max(multipliers))
            print(f"  {ss_name}: mean_top1_mult={mean_mult:.2f}, max_top1_mult={max_mult:.2f}")
    
    # 比较W_U vs 其他子空间
    wu_mults = []
    pca_mults = []
    wo_mults = []
    rand_mults = []
    
    for li_str, li_data in results["alignment_data"].items():
        for key, val in li_data.items():
            if key.endswith("_W_U"):
                wu_mults.append(val["top1_multiplier"])
            elif key.endswith("_FinalPCA"):
                pca_mults.append(val["top1_multiplier"])
            elif key.endswith("_AttnWO"):
                wo_mults.append(val["top1_multiplier"])
            elif key.endswith("_Random"):
                rand_mults.append(val["top1_multiplier"])
    
    print(f"\n  W_U vs Random:")
    if wu_mults and rand_mults:
        wu_vs_rand = float(np.mean(wu_mults) / (np.mean(rand_mults) + 0.001))
        print(f"    W_U mean mult: {np.mean(wu_mults):.2f}")
        print(f"    Random mean mult: {np.mean(rand_mults):.2f}")
        print(f"    W_U/Random ratio: {wu_vs_rand:.2f}")
    
    if pca_mults:
        print(f"\n  W_U vs FinalPCA:")
        pca_vs_rand = float(np.mean(pca_mults) / (np.mean(rand_mults) + 0.001))
        wu_vs_pca = float(np.mean(wu_mults) / (np.mean(pca_mults) + 0.001))
        print(f"    FinalPCA mean mult: {np.mean(pca_mults):.2f}")
        print(f"    FinalPCA/Random: {pca_vs_rand:.2f}")
        print(f"    W_U/FinalPCA: {wu_vs_pca:.2f}")
    
    if wo_mults:
        print(f"\n  W_U vs AttnWO:")
        wo_vs_rand = float(np.mean(wo_mults) / (np.mean(rand_mults) + 0.001))
        wu_vs_wo = float(np.mean(wu_mults) / (np.mean(wo_mults) + 0.001))
        print(f"    AttnWO mean mult: {np.mean(wo_mults):.2f}")
        print(f"    AttnWO/Random: {wo_vs_rand:.2f}")
        print(f"    W_U/AttnWO: {wu_vs_wo:.2f}")
    
    # 判别
    print(f"\n  ★ 判别:")
    if wu_mults and rand_mults:
        wu_exceeds_random = np.mean(wu_mults) > np.mean(rand_mults) * 1.5
        if pca_mults:
            pca_exceeds_random = np.mean(pca_mults) > np.mean(rand_mults) * 1.5
            pca_similar_to_wu = abs(np.mean(pca_mults) - np.mean(wu_mults)) < 0.3 * max(np.mean(pca_mults), np.mean(wu_mults))
        else:
            pca_exceeds_random = False
            pca_similar_to_wu = False
        
        if wu_exceeds_random and pca_similar_to_wu:
            print(f"    ★★ 多子空间都特殊 → 不是routing toward W_U, 是general spectral alignment")
        elif wu_exceeds_random and not pca_similar_to_wu:
            print(f"    ★★ 只有W_U特殊 → 支持routing toward readout")
        elif not wu_exceeds_random:
            print(f"    ✗ W_U不比random特殊 → 无特殊对齐")
    
    return results


# ============================================================================
# DS7B PR≈1 验证
# ============================================================================

def expC_verify_extreme_PR(model_name, model, tokenizer, device):
    """
    验证DS7B最后一层PR≈1是否真实,还是数值/测量伪象
    
    方法:
    1. 用不同扰动尺度测Jacobian
    2. 用不同概念测
    3. 检查是否是量化(8bit)导致的
    """
    from sklearn.decomposition import TruncatedSVD
    
    print(f"\n{'='*70}")
    print(f"37C: 极端PR验证 — 是否是数值伪象?")
    print(f"{'='*70}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))
    
    # 测试最后一层和倒数第二层
    test_layers = [n_layers - 2, n_layers - 3] if n_layers > 3 else [n_layers - 2]
    concepts = ["apple", "dog", "hammer"]
    eps_values = [0.001, 0.01, 0.05]  # 不同扰动尺度
    
    results = {
        "model": model_name, "exp": "C",
        "experiment": "verify_extreme_PR",
        "test_data": {},
    }
    
    for li in test_layers:
        results["test_data"][str(li)] = {}
        
        for concept in concepts:
            baseline_hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
            if baseline_hs is None or li not in baseline_hs or (li + 1) not in baseline_hs:
                continue
            
            h_scale = np.linalg.norm(baseline_hs[li])
            
            for eps_val in eps_values:
                actual_eps = eps_val * h_scale
                
                np.random.seed(42 + li)
                jacobian_cols = []
                
                for trial in range(20):
                    rand_dir = np.random.randn(d_model).astype(np.float32)
                    rand_dir /= np.linalg.norm(rand_dir)
                    
                    perturbed_hs, _, _ = inject_and_collect(
                        model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                        li, rand_dir, actual_eps, n_layers)
                    
                    if perturbed_hs is not None and (li + 1) in perturbed_hs:
                        col = (perturbed_hs[li + 1] - baseline_hs[li + 1]) / actual_eps
                        jacobian_cols.append(col)
                
                if len(jacobian_cols) < 5:
                    continue
                
                J_cols = np.array(jacobian_cols)
                
                # 计算SVD和PR
                from sklearn.decomposition import TruncatedSVD
                n_comp = min(20, len(jacobian_cols) - 1, J_cols.shape[1])
                svd_j = TruncatedSVD(n_components=n_comp)
                svd_j.fit(J_cols)
                S = svd_j.singular_values_
                
                # Participation Ratio
                PR = (np.sum(S) ** 2) / (np.sum(S ** 2) + 1e-10)
                
                # Isotropy
                if S[0] > 0:
                    isotropy = 1.0 - (S[0] - np.mean(S)) / (S[0] + np.mean(S) + 1e-10)
                else:
                    isotropy = 1.0
                
                key = f"{concept}_eps{eps_val}"
                results["test_data"][str(li)][key] = {
                    "PR": float(PR),
                    "isotropy": float(isotropy),
                    "top5_S": S[:5].tolist(),
                    "top1_mean_ratio": float(S[0] / (np.mean(S) + 1e-10)),
                    "n_cols": len(jacobian_cols),
                }
                
                print(f"  L{li} {concept} eps={eps_val}: PR={PR:.1f}, iso={isotropy:.3f}, "
                      f"top1/mean={S[0]/(np.mean(S)+1e-10):.1f}, top5 S={S[:5].round(1)}")
    
    # 判别
    print(f"\n  ★ 判别:")
    all_prs = []
    for li_str, li_data in results["test_data"].items():
        for key, val in li_data.items():
            all_prs.append(val["PR"])
    
    if all_prs:
        min_pr = min(all_prs)
        max_pr = max(all_prs)
        mean_pr = float(np.mean(all_prs))
        
        # PR是否随eps变化很大?
        eps1_prs = [v["PR"] for li_data in results["test_data"].values() 
                    for k, v in li_data.items() if "eps0.001" in k]
        eps2_prs = [v["PR"] for li_data in results["test_data"].values() 
                    for k, v in li_data.items() if "eps0.05" in k]
        
        print(f"    PR range: {min_pr:.1f} - {max_pr:.1f}, mean={mean_pr:.1f}")
        
        if eps1_prs and eps2_prs:
            pr_change = abs(np.mean(eps1_prs) - np.mean(eps2_prs))
            print(f"    PR change with eps: {np.mean(eps1_prs):.1f} (eps=0.001) vs {np.mean(eps2_prs):.1f} (eps=0.05), Δ={pr_change:.1f}")
            if pr_change > 5:
                print(f"    ★ PR强烈依赖于扰动尺度 → 可能是数值伪象!")
            elif min_pr < 3:
                print(f"    ★ PR确实很低 → 极端各向异性是真实的")
            else:
                print(f"    PR适中 → 极端PR可能是特定条件下的结果")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

EXPERIMENTS = {
    "1": ("37A: 正交方向生存", expA_orthogonal_survival),
    "2": ("37B: 多子空间对齐", expB_multi_subspace_alignment),
    "3": ("37C: 极端PR验证", expC_verify_extreme_PR),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3"])
    args = parser.parse_args()
    
    model_name = args.model
    exp_id = args.exp
    
    exp_name, exp_fn = EXPERIMENTS[exp_id]
    print(f"Model: {model_name}, Experiment: {exp_name}")
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    model, tokenizer, device = load_model(model_name)
    model.eval()
    print(f"Model loaded: {model_name}")
    
    try:
        result = exp_fn(model_name, model, tokenizer, device)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        result = {"model": model_name, "exp": exp_id, "error": str(e)}
    
    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    exp_letters = {"1": "A", "2": "B", "3": "C"}
    out_file = os.path.join(out_dir, f"ccml_phase37_exp{exp_letters[exp_id]}_{model_name}_results.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Model released, GPU cache cleared.")


if __name__ == "__main__":
    main()
