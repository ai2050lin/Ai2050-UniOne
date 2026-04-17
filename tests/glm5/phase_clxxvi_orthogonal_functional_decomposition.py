"""
Phase CLXXVI: 正交功能分解 — 寻找变换矩阵Q
=============================================
核心假说(来自CLXXV):
  语言功能在残差流中是正交编码的
  → 存在正交变换Q, 使得 h_final = Q @ [h_syntax; h_semantic; h_style; ...]
  → 找到Q就是破解语言数学结构的关键

实验设计:
  P761: 功能方向的精确提取与归一化
    - 用大量句子对提取5个功能方向
    - 对方向做Gram-Schmidt正交化, 验证正交化前后的信息保留
    - 测量每个功能子空间的精确维度

  P762: 正交Procrustes分析
    - 给定功能方向矩阵D=[d_syntax, d_semantic, ...], 找到正交矩阵Q
    - 使得 D^T @ Q 最近似对角阵
    - 验证Q的稳定性(跨句子、跨模型)

  P763: 功能子空间维度测量
    - 对每个功能维度, 用多个句子对的差异向量做PCA
    - 测量有效秩(需要多少维来编码每个功能)
    - 验证: 语法3维? 语义10维? 风格5维?

  P764: 正交干预验证
    - 在正交化后的方向上做独立干预
    - 验证: 扰动语法方向只影响语法, 不影响语义/风格
    - 这是对"正交功能子空间"假说的最强验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.linalg import orthogonal_procrustes

from model_utils import load_model, get_model_info, get_layers


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# 扩展的功能句子对 (每维度更多对, 提高统计可靠性)
# ============================================================

EXTENDED_FUNCTIONAL_PAIRS = {
    'syntax': [
        ("The cat sits on the mat", "The cats sit on the mat"),
        ("She walks to school", "She walked to school"),
        ("The dog chased the cat", "The cat was chased by the dog"),
        ("He is running fast", "Is he running fast?"),
        ("A bird flies in the sky", "Birds fly in the sky"),
        ("The man reads a book", "The men read books"),
        ("She has eaten dinner", "She had eaten dinner"),
        ("They will go home", "They went home"),
        ("I can see the mountain", "Can I see the mountain?"),
        ("The child plays outside", "The children play outside"),
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),
        ("She drank cold water", "She drank hot tea"),
        ("The car drove fast", "The bird flew high"),
        ("He read a book about history", "He read a book about math"),
        ("The sun shines brightly", "The rain falls heavily"),
        ("She bought a red dress", "She bought a blue shirt"),
        ("The teacher explained the problem", "The doctor examined the patient"),
        ("They built a tall building", "They planted a large tree"),
        ("The river flows to the sea", "The road leads to the city"),
        ("He cooked delicious food", "He wrote beautiful music"),
    ],
    'style': [
        ("The cat sat on the mat", "The feline settled upon the rug"),
        ("She went to the store", "She proceeded to the establishment"),
        ("It is very cold", "It is exceedingly frigid"),
        ("The dog ran quickly", "The canine hastened with alacrity"),
        ("He ate a lot of food", "He consumed a substantial quantity of nourishment"),
        ("She said hello", "She extended her greetings"),
        ("The house is big", "The residence is of considerable magnitude"),
        ("They came back soon", "They returned with celerity"),
        ("I think this is good", "I am of the opinion that this is satisfactory"),
        ("The movie was fun", "The cinematic production proved entertaining"),
    ],
    'tense': [
        ("She walks to school every day", "She walked to school every day"),
        ("He will finish the work", "He finished the work"),
        ("They are playing outside", "They were playing outside"),
        ("I eat breakfast at seven", "I ate breakfast at seven"),
        ("The train arrives at noon", "The train arrived at noon"),
        ("She writes letters often", "She wrote letters often"),
        ("We go to the park", "We went to the park"),
        ("He runs five miles daily", "He ran five miles daily"),
        ("The sun rises in the east", "The sun rose in the east"),
        ("They build houses here", "They built houses here"),
    ],
    'polarity': [
        ("This is a good idea", "This is not a good idea"),
        ("She likes the movie", "She dislikes the movie"),
        ("He always comes on time", "He never comes on time"),
        ("The weather is pleasant", "The weather is unpleasant"),
        ("They agree with the plan", "They disagree with the plan"),
        ("The answer is correct", "The answer is incorrect"),
        ("She is happy about it", "She is unhappy about it"),
        ("He trusts the system", "He distrusts the system"),
        ("The result was successful", "The result was unsuccessful"),
        ("They approve the proposal", "They reject the proposal"),
    ],
}


# ============================================================
# 辅助函数
# ============================================================

def _collect_residual_stream(model, tokenizer, device, sentence, target_layer=None):
    """收集指定层的残差流最后一个token"""
    ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    layers_out = {}

    def make_hook(storage, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[layer_idx] = to_numpy(output[0])
            else:
                storage[layer_idx] = to_numpy(output)
        return hook

    hooks = []
    for i, layer in enumerate(get_layers(model)):
        if target_layer is None or i == target_layer:
            h = layer.register_forward_hook(make_hook(layers_out, i))
            hooks.append(h)

    with torch.no_grad():
        _ = model(ids)

    for h in hooks:
        h.remove()

    residual = {}
    for layer_idx in sorted(layers_out.keys()):
        arr = layers_out[layer_idx]
        if arr.ndim == 3:
            residual[layer_idx] = arr[0, -1, :]
        elif arr.ndim == 2:
            residual[layer_idx] = arr[-1, :]
        else:
            residual[layer_idx] = arr
    return residual


# ============================================================
# P761: 功能方向的精确提取与正交化
# ============================================================

def P761_precise_functional_directions(model, tokenizer, device, model_name, results):
    """
    用大量句子对提取功能方向, 验证正交性, Gram-Schmidt正交化
    """
    print("\n--- P761: 功能方向的精确提取与正交化 ---")

    n_layers = len(get_layers(model))
    d_model = model.lm_head.weight.shape[1]

    # 选择分析层: 初始层, 中间层, 后层
    analysis_layers = [
        0, 
        n_layers // 4, 
        n_layers // 2, 
        3 * n_layers // 4,
        n_layers - 3,
        n_layers - 1,
    ]
    analysis_layers = sorted(set([l for l in analysis_layers if l < n_layers]))

    print(f"  分析层: {analysis_layers}")

    # 对每个分析层, 提取所有功能方向
    all_layer_results = {}

    for target_layer in analysis_layers:
        print(f"\n  === Layer {target_layer} ===")

        # 对每个功能维度, 提取差异向量
        func_directions = {}  # func_dim -> list of diff vectors
        func_direction_raw = {}

        for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
            diffs = []
            for s1, s2 in pairs:
                r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
                r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)

                if target_layer in r1 and target_layer in r2:
                    diff = r1[target_layer] - r2[target_layer]
                    diffs.append(diff)

            if diffs:
                # 堆叠为矩阵 [n_pairs, d_model]
                diff_matrix = np.stack(diffs, axis=0)
                func_direction_raw[func_dim] = diff_matrix

                # 平均差异方向
                avg_diff = np.mean(diff_matrix, axis=0)
                norm = np.linalg.norm(avg_diff)
                if norm > 1e-10:
                    func_directions[func_dim] = avg_diff / norm

        # 1. 验证原始方向的正交性
        print(f"\n  原始方向正交性:")
        dim_names = sorted(func_directions.keys())
        orig_orthogonality = {}
        for i, d1 in enumerate(dim_names):
            for j, d2 in enumerate(dim_names):
                if i < j:
                    cos_val = float(np.dot(func_directions[d1], func_directions[d2]))
                    orig_orthogonality[f"{d1}_vs_{d2}"] = cos_val
                    print(f"    {d1} vs {d2}: cos={cos_val:.4f}")

        # 2. Gram-Schmidt正交化
        direction_matrix = np.stack([func_directions[d] for d in dim_names], axis=1)  # [d_model, n_dims]
        n_dims = len(dim_names)

        # Q, R = QR分解 (等价于Gram-Schmidt)
        Q, R = np.linalg.qr(direction_matrix)

        # 正交化后的方向
        orth_directions = {}
        for i, dim_name in enumerate(dim_names):
            orth_directions[dim_name] = Q[:, i]

        # 3. 验证正交化后的正交性(应该完全正交)
        orth_orthogonality = {}
        for i, d1 in enumerate(dim_names):
            for j, d2 in enumerate(dim_names):
                if i < j:
                    cos_val = float(np.dot(orth_directions[d1], orth_directions[d2]))
                    orth_orthogonality[f"{d1}_vs_{d2}"] = cos_val

        # 4. 正交化信息保留率
        # 计算原始方向在正交化方向上的投影保留
        retention = {}
        for dim_name in dim_names:
            orig = func_directions[dim_name]
            orth = orth_directions[dim_name]
            # 投影保留 = |cos(angle)|
            retention[dim_name] = float(abs(np.dot(orig, orth)))

        print(f"\n  正交化信息保留率:")
        for dim_name, ret in retention.items():
            print(f"    {dim_name}: {ret:.4f}")

        # 5. 每个功能维度的有效秩(子空间维度)
        func_effective_ranks = {}
        for dim_name, diff_matrix in func_direction_raw.items():
            # diff_matrix: [n_pairs, d_model]
            # SVD分析
            if diff_matrix.shape[0] > 1:
                U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
                # 有效秩 = (sum(S))^2 / sum(S^2)
                p = S / (S.sum() + 1e-30)
                eff_rank = 1.0 / (np.sum(p**2) + 1e-30)
                func_effective_ranks[dim_name] = float(eff_rank)

                # 前3个奇异值占比
                top3_ratio = float(S[:3].sum() / (S.sum() + 1e-30))
                print(f"    {dim_name}: eff_rank={eff_rank:.1f}, top3_SV={top3_ratio:.1%}")
            else:
                func_effective_ranks[dim_name] = 1.0

        # 6. 正交Procrustes: 找最优正交变换
        # 目标: 找正交矩阵Q, 使得 D^T @ Q 最近似对角阵
        # 其中 D = direction_matrix [d_model, n_dims]
        # 这等价于找Q使得 Q^T @ D 的列尽可能正交
        # 实际上QR分解已经给出了这个Q

        # 额外: 计算R矩阵的结构
        R_matrix = R[:n_dims, :n_dims]  # [n_dims, n_dims]
        R_normalized = np.abs(R_matrix) / (np.abs(R_matrix).max() + 1e-30)

        print(f"\n  R矩阵(上三角, 表示正交化中的混合系数):")
        for i, d1 in enumerate(dim_names):
            row = [f"{R_matrix[i,j]:.3f}" for j in range(n_dims)]
            print(f"    {d1}: {row}")

        all_layer_results[str(target_layer)] = {
            'orig_orthogonality': orig_orthogonality,
            'orth_orthogonality': orth_orthogonality,
            'retention': retention,
            'func_effective_ranks': func_effective_ranks,
            'R_matrix': R_matrix.tolist(),
            'dim_names': dim_names,
        }

    results["p761_precise_directions"] = all_layer_results
    return results, func_directions, analysis_layers


# ============================================================
# P762: 正交Procrustes分析
# ============================================================

def P762_orthogonal_procrustes(model, tokenizer, device, model_name, results, func_directions, analysis_layers):
    """
    正交Procrustes: 找到最优正交变换Q
    验证Q的稳定性(跨句子组, 跨层)
    """
    print("\n--- P762: 正交Procrustes分析 ---")

    d_model = model.lm_head.weight.shape[1]
    dim_names = sorted(func_directions.keys())
    n_dims = len(dim_names)

    # 收集每层的功能方向矩阵
    layer_direction_matrices = {}

    for target_layer in analysis_layers:
        directions = {}
        for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
            diffs = []
            for s1, s2 in pairs[:5]:  # 用前5对
                r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
                r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)
                if target_layer in r1 and target_layer in r2:
                    diff = r1[target_layer] - r2[target_layer]
                    diffs.append(diff)
            if diffs:
                avg_diff = np.mean(diffs, axis=0)
                norm = np.linalg.norm(avg_diff)
                if norm > 1e-10:
                    directions[func_dim] = avg_diff / norm

        if len(directions) == n_dims:
            D = np.stack([directions[d] for d in dim_names], axis=1)  # [d_model, n_dims]
            layer_direction_matrices[target_layer] = D

    # 1. 跨层Procrustes对齐
    print(f"\n  === 跨层Procrustes对齐 ===")
    
    if len(layer_direction_matrices) >= 2:
        layers_list = sorted(layer_direction_matrices.keys())
        cross_layer_alignment = {}

        for i, l1 in enumerate(layers_list):
            for j, l2 in enumerate(layers_list):
                if i < j:
                    D1 = layer_direction_matrices[l1]
                    D2 = layer_direction_matrices[l2]

                    # Procrustes: 找Q使得 D2 ≈ D1 @ Q
                    # Q = U @ V^T where D1^T @ D2 = U @ S @ V^T
                    M = D1.T @ D2  # [n_dims, n_dims]
                    U, S, Vt = np.linalg.svd(M)
                    Q = U @ Vt

                    # 对齐误差
                    D2_aligned = D1 @ Q
                    error = float(np.linalg.norm(D2 - D2_aligned) / np.linalg.norm(D2))

                    # 对角性: D1^T @ D2_aligned的对角线 vs 非对角线
                    diag_matrix = D1.T @ D2_aligned
                    diag_vals = np.diag(diag_matrix)
                    off_diag = diag_matrix - np.diag(diag_vals)
                    diag_dominance = float(np.mean(np.abs(diag_vals)) / (np.mean(np.abs(off_diag)) + 1e-30))

                    cross_layer_alignment[f"L{l1}_vs_L{l2}"] = {
                        'procrustes_error': error,
                        'diag_dominance': diag_dominance,
                        'singular_values': S.tolist(),
                    }
                    print(f"    L{l1} vs L{l2}: error={error:.4f}, diag_dominance={diag_dominance:.2f}x")

        results["p762_procrustes"] = {
            "cross_layer_alignment": cross_layer_alignment,
        }
    else:
        results["p762_procrustes"] = {"error": "insufficient layers"}

    # 2. 稳定性测试: 分半验证
    print(f"\n  === 分半稳定性验证 ===")
    
    stability_results = {}
    for target_layer in analysis_layers[:3]:  # 只测试3个层
        # 第一半
        dirs_half1 = {}
        for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
            diffs = []
            for s1, s2 in pairs[:5]:
                r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
                r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)
                if target_layer in r1 and target_layer in r2:
                    diffs.append(r1[target_layer] - r2[target_layer])
            if diffs:
                avg = np.mean(diffs, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-10:
                    dirs_half1[func_dim] = avg / norm

        # 第二半
        dirs_half2 = {}
        for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
            diffs = []
            for s1, s2 in pairs[5:]:
                r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
                r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)
                if target_layer in r1 and target_layer in r2:
                    diffs.append(r1[target_layer] - r2[target_layer])
            if diffs:
                avg = np.mean(diffs, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-10:
                    dirs_half2[func_dim] = avg / norm

        # 计算同维度跨半一致性
        consistencies = {}
        for dim_name in dim_names:
            if dim_name in dirs_half1 and dim_name in dirs_half2:
                cos = float(np.dot(dirs_half1[dim_name], dirs_half2[dim_name]))
                consistencies[dim_name] = cos

        if consistencies:
            stability_results[str(target_layer)] = consistencies
            avg_cons = float(np.mean(list(consistencies.values())))
            print(f"    Layer {target_layer}: avg_consistency={avg_cons:.4f}")
            for dim_name, cos_val in consistencies.items():
                print(f"      {dim_name}: cos={cos_val:.4f}")

    results["p762_procrustes"]["stability"] = stability_results

    return results


# ============================================================
# P763: 功能子空间维度测量
# ============================================================

def P763_functional_subspace_dimension(model, tokenizer, device, model_name, results):
    """
    对每个功能维度, 用多个句子对的差异向量做PCA
    测量有效秩(需要多少维来编码每个功能)
    """
    print("\n--- P763: 功能子空间维度测量 ---")

    n_layers = len(get_layers(model))
    d_model = model.lm_head.weight.shape[1]

    # 选择中间层进行分析(功能信息最丰富的层)
    target_layer = n_layers // 2
    print(f"  目标层: {target_layer} (中间层)")

    # 对每个功能维度收集差异向量
    func_subspace_info = {}

    for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
        diffs = []
        for s1, s2 in pairs:
            r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
            r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)
            if target_layer in r1 and target_layer in r2:
                diff = r1[target_layer] - r2[target_layer]
                diffs.append(diff)

        if len(diffs) < 2:
            continue

        diff_matrix = np.stack(diffs, axis=0)  # [n_pairs, d_model]

        # PCA / SVD
        # 中心化
        mean_diff = np.mean(diff_matrix, axis=0)
        centered = diff_matrix - mean_diff

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # 有效秩
        p = S / (S.sum() + 1e-30)
        eff_rank = 1.0 / (np.sum(p**2) + 1e-30)

        # 累积方差
        cumvar = np.cumsum(S**2) / (np.sum(S**2) + 1e-30)

        # 找到90%和95%方差需要的维度数
        n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n_99 = int(np.searchsorted(cumvar, 0.99)) + 1

        # 前5个主成分的方差解释比
        top5_var = [float(v) for v in (S**2 / (S.sum()**2 + 1e-30))[:5]]

        func_subspace_info[func_dim] = {
            'effective_rank': float(eff_rank),
            'n_90': n_90,
            'n_95': n_95,
            'n_99': n_99,
            'top5_variance_ratio': top5_var,
            'n_pairs': len(diffs),
            'singular_values': [float(s) for s in S[:10]],
        }

        print(f"\n  {func_dim}:")
        print(f"    有效秩: {eff_rank:.1f}")
        print(f"    90%方差需 {n_90} 维, 95%需 {n_95} 维, 99%需 {n_99} 维")
        print(f"    Top-5方差比: {[f'{v:.4f}' for v in top5_var]}")

    # 2. 跨维度的子空间重叠度
    print(f"\n  === 跨维度子空间重叠度 ===")
    
    # 对每对功能维度, 计算主成分子空间的重叠度
    dim_names = sorted(func_subspace_info.keys())
    subspace_overlap = {}

    # 重新收集每个维度的主成分
    func_pcs = {}
    for func_dim, pairs in EXTENDED_FUNCTIONAL_PAIRS.items():
        diffs = []
        for s1, s2 in pairs:
            r1 = _collect_residual_stream(model, tokenizer, device, s1, target_layer)
            r2 = _collect_residual_stream(model, tokenizer, device, s2, target_layer)
            if target_layer in r1 and target_layer in r2:
                diffs.append(r1[target_layer] - r2[target_layer])
        if len(diffs) >= 2:
            diff_matrix = np.stack(diffs, axis=0)
            centered = diff_matrix - np.mean(diff_matrix, axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # 取前3个主成分
            func_pcs[func_dim] = Vt[:3]  # [3, d_model]

    for i, d1 in enumerate(dim_names):
        for j, d2 in enumerate(dim_names):
            if i < j and d1 in func_pcs and d2 in func_pcs:
                # 子空间重叠 = ||P1 @ P2||_F / sqrt(||P1||_F * ||P2||_F)
                # P1, P2是投影矩阵
                pc1 = func_pcs[d1]  # [3, d_model]
                pc2 = func_pcs[d2]  # [3, d_model]

                # 子空间重叠度: Principal Angles
                # cos(θ_k) = σ_k where σ_k are singular values of pc1 @ pc2^T
                M = pc1 @ pc2.T  # [3, 3]
                sv_overlap = np.linalg.svd(M, compute_uv=False)
                max_overlap = float(sv_overlap[0])  # 最大主角度的cos
                mean_overlap = float(np.mean(sv_overlap))

                subspace_overlap[f"{d1}_vs_{d2}"] = {
                    'max_principal_cos': max_overlap,
                    'mean_principal_cos': mean_overlap,
                }
                print(f"    {d1} vs {d2}: max_cos={max_overlap:.4f}, mean_cos={mean_overlap:.4f}")

    results["p763_subspace_dimension"] = {
        "target_layer": target_layer,
        "func_subspace_info": func_subspace_info,
        "subspace_overlap": subspace_overlap,
    }

    return results


# ============================================================
# P764: 正交干预验证
# ============================================================

def P764_orthogonal_intervention(model, tokenizer, device, model_name, results, func_directions):
    """
    在正交化后的方向上做独立干预, 验证功能独立性
    """
    print("\n--- P764: 正交干预验证 ---")

    n_layers = len(get_layers(model))
    d_model = model.lm_head.weight.shape[1]

    # Gram-Schmidt正交化功能方向
    dim_names = sorted(func_directions.keys())
    direction_matrix = np.stack([func_directions[d] for d in dim_names], axis=1)
    Q, R = np.linalg.qr(direction_matrix)
    orth_directions = {dim_names[i]: Q[:, i] for i in range(len(dim_names))}

    print(f"  正交方向: {dim_names}")

    # 测试句子
    test_sentences = [
        "The cat sat on the mat and",
        "She walked to the store to buy",
        "He carefully opened the old door and",
    ]

    # 干预设置
    target_layer = n_layers - 4  # 在后层但不是最后
    intervention_scale = 1.0

    print(f"  干预层: {target_layer}")
    print(f"  干预强度: {intervention_scale}")

    # 对每个句子, 做正交干预并测量各维度的影响
    intervention_effects = defaultdict(lambda: defaultdict(list))

    for sent in test_sentences:
        # Baseline
        ids = tokenizer.encode(sent, return_tensors='pt').to(device)
        with torch.no_grad():
            baseline_output = model(ids)
            baseline_logits = to_numpy(baseline_output.logits[0, -1, :])

        baseline_top10 = set(np.argsort(baseline_logits)[-10:][::-1])

        # 对每个正交方向做干预
        for intervene_dim in dim_names:
            direction = orth_directions[intervene_dim]
            direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

            def make_intervention_hook(dir_t, scale):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    residual_norm = torch.norm(hidden[0, -1, :]).item()
                    perturbation = dir_t * (scale * residual_norm)
                    hidden_modified = hidden.clone()
                    hidden_modified[0, -1, :] += perturbation

                    if isinstance(output, tuple):
                        return (hidden_modified,) + output[1:]
                    return hidden_modified
                return hook

            layers = get_layers(model)
            if target_layer < len(layers):
                hook = layers[target_layer].register_forward_hook(
                    make_intervention_hook(direction_t, intervention_scale)
                )

                with torch.no_grad():
                    modified_output = model(ids)
                    modified_logits = to_numpy(modified_output.logits[0, -1, :])

                hook.remove()

                # 测量对每个功能维度的影响
                logit_diff = modified_logits - baseline_logits
                
                baseline_top10 = set(np.argsort(baseline_logits)[-10:][::-1])
                modified_top10 = set(np.argsort(modified_logits)[-10:][::-1])
                top10_overlap = len(baseline_top10 & modified_top10) / 10.0
                
                logit_cos = float(np.dot(baseline_logits, modified_logits) / (
                    np.linalg.norm(baseline_logits) * np.linalg.norm(modified_logits) + 1e-30
                ))
                logit_l2 = float(np.linalg.norm(modified_logits - baseline_logits))
                
                # 用W_U将功能方向投影到logit空间, 然后测量logit变化在该方向上的投影
                W_U = model.lm_head.weight.data  # [n_vocab, d_model]
                for measure_dim in dim_names:
                    measure_dir = orth_directions[measure_dim]  # [d_model]
                    # logit方向的投影: W_U @ measure_dir -> [n_vocab]
                    measure_dir_t = torch.tensor(measure_dir, dtype=W_U.dtype, device=W_U.device)
                    with torch.no_grad():
                        logit_direction = to_numpy(W_U @ measure_dir_t)  # [n_vocab]
                    logit_dir_norm = np.linalg.norm(logit_direction)
                    if logit_dir_norm > 1e-10:
                        logit_direction_normalized = logit_direction / logit_dir_norm
                        proj_change = float(np.dot(logit_diff, logit_direction_normalized))
                    else:
                        proj_change = 0.0
                    intervention_effects[intervene_dim][measure_dim].append(proj_change)

    # 汇总
    print(f"\n  === 正交干预效果矩阵 ===")
    print(f"  (行=干预维度, 列=测量维度, 值=logit投影变化)")

    effect_matrix = {}
    for intervene_dim in dim_names:
        effect_matrix[intervene_dim] = {}
        row_str = f"  {intervene_dim:>10}: "
        for measure_dim in dim_names:
            values = intervention_effects[intervene_dim][measure_dim]
            avg_effect = float(np.mean(np.abs(values))) if values else 0
            effect_matrix[intervene_dim][measure_dim] = avg_effect
            row_str += f"{avg_effect:>8.2f} "
        print(row_str)

    # 计算选择性指数: 对角线 / 非对角线
    diagonal_effects = [effect_matrix[d][d] for d in dim_names]
    off_diagonal_effects = []
    for d1 in dim_names:
        for d2 in dim_names:
            if d1 != d2:
                off_diagonal_effects.append(effect_matrix[d1][d2])

    avg_diag = float(np.mean(diagonal_effects)) if diagonal_effects else 0
    avg_off_diag = float(np.mean(off_diagonal_effects)) if off_diagonal_effects else 0
    selectivity = avg_diag / (avg_off_diag + 1e-30)

    print(f"\n  对角线平均: {avg_diag:.4f}")
    print(f"  非对角线平均: {avg_off_diag:.4f}")
    print(f"  选择性指数(对角/非对角): {selectivity:.2f}x")

    # 如果选择性指数>1, 说明干预一个方向主要影响该方向
    # 如果≈1, 说明方向之间没有独立性

    results["p764_orthogonal_intervention"] = {
        "effect_matrix": effect_matrix,
        "selectivity_index": selectivity,
        "diagonal_avg": avg_diag,
        "off_diagonal_avg": avg_off_diag,
        "target_layer": target_layer,
        "intervention_scale": intervention_scale,
        "test_sentences": test_sentences,
    }

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    model_name = args.model
    print(f"Phase CLXXVI: 正交功能分解 — 寻找变换矩阵Q -- {model_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)

    results = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # P761: 精确功能方向
    results, func_directions, analysis_layers = P761_precise_functional_directions(
        model, tokenizer, device, model_name, results
    )

    # P762: 正交Procrustes
    results = P762_orthogonal_procrustes(
        model, tokenizer, device, model_name, results, func_directions, analysis_layers
    )

    # P763: 功能子空间维度
    results = P763_functional_subspace_dimension(
        model, tokenizer, device, model_name, results
    )

    # P764: 正交干预验证
    results = P764_orthogonal_intervention(
        model, tokenizer, device, model_name, results, func_directions
    )

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 保存
    out_dir = Path(f"results/phase_clxxvi")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
