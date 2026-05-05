"""
CCML(Phase 36): 判别实验 — 信息路由(H1) vs 高维几何(H0)
========================================================

核心问题: W_U子空间是否是唯一被特殊对待的子空间?
  YES → 模型存在"读出对齐机制" (接近routing)
  NO  → 没有路由, 只是高维线性混合+各向异性

5个判别实验:

  36A: ★★★★★★★★★ 随机子空间对照 (THE most critical)
    构造3类子空间: W_U, random同维, orthogonal complement
    如果只有W_U的投影比上升 → routing; 如果所有子空间都类似 → geometry

  36B: ★★★★★★★★ 未训练模型对照
    训练好的模型 vs 随机初始化模型
    如果训练模型有路由但未训练没有 → routing是学来的
    如果两者类似 → 只是矩阵乘法的自然结果

  36C: ★★★★★★★ Jacobian主方向对齐
    J = UΣV^T 的top left singular vectors是否对齐W_U行空间?
    如果top-U逐层对齐W_U → routing; 如果无系统对齐 → geometry

  36D: ★★★★★ 谱结构分析 (singular value decay)
    是否存在few dominant directions? 还是truly isotropic?
    有强主方向 → routing可能沿这些方向; 完全平 → 不可能有目标导向机制

  36E: ★★★★ 子空间破坏实验 (因果验证)
    在某层去掉W_U投影, 继续前向传播, 看logit变化
    早期去掉→后面能恢复(被重新路由) vs 晚期去掉→输出崩溃 → routing
    行为不可解释/无系统模式 → geometry

判别标准汇总:
  H1(routing成立): W_U特殊, 训练创造, 主方向对齐, 有dominant directions, 因果可恢复
  H0(几何效应):   所有子空间类似, 未训练也有, 无系统对齐, 平坦谱, 无因果模式
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from collections import defaultdict
import time

from model_utils import (load_model, get_layers, get_model_info, release_model,
                         safe_decode, MODEL_CONFIGS)
# get_W_U only takes model (not model_name)
def get_W_U_np(model):
    """获取lm_head权重矩阵 [vocab_size, d_model]"""
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output_layer'):
        return model.transformer.output_layer.weight.detach().cpu().float().numpy()
    else:
        return model.get_output_embeddings().weight.detach().cpu().float().numpy()


# ============================================================================
# 数据定义 (同Phase 35)
# ============================================================================

CONCEPT_DATASET = {
    "apple":      {"edible":1, "animacy":0, "size":0.5},
    "orange":     {"edible":1, "animacy":0, "size":0.5},
    "dog":        {"edible":0, "animacy":1, "size":0.5},
    "cat":        {"edible":0, "animacy":1, "size":0.3},
    "hammer":     {"edible":0, "animacy":0, "size":0.5},
    "knife":      {"edible":0, "animacy":0, "size":0.3},
    "salmon":     {"edible":1, "animacy":1, "size":0.5},
    "water":      {"edible":1, "animacy":0, "size":0.5},
    "tree":       {"edible":0, "animacy":0, "size":1.0},
    "car":        {"edible":0, "animacy":0, "size":1.0},
}

ATTR_NAMES = ["edible", "animacy", "size"]

CONTEXT_TEMPLATES = [
    "The {word} is here",
    "I see a {word}",
    "The {word} was found",
]

TEST_CONCEPTS = ["apple", "dog", "hammer"]


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
# 核心函数 (复用Phase 35)
# ============================================================================

def collect_all_layer_hs(model, tokenizer, device, word, template, n_layers):
    """收集ALL层的hidden states"""
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
                captured[li] = output.detach().float().numpy()
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
    """在source_layer注入扰动, 收集ALL后续层的hidden states和logits"""
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


def modify_and_collect(model, tokenizer, device, word, template,
                       modify_layer, modify_fn, n_layers):
    """在modify_layer修改hidden state, 继续前向传播, 收集logits

    modify_fn: function(h) -> modified_h, where h is [d_model] numpy array
    """
    layers = get_layers(model)
    sent = template.format(word=word)
    toks = tokenizer(sent, return_tensors="pt").to(device)
    tokens_list = [safe_decode(tokenizer, t) for t in toks.input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, word)
    if dep_idx < 0:
        return None, dep_idx

    def make_modify_hook(li, fn):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            h_np = out[0, dep_idx, :].detach().float().cpu().numpy()
            h_mod = fn(h_np)
            out[0, dep_idx, :] = torch.tensor(h_mod, dtype=out.dtype, device=device)
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return hook

    hook = layers[modify_layer].register_forward_hook(
        make_modify_hook(modify_layer, modify_fn))

    with torch.no_grad():
        outputs = model(**toks)
        logits = outputs.logits[0, dep_idx, :].detach().float().cpu().numpy()

    hook.remove()
    return logits, dep_idx


def get_subspace_basis(W_U, d_model, n_components=200):
    """获取W_U行空间的正交基 (前n_components个右奇异向量)
    使用截断SVD避免内存问题
    """
    # W_U: [vocab, d_model]
    # 使用sklearn的截断SVD (只计算前n_components个分量)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_U)  # 这只计算前n_components个奇异向量
    basis = svd.components_  # [n_components, d_model]
    S = svd.singular_values_  # [n_components]
    return basis, S, n_components


def make_random_subspace(d_model, dim, seed=12345):
    """构造随机正交子空间基 [dim, d_model]"""
    np.random.seed(seed)
    R = np.random.randn(dim, d_model).astype(np.float32)
    Q, _ = np.linalg.qr(R.T)  # [d_model, dim]
    return Q.T[:dim, :]  # [dim, d_model]


def make_orthogonal_subspace(basis, d_model, dim):
    """构造与basis正交的子空间基 [dim, d_model]"""
    # basis: [k, d_model], 构造与所有basis行正交的随机子空间
    np.random.seed(54321)
    R = np.random.randn(dim + basis.shape[0], d_model).astype(np.float32)
    # Gram-Schmidt: 把R投影到basis的正交补
    # P_perp = I - basis^T @ basis (投影到basis的正交补)
    # R_perp = R @ P_perp = R - R @ basis^T @ basis
    proj_coeffs = R @ basis.T  # [dim+k, k]
    R_orth = R - proj_coeffs @ basis  # [dim+k, d_model]
    # QR分解取前dim个正交基
    Q, _ = np.linalg.qr(R_orth.T)
    result = Q.T[:dim, :]
    # 验证正交性
    orth_err = np.max(np.abs(result @ basis.T))
    if orth_err > 0.01:
        print(f"    WARNING: orthogonal subspace not perfectly orthogonal, max err={orth_err:.6f}")
    return result


def compute_proj_ratio(vec, basis):
    """计算向量在子空间(basis行向量张成)中的投影比 = ||P_S(v)||^2 / ||v||^2"""
    if np.linalg.norm(vec) < 1e-10:
        return 0.0
    # P_S(v) = basis^T @ (basis @ v^T)^T = basis^T @ basis @ v
    # ||P_S(v)||^2 = v^T @ basis^T @ basis @ v
    proj_coeffs = basis @ vec  # [k]
    proj_norm_sq = np.sum(proj_coeffs ** 2)
    vec_norm_sq = np.sum(vec ** 2)
    return proj_norm_sq / vec_norm_sq


# ============================================================================
# 36A: ★★★★★★★★★ 随机子空间对照 (THE most critical)
# ============================================================================

def expA_subspace_control(model_name, model, tokenizer, device):
    """
    核心判别: W_U子空间是否特殊?

    构造3类子空间:
    1. S_WU = span(W_U行空间) - 真实子空间
    2. S_rand = 同维随机正交子空间 - 几何对照
    3. S_orth = 与W_U正交的子空间 - 对抗对照

    对每类子空间:
    - 注入扰动, 跟踪投影比随层变化
    - 如果只有W_U投影比上升 → routing
    - 如果所有子空间都类似 → geometry
    """
    print(f"\n{'='*70}")
    print(f"36A: ★★★★★★★★★ 随机子空间对照 — W_U是否特殊?")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 获取W_U
    W_U = get_W_U_np(model).astype(np.float32)
    print(f"W_U shape: {W_U.shape}")

    # W_U行空间基 (前200个SVD分量, 覆盖约14-28%能量)
    n_components = min(200, W_U.shape[0], d_model)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, n_components)
    print(f"W_U SVD: top-{k_wu} components cover {np.sum(S_wu[:k_wu]**2)/np.sum(S_wu**2)*100:.1f}% energy")

    # 构造3类子空间
    subspaces = {}

    # 1. W_U子空间
    subspaces["W_U"] = {
        "basis": basis_wu,
        "dim": k_wu,
        "desc": f"W_U row space (top-{k_wu} SVs)"
    }

    # 2. 随机子空间 (同维)
    subspaces["random"] = {
        "basis": make_random_subspace(d_model, k_wu, seed=12345),
        "dim": k_wu,
        "desc": f"Random orthonormal (dim={k_wu})"
    }

    # 3. 正交补子空间 (与W_U正交)
    subspaces["orthogonal"] = {
        "basis": make_orthogonal_subspace(basis_wu, d_model, k_wu),
        "dim": k_wu,
        "desc": f"Orthogonal to W_U (dim={k_wu})"
    }

    # 验证: 3类子空间的初始投影比 (对随机向量)
    np.random.seed(999)
    test_vec = np.random.randn(d_model).astype(np.float32)
    test_vec /= np.linalg.norm(test_vec)
    for name, ss in subspaces.items():
        ratio = compute_proj_ratio(test_vec, ss["basis"])
        print(f"  随机向量在{name}子空间的投影比: {ratio:.4f} (期望≈{ss['dim']/d_model:.4f})")

    # 采样层
    if n_layers <= 12:
        source_layers = [0]
        target_layers = list(range(n_layers))
    else:
        source_layers = [0, n_layers//4, n_layers//2]
        step = max(1, n_layers // 8)
        target_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))

    print(f"源层: {source_layers}")
    print(f"目标层: {target_layers}")

    eps = 0.01
    concepts = TEST_CONCEPTS  # apple, dog, hammer
    n_random_dirs = 10  # 每个子空间测10个随机方向

    results = {
        "model": model_name, "exp": "A",
        "experiment": "subspace_control",
        "n_layers": n_layers, "d_model": d_model,
        "n_components": k_wu,
        "subspace_dims": {name: ss["dim"] for name, ss in subspaces.items()},
        "eps": eps,
        "projection_trajectories": {},  # {concept: {source: {subspace: {target: ratio}}}}
        "baseline_proj_ratios": {},  # {concept: {subspace: {layer: ratio}}}
    }

    for concept in concepts:
        results["projection_trajectories"][concept] = {}
        results["baseline_proj_ratios"][concept] = {}

        # 收集baseline (无扰动的各层hidden state)
        print(f"\n概念: {concept}")
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            print(f"  baseline收集失败, 跳过")
            continue

        # 先测baseline hidden state在各子空间的投影比
        for ss_name, ss in subspaces.items():
            results["baseline_proj_ratios"][concept][ss_name] = {}
            for li in target_layers:
                if li in baseline_hs:
                    ratio = compute_proj_ratio(baseline_hs[li], ss["basis"])
                    results["baseline_proj_ratios"][concept][ss_name][str(li)] = float(ratio)

        # 对每个源层, 注入扰动, 跟踪投影比
        for src_l in source_layers:
            if src_l not in baseline_hs:
                continue

            results["projection_trajectories"][concept][str(src_l)] = {}
            h_scale = np.linalg.norm(baseline_hs[src_l])
            actual_eps = eps * h_scale

            for ss_name, ss in subspaces.items():
                results["projection_trajectories"][concept][str(src_l)][ss_name] = {}
                print(f"  L{src_l}注入, 子空间={ss_name}:")

                # 从该子空间中采n_random_dirs个方向
                np.random.seed(42)
                rand_coeffs = np.random.randn(n_random_dirs, ss["dim"]).astype(np.float32)
                for i in range(n_random_dirs):
                    rand_coeffs[i] /= np.linalg.norm(rand_coeffs[i])

                # 平均投影比
                avg_proj_at_targets = defaultdict(list)

                for i in range(n_random_dirs):
                    # 构造注入方向: 子空间基的线性组合
                    inject_dir = rand_coeffs[i] @ ss["basis"]  # [d_model]
                    inject_dir /= np.linalg.norm(inject_dir)

                    # 注入扰动, 收集各层
                    perturbed_hs, _, _ = inject_and_collect(
                        model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                        src_l, inject_dir, actual_eps, n_layers)

                    if perturbed_hs is None:
                        continue

                    # 计算delta在各目标层的投影比
                    for tgt_l in target_layers:
                        if tgt_l in perturbed_hs and tgt_l in baseline_hs:
                            delta = perturbed_hs[tgt_l] - baseline_hs[tgt_l]
                            for ss_name2, ss2 in subspaces.items():
                                ratio = compute_proj_ratio(delta, ss2["basis"])
                                avg_proj_at_targets[(tgt_l, ss_name2)].append(float(ratio))

                # 汇总: 对每个目标层, 报告注入到ss_name时, delta在3类子空间的投影比
                for tgt_l in target_layers:
                    entry = {}
                    for ss_name2 in subspaces:
                        key = (tgt_l, ss_name2)
                        if key in avg_proj_at_targets and len(avg_proj_at_targets[key]) > 0:
                            entry[ss_name2] = {
                                "mean": float(np.mean(avg_proj_at_targets[key])),
                                "std": float(np.std(avg_proj_at_targets[key])),
                                "n": len(avg_proj_at_targets[key])
                            }
                    results["projection_trajectories"][concept][str(src_l)][ss_name][str(tgt_l)] = entry

                    # 打印关键层的结果
                    if tgt_l in [src_l, n_layers-1] or abs(tgt_l - src_l) <= 4:
                        parts = []
                        for ss_name2 in subspaces:
                            if ss_name2 in entry:
                                parts.append(f"{ss_name2}={entry[ss_name2]['mean']:.4f}")
                        if parts:
                            print(f"    L{tgt_l}: {', '.join(parts)}")

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36A 判别分析: W_U是否特殊?")
    print(f"{'='*60}")

    for concept in concepts:
        for src_l in source_layers:
            src_key = str(src_l)
            if src_key not in results["projection_trajectories"].get(concept, {}):
                continue

            # W_U注入时: 看W_U投影比 vs random投影比 vs orth投影比
            wu_traj = results["projection_trajectories"][concept][src_key].get("W_U", {})

            for tgt_l in target_layers:
                tgt_key = str(tgt_l)
                if tgt_key not in wu_traj:
                    continue
                entry = wu_traj[tgt_key]

                wu_proj = entry.get("W_U", {}).get("mean", 0)
                rand_proj = entry.get("random", {}).get("mean", 0)
                orth_proj = entry.get("orthogonal", {}).get("mean", 0)

                if tgt_l == n_layers - 1 or abs(tgt_l - src_l) <= 1:
                    print(f"  {concept} L{src_l}→L{tgt_l} (W_U注入): "
                          f"W_U={wu_proj:.4f} rand={rand_proj:.4f} orth={orth_proj:.4f}")

                    if wu_proj > rand_proj * 1.5 and wu_proj > orth_proj * 1.5:
                        print(f"    ★ W_U特殊! → 支持H1(routing)")
                    elif abs(wu_proj - rand_proj) / max(rand_proj, 0.001) < 0.3:
                        print(f"    ✗ W_U不特殊 → 支持H0(geometry)")

    return results


# ============================================================================
# 36B: ★★★★★★★★ 未训练模型对照
# ============================================================================

def expB_untrained_model(model_name, model, tokenizer, device):
    """
    训练好的模型 vs 随机初始化模型

    对比: 注入扰动后, W_U投影比是否上升
    - 训练模型: 投影比上升 → 可能是routing
    - 未训练模型: 如果也上升 → 只是几何效应
    """
    print(f"\n{'='*70}")
    print(f"36B: ★★★★★★★★ 未训练模型对照 — routing是学来的吗?")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 获取W_U (从训练好的模型)
    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))

    # ===== 训练模型的W_U投影跟踪 (复用Phase 35B的核心逻辑) =====
    print(f"\n--- 训练模型: W_U投影跟踪 ---")

    source_layers = [0, n_layers//2]
    target_layers = sorted(set(list(range(0, n_layers, max(1, n_layers//8))) + [n_layers-1]))
    eps = 0.01
    concepts = TEST_CONCEPTS

    trained_results = {}

    for concept in concepts:
        trained_results[concept] = {}
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            continue

        for src_l in source_layers:
            if src_l not in baseline_hs:
                continue

            h_scale = np.linalg.norm(baseline_hs[src_l])
            actual_eps = eps * h_scale

            # 注入随机方向
            np.random.seed(42)
            proj_ratios = defaultdict(list)

            for trial in range(10):
                rand_dir = np.random.randn(d_model).astype(np.float32)
                rand_dir /= np.linalg.norm(rand_dir)

                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    src_l, rand_dir, actual_eps, n_layers)

                if perturbed_hs is None:
                    continue

                for tgt_l in target_layers:
                    if tgt_l in perturbed_hs and tgt_l in baseline_hs:
                        delta = perturbed_hs[tgt_l] - baseline_hs[tgt_l]
                        ratio = compute_proj_ratio(delta, basis_wu)
                        proj_ratios[tgt_l].append(float(ratio))

            trained_results[concept][src_l] = {
                "inject_ratio": float(compute_proj_ratio(rand_dir, basis_wu)),  # 注入时的投影比
                "target_ratios": {
                    str(tl): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for tl, v in proj_ratios.items()
                }
            }

            # 打印
            inject_r = trained_results[concept][src_l]["inject_ratio"]
            final_r = trained_results[concept][src_l]["target_ratios"].get(
                str(n_layers-1), {}).get("mean", 0)
            print(f"  {concept} L{src_l}: inject={inject_r:.4f} → L{n_layers-1}={final_r:.4f} "
                  f"(Δ={final_r - inject_r:+.4f})")

    # 保存trained W_U (在释放模型前)
    trained_W_U = get_W_U_np(model).astype(np.float32)
    print(f"  已保存训练模型的W_U (shape={trained_W_U.shape})")

    # ===== 未训练模型 =====
    print(f"\n--- 构造未训练模型 ---")

    # ★关键: 先释放训练模型, 否则GPU内存不够★
    print(f"  释放训练模型以腾出GPU内存...")
    # 保存W_U基 (在释放模型前已保存)
    # 释放模型
    release_model(model)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  训练模型已释放")

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    model_path = MODEL_CONFIGS[model_name]["path"]
    print(f"  加载config: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 创建随机初始化的模型
    print(f"  创建随机初始化模型...")
    untrained_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    untrained_model = untrained_model.to(device)
    untrained_model.eval()

    # 但保留训练好的lm_head权重! (因为我们测的是对W_U的投影)
    # 把训练好的lm_head权重复制过来
    print(f"  复制训练好的lm_head权重...")
    # trained_W_U 已经在上面保存了

    # 设置untrained model的lm_head
    if hasattr(untrained_model, 'lm_head') and untrained_model.lm_head is not None:
        with torch.no_grad():
            untrained_model.lm_head.weight.copy_(
                torch.tensor(trained_W_U, dtype=untrained_model.lm_head.weight.dtype))
    # 有些模型没有单独的lm_head, 用embed_tokens
    elif hasattr(untrained_model.model, 'embed_tokens'):
        # tied weights - 不需要单独设置
        pass

    untrained_results = {}

    for concept in concepts:
        untrained_results[concept] = {}

        # 收集baseline
        baseline_hs, _ = collect_all_layer_hs(
            untrained_model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            print(f"  {concept}: untrained baseline失败")
            continue

        for src_l in source_layers:
            if src_l not in baseline_hs:
                continue

            h_scale = np.linalg.norm(baseline_hs[src_l])
            actual_eps = eps * max(h_scale, 0.01)  # 避免除以0

            np.random.seed(42)
            proj_ratios = defaultdict(list)

            for trial in range(10):
                rand_dir = np.random.randn(d_model).astype(np.float32)
                rand_dir /= np.linalg.norm(rand_dir)

                perturbed_hs, _, _ = inject_and_collect(
                    untrained_model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    src_l, rand_dir, actual_eps, n_layers)

                if perturbed_hs is None:
                    continue

                for tgt_l in target_layers:
                    if tgt_l in perturbed_hs and tgt_l in baseline_hs:
                        delta = perturbed_hs[tgt_l] - baseline_hs[tgt_l]
                        if np.linalg.norm(delta) > 1e-10:
                            ratio = compute_proj_ratio(delta, basis_wu)
                            proj_ratios[tgt_l].append(float(ratio))

            untrained_results[concept][src_l] = {
                "inject_ratio": float(compute_proj_ratio(rand_dir, basis_wu)),
                "target_ratios": {
                    str(tl): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for tl, v in proj_ratios.items() if len(v) > 0
                }
            }

            inject_r = untrained_results[concept][src_l]["inject_ratio"]
            final_key = str(n_layers-1)
            final_r = untrained_results[concept][src_l]["target_ratios"].get(
                final_key, {}).get("mean", 0)
            print(f"  {concept} L{src_l}: inject={inject_r:.4f} → L{n_layers-1}={final_r:.4f} "
                  f"(Δ={final_r - inject_r:+.4f})")

    # 释放未训练模型
    del untrained_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36B 判别分析: routing是学来的还是架构固有的?")
    print(f"{'='*60}")

    for concept in concepts:
        for src_l in source_layers:
            tr = trained_results.get(concept, {}).get(src_l, {})
            ut = untrained_results.get(concept, {}).get(src_l, {})

            if not tr or not ut:
                continue

            tr_inject = tr.get("inject_ratio", 0)
            tr_final = tr.get("target_ratios", {}).get(str(n_layers-1), {}).get("mean", 0)
            ut_inject = ut.get("inject_ratio", 0)
            ut_final = ut.get("target_ratios", {}).get(str(n_layers-1), {}).get("mean", 0)

            tr_delta = tr_final - tr_inject
            ut_delta = ut_final - ut_inject

            print(f"  {concept} L{src_l}:")
            print(f"    训练模型: {tr_inject:.4f} → {tr_final:.4f} (Δ={tr_delta:+.4f})")
            print(f"    未训练模型: {ut_inject:.4f} → {ut_final:.4f} (Δ={ut_delta:+.4f})")

            if tr_delta > 0.02 and abs(ut_delta) < 0.01:
                print(f"    ★ 训练创造了路由! → 支持H1")
            elif abs(tr_delta - ut_delta) < 0.01:
                print(f"    ✗ 训练和未训练差不多 → 支持H0(geometry)")
            elif tr_delta > ut_delta + 0.01:
                print(f"    ★ 训练增强了路由 → 部分支持H1")
            else:
                print(f"    ? 模式不清晰, 需要更多分析")

    return {
        "model": model_name, "exp": "B",
        "experiment": "untrained_model",
        "trained_results": trained_results,
        "untrained_results": untrained_results,
    }


# ============================================================================
# 36C: ★★★★★★★ Jacobian主方向对齐
# ============================================================================

def expC_jacobian_alignment(model_name, model, tokenizer, device):
    """
    J = UΣV^T 的 top left singular vectors 是否对齐 W_U 行空间?

    如果 top-U 逐层对齐 W_U → routing (模型主动把能量集中到W_U可读方向)
    如果无系统对齐 → geometry
    """
    print(f"\n{'='*70}")
    print(f"36C: ★★★★★★★ Jacobian主方向对齐 — top-U vs W_U")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(0, n_layers - 1))
    else:
        step = max(1, (n_layers - 1) // 8)
        sample_layers = sorted(set(list(range(0, n_layers - 1, step)) + [n_layers - 2]))

    print(f"采样层: {sample_layers}")

    n_random_dirs = 50
    eps = 0.01
    concepts_for_jacobian = ["apple", "dog"]

    results = {
        "model": model_name, "exp": "C",
        "experiment": "jacobian_alignment",
        "n_layers": n_layers, "d_model": d_model,
        "k_wu": k_wu,
        "alignment_data": {},  # {layer: {concept: {top_k: alignment}}}
    }

    for li in sample_layers:
        results["alignment_data"][str(li)] = {}

        for concept in concepts_for_jacobian:
            # 收集baseline
            baseline_hs, _ = collect_all_layer_hs(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
            if baseline_hs is None or li not in baseline_hs or (li+1) not in baseline_hs:
                continue

            h_l_base = baseline_hs[li]
            h_l1_base = baseline_hs[li + 1]

            # 注入随机方向, 测量下一层变化
            np.random.seed(42)
            delta_at_next = np.zeros((n_random_dirs, d_model), dtype=np.float32)
            valid_count = 0

            for i in range(n_random_dirs):
                dir_vec = np.random.randn(d_model).astype(np.float32)
                dir_vec /= np.linalg.norm(dir_vec)
                actual_eps = eps * np.linalg.norm(h_l_base)

                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    li, dir_vec, actual_eps, n_layers)

                if perturbed_hs is not None and (li + 1) in perturbed_hs:
                    delta_next = perturbed_hs[li + 1] - h_l1_base
                    if np.linalg.norm(delta_next) > 1e-10:
                        delta_at_next[valid_count] = delta_next
                        valid_count += 1

            if valid_count < 10:
                print(f"  L{li} {concept}: 有效样本不足 ({valid_count})")
                continue

            delta_at_next = delta_at_next[:valid_count]

            # SVD of delta_at_next: [valid_count, d_model]
            U_j, S_j, Vt_j = np.linalg.svd(delta_at_next, full_matrices=False)

            # 计算 top-k left singular vectors 与 W_U 行空间的对齐
            # U_j: [valid_count, valid_count] 的左奇异向量
            # 但实际上 delta_at_next = U @ diag(S) @ Vt
            # U_j 的每一行是输出空间的方向 (在output feature space中的方向)
            # 不对: U_j: [valid_count, valid_count], Vt_j: [valid_count, d_model]
            # delta_at_next[i] ≈ sum_j U_j[i,j] * S_j[j] * Vt_j[j,:]
            # U_j 是样本空间 (50个方向的空间), Vt_j 是特征空间

            # 我们需要的是: 在输出特征空间中, 哪些方向被Jacobian放大?
            # 这需要计算 delta_at_next^T @ delta_at_next 的特征向量
            # = Vt^T @ diag(S^2) @ Vt
            # → 输出方差的主方向 = Vt_j 的行 (在d_model空间中)

            # 所以: 输出方向的主成分 = Vt_j 的行向量
            # 检查这些主方向是否对齐 W_U 行空间

            alignment_scores = {}
            for top_k in [1, 3, 5, 10, 20]:
                # top-k 输出主方向
                top_directions = Vt_j[:top_k, :]  # [top_k, d_model]

                # 每个主方向在W_U子空间的投影比
                proj_ratios = []
                for d in range(min(top_k, top_directions.shape[0])):
                    ratio = compute_proj_ratio(top_directions[d], basis_wu)
                    proj_ratios.append(float(ratio))

                mean_ratio = float(np.mean(proj_ratios))
                alignment_scores[f"top_{top_k}"] = {
                    "mean_proj_in_WU": mean_ratio,
                    "individual_ratios": proj_ratios,
                    "expected_random": float(k_wu / d_model),  # 随机期望
                }

            results["alignment_data"][str(li)][concept] = alignment_scores

            # 打印
            print(f"  L{li}→L{li+1} {concept}:")
            for top_k in [1, 5, 10]:
                key = f"top_{top_k}"
                if key in alignment_scores:
                    mean_r = alignment_scores[key]["mean_proj_in_WU"]
                    expected = alignment_scores[key]["expected_random"]
                    ratio_to_expected = mean_r / expected if expected > 0 else float('inf')
                    print(f"    top-{top_k} output PCs: W_U投影={mean_r:.4f} "
                          f"(随机期望={expected:.4f}, 倍数={ratio_to_expected:.2f}x)")

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36C 判别分析: Jacobian主方向是否对齐W_U?")
    print(f"{'='*60}")

    for li_str, concepts_data in results["alignment_data"].items():
        for concept, scores in concepts_data.items():
            top1 = scores.get("top_1", {}).get("mean_proj_in_WU", 0)
            top5 = scores.get("top_5", {}).get("mean_proj_in_WU", 0)
            expected = scores.get("top_1", {}).get("expected_random", 0)

            if top1 > expected * 3:
                print(f"  L{li_str} {concept}: top-1={top1:.4f} vs expected={expected:.4f} "
                      f"({top1/expected:.1f}x) ★ 强对齐! → 支持H1")
            elif top1 < expected * 1.5:
                print(f"  L{li_str} {concept}: top-1={top1:.4f} vs expected={expected:.4f} "
                      f"({top1/expected:.1f}x) ✗ 无对齐 → 支持H0")
            else:
                print(f"  L{li_str} {concept}: top-1={top1:.4f} vs expected={expected:.4f} "
                      f"({top1/expected:.1f}x) ? 弱对齐, 不确定")

    return results


# ============================================================================
# 36D: ★★★★★ 谱结构分析
# ============================================================================

def expD_spectral_analysis(model_name, model, tokenizer, device):
    """
    Singular value decay: 是否存在 few dominant directions?

    如果有强主方向 → routing可能沿这些方向
    如果完全平 → 不可能有目标导向机制
    """
    print(f"\n{'='*70}")
    print(f"36D: ★★★★★ 谱结构分析 — singular value decay")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样层
    if n_layers <= 12:
        sample_layers = list(range(0, n_layers - 1))
    else:
        step = max(1, (n_layers - 1) // 6)
        sample_layers = sorted(set(list(range(0, n_layers - 1, step)) + [n_layers - 2]))

    print(f"采样层: {sample_layers}")

    n_random_dirs = 50
    eps = 0.01
    concept = "apple"  # 只用一个概念 (谱结构跨概念一致)

    results = {
        "model": model_name, "exp": "D",
        "experiment": "spectral_analysis",
        "n_layers": n_layers, "d_model": d_model,
        "spectra": {},  # {layer: {sv_top10: [...], sv_all: [...], metrics: {...}}}
    }

    for li in sample_layers:
        # 收集baseline
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None or li not in baseline_hs or (li+1) not in baseline_hs:
            continue

        h_l_base = baseline_hs[li]
        h_l1_base = baseline_hs[li + 1]

        np.random.seed(42)
        delta_at_next = np.zeros((n_random_dirs, d_model), dtype=np.float32)
        valid_count = 0

        for i in range(n_random_dirs):
            dir_vec = np.random.randn(d_model).astype(np.float32)
            dir_vec /= np.linalg.norm(dir_vec)
            actual_eps = eps * np.linalg.norm(h_l_base)

            perturbed_hs, _, _ = inject_and_collect(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                li, dir_vec, actual_eps, n_layers)

            if perturbed_hs is not None and (li + 1) in perturbed_hs:
                delta_next = perturbed_hs[li + 1] - h_l1_base
                if np.linalg.norm(delta_next) > 1e-10:
                    delta_at_next[valid_count] = delta_next
                    valid_count += 1

        if valid_count < 10:
            continue

        delta_at_next = delta_at_next[:valid_count]

        # SVD
        U, S, Vt = np.linalg.svd(delta_at_next, full_matrices=False)

        # 谱分析指标
        S_sq = S ** 2
        total_energy = np.sum(S_sq)

        # Participation ratio (有效维度)
        pr = (np.sum(S_sq) ** 2) / (np.sum(S_sq ** 2)) if np.sum(S_sq) > 0 else 0

        # 90% energy rank
        cum_energy = np.cumsum(S_sq) / total_energy
        rank_90 = int(np.searchsorted(cum_energy, 0.9)) + 1

        # Top-1 / mean ratio (主方向优势度)
        top1_ratio = S[0] / np.mean(S) if np.mean(S) > 0 else float('inf')

        # Top-5 / total ratio
        top5_energy = float(np.sum(S_sq[:5]) / total_energy) if len(S) >= 5 else 1.0

        # Isotropy index: 1.0 = 完全等方, 0.0 = 单一方向
        # = (sum(S^2))^2 / (k * sum(S^4))
        isotropy = (np.sum(S_sq)**2) / (len(S) * np.sum(S_sq**2)) if np.sum(S_sq**2) > 0 else 0

        entry = {
            "singular_values": [float(x) for x in S],
            "top10_sv": [float(x) for x in S[:10]],
            "participation_ratio": float(pr),
            "rank_90": int(rank_90),
            "top1_to_mean_ratio": float(top1_ratio),
            "top5_energy_fraction": float(top5_energy),
            "isotropy_index": float(isotropy),
            "n_valid": valid_count,
            "condition_number": float(S[0] / S[-1]) if S[-1] > 0 else float('inf'),
        }
        results["spectra"][str(li)] = entry

        print(f"  L{li}→L{li+1}: PR={pr:.1f} rank90={rank_90} "
              f"top1/mean={top1_ratio:.2f} top5_energy={top5_energy:.3f} "
              f"isotropy={isotropy:.3f} cond={entry['condition_number']:.2f}")
        print(f"    top-10 SVs: {[f'{x:.3f}' for x in S[:10]]}")

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36D 判别分析: 是否存在dominant directions?")
    print(f"{'='*60}")

    for li_str, spec in results["spectra"].items():
        pr = spec["participation_ratio"]
        iso = spec["isotropy_index"]
        top5 = spec["top5_energy_fraction"]
        top1r = spec["top1_to_mean_ratio"]

        if top5 > 0.5 and top1r > 5:
            print(f"  L{li_str}: top5_energy={top5:.3f} top1/mean={top1r:.1f} "
                  f"★ 强主方向! → routing可能沿主方向")
        elif iso > 0.8 and top1r < 2:
            print(f"  L{li_str}: isotropy={iso:.3f} top1/mean={top1r:.1f} "
                  f"✗ 高度等方 → 不可能有目标导向")
        else:
            print(f"  L{li_str}: isotropy={iso:.3f} top1/mean={top1r:.1f} "
                  f"? 中等anisotropy, 需要更多分析")

    return results


# ============================================================================
# 36E: ★★★★ 子空间破坏实验 (因果验证)
# ============================================================================

def expE_subspace_ablation(model_name, model, tokenizer, device):
    """
    在某层去掉W_U投影, 继续前向传播, 看logit变化

    H1(routing): 早期去掉 → 后面能恢复(被重新路由); 晚期去掉 → 输出崩溃
    H0(geometry): 行为不可解释 / 无系统模式

    同时做random子空间破坏作为对照
    """
    print(f"\n{'='*70}")
    print(f"36E: ★★★★ 子空间破坏实验 — 因果验证")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))

    # 随机子空间对照
    basis_rand = make_random_subspace(d_model, k_wu, seed=99999)

    # 采样层
    if n_layers <= 12:
        ablation_layers = list(range(n_layers))
    else:
        step = max(1, n_layers // 6)
        ablation_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))

    print(f"破坏层: {ablation_layers}")

    concepts = TEST_CONCEPTS

    results = {
        "model": model_name, "exp": "E",
        "experiment": "subspace_ablation",
        "n_layers": n_layers, "d_model": d_model,
        "k_wu": k_wu,
        "ablation_results": {},  # {concept: {layer: {type: logit_change}}}
    }

    for concept in concepts:
        results["ablation_results"][concept] = {}
        print(f"\n概念: {concept}")

        # 先获取baseline logits
        baseline_logits, _ = modify_and_collect(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
            0, lambda h: h, n_layers)  # 不修改

        if baseline_logits is None:
            print(f"  baseline获取失败")
            continue

        # baseline W_U投影
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)

        for abl_l in ablation_layers:
            abl_key = str(abl_l)
            results["ablation_results"][concept][abl_key] = {}

            # 1. 去掉W_U投影
            def remove_WU_proj(h):
                # h: [d_model]
                proj_coeffs = basis_wu @ h  # [k_wu]
                proj = proj_coeffs @ basis_wu  # [d_model] = P_{W_U}(h)
                return h - proj

            try:
                logits_no_wu, _ = modify_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    abl_l, remove_WU_proj, n_layers)

                if logits_no_wu is not None:
                    logit_change = float(np.linalg.norm(logits_no_wu - baseline_logits))
                    # 也测W_U投影恢复了没有
                    abl_logits = logits_no_wu

                    results["ablation_results"][concept][abl_key]["remove_WU"] = {
                        "logit_change_l2": logit_change,
                        "logit_change_mean": float(np.mean(np.abs(logits_no_wu - baseline_logits))),
                        "logit_change_max": float(np.max(np.abs(logits_no_wu - baseline_logits))),
                    }
                    print(f"  L{abl_l} 去W_U: logit_change={logit_change:.2f}")
            except Exception as e:
                print(f"  L{abl_l} 去W_U 失败: {e}")

            # 2. 去掉random子空间投影 (对照)
            def remove_rand_proj(h):
                proj_coeffs = basis_rand @ h
                proj = proj_coeffs @ basis_rand
                return h - proj

            try:
                logits_no_rand, _ = modify_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    abl_l, remove_rand_proj, n_layers)

                if logits_no_rand is not None:
                    logit_change_rand = float(np.linalg.norm(logits_no_rand - baseline_logits))
                    results["ablation_results"][concept][abl_key]["remove_random"] = {
                        "logit_change_l2": logit_change_rand,
                    }
                    print(f"  L{abl_l} 去rand: logit_change={logit_change_rand:.2f}")

                    # 比较: W_U破坏 vs random破坏
                    wu_change = results["ablation_results"][concept][abl_key].get(
                        "remove_WU", {}).get("logit_change_l2", 0)
                    ratio = wu_change / logit_change_rand if logit_change_rand > 0 else float('inf')
                    results["ablation_results"][concept][abl_key]["WU_to_random_ratio"] = float(ratio)
                    print(f"  L{abl_l} WU破坏/rand破坏 = {ratio:.2f}x")

            except Exception as e:
                print(f"  L{abl_l} 去rand 失败: {e}")

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36E 判别分析: W_U子空间破坏的因果效应")
    print(f"{'='*60}")

    for concept in concepts:
        print(f"\n  {concept}:")
        for abl_l in ablation_layers:
            abl_key = str(abl_l)
            data = results["ablation_results"].get(concept, {}).get(abl_key, {})

            wu_change = data.get("remove_WU", {}).get("logit_change_l2", 0)
            rand_change = data.get("remove_random", {}).get("logit_change_l2", 1)
            ratio = data.get("WU_to_random_ratio", 0)

            # 判断模式
            if ratio > 3:
                specificity = "★ W_U特异性破坏! → 支持H1"
            elif ratio < 1.5:
                specificity = "✗ W_U不比random特殊 → 支持H0"
            else:
                specificity = "? 弱特异性, 不确定"

            # 判断恢复模式 (早期 vs 晚期)
            if abl_l < n_layers // 2:
                position = "早期"
            else:
                position = "晚期"

            print(f"    L{abl_l} ({position}): WU={wu_change:.2f} rand={rand_change:.2f} "
                  f"ratio={ratio:.2f}x {specificity}")

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 36: 判别实验 — routing vs geometry")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                       choices=["1", "2", "3", "4", "5"])
    args = parser.parse_args()

    model_name = args.model
    exp_id = args.exp

    print(f"Phase 36: 判别实验 — routing vs geometry")
    print(f"模型: {model_name}, 实验: {exp_id}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)

    try:
        if exp_id == "1":
            results = expA_subspace_control(model_name, model, tokenizer, device)
            exp_tag = "expA"
        elif exp_id == "2":
            results = expB_untrained_model(model_name, model, tokenizer, device)
            exp_tag = "expB"
        elif exp_id == "3":
            results = expC_jacobian_alignment(model_name, model, tokenizer, device)
            exp_tag = "expC"
        elif exp_id == "4":
            results = expD_spectral_analysis(model_name, model, tokenizer, device)
            exp_tag = "expD"
        elif exp_id == "5":
            results = expE_subspace_ablation(model_name, model, tokenizer, device)
            exp_tag = "expE"

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"ccml_phase36_{exp_tag}_{model_name}_results.json")

        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            return obj

        results = convert(results)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_file}")

    finally:
        release_model(model)


if __name__ == "__main__":
    main()
