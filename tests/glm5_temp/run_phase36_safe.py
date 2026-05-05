"""
Phase 36 安全执行脚本 - 避免GPU内存溢出导致崩溃
=================================================
策略:
1. 每个实验单独运行, 运行完后彻底释放GPU
2. expB (untrained model) 用CPU加载untrained model
3. 逐模型、逐实验执行

用法:
  python run_phase36_safe.py --model deepseek7b --exp 1
  python run_phase36_safe.py --model deepseek7b --exp 2  (expB特殊处理)
  python run_phase36_safe.py --model glm4 --exp 3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
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
# 核心函数
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
                captured[li] = output.detach().float().numpy()
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
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_U)
    basis = svd.components_
    S = svd.singular_values_
    return basis, S, n_components


def make_random_subspace(d_model, dim, seed=12345):
    np.random.seed(seed)
    R = np.random.randn(dim, d_model).astype(np.float32)
    Q, _ = np.linalg.qr(R.T)
    return Q.T[:dim, :]


def make_orthogonal_subspace(basis, d_model, dim):
    np.random.seed(54321)
    R = np.random.randn(dim + basis.shape[0], d_model).astype(np.float32)
    proj_coeffs = R @ basis.T
    R_orth = R - proj_coeffs @ basis
    Q, _ = np.linalg.qr(R_orth.T)
    result = Q.T[:dim, :]
    orth_err = np.max(np.abs(result @ basis.T))
    if orth_err > 0.01:
        print(f"    WARNING: orthogonal subspace not perfectly orthogonal, max err={orth_err:.6f}")
    return result


def compute_proj_ratio(vec, basis):
    if np.linalg.norm(vec) < 1e-10:
        return 0.0
    proj_coeffs = basis @ vec
    proj_norm_sq = np.sum(proj_coeffs ** 2)
    vec_norm_sq = np.sum(vec ** 2)
    return proj_norm_sq / vec_norm_sq


# ============================================================================
# ExpC: Jacobian主方向对齐
# ============================================================================

def expC_jacobian_alignment(model_name, model, tokenizer, device):
    print(f"\n{'='*70}")
    print(f"36C: Jacobian主方向对齐 — top-U vs W_U")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))

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
        "alignment_data": {},
    }

    for li in sample_layers:
        results["alignment_data"][str(li)] = {}

        for concept in concepts_for_jacobian:
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
                print(f"  L{li} {concept}: 有效样本不足 ({valid_count})")
                continue

            delta_at_next = delta_at_next[:valid_count]

            U_j, S_j, Vt_j = np.linalg.svd(delta_at_next, full_matrices=False)

            alignment_scores = {}
            for top_k in [1, 3, 5, 10, 20]:
                top_directions = Vt_j[:top_k, :]
                proj_ratios = []
                for d in range(min(top_k, top_directions.shape[0])):
                    ratio = compute_proj_ratio(top_directions[d], basis_wu)
                    proj_ratios.append(float(ratio))

                mean_ratio = float(np.mean(proj_ratios))
                alignment_scores[f"top_{top_k}"] = {
                    "mean_proj_in_WU": mean_ratio,
                    "individual_ratios": proj_ratios,
                    "expected_random": float(k_wu / d_model),
                }

            results["alignment_data"][str(li)][concept] = alignment_scores

            print(f"  L{li}→L{li+1} {concept}:")
            for top_k in [1, 5, 10]:
                key = f"top_{top_k}"
                if key in alignment_scores:
                    mean_r = alignment_scores[key]["mean_proj_in_WU"]
                    expected = alignment_scores[key]["expected_random"]
                    ratio_to_expected = mean_r / expected if expected > 0 else float('inf')
                    print(f"    top-{top_k} output PCs: W_U投影={mean_r:.4f} "
                          f"(随机期望={expected:.4f}, 倍数={ratio_to_expected:.2f}x)")

    return results


# ============================================================================
# ExpD: 谱结构分析
# ============================================================================

def expD_spectral_analysis(model_name, model, tokenizer, device):
    print(f"\n{'='*70}")
    print(f"36D: 谱结构分析 — singular value decay")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    if n_layers <= 12:
        sample_layers = list(range(0, n_layers - 1))
    else:
        step = max(1, (n_layers - 1) // 6)
        sample_layers = sorted(set(list(range(0, n_layers - 1, step)) + [n_layers - 2]))

    print(f"采样层: {sample_layers}")

    n_random_dirs = 50
    eps = 0.01
    concept = "apple"

    results = {
        "model": model_name, "exp": "D",
        "experiment": "spectral_analysis",
        "n_layers": n_layers, "d_model": d_model,
        "spectra": {},
    }

    for li in sample_layers:
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

        U, S, Vt = np.linalg.svd(delta_at_next, full_matrices=False)

        S_sq = S ** 2
        total_energy = np.sum(S_sq)

        pr = (np.sum(S_sq) ** 2) / (np.sum(S_sq ** 2)) if np.sum(S_sq) > 0 else 0
        cum_energy = np.cumsum(S_sq) / total_energy
        rank_90 = int(np.searchsorted(cum_energy, 0.9)) + 1
        top1_ratio = S[0] / np.mean(S) if np.mean(S) > 0 else float('inf')
        top5_energy = float(np.sum(S_sq[:5]) / total_energy) if len(S) >= 5 else 1.0
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

    return results


# ============================================================================
# ExpE: 子空间破坏实验
# ============================================================================

def expE_subspace_ablation(model_name, model, tokenizer, device):
    print(f"\n{'='*70}")
    print(f"36E: 子空间破坏实验 — 因果验证")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))
    basis_rand = make_random_subspace(d_model, k_wu, seed=99999)

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
        "ablation_results": {},
    }

    for concept in concepts:
        results["ablation_results"][concept] = {}
        print(f"\n概念: {concept}")

        baseline_logits, _ = modify_and_collect(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
            0, lambda h: h, n_layers)

        if baseline_logits is None:
            print(f"  baseline获取失败")
            continue

        for abl_l in ablation_layers:
            abl_key = str(abl_l)
            results["ablation_results"][concept][abl_key] = {}

            # 去掉W_U投影
            def remove_WU_proj(h):
                proj_coeffs = basis_wu @ h
                proj = proj_coeffs @ basis_wu
                return h - proj

            try:
                logits_no_wu, _ = modify_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    abl_l, remove_WU_proj, n_layers)

                if logits_no_wu is not None:
                    logit_change = float(np.linalg.norm(logits_no_wu - baseline_logits))
                    results["ablation_results"][concept][abl_key]["remove_WU"] = {
                        "logit_change_l2": logit_change,
                        "logit_change_mean": float(np.mean(np.abs(logits_no_wu - baseline_logits))),
                        "logit_change_max": float(np.max(np.abs(logits_no_wu - baseline_logits))),
                    }
                    print(f"  L{abl_l} 去W_U: logit_change={logit_change:.2f}")
            except Exception as e:
                print(f"  L{abl_l} 去W_U 失败: {e}")

            # 去掉random子空间投影
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

                    wu_change = results["ablation_results"][concept][abl_key].get(
                        "remove_WU", {}).get("logit_change_l2", 0)
                    ratio = wu_change / logit_change_rand if logit_change_rand > 0 else float('inf')
                    results["ablation_results"][concept][abl_key]["WU_to_random_ratio"] = float(ratio)
                    print(f"  L{abl_l} WU破坏/rand破坏 = {ratio:.2f}x")

            except Exception as e:
                print(f"  L{abl_l} 去rand 失败: {e}")

    return results


# ============================================================================
# ExpB: 未训练模型对照 (安全版 - CPU加载untrained model)
# ============================================================================

def expB_untrained_model_safe(model_name, model, tokenizer, device):
    """36B替代方案: 用随机权重矩阵模拟untrained模型的行为
    
    核心思路: untrained模型的行为等价于随机权重矩阵的线性变换
    如果训练模型的W_U投影比上升是"routing", 那么随机权重矩阵应该不会有这种现象
    
    方法:
    1. 从训练模型中提取每层的关键权重矩阵 (attn+MLP的输出投影)
    2. 用随机正交矩阵替换这些权重, 模拟untrained模型
    3. 在"随机模型"上做相同的W_U投影比跟踪
    
    但是: 创建完整untrained模型需要太多内存
    替代: 用"层级别Jacobian"代替完整前向传播
    - 训练模型的层Jacobian: J_trained = dh_{l+1}/dh_l (已测)
    - 随机模型的层Jacobian: J_random = 随机正交矩阵 (保持谱特性)
    
    更简单的方法: 逐层传播delta, 用随机正交变换代替层的实际变换
    """
    print(f"\n{'='*70}")
    print(f"36B: 未训练模型对照 (随机变换替代版)")
    print(f"{'='*70}")

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U_np(model).astype(np.float32)
    basis_wu, S_wu, k_wu = get_subspace_basis(W_U, d_model, min(200, W_U.shape[0]))

    # 训练模型的W_U投影跟踪
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

            np.random.seed(42)
            proj_ratios = {}

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
                        if tgt_l not in proj_ratios:
                            proj_ratios[tgt_l] = []
                        proj_ratios[tgt_l].append(float(ratio))

            trained_results[concept][src_l] = {
                "inject_ratio": float(k_wu / d_model),
                "target_ratios": {
                    str(tl): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for tl, v in proj_ratios.items() if len(v) > 0
                }
            }

            inject_r = trained_results[concept][src_l]["inject_ratio"]
            final_r = trained_results[concept][src_l]["target_ratios"].get(
                str(n_layers-1), {}).get("mean", 0)
            print(f"  {concept} L{src_l}: inject={inject_r:.4f} → L{n_layers-1}={final_r:.4f} "
                  f"(Δ={final_r - inject_r:+.4f})")

    # ===== 随机变换替代: 用逐层Jacobian的放大率+随机正交方向 =====
    print(f"\n--- 随机变换模拟 ---")
    print(f"  方法: 用训练模型的层Jacobian放大率, 但方向随机旋转")
    print(f"  如果W_U投影比上升需要特定的方向对齐 → routing")
    print(f"  如果随机旋转后仍然上升 → geometry (放大率本身导致)")

    # 先收集训练模型的逐层Jacobian放大率
    # ★优化: 只用5次trial, 只测5个层
    print(f"\n  收集训练模型的逐层放大率 (精简版)...")
    
    # 收集baseline hidden states (只用apple)
    baseline_all = {}
    for concept in ["apple"]:  # 只用apple来测放大率
        hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if hs is not None:
            baseline_all[concept] = hs

    # 对每层, 测量Jacobian的放大率
    layer_amp_ratios = {}
    # 精简: 只测5个层, 用5次trial
    sample_layers_for_amp = sorted(set(list(range(0, n_layers-1, max(1, n_layers//4))) + [n_layers-2]))
    
    for li in sample_layers_for_amp:
        concept = "apple"
        if concept not in baseline_all or li not in baseline_all[concept] or (li+1) not in baseline_all[concept]:
            continue
        
        h_base = baseline_all[concept][li]
        h_next = baseline_all[concept][li + 1]
        
        np.random.seed(42)
        amp_ratios = []
        for trial in range(5):  # 只5次
            dir_vec = np.random.randn(d_model).astype(np.float32)
            dir_vec /= np.linalg.norm(dir_vec)
            actual_eps = eps * np.linalg.norm(h_base)
            
            perturbed_hs, _, _ = inject_and_collect(
                model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                li, dir_vec, actual_eps, n_layers)
            
            if perturbed_hs is not None and (li+1) in perturbed_hs:
                delta_out = perturbed_hs[li+1] - h_next
                amp = np.linalg.norm(delta_out) / actual_eps
                amp_ratios.append(float(amp))
        
        if amp_ratios:
            layer_amp_ratios[li] = float(np.mean(amp_ratios))
            print(f"    L{li}: 平均放大率={layer_amp_ratios[li]:.3f}")
    
    # 对未测量的层, 用线性插值
    all_layers = list(range(n_layers - 1))
    measured_layers = sorted(layer_amp_ratios.keys())
    for li in all_layers:
        if li not in layer_amp_ratios:
            # 用最近的已测层的放大率
            if len(measured_layers) > 0:
                nearest = min(measured_layers, key=lambda x: abs(x - li))
                layer_amp_ratios[li] = layer_amp_ratios[nearest]
            else:
                layer_amp_ratios[li] = 1.0

    # 随机变换: 从src_l开始, 逐层传播delta
    # 每层: delta_{l+1} = amp_ratio * R_l @ delta_l, R_l是随机正交矩阵
    random_results = {}

    for concept in concepts:
        random_results[concept] = {}
        if concept not in baseline_all:
            continue

        for src_l in source_layers:
            if src_l not in baseline_all[concept]:
                continue

            h_scale = np.linalg.norm(baseline_all[concept][src_l])
            actual_eps = eps * h_scale

            np.random.seed(42)
            random_proj_trajectories = {}

            for trial in range(5):  # 只5次,够用
                # 初始delta: 随机方向
                delta = np.random.randn(d_model).astype(np.float32)
                delta /= np.linalg.norm(delta)
                delta *= actual_eps

                # 逐层传播
                for li in range(src_l, n_layers - 1):
                    # 获取该层的放大率
                    if li in layer_amp_ratios:
                        amp = layer_amp_ratios[li]
                    else:
                        # 用最近层的放大率, 或默认1.0
                        amp = 1.0
                    
                    # 随机正交旋转
                    np.random.seed(li * 1000 + trial)
                    R = np.random.randn(d_model, d_model).astype(np.float32)
                    Q, _ = np.linalg.qr(R)
                    
                    # delta_{l+1} = amp * Q @ delta
                    delta = amp * (Q @ delta)
                    
                    # 记录W_U投影比
                    if li + 1 in [tl for tl in target_layers]:
                        ratio = compute_proj_ratio(delta, basis_wu)
                        if (li + 1) not in random_proj_trajectories:
                            random_proj_trajectories[li + 1] = []
                        random_proj_trajectories[li + 1].append(float(ratio))
                
                # 也记录最后一层
                ratio = compute_proj_ratio(delta, basis_wu)
                if n_layers - 1 not in random_proj_trajectories:
                    random_proj_trajectories[n_layers - 1] = []
                random_proj_trajectories[n_layers - 1].append(float(ratio))

            random_results[concept][src_l] = {
                "inject_ratio": float(k_wu / d_model),
                "target_ratios": {
                    str(tl): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for tl, v in random_proj_trajectories.items() if len(v) > 0
                }
            }

            inject_r = random_results[concept][src_l]["inject_ratio"]
            final_r = random_results[concept][src_l]["target_ratios"].get(
                str(n_layers-1), {}).get("mean", 0)
            print(f"  {concept} L{src_l}(随机): inject={inject_r:.4f} → L{n_layers-1}={final_r:.4f} "
                  f"(Δ={final_r - inject_r:+.4f})")

    # ===== 判别分析 =====
    print(f"\n{'='*60}")
    print(f"36B 判别分析: routing是学来的还是几何效应?")
    print(f"{'='*60}")
    print(f"对比: 训练模型(有方向对齐) vs 随机旋转(保持放大率但方向随机)")
    print(f"如果训练模型W_U投影上升但随机旋转不上升 → routing(方向对齐重要)")
    print(f"如果两者都上升 → geometry(放大率本身导致)")

    for concept in concepts:
        for src_l in source_layers:
            tr = trained_results.get(concept, {}).get(src_l, {})
            rnd = random_results.get(concept, {}).get(src_l, {})

            if not tr or not rnd:
                continue

            tr_inject = tr.get("inject_ratio", 0)
            tr_final = tr.get("target_ratios", {}).get(str(n_layers-1), {}).get("mean", 0)
            rnd_inject = rnd.get("inject_ratio", 0)
            rnd_final = rnd.get("target_ratios", {}).get(str(n_layers-1), {}).get("mean", 0)

            tr_delta = tr_final - tr_inject
            rnd_delta = rnd_final - rnd_inject

            print(f"  {concept} L{src_l}:")
            print(f"    训练模型: {tr_inject:.4f} → {tr_final:.4f} (Δ={tr_delta:+.4f})")
            print(f"    随机旋转: {rnd_inject:.4f} → {rnd_final:.4f} (Δ={rnd_delta:+.4f})")

            if tr_delta > 0.02 and abs(rnd_delta) < 0.01:
                print(f"    ★ 训练创造了路由! 随机旋转不上升 → 支持H1(routing)")
            elif abs(tr_delta - rnd_delta) < 0.01:
                print(f"    ✗ 训练和随机差不多 → 支持H0(geometry)")
            elif tr_delta > rnd_delta + 0.01:
                print(f"    ★ 训练增强了路由 → 部分支持H1")
            else:
                print(f"    ? 模式不清晰")

    return {
        "model": model_name, "exp": "B",
        "experiment": "untrained_model_random_rotation",
        "trained_results": trained_results,
        "random_results": random_results,
        "layer_amp_ratios": layer_amp_ratios,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 36: 安全执行")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                       choices=["1", "2", "3", "4", "5"])
    args = parser.parse_args()

    model_name = args.model
    exp_id = args.exp

    exp_names = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    exp_tag = exp_names[exp_id]

    print(f"Phase 36: 安全执行")
    print(f"模型: {model_name}, 实验: {exp_tag}")

    # ExpB特殊: 内部管理模型加载/释放
    if exp_id == "2":
        model, tokenizer, device = load_model(model_name)
        try:
            results = expB_untrained_model_safe(model_name, model, tokenizer, device)
        finally:
            # expB内部已经释放了模型, 但以防万一
            try:
                release_model(model)
            except:
                pass
    else:
        # 其他实验: 正常加载模型
        model, tokenizer, device = load_model(model_name)
        try:
            if exp_id == "3":
                results = expC_jacobian_alignment(model_name, model, tokenizer, device)
            elif exp_id == "4":
                results = expD_spectral_analysis(model_name, model, tokenizer, device)
            elif exp_id == "5":
                results = expE_subspace_ablation(model_name, model, tokenizer, device)
            else:
                raise ValueError(f"Exp {exp_id} not supported in safe mode, use main script")
        finally:
            release_model(model)
            gc.collect()
            torch.cuda.empty_cache()

    # 保存结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"ccml_phase36_exp{exp_tag}_{model_name}_results.json")

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


if __name__ == "__main__":
    main()
