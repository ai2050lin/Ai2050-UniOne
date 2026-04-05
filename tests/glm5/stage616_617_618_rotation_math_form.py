#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage616-617-618: 旋转数学形式 + Gemma4补偿机制 + 权重有效维度

Stage616: 旋转的数学形式提取
  原理：已知MLP在极低维子空间(1-5维)内旋转(Stage610/613)，
  但不知道旋转的精确数学形式。
  - 对每层MLP的Δh矩阵做SVD，提取前两个主方向
  - 在该2D子空间内检查旋转是否满足Givens旋转性质：
    (a) 矩阵形式 [[cos θ, -sin θ], [sin θ, cos θ]]
    (b) 行列式 = 1
    (c) 正交性（行向量互相垂直且归一）
  - 同时检查指数映射形式：R = exp(A), A反对称
  - 对消歧方向和随机方向分别验证

Stage617: Gemma4 Attn-MLP互相补偿机制
  原理：Stage612发现Gemma4的Attn和MLP不是功能等价的(其他3模型等价)，
  零化两者效果不同，说明存在补偿关系。
  - 交叉实验：零化Attn后测量MLP贡献变化，零化MLP后测量Attn贡献变化
  - 逐层分析：哪些层的补偿最显著
  - 分析补偿的方向：是否在旋转子空间内

Stage618: MLP权重有效维度分析（轻量版Stage614补充）
  原理：Stage614因内存问题未完成GLM4/Gemma4。
  - 使用svdvals替代全SVD，降低内存占用
  - 分析up_proj/down_proj的奇异值分布
  - 计算"权重有效维度" vs "实际旋转维度"的比值
  - 探索SwiGLU激活如何把满秩权重压缩到低维变换

用法: python stage616_617_618_rotation_math_form.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, encode_to_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def safe_get_device(model):
    for attr in [None, 'model', 'model.model']:
        try:
            obj = model
            if attr:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
            return next(obj.parameters()).device
        except (StopIteration, AttributeError):
            continue
    return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def find_components(layer_module):
    attn_mod = None
    mlp_mod = None
    for name, child in layer_module.named_children():
        nl = name.lower()
        if 'attn' in nl or 'attention' in nl or 'self_attn' in nl:
            attn_mod = child
        if 'mlp' in nl or 'feed_forward' in nl or 'ffn' in nl:
            mlp_mod = child
    return attn_mod, mlp_mod


def make_zero_hook_fn(return_tuple=False):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            x = output[0]
            zeros = torch.zeros_like(x)
            return (zeros,) + output[1:]
        else:
            return torch.zeros_like(output)
    return hook_fn


DISAMB_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank"),
    ("She ate a red apple.", "Apple released the iPhone.", "apple"),
    ("The factory plant employs workers.", "She watered the plant.", "plant"),
]

PROBE_SENTENCES = [
    "The cat sat on the mat.",
    "A beautiful sunset over the ocean.",
    "The quantum physics lecture was fascinating.",
    "She traveled to Paris last summer.",
    "The company reported strong earnings.",
    "He played guitar in the band.",
    "The recipe calls for fresh ingredients.",
    "Climate change affects global weather patterns.",
    "The museum exhibits ancient artifacts.",
    "Digital technology transforms education.",
    "The old bridge crossed the river.",
    "Music brings joy to everyone.",
]


# ============ Stage616: 旋转数学形式提取 ============

def run_stage616(model, tokenizer, model_key):
    """
    分析每层MLP在消歧方向上的旋转是否满足Givens旋转性质。

    原理：
    1. 收集每层MLP对多个输入的Δh（MLP纯贡献）
    2. 对Δh矩阵做SVD，取前2个主方向 u1, u2
    3. 对消歧方向v，投影到u1-u2平面：p = [v·u1, v·u2]
    4. 对比MLP前后的投影变化，检查是否为2D旋转：
       - 正交变换：|p_out| ≈ |p_in|
       - 行列式 ≈ 1（保向）
       - 角度变化均匀（非剪切）
    5. 同时对随机方向做同样分析，检查旋转是否通用
    """
    print(f"\n  --- Stage616: 旋转数学形式提取 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)
    hidden_dim = None

    # Sample ~8 layers evenly
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    # Step 1: Collect MLP deltas for all probe sentences
    layer_deltas = {}  # layer_idx -> list of delta vectors
    layer_inputs = {}  # layer_idx -> list of input vectors
    layer_outputs = {}  # layer_idx -> list of output vectors

    for sent in PROBE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        mlp_inputs_local = {}
        mlp_outputs_local = {}

        hooks = []
        for li in sample_layers:
            layer = layers[li]
            _, mlp_mod = find_components(layer)
            if mlp_mod is None:
                continue

            if hidden_dim is None:
                for p in mlp_mod.parameters():
                    hidden_dim = p.shape[0] if p.dim() >= 2 else None
                    if hidden_dim and hidden_dim > 100:
                        break

            def make_hooks(layer_idx):
                def pre_hook(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        mlp_inputs_local[layer_idx] = input[0][0, -1, :].float().detach().cpu()
                def post_hook(module, input, output):
                    if isinstance(output, tuple):
                        mlp_outputs_local[layer_idx] = output[0][0, -1, :].float().detach().cpu()
                    else:
                        mlp_outputs_local[layer_idx] = output[0, -1, :].float().detach().cpu()
                return pre_hook, post_hook

            pre_h, post_h = make_hooks(li)
            hooks.append(mlp_mod.register_forward_pre_hook(pre_h))
            hooks.append(mlp_mod.register_forward_hook(post_h))

        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hooks:
                h.remove()

        for li in sample_layers:
            if li in mlp_inputs_local and li in mlp_outputs_local:
                inp = mlp_inputs_local[li]
                out = mlp_outputs_local[li]
                delta = out - inp
                if li not in layer_deltas:
                    layer_deltas[li] = []
                    layer_inputs[li] = []
                    layer_outputs[li] = []
                layer_deltas[li].append(delta)
                layer_inputs[li].append(inp)
                layer_outputs[li].append(out)

    if not layer_deltas or hidden_dim is None:
        print("  WARNING: No MLP data")
        return {"error": "no data"}

    # Step 2: For each sampled layer, build delta matrix and analyze 2D rotation
    layer_results = {}

    for li in sorted(layer_deltas.keys()):
        deltas = layer_deltas[li]
        inputs = layer_inputs[li]
        outputs = layer_outputs[li]
        n_s = len(deltas)

        delta_matrix = torch.stack(deltas, dim=0)  # [n_samples, hidden_dim]

        # SVD to find rotation subspace
        U, S, Vt = torch.linalg.svd(delta_matrix.float(), full_matrices=False)
        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-10:
            continue

        cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
        n2d = min(2, len(S))  # Top 2 directions

        # Top-2 principal directions of delta space
        u1 = Vt[0]  # [hidden_dim]
        u2 = Vt[1] if len(Vt) > 1 else torch.zeros_like(u1)

        # For each input-output pair, project onto u1-u2 plane
        # and check if the transformation is a 2D rotation
        angles_in = []
        angles_out = []
        norms_in = []
        norms_out = []
        determinants = []

        for k in range(n_s):
            inp_k = inputs[k]
            out_k = outputs[k]

            # Project input onto u1-u2
            p_in = torch.stack([torch.dot(inp_k, u1), torch.dot(inp_k, u2)])
            p_out = torch.stack([torch.dot(out_k, u1), torch.dot(out_k, u2)])

            # Angle of projection
            angle_in = torch.atan2(p_in[1], p_in[0]).item()
            angle_out = torch.atan2(p_out[1], p_out[0]).item()
            norm_in = torch.norm(p_in).item()
            norm_out = torch.norm(p_out).item()

            angles_in.append(angle_in)
            angles_out.append(angle_out)
            norms_in.append(norm_in)
            norms_out.append(norm_out)

            # 2x2 transformation matrix (approximate)
            # Build from least squares if we had enough samples
            # For now, compute cross-product (determinant proxy)
            det = p_in[0] * p_out[1] - p_in[1] * p_out[0]
            determinants.append(det.item())

        # Compute 2x2 transformation matrix via least squares
        # p_out = M @ p_in, solve M from all samples
        P_in = torch.stack([
            torch.stack([torch.dot(inputs[k], u1), torch.dot(inputs[k], u2)]) 
            for k in range(n_s)
        ])  # [n_s, 2]
        P_out = torch.stack([
            torch.stack([torch.dot(outputs[k], u1), torch.dot(outputs[k], u2)]) 
            for k in range(n_s)
        ])  # [n_s, 2]

        # Least squares: M = (P_in^T P_in)^{-1} P_in^T P_out
        try:
            M_2x2 = torch.linalg.lstsq(P_in, P_out).solution  # [2, 2]
        except:
            M_2x2 = torch.eye(2)

        # Check Givens rotation properties
        det_M = float(torch.det(M_2x2).item())
        # For Givens: M = [[cos θ, -sin θ], [sin θ, cos θ]]
        # det = cos²θ + sin²θ = 1
        # M^T M = I (orthogonal)

        MMT = M_2x2 @ M_2x2.T
        orth_error = float(torch.norm(MMT - torch.eye(2)).item())

        # Extract rotation angle from M
        # If M = [[a, -b], [b, a]], then θ = atan2(b, a)
        a, neg_b = M_2x2[0, 0].item(), M_2x2[0, 1].item()
        b, d = M_2x2[1, 0].item(), M_2x2[1, 1].item()
        extracted_angle = np.degrees(np.arctan2(b, a))
        givens_residual = np.sqrt((a - d)**2 + (neg_b + b)**2)  # should be ~0 for Givens

        # Check exponential map: R = exp(A), A = [[0, -θ], [θ, 0]]
        # For 2D, exp map and Givens are equivalent, so check 3D extension
        # Use top-3 directions if available
        n3d = min(3, len(S))
        u3 = Vt[2] if len(Vt) > 2 else torch.zeros_like(u1)

        # Build 3x3 transformation
        P_in_3d = torch.stack([
            torch.stack([torch.dot(inputs[k], u1), torch.dot(inputs[k], u2), torch.dot(inputs[k], u3)])
            for k in range(min(n_s, len(inputs)))
        ])[:n3d + 1]  # Need at least 3 samples for 3D
        P_out_3d = torch.stack([
            torch.stack([torch.dot(outputs[k], u1), torch.dot(outputs[k], u2), torch.dot(outputs[k], u3)])
            for k in range(min(n_s, len(outputs)))
        ])[:n3d + 1]

        is_givens = abs(givens_residual) < 0.3 and abs(det_M - 1.0) < 0.3 and orth_error < 0.3
        is_orthogonal = orth_error < 0.3

        # Norm preservation (for rotation, |p_out| ≈ |p_in|)
        norm_ratio = np.mean([norms_out[k] / max(norms_in[k], 1e-10) for k in range(len(norms_in))])

        # Energy in top-2
        energy_2d = float(cum_energy[1].item()) if len(cum_energy) > 1 else 1.0

        layer_results[str(li)] = {
            "effective_rank_2d": n2d,
            "energy_in_2d": round(energy_2d, 4),
            "rotation_angle_deg": round(extracted_angle, 2),
            "det_M": round(det_M, 4),
            "orth_error": round(orth_error, 4),
            "givens_residual": round(givens_residual, 4),
            "is_givens": bool(is_givens),
            "is_orthogonal": bool(is_orthogonal),
            "norm_preservation": round(norm_ratio, 4),
            "M_2x2": [[round(M_2x2[i, j].item(), 4) for j in range(2)] for i in range(2)],
            "n_samples": n_s,
        }

        if li % (max(1, n_layers // 8)) == 0 or li == sample_layers[-1]:
            print(f"    L{li}: angle={extracted_angle:.1f}°, det={det_M:.3f}, "
                  f"orth_err={orth_error:.4f}, givens_res={givens_residual:.3f}, "
                  f"is_givens={is_givens}, norm_preserve={norm_ratio:.3f}")

    # Also test on disambiguation direction
    disamb_results = {}
    for s1, s2, word in DISAMB_PAIRS:
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=128)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=128)
        enc1 = move_to_device(enc1, model)
        enc2 = move_to_device(enc2, model)

        with torch.no_grad():
            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

        word_disamb_rots = {}
        for li in sample_layers:
            h1 = out1.hidden_states[li + 1][0, -1, :].float().cpu()
            h2 = out2.hidden_states[li + 1][0, -1, :].float().cpu()
            h1_prev = out1.hidden_states[li][0, -1, :].float().cpu()
            h2_prev = out2.hidden_states[li][0, -1, :].float().cpu()

            d_prev = h1_prev - h2_prev
            d_curr = h1 - h2

            if str(li) in layer_results and torch.norm(d_prev) > 1e-10 and torch.norm(d_curr) > 1e-10:
                lr = layer_results[str(li)]
                u1_v = None
                u2_v = None

                # Recompute u1, u2 for this layer
                if li in layer_deltas:
                    delta_m = torch.stack(layer_deltas[li], dim=0)
                    _, _, Vt_local = torch.linalg.svd(delta_m.float(), full_matrices=False)
                    u1_v = Vt_local[0]
                    u2_v = Vt_local[1] if len(Vt_local) > 1 else torch.zeros_like(u1_v)

                    # Project disamb difference onto rotation subspace
                    proj_prev = torch.stack([torch.dot(d_prev, u1_v), torch.dot(d_prev, u2_v)])
                    proj_curr = torch.stack([torch.dot(d_curr, u1_v), torch.dot(d_curr, u2_v)])

                    # Check if disamb direction is in rotation subspace
                    disamb_in_subspace = (torch.norm(proj_prev) / torch.norm(d_prev)).item()

                    # Rotation of disamb direction
                    if torch.norm(proj_prev) > 1e-10 and torch.norm(proj_curr) > 1e-10:
                        disamb_rot_angle = np.degrees(np.arccos(np.clip(
                            cos_sim(proj_prev, proj_curr), -1, 1)))
                    else:
                        disamb_rot_angle = 0

                    word_disamb_rots[str(li)] = {
                        "disamb_in_subspace": round(disamb_in_subspace, 4),
                        "disamb_rot_angle": round(disamb_rot_angle, 2),
                    }

        disamb_results[word] = word_disamb_rots

    elapsed = time.time() - t0
    print(f"  Stage616 done in {elapsed:.1f}s")

    summary = {
        "layer_analysis": layer_results,
        "disamb_direction_analysis": disamb_results,
        "overall_givens_fraction": round(
            sum(1 for v in layer_results.values() if v["is_givens"]) / max(len(layer_results), 1), 3
        ),
        "overall_orthogonal_fraction": round(
            sum(1 for v in layer_results.values() if v["is_orthogonal"]) / max(len(layer_results), 1), 3
        ),
    }
    return summary


# ============ Stage617: Gemma4 Attn-MLP补偿机制 ============

def run_stage617(model, tokenizer, model_key):
    """
    分析Attn和MLP之间的补偿关系。

    原理：
    Stage612发现Gemma4的Attn和MLP不是功能等价的——零化两者效果不同。
    这说明它们之间存在补偿关系：当一方被移除时，另一方会"代偿"。

    实验设计：
    1. 正常运行 → baseline消歧度
    2. 零化单层Attn → 该层MLP的"独自贡献"
    3. 零化单层MLP → 该层Attn的"独自贡献"
    4. 零化单层Attn+MLP → 该层总贡献
    5. 补偿量 = (Attn独自 + MLP独自) - (两者一起) - baseline变化

    如果补偿量 > 0：存在超加性（互相增强）
    如果补偿量 < 0：存在互相抑制
    如果补偿量 ≈ 0：独立作用（加性）
    """
    print(f"\n  --- Stage617: Attn-MLP补偿机制 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    # Focus on layers around the disambiguation peak (~25% of network)
    peak_region = list(range(max(0, n_layers // 4 - 2), min(n_layers, n_layers // 4 + 3)))

    results = {}

    for s1, s2, word in DISAMB_PAIRS:
        print(f"\n    Processing '{word}'...")

        def run_with_config(zero_layers=None, zero_type="none"):
            """Run model with specified zeroing configuration."""
            hooks = []
            if zero_layers and zero_type != "none":
                for li in zero_layers:
                    layer = layers[li]
                    attn_mod, mlp_mod = find_components(layer)
                    if zero_type == "attn" and attn_mod is not None:
                        hooks.append(attn_mod.register_forward_hook(make_zero_hook_fn(return_tuple=True)))
                    elif zero_type == "mlp" and mlp_mod is not None:
                        hooks.append(mlp_mod.register_forward_hook(make_zero_hook_fn(return_tuple=False)))
                    elif zero_type == "both":
                        if attn_mod is not None:
                            hooks.append(attn_mod.register_forward_hook(make_zero_hook_fn(return_tuple=True)))
                        if mlp_mod is not None:
                            hooks.append(mlp_mod.register_forward_hook(make_zero_hook_fn(return_tuple=False)))

            try:
                disamb_per_layer = []
                for s in [s1, s2]:
                    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
                    enc = move_to_device(enc, model)
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                    h_last = out.hidden_states[-1][0, -1, :].float().cpu()
                    disamb_per_layer.append(h_last)
                return 1 - cos_sim(disamb_per_layer[0], disamb_per_layer[1])
            finally:
                for h in hooks:
                    h.remove()

        # Baseline
        baseline_disamb = run_with_config()

        # Per-layer analysis around peak
        layer_analysis = {}
        for li in peak_region:
            # Zero only Attn at layer li
            disamb_no_attn = run_with_config(zero_layers=[li], zero_type="attn")
            # Zero only MLP at layer li
            disamb_no_mlp = run_with_config(zero_layers=[li], zero_type="mlp")
            # Zero both
            disamb_no_both = run_with_config(zero_layers=[li], zero_type="both")

            # Individual contributions (change from baseline)
            attn_alone_effect = disamb_no_attn - baseline_disamb
            mlp_alone_effect = disamb_no_mlp - baseline_disamb
            both_effect = disamb_no_both - baseline_disamb

            # Compensation = expected additive effect - actual combined effect
            expected_additive = attn_alone_effect + mlp_alone_effect
            compensation = expected_additive - both_effect

            # Synergy: if compensation > 0, they enhance each other
            # Antagonism: if compensation < 0, they suppress each other
            layer_analysis[str(li)] = {
                "baseline": round(baseline_disamb, 4),
                "no_attn": round(disamb_no_attn, 4),
                "no_mlp": round(disamb_no_mlp, 4),
                "no_both": round(disamb_no_both, 4),
                "attn_effect": round(attn_alone_effect, 4),
                "mlp_effect": round(mlp_alone_effect, 4),
                "both_effect": round(both_effect, 4),
                "compensation": round(compensation, 4),
            }

            print(f"      L{li}: baseline={baseline_disamb:.4f}, "
                  f"Δattn={attn_alone_effect:+.4f}, Δmlp={mlp_alone_effect:+.4f}, "
                  f"Δboth={both_effect:+.4f}, compens={compensation:+.4f}")

        # Also do multi-layer analysis: zero all post-peak Attn, then add back one MLP at a time
        post_peak_start = n_layers // 2
        print(f"\n    Multi-layer cross analysis (post-peak L{post_peak_start}+):")

        # Zero all post-peak Attn
        post_layers = list(range(post_peak_start, n_layers))

        disamb_zero_all_attn = run_with_config(zero_layers=post_layers, zero_type="attn")
        disamb_zero_all_mlp = run_with_config(zero_layers=post_layers, zero_type="mlp")

        # Now zero all post-peak Attn AND all post-peak MLP except one at a time
        rescue_effects = []
        for rescue_li in post_layers:
            # Zero all post-peak Attn + all post-peak MLP except rescue_li
            hooks = []
            for li in post_layers:
                layer = layers[li]
                attn_mod, mlp_mod = find_components(layer)
                if attn_mod is not None:
                    hooks.append(attn_mod.register_forward_hook(make_zero_hook_fn(return_tuple=True)))
                if mlp_mod is not None and li != rescue_li:
                    hooks.append(mlp_mod.register_forward_hook(make_zero_hook_fn(return_tuple=False)))

            try:
                d1 = []
                for s in [s1, s2]:
                    enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
                    enc = move_to_device(enc, model)
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                    d1.append(out.hidden_states[-1][0, -1, :].float().cpu())
                rescue_disamb = 1 - cos_sim(d1[0], d1[1])
            finally:
                for h in hooks:
                    h.remove()

            rescue_effect = rescue_disamb - disamb_zero_all_attn  # How much this single MLP rescues
            rescue_effects.append({
                "layer": rescue_li,
                "rescue_effect": round(rescue_effect, 4),
            })

        results[word] = {
            "baseline_disamb": round(baseline_disamb, 4),
            "per_layer_analysis": layer_analysis,
            "multi_layer": {
                "zero_all_post_attn": round(disamb_zero_all_attn, 4),
                "zero_all_post_mlp": round(disamb_zero_all_mlp, 4),
                "best_rescue_layer": max(rescue_effects, key=lambda x: x["rescue_effect"]),
                "worst_rescue_layer": min(rescue_effects, key=lambda x: x["rescue_effect"]),
                "avg_rescue": round(np.mean([r["rescue_effect"] for r in rescue_effects]), 4),
            },
        }

    elapsed = time.time() - t0
    print(f"  Stage617 done in {elapsed:.1f}s")

    # Summary statistics
    all_compensations = []
    for word_data in results.values():
        for la in word_data.get("per_layer_analysis", {}).values():
            all_compensations.append(la["compensation"])

    return {
        "compensation_analysis": results,
        "mean_compensation": round(np.mean(all_compensations), 4) if all_compensations else 0,
        "std_compensation": round(np.std(all_compensations), 4) if all_compensations else 0,
        "dominant_interaction": "synergy" if np.mean(all_compensations) > 0.01 else (
            "antagonism" if np.mean(all_compensations) < -0.01 else "additive"),
    }


# ============ Stage618: MLP权重有效维度（轻量版Stage614） ============

def run_stage618(model, tokenizer, model_key):
    """
    分析MLP权重矩阵的奇异值分布，用svdvals替代全SVD减少内存。

    原理：
    Stage614发现权重矩阵是满秩的，但Stage613发现实际变换是低维的。
    SwiGLU的非线性激活把满秩权重"投影"到低维子空间。

    本实验：
    1. 对每层MLP的up_proj, down_proj做svdvals（只返回奇异值，不返回U/V）
    2. 计算权重有效维度（与Stage613的实际旋转维度对比）
    3. 分析SwiGLU gate_proj的奇异值分布
    4. 检查gate_proj与up_proj的奇异值对齐度
    """
    print(f"\n  --- Stage618: MLP权重有效维度分析 ---")
    t0 = time.time()
    layers = discover_layers(model)
    n_layers = len(layers)

    # Sample every 4th layer to save time
    sample_layers = list(range(0, n_layers, 4))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    layer_results = {}

    for li in sample_layers:
        layer = layers[li]
        _, mlp_mod = find_components(layer)
        if mlp_mod is None:
            continue

        # Find weight matrices - search recursively through submodules
        weights = {}
        for name, param in mlp_mod.named_parameters():
            if param.dim() >= 2 and param.shape[0] > 50:
                # Use full relative name (e.g. "gate_proj.weight" or "up_proj.weight")
                weights[name] = param

        layer_data = {"layer": li}

        for wname, w in weights.items():
            w_small = w.float().cpu()

            # Use svdvals to save memory (no U/V matrices)
            try:
                sv = torch.linalg.svdvals(w_small.detach())
            except Exception as e:
                print(f"    L{li} {wname}: SVD failed: {e}")
                continue

            sv_np = sv.numpy()
            total_e = (sv_np ** 2).sum()
            if total_e < 1e-10:
                continue

            # Effective rank
            p = (sv_np ** 2) / total_e
            p = p[p > 1e-10]
            eff_rank = np.exp(-float((p * np.log(p)).sum()))

            # Dimensions for energy thresholds
            cum_e = np.cumsum(sv_np ** 2) / total_e
            dims = {}
            for thr in [0.5, 0.8, 0.9, 0.95, 0.99]:
                dims[f"{int(thr*100)}%"] = int(np.searchsorted(cum_e, thr)) + 1

            # Condition number
            cond = float(sv_np[0] / max(sv_np[-1], 1e-10))

            layer_data[wname] = {
                "shape": list(w.shape),
                "effective_rank": round(eff_rank, 2),
                "condition_number": round(cond, 2),
                "dim_50%": dims.get("50%"),
                "dim_80%": dims.get("80%"),
                "dim_90%": dims.get("90%"),
                "dim_95%": dims.get("95%"),
                "dim_99%": dims.get("99%"),
                "top5_sv": [round(float(s), 4) for s in sv_np[:5]],
                "bottom5_sv": [round(float(s), 6) for s in sv_np[-5:]],
                "sv_decay_ratio": round(float(sv_np[4] / max(sv_np[0], 1e-10)), 4),
            }

        # Check gate-up alignment if both exist
        gate_key = [k for k in weights.keys() if 'gate' in k.lower()]
        up_key = [k for k in weights.keys() if 'up' in k.lower()]
        if gate_key and up_key:
            gate_w = weights[gate_key[0]].float().cpu().detach()
            up_w = weights[up_key[0]].float().cpu().detach()
            # Flatten and compare top singular vectors direction
            # Use svdvals on concatenated to check if they share a subspace
            gate_sv = torch.linalg.svdvals(gate_w).numpy()
            up_sv = torch.linalg.svdvals(up_w).numpy()
            # Correlation of singular value profiles
            min_len = min(len(gate_sv), len(up_sv))
            gate_norm = gate_sv[:min_len] / (gate_sv[0] + 1e-10)
            up_norm = up_sv[:min_len] / (up_sv[0] + 1e-10)
            sv_corr = float(np.corrcoef(gate_norm, up_norm)[0, 1])
            layer_data["gate_up_sv_corr"] = round(sv_corr, 4)

        layer_results[str(li)] = layer_data

        # Print summary
        up_keys = [k for k in layer_data.keys() if 'up' in k.lower()]
        gate_keys = [k for k in layer_data.keys() if 'gate' in k.lower()]
        down_keys = [k for k in layer_data.keys() if 'down' in k.lower()]
        sample_key = (up_keys + gate_keys + down_keys + list(weights.keys()))[:1]
        if sample_key:
            sk = sample_key[0]
            info = layer_data.get(sk, {})
            print(f"    L{li} {sk}: rank={info.get('effective_rank', 'N/A')}, "
                  f"cond={info.get('condition_number', 'N/A')}, "
                  f"gate_up_corr={layer_data.get('gate_up_sv_corr', 'N/A')}")

    elapsed = time.time() - t0
    print(f"  Stage618 done in {elapsed:.1f}s")

    # Aggregate
    all_eff_ranks = []
    all_conds = []
    for lr in layer_results.values():
        for key, val in lr.items():
            if isinstance(val, dict) and 'effective_rank' in val:
                all_eff_ranks.append(val['effective_rank'])
                all_conds.append(val['condition_number'])

    return {
        "weight_analysis": layer_results,
        "mean_effective_rank": round(np.mean(all_eff_ranks), 2) if all_eff_ranks else 0,
        "mean_condition_number": round(np.mean(all_conds), 2) if all_conds else 0,
    }


# ============ Main ============

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"=== Stage616-617-618: {model_key} ===")
    print(f"Timestamp: {TIMESTAMP}")

    model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
    try:
        r616 = run_stage616(model, tokenizer, model_key)
        gc.collect()
        torch.cuda.empty_cache()

        r617 = run_stage617(model, tokenizer, model_key)
        gc.collect()
        torch.cuda.empty_cache()

        r618 = run_stage618(model, tokenizer, model_key)
    finally:
        free_model(model)
        gc.collect()

    all_results = {
        "model": model_key,
        "timestamp": TIMESTAMP,
        "stage616_rotation_math_form": r616,
        "stage617_compensation": r617,
        "stage618_weight_dims": r618,
    }

    out_path = OUTPUT_DIR / f"stage616_617_618_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
