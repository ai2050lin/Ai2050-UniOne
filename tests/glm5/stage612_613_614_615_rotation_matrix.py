#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage612-613-614-615: 旋转矩阵数学刻画 + MLP权重分析 + unembed对齐

Stage612: 多层MLP/Attn零化分离旋转贡献
  - 同时零化后半所有层MLP → 只剩Attn贡献
  - 同时零化后半所有层Attn → 只剩MLP贡献
  - 测量整体旋转角度差异

Stage613: MLP等效旋转矩阵提取
  - 对每层MLP，收集多个输入向量，拟合线性映射 R: x → f(x) - x
  - 分析R的SVD结构：是否低秩？是否接近正交？
  - 提取"旋转子空间"的基向量

Stage614: 旋转子空间维度 vs MLP权重SVD
  - 分析每层MLP的up_proj/down_proj权重矩阵的奇异值分布
  - 检查权重SVD与实际旋转子空间维度的相关性
  - 探索DS7B=1维 vs Gemma4=29维的架构根因

Stage615: unembed矩阵与旋转子空间对齐
  - 计算unembed矩阵行空间与旋转子空间的交角
  - 如果unembed行空间包含旋转子空间 → 旋转不影响读出
  - 如果unembed行空间与旋转子空间正交 → 旋转会破坏读出

用法: python stage612_613_614_615_rotation_matrix.py [qwen3|deepseek7b|glm4|gemma4]
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

# 用于Stage613的随机探测句子
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
]


# ============ Stage612: 多层MLP/Attn零化分离旋转贡献 ============

def run_stage612(model, tokenizer, model_key):
    """
    同时零化后半部分所有层的MLP或Attn，测量整体旋转贡献。
    
    原理：Stage609单层零化无效（残差流绕过），改为同时零化多层。
    - 正常运行：所有层正常
    - 零化后半MLP：后半所有层的MLP输出=0 → 只剩Attn在旋转
    - 零化后半Attn：后半所有层的Attn输出=0 → 只剩MLP在旋转
    """
    print(f"\n  --- Stage612: 多层MLP/Attn零化分离旋转贡献 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    def run_with_multi_zero(s1, s2, zero_half="none", half="post"):
        """
        zero_half: "none", "mlp", "attn"
        half: "post" (后半) or "pre" (前半)
        """
        if half == "post":
            target_range = range(n_layers // 2, n_layers)
        else:
            target_range = range(0, n_layers // 2)

        hooks = []
        for li in target_range:
            layer = layers[li]
            attn_mod, mlp_mod = find_components(layer)
            if zero_half == "mlp" and mlp_mod is not None:
                hooks.append(mlp_mod.register_forward_hook(make_zero_hook_fn(return_tuple=False)))
            elif zero_half == "attn" and attn_mod is not None:
                hooks.append(attn_mod.register_forward_hook(make_zero_hook_fn(return_tuple=True)))

        try:
            h1s, h2s = [], []
            for s in [s1, s2]:
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
                enc = move_to_device(enc, model)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                for h in out.hidden_states:
                    hv = h[0, -1, :].float().cpu()
                    if s == s1:
                        h1s.append(hv)
                    else:
                        h2s.append(hv)
        finally:
            for h in hooks:
                h.remove()

        return h1s, h2s

    results = {}

    for s1, s2, word in DISAMB_PAIRS:
        print(f"\n    Processing '{word}'...")

        # Get all 4 configurations
        configs = [
            ("normal", "none", "post"),
            ("zero_post_mlp", "mlp", "post"),
            ("zero_post_attn", "attn", "post"),
            ("zero_pre_mlp", "mlp", "pre"),
            ("zero_pre_attn", "attn", "pre"),
        ]

        config_data = {}
        for cfg_name, zero_half, half in configs:
            h1s, h2s = run_with_multi_zero(s1, s2, zero_half=zero_half, half=half)
            
            # Find peak and measure overall rotation (L1 to last)
            peak_l = 0
            max_d = 0
            for li in range(n_layers):
                d = 1 - cos_sim(h1s[li], h2s[li])
                if d > max_d:
                    max_d = d
                    peak_l = li

            # Measure rotation from peak to last layer
            if peak_l < n_layers - 1:
                d_peak = h1s[peak_l] - h2s[peak_l]
                d_last = h1s[-1] - h2s[-1]
                dp_n = torch.norm(d_peak).item()
                dl_n = torch.norm(d_last).item()
                if dp_n > 1e-10 and dl_n > 1e-10:
                    total_rot = np.degrees(np.arccos(np.clip(cos_sim(d_peak, d_last), -1, 1)))
                else:
                    total_rot = 0
            else:
                total_rot = 0

            # Measure layer-by-layer rotation angles (post-peak)
            post_peak_rots = []
            for li in range(peak_l + 1, n_layers):
                d_prev = h1s[li-1] - h2s[li-1]
                d_curr = h1s[li] - h2s[li]
                dp_n = torch.norm(d_prev).item()
                dc_n = torch.norm(d_curr).item()
                if dp_n > 1e-10 and dc_n > 1e-10:
                    rot = np.degrees(np.arccos(np.clip(cos_sim(d_prev, d_curr), -1, 1)))
                    post_peak_rots.append(rot)
                else:
                    post_peak_rots.append(0)

            avg_post_rot = np.mean(post_peak_rots) if post_peak_rots else 0

            config_data[cfg_name] = {
                "peak_layer": peak_l,
                "peak_disamb": round(max_d, 4),
                "last_disamb": round(1 - cos_sim(h1s[-1], h2s[-1]), 4),
                "total_rot_peak_to_last": round(total_rot, 2),
                "avg_post_peak_rot": round(avg_post_rot, 2),
            }

        # Compute contributions
        normal = config_data["normal"]
        zero_post_mlp = config_data["zero_post_mlp"]
        zero_post_attn = config_data["zero_post_attn"]

        # If zeroing post-half MLP reduces rotation → MLP contributes to post-peak rotation
        mlp_rot_reduction = normal["avg_post_peak_rot"] - zero_post_mlp["avg_post_peak_rot"]
        attn_rot_reduction = normal["avg_post_peak_rot"] - zero_post_attn["avg_post_peak_rot"]

        # Disambiguation preservation
        mlp_disamb_preserve = zero_post_mlp["last_disamb"] - normal["last_disamb"]
        attn_disamb_preserve = zero_post_attn["last_disamb"] - normal["last_disamb"]

        results[word] = {
            "configs": config_data,
            "post_peak_mlp_rot_contribution": round(mlp_rot_reduction, 2),
            "post_peak_attn_rot_contribution": round(attn_rot_reduction, 2),
            "mlp_disamb_preservation": round(mlp_disamb_preserve, 4),
            "attn_disamb_preservation": round(attn_disamb_preserve, 4),
        }

        print(f"    {word}: peak=L{normal['peak_layer']}, "
              f"normal_rot={normal['avg_post_peak_rot']:.1f}deg, "
              f"zero_post_mlp_rot={zero_post_mlp['avg_post_peak_rot']:.1f}deg (Δ={mlp_rot_reduction:+.1f}), "
              f"zero_post_attn_rot={zero_post_attn['avg_post_peak_rot']:.1f}deg (Δ={attn_rot_reduction:+.1f})")
        print(f"      disamb: normal={normal['last_disamb']:.4f}, "
              f"zero_mlp={zero_post_mlp['last_disamb']:.4f} (Δ={mlp_disamb_preserve:+.4f}), "
              f"zero_attn={zero_post_attn['last_disamb']:.4f} (Δ={attn_disamb_preserve:+.4f})")

    elapsed = time.time() - t0
    print(f"  Stage612 done in {elapsed:.1f}s")
    return {"multi_layer_ablation": results}


# ============ Stage613: MLP等效旋转矩阵提取 ============

def run_stage613(model, tokenizer, model_key):
    """
    对每层MLP，收集多个随机输入向量，计算MLP的"纯贡献" f(x)-x，
    拟合线性映射来逼近MLP的非线性变换。

    原理：
    - MLP层的纯贡献：Δh = MLP(x) = output - input
    - 用多个随机输入收集Δh，堆叠成矩阵
    - 对[Δh矩阵]做SVD → 分析MLP的"等效线性旋转"
    - 如果Δh的秩远小于hidden_dim → MLP在一个低维子空间内旋转
    """
    print(f"\n  --- Stage613: MLP等效旋转矩阵提取 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)
    hidden_dim = None

    # Collect residual stream activations at each layer for multiple sentences
    layer_deltas = {}  # layer_idx -> list of delta vectors (MLP output)

    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))  # ~10 evenly spaced layers

    for sent in PROBE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        # We need to capture the input and output of each MLP layer
        # Use hooks to capture MLP input and output
        mlp_inputs = {}
        mlp_outputs = {}

        hooks = []
        for li in sample_layers:
            layer = layers[li]
            _, mlp_mod = find_components(layer)

            if mlp_mod is None:
                continue

            if hidden_dim is None:
                # Infer hidden dim from MLP input
                for p in mlp_mod.parameters():
                    hidden_dim = p.shape[0] if p.dim() >= 2 else None
                    if hidden_dim and hidden_dim > 100:
                        break

            def make_hooks(layer_idx):
                def pre_hook(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        mlp_inputs[layer_idx] = input[0][0, -1, :].float().detach().cpu()
                def post_hook(module, input, output):
                    if isinstance(output, tuple):
                        mlp_outputs[layer_idx] = output[0][0, -1, :].float().detach().cpu()
                    else:
                        mlp_outputs[layer_idx] = output[0, -1, :].float().detach().cpu()
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

        # Compute deltas and store
        for li in sample_layers:
            if li in mlp_inputs and li in mlp_outputs:
                delta = mlp_outputs[li] - mlp_inputs[li]  # MLP pure contribution
                if li not in layer_deltas:
                    layer_deltas[li] = []
                layer_deltas[li].append(delta)

    if not layer_deltas or hidden_dim is None:
        print("  WARNING: Could not collect MLP data")
        return {"error": "no mlp data collected"}

    # Analyze each layer's delta matrix
    layer_analysis = {}
    all_singular_values = {}

    for li in sorted(layer_deltas.keys()):
        deltas = layer_deltas[li]
        if len(deltas) < 3:
            continue

        delta_matrix = torch.stack(deltas, dim=0)  # [n_samples, hidden_dim]
        n_samples, hd = delta_matrix.shape

        # SVD of delta matrix
        U, S, Vt = torch.linalg.svd(delta_matrix.float(), full_matrices=False)

        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-10:
            continue

        cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy

        # Effective rank
        thresholds = [0.5, 0.8, 0.95, 0.99]
        dims = {}
        for thr in thresholds:
            dims[f"{int(thr*100)}%"] = min(int((cum_energy < thr).sum().item()) + 1, len(S))

        # Check if delta matrix is approximately low-rank
        # Effective rank = exp(-sum(p_i * log(p_i))) where p_i = s_i^2 / sum(s^2)
        p = (S ** 2) / total_energy
        p = p[p > 1e-10]
        eff_rank = np.exp(-float((p * torch.log(p)).sum()))

        # Is the transformation approximately orthogonal?
        # For orthogonal: Vt @ Vt.T ≈ I (but Vt is [k, hidden], so Vt @ Vt.T is [k, k])
        # Instead check: singular values ≈ 1 (normalized)
        if len(S) > 0:
            sv_std = float(torch.std(S / S[0]).item()) if S[0] > 0 else 0
        else:
            sv_std = 0

        # Top-5 singular values
        top5_sv = [round(float(s), 4) for s in S[:5].tolist()]
        top5_ratio = [round(float(s**2 / total_energy), 4) for s in S[:5]]

        layer_analysis[li] = {
            "n_samples": n_samples,
            "effective_rank": round(eff_rank, 2),
            "dims_for_energy": dims,
            "top5_singular_values": top5_sv,
            "top5_energy_ratio": top5_ratio,
            "sv_normalized_std": round(sv_std, 4),
        }
        all_singular_values[li] = [round(float(s), 2) for s in S.tolist()]

    # Summary statistics
    eff_ranks = [la["effective_rank"] for la in layer_analysis.values()]
    dim95_list = [la["dims_for_energy"]["95%"] for la in layer_analysis.values()]

    results = {
        "hidden_dim": hidden_dim,
        "n_sample_layers": len(layer_analysis),
        "sample_layers": sorted(layer_analysis.keys()),
        "avg_effective_rank": round(float(np.mean(eff_ranks)), 2) if eff_ranks else 0,
        "avg_dim_95pct": round(float(np.mean(dim95_list)), 2) if dim95_list else 0,
        "layer_analysis": layer_analysis,
    }

    print(f"    hidden_dim={hidden_dim}, {len(layer_analysis)} layers analyzed")
    print(f"    avg_effective_rank={results['avg_effective_rank']:.2f}, "
          f"avg_dim_95pct={results['avg_dim_95pct']:.2f}")

    elapsed = time.time() - t0
    print(f"  Stage613 done in {elapsed:.1f}s")
    return {"mlp_rotation_matrix": results}


# ============ Stage614: 旋转子空间 vs MLP权重SVD ============

def run_stage614(model, tokenizer, model_key):
    """
    分析每层MLP权重矩阵(up_proj, down_proj)的SVD结构，
    与实际旋转子空间维度(Stage610)对比。

    原理：
    - MLP的权重矩阵决定了其变换能力
    - up_proj: hidden → intermediate (扩展)
    - down_proj: intermediate → hidden (压缩)
    - 如果down_proj的有效秩很低 → MLP输出被限制在低维子空间
    - 这可以解释DS7B=1维 vs Gemma4=29维的差异
    """
    print(f"\n  --- Stage614: 旋转子空间 vs MLP权重SVD ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    weight_analysis = []

    # Sample every 4th layer for speed (full analysis takes >10min for large models)
    sample_indices = list(range(0, n_layers, max(1, n_layers // 8)))

    for li in sample_indices:
        layer = layers[li]
        _, mlp_mod = find_components(layer)

        if mlp_mod is None:
            continue

        # Extract weight matrices
        up_w = None
        down_w = None
        gate_w = None

        for name, param in mlp_mod.named_parameters():
            nl = name.lower()
            if 'up' in nl or 'w2' in nl:
                up_w = param.data.float().cpu()
            elif 'down' in nl or 'w1' in nl or 'o_proj' in nl:
                down_w = param.data.float().cpu()
            elif 'gate' in nl or 'w3' in nl:
                gate_w = param.data.float().cpu()

        if up_w is None or down_w is None:
            continue

        # Use svdvals for speed (only computes singular values, not full SVD)
        try:
            up_svd_S = torch.linalg.svdvals(up_w.float())
            down_svd_S = torch.linalg.svdvals(down_w.float())
        except Exception:
            continue

        # Effective rank of each
        def eff_rank(s):
            s2 = s ** 2
            total = s2.sum()
            if total < 1e-10:
                return 0
            p = s2 / total
            p = p[p > 1e-10]
            return float(np.exp(-float((p * torch.log(p)).sum())))

        up_eff_rank = eff_rank(up_svd_S)
        down_eff_rank = eff_rank(down_svd_S)

        # Dimension for 95% energy
        def dim_for_energy(s, thr=0.95):
            s2 = s ** 2
            cum = torch.cumsum(s2, dim=0) / s2.sum()
            return min(int((cum < thr).sum().item()) + 1, len(s))

        up_dim95 = dim_for_energy(up_svd_S)
        down_dim95 = dim_for_energy(down_svd_S)

        # Extract singular values for top5 reporting
        up_S = up_svd_S
        down_S = down_svd_S

        # Check gate projection if exists
        gate_dim95 = None
        gate_eff_rank = None
        if gate_w is not None:
            try:
                gate_svd_S = torch.linalg.svdvals(gate_w.float())
                gate_dim95 = dim_for_energy(gate_svd_S)
                gate_eff_rank = eff_rank(gate_svd_S)
            except Exception:
                gate_dim95 = None
                gate_eff_rank = None

        # MLP ratio
        mlp_ratio = up_w.shape[1] / up_w.shape[0] if up_w.shape[0] > 0 else 0

        layer_info = {
            "layer": li,
            "up_proj_shape": list(up_w.shape),
            "down_proj_shape": list(down_w.shape),
            "mlp_ratio": round(mlp_ratio, 2),
            "up_eff_rank": round(up_eff_rank, 2),
            "down_eff_rank": round(down_eff_rank, 2),
            "up_dim_95pct": up_dim95,
            "down_dim_95pct": down_dim95,
            "up_top5_sv": [round(float(s), 4) for s in up_S[:5].tolist()],
            "down_top5_sv": [round(float(s), 4) for s in down_S[:5].tolist()],
        }
        if gate_dim95 is not None:
            layer_info["gate_dim_95pct"] = gate_dim95
            layer_info["gate_eff_rank"] = round(gate_eff_rank, 2)

        weight_analysis.append(layer_info)

    # Summary
    if weight_analysis:
        avg_up_rank = np.mean([wa["up_eff_rank"] for wa in weight_analysis])
        avg_down_rank = np.mean([wa["down_eff_rank"] for wa in weight_analysis])
        avg_up_dim95 = np.mean([wa["up_dim_95pct"] for wa in weight_analysis])
        avg_down_dim95 = np.mean([wa["down_dim_95pct"] for wa in weight_analysis])
        avg_mlp_ratio = np.mean([wa["mlp_ratio"] for wa in weight_analysis])

        # Check correlation: down_eff_rank vs layer position
        early = [wa["down_eff_rank"] for wa in weight_analysis[:len(weight_analysis)//2]]
        late = [wa["down_eff_rank"] for wa in weight_analysis[len(weight_analysis)//2:]]

        results = {
            "n_layers_analyzed": len(weight_analysis),
            "avg_mlp_ratio": round(float(avg_mlp_ratio), 2),
            "avg_up_eff_rank": round(float(avg_up_rank), 2),
            "avg_down_eff_rank": round(float(avg_down_rank), 2),
            "avg_up_dim_95pct": round(float(avg_up_dim95), 2),
            "avg_down_dim_95pct": round(float(avg_down_dim95), 2),
            "early_half_down_eff_rank": round(float(np.mean(early)), 2) if early else 0,
            "late_half_down_eff_rank": round(float(np.mean(late)), 2) if late else 0,
            "layer_data": weight_analysis,
        }

        print(f"    {len(weight_analysis)} layers analyzed")
        print(f"    avg mlp_ratio={avg_mlp_ratio:.2f}, "
              f"up_eff_rank={avg_up_rank:.1f}, down_eff_rank={avg_down_rank:.1f}")
        print(f"    up_dim95={avg_up_dim95:.1f}, down_dim95={avg_down_dim95:.1f}")
    else:
        results = {"error": "no weight data"}

    elapsed = time.time() - t0
    print(f"  Stage614 done in {elapsed:.1f}s")
    return {"mlp_weight_analysis": results}


# ============ Stage615: unembed与旋转子空间对齐 ============

def run_stage615(model, tokenizer, model_key):
    """
    分析unembed矩阵的行空间与旋转子空间的对齐关系。

    原理：
    - 旋转子空间(Stage610) = 消歧方向变化的主要方向
    - unembed矩阵行空间 = 可以从hidden state读出的方向
    - 如果旋转子空间 ⊆ unembed行空间 → 旋转不破坏读出（消歧度保持高）
    - 如果旋转子空间 ⊥ unembed行空间的一部分 → 旋转会降低消歧度

    算法：
    1. 从Stage610获取旋转子空间的基向量（右奇异向量Vt的top-k行）
    2. 获取unembed矩阵
    3. 计算旋转子空间在unembed行空间中的投影比例
    4. 计算旋转子空间与unembed行空间的夹角
    """
    print(f"\n  --- Stage615: unembed与旋转子空间对齐 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    # Get unembed matrix
    unembed = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        unembed = model.lm_head.weight.data.float().cpu()
    elif hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe is not None and hasattr(oe, 'weight'):
            unembed = oe.weight.data.float().cpu()

    if unembed is None:
        print("  WARNING: No unembed matrix found")
        return {"error": "no unembed matrix"}

    unembed_cpu = unembed.cpu()
    vocab_size, hidden_dim = unembed_cpu.shape

    # 1. Compute rotation subspace from disambiguation direction deltas
    all_deltas = []
    for s1, s2, word in DISAMB_PAIRS:
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            all_deltas.append(out.hidden_states[-1][0, -1, :].float().cpu())

    # Compute deltas between contexts for same word
    context_deltas = []
    for i in range(0, len(DISAMB_PAIRS) * 2, 2):
        if i + 1 < len(all_deltas):
            delta = all_deltas[i] - all_deltas[i+1]
            context_deltas.append(delta)

    # Also compute per-layer deltas (rotation deltas)
    layer_rotation_deltas = []
    for s1, s2, word in DISAMB_PAIRS:
        h1s, h2s = [], []
        for s in [s1, s2]:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            enc = move_to_device(enc, model)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for h in out.hidden_states:
                hv = h[0, -1, :].float().cpu()
                if s == s1:
                    h1s.append(hv)
                else:
                    h2s.append(hv)
        for li in range(1, n_layers):
            d = (h1s[li] - h2s[li]) - (h1s[li-1] - h2s[li-1])
            layer_rotation_deltas.append(d)

    # Combine both types of deltas
    all_rot_deltas = context_deltas + layer_rotation_deltas
    rot_matrix = torch.stack(all_rot_deltas, dim=0)  # [n, hidden_dim]

    # SVD to get rotation subspace
    _, S_rot, Vt_rot = torch.linalg.svd(rot_matrix.float(), full_matrices=False)
    total_energy = (S_rot ** 2).sum().item()
    cum_ratio = torch.cumsum(S_rot ** 2, dim=0) / total_energy

    # Top-k rotation subspace basis vectors
    rot_dim_95 = int((cum_ratio < 0.95).sum().item()) + 1
    rot_dim_95 = min(rot_dim_95, 50, len(Vt_rot))
    rot_basis = Vt_rot[:rot_dim_95, :]  # [k, hidden_dim]

    # 2. Compute unembed subspace using right singular vectors (need full SVD for this)
    # Use truncated approach: only compute top-k components
    ue_k = min(50, hidden_dim)
    try:
        # Use economy SVD: unembed is [vocab, hidden], we want Vt (right singular vectors)
        # For speed, compute U, S, Vt using truncated SVD approximation
        _, S_ue_full, Vt_ue = torch.linalg.svd(unembed_cpu.float(), full_matrices=False)
        ue_cum = torch.cumsum(S_ue_full ** 2, dim=0) / (S_ue_full ** 2).sum()
        ue_dim_95 = int((ue_cum < 0.95).sum().item()) + 1
        ue_dim_95 = min(ue_dim_95, ue_k, len(Vt_ue))
        ue_basis = Vt_ue[:ue_dim_95, :]
    except Exception:
        ue_basis = None
        ue_dim_95 = hidden_dim

    # 3. Compute alignment between rotation subspace and unembed subspace
    if ue_basis is not None:
        rot_in_ue = []
        for i in range(rot_dim_95):
            rv = rot_basis[i]  # [hidden_dim]
            proj_coeffs = ue_basis @ rv  # [ue_dim_95]
            reconstructed = proj_coeffs @ ue_basis  # [hidden_dim]
            cos_angle = cos_sim(rv, reconstructed)
            rot_in_ue.append(abs(cos_angle))
        avg_alignment = float(np.mean(rot_in_ue))
        min_alignment = float(np.min(rot_in_ue))
        max_alignment = float(np.max(rot_in_ue))
    else:
        avg_alignment = 0
        min_alignment = 0
        max_alignment = 0

    # 4. For each context delta, compute how well unembed can capture it
    # unembed can capture a direction d if d is in unembed's row space
    # Since vocab >> hidden, unembed row space = full hidden space
    # But the "sensitivity" of unembed to different directions varies
    # We measure this by: for direction d, what's the norm of unembed @ d?
    # Higher norm = unembed is more sensitive to this direction

    delta_sensitivities = []
    for delta in context_deltas:
        delta_norm = torch.norm(delta).item()
        if delta_norm < 1e-10:
            delta_sensitivities.append(0)
            continue
        logit_response = unembed_cpu @ delta  # [vocab_size]
        logit_energy = torch.norm(logit_response).item()
        # Normalize by delta norm and vocab size
        sensitivity = logit_energy / (delta_norm * np.sqrt(vocab_size))
        delta_sensitivities.append(sensitivity)

    # 5. Compare: rotation direction sensitivities vs random direction sensitivities
    n_random = 100
    random_sensitivities = []
    for _ in range(n_random):
        rand_dir = torch.randn(hidden_dim)
        rand_dir = rand_dir / torch.norm(rand_dir)
        logit_resp = unembed_cpu @ rand_dir
        sens = torch.norm(logit_resp).item() / np.sqrt(vocab_size)
        random_sensitivities.append(sens)

    avg_rot_sens = float(np.mean(delta_sensitivities))
    avg_rand_sens = float(np.mean(random_sensitivities))
    sens_ratio = avg_rot_sens / max(avg_rand_sens, 1e-10)

    # 6. Check: do rotation basis vectors align with unembed's top singular vectors?
    top_ue_alignment = []
    if ue_basis is not None:
        for i in range(min(10, rot_dim_95)):
            rv = rot_basis[i]
            aligns = []
            for j in range(min(10, ue_basis.shape[0])):
                cos_a = abs(cos_sim(rv, ue_basis[j]))
                aligns.append(round(cos_a, 4))
            top_ue_alignment.append(aligns)

    results = {
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "rotation_subspace_dim_95pct": rot_dim_95,
        "unembed_subspace_dim_95pct": ue_dim_95,
        "avg_rot_ue_alignment": round(avg_alignment, 4),
        "min_rot_ue_alignment": round(min_alignment, 4),
        "max_rot_ue_alignment": round(max_alignment, 4),
        "avg_rotation_direction_sensitivity": round(avg_rot_sens, 4),
        "avg_random_direction_sensitivity": round(avg_rand_sens, 4),
        "sensitivity_ratio": round(sens_ratio, 4),
        "rotation_vs_unembed_top10_alignment": top_ue_alignment[:5],
    }

    print(f"    rot_subspace_dim={rot_dim_95}, ue_subspace_dim={ue_dim_95}")
    print(f"    avg rot-ue alignment={avg_alignment:.4f} "
          f"(min={min_alignment:.4f}, max={max_alignment:.4f})")
    print(f"    rot_sensitivity={avg_rot_sens:.4f}, rand_sensitivity={avg_rand_sens:.4f}, "
          f"ratio={sens_ratio:.4f}")

    elapsed = time.time() - t0
    print(f"  Stage615 done in {elapsed:.1f}s")
    return {"unembed_rotation_alignment": results}


# ============ Main ============

MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]


def run_single_model(mk):
    print(f"\n{'='*60}")
    print(f"  Loading {mk}...")
    print(f"{'='*60}")
    t0 = time.time()
    model, tokenizer = load_model_bundle(mk)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    try:
        s612 = run_stage612(model, tokenizer, mk)
        s613 = run_stage613(model, tokenizer, mk)
        s614 = run_stage614(model, tokenizer, mk)
        s615 = run_stage615(model, tokenizer, mk)
        result = {"stage612": s612, "stage613": s613, "stage614": s614, "stage615": s615}
    except Exception as e:
        import traceback
        print(f"  ERROR in {mk}: {e}")
        traceback.print_exc()
        result = {"error": str(e)}
    finally:
        free_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(3)
    return result


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target not in MODEL_KEYS:
            print(f"Unknown model: {target}. Use one of: {MODEL_KEYS}")
            return
        models_to_run = [target]
    else:
        models_to_run = MODEL_KEYS

    combined_path = OUTPUT_DIR / f"stage612_613_614_615_combined_{TIMESTAMP}.json"
    combined = {"timestamp": TIMESTAMP, "models": {}}

    existing_files = sorted(OUTPUT_DIR.glob("stage612_613_614_615_combined_*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if existing_files and len(sys.argv) == 1:
        try:
            with open(existing_files[0], "r", encoding="utf-8") as f:
                prev = json.load(f)
            combined = prev
            combined_path = existing_files[0]
            print(f"Resuming from {existing_files[0].name}")
        except:
            pass

    for mk in models_to_run:
        if mk in combined["models"] and "error" not in combined["models"][mk]:
            print(f"\n  Skipping {mk} (already completed)")
            continue

        result = run_single_model(mk)
        combined["models"][mk] = result

        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {combined_path}")

    print(f"\nAll done. Results: {combined_path}")


if __name__ == "__main__":
    main()
