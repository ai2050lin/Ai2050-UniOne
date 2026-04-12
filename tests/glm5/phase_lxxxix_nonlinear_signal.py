"""
Phase LXXXIX: 非线性信号传播理论 + W_U解码器分析
================================================================================

核心问题: P434发现信号从L1开始方向被完全散射(cos≈0), 但模型仍正确输出
→ 散射不是"信息丢失", 而是"信息重编码"
→ W_U扮演"解码器"角色, 将散射信号解码为目标词

实验设计:
P435: 信号散射结构分析——散射是随机的还是有结构的?
  - 每层散射后的信号方向 vs W_U目标行的余弦相似度
  - 散射方向在V_lang子空间中的投影比例
  - 核心问题: 散射是否"瞄准"W_U的解码空间?

P436: W_U解码器分析——W_U如何解码散射的信号?
  - W_U的行空间(row space)结构: SVD谱, PR
  - 每层hidden state在W_U行空间中的投影比例
  - 核心问题: W_U的解码能力有多强?

P437: 非线性信号传播数学模型
  - 逐层雅可比矩阵的近似: 用N个随机方向估计
  - 信号传播模型: dh_l ≈ R_l × dh_{l-1} (旋转+散射)
  - R_l的特征值分布: 是否有方向选择性?

P438: 信息重编码量化
  - 每层信号在V_lang中的能量比例
  - 信号"进出V_lang"的动态过程
  - 核心问题: 信息在传播中是否在V_lang内外"振荡"?

模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True, "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True, "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True, "use_fast": False,
    },
}

PROMPTS = [
    "The apple is", "The scientist explained that",
    "If all humans are mortal and Socrates is human, then",
    "She felt deeply moved by", "In the future, people will",
    "The government announced that", "Research shows that the brain",
    "The evidence suggests that the conclusion",
    "She looked at the sky and", "Throughout history, civilizations have",
]

DIM_PAIRS = {
    "style": ("formal", "informal"),
    "logic": ("true", "false"),
    "sentiment": ("happy", "sad"),
    "tense": ("was", "is"),
    "certainty": ("definitely", "maybe"),
    "quantity": ("many", "few"),
    "complexity": ("complex", "simple"),
    "formality": ("professional", "casual"),
    "size": ("large", "small"),
    "strength": ("strong", "weak"),
}


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"Loading {model_name}...")
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu",
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = mdl.device
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, use_fast=cfg["use_fast"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok, device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


def get_dimension_direction(model, tokenizer, w1, w2):
    embed = model.get_input_embeddings()
    t1 = tokenizer.encode(w1, add_special_tokens=False)
    t2 = tokenizer.encode(w2, add_special_tokens=False)
    if not t1 or not t2:
        return np.zeros(embed.weight.shape[1]), 0
    e1 = embed.weight[t1[0]].detach().cpu().float().numpy()
    e2 = embed.weight[t2[0]].detach().cpu().float().numpy()
    d = e1 - e2
    return d / (np.linalg.norm(d) + 1e-10), np.linalg.norm(d)


# ========== P435: 信号散射结构分析 ==========

def run_p435(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P435: Signal Scattering Structure Analysis - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab_size, d_model]

    beta = 1.0  # 小信号

    prompts = PROMPTS[:10]
    test_dims = ["logic", "sentiment", "style"]

    results = {"model": model_name, "exp": "p435", "n_layers": n_layers, "d_model": d_model}

    # ---- 1. W_U的SVD分析 (只做一次) ----
    print("  Computing W_U SVD (this may take a moment)...")
    # W_U可能很大, 使用scipy.sparse.linalg.svds只取前500个奇异值
    from scipy.sparse.linalg import svds
    n_svd = min(500, min(W_U.shape) - 1)
    try:
        # svds要求k < min(m,n)
        k_svd = min(n_svd, min(W_U.shape) - 1)
        U_svd, S_wu, Vt_svd = svds(W_U.astype(np.float32), k=k_svd)
        # svds返回的是升序, 需要反转
        S_wu = S_wu[::-1]
        U_svd = U_svd[:, ::-1]
        Vt_svd = Vt_svd[::-1, :]
    except Exception as e:
        print(f"  svds failed ({e}), falling back to truncated SVD")
        # fallback: 用W_U @ W_U.T的前k个特征值
        WtW = (W_U[:1000].astype(np.float32)) @ W_U[:1000].astype(np.float32).T
        evals = np.linalg.eigvalsh(WtW)
        S_wu = np.sqrt(np.maximum(evals, 0))[::-1][:n_svd]
        Vt_svd = None
        U_svd = None
    # Participation Ratio
    PR_wu = (S_wu.sum() ** 2) / (S_wu ** 2).sum()
    results["W_U_svd_top10"] = [round(float(x), 4) for x in S_wu[:10]]
    results["W_U_PR"] = round(float(PR_wu), 4)
    results["W_U_svd_top500_sum_ratio"] = round(float(S_wu.sum() / S_wu.sum()), 4)  # trivially 1.0 since we only have partial

    # W_U的前k个右奇异向量 (解码空间的基)
    # 由于Vt太大, 我们用更高效的方法
    # 只需要Vt的前PR维 (约130维 for Qwen3)
    n_basis = min(int(PR_wu) * 2, 300, d_model)
    # 使用之前计算的svds结果
    if Vt_svd is not None:
        n_basis = min(int(PR_wu) * 2, 300, d_model, Vt_svd.shape[0])
        decode_basis = Vt_svd[:n_basis, :]  # [n_basis, d_model]
    else:
        # fallback: 用随机正交基
        n_basis = min(100, d_model)
        decode_basis = np.random.randn(n_basis, d_model).astype(np.float32)
        decode_basis, _ = np.linalg.qr(decode_basis.T)
        decode_basis = decode_basis.T[:n_basis]
    results["decode_basis_dim"] = n_basis

    print(f"  W_U PR={PR_wu:.1f}, decode_basis_dim={n_basis}")

    # ---- 2. 信号散射方向 vs W_U解码空间 ----
    # 对每个prompt和维度, 注入信号, 收集每层的delta_h
    # 然后检查: delta_h在decode_basis中的投影比例

    all_layer_scatter = {str(l): [] for l in range(n_layers)}

    for dim_name in test_dims:
        w1, w2 = DIM_PAIRS[dim_name]
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)

        # 获取目标词的W_U行向量
        t1 = tokenizer.encode(w1, add_special_tokens=False)
        t2 = tokenizer.encode(w2, add_special_tokens=False)
        target_row_w1 = W_U[t1[0]] if t1 else None  # d_model维
        target_row_w2 = W_U[t2[0]] if t2 else None

        for pi, prompt in enumerate(prompts):
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            # 收集baseline和intervened每层hidden state
            baseline_hs = {}
            intervened_hs = {}

            def make_hs_hook(store, layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    store[layer_idx] = x[0, -1, :].detach().cpu().float()
                    return args
                return hook_fn

            # Baseline
            hooks_b = []
            for l in range(n_layers):
                hooks_b.append(layers[l].register_forward_pre_hook(make_hs_hook(baseline_hs, l)))
            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks_b:
                h.remove()

            # Intervened
            hooks_i = []
            for l in range(n_layers):
                hooks_i.append(layers[l].register_forward_pre_hook(make_hs_hook(intervened_hs, l)))

            def inj_hook_fn(module, input, output, d=direction_t):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(inj_hook_fn)
            with torch.no_grad():
                intervened_out = model(input_ids)
            h_embed.remove()
            for h in hooks_i:
                h.remove()

            # 获取intervened logits的目标词差
            il = intervened_out.logits[0, -1].detach().cpu().float()
            with torch.no_grad():
                bl_logits = model(input_ids).logits[0, -1].detach().cpu().float()

            # ---- 逐层分析 ----
            for l in range(n_layers):
                if l not in baseline_hs or l not in intervened_hs:
                    continue
                delta_h = intervened_hs[l] - baseline_hs[l]  # d_model维
                delta_np = delta_h.numpy()

                # 1. 信号在W_U解码空间中的投影比例
                proj_decode = decode_basis @ delta_np  # [n_basis]
                energy_in_decode = float(np.sum(proj_decode ** 2))
                energy_total = float(np.sum(delta_np ** 2))
                decode_ratio = energy_in_decode / energy_total if energy_total > 1e-20 else 0

                # 2. 信号方向 vs 目标词W_U行的余弦
                cos_w1 = float(np.dot(delta_np, target_row_w1) / (np.linalg.norm(delta_np) * np.linalg.norm(target_row_w1) + 1e-10)) if target_row_w1 is not None else 0
                cos_w2 = float(np.dot(delta_np, target_row_w2) / (np.linalg.norm(delta_np) * np.linalg.norm(target_row_w2) + 1e-10)) if target_row_w2 is not None else 0

                # 3. 信号幅度
                delta_norm = float(np.linalg.norm(delta_np))

                # 4. 信号在注入方向上的投影 (方向保持度)
                cos_inject = float(np.dot(delta_np, direction) / (delta_norm * np.linalg.norm(direction) + 1e-10))

                all_layer_scatter[str(l)].append({
                    "dim": dim_name,
                    "prompt_idx": pi,
                    "delta_norm": round(delta_norm, 6),
                    "decode_ratio": round(decode_ratio, 6),
                    "cos_target_w1": round(cos_w1, 6),
                    "cos_target_w2": round(cos_w2, 6),
                    "cos_inject": round(cos_inject, 6),
                })

            # 释放GPU
            del baseline_hs, intervened_hs

    # 汇总
    layer_summary = {}
    for l in range(n_layers):
        samples = all_layer_scatter[str(l)]
        if not samples:
            continue
        avg_decode_ratio = np.mean([s["decode_ratio"] for s in samples])
        avg_cos_w1 = np.mean([s["cos_target_w1"] for s in samples])
        avg_cos_w2 = np.mean([s["cos_target_w2"] for s in samples])
        avg_cos_inject = np.mean([s["cos_inject"] for s in samples])
        avg_delta_norm = np.mean([s["delta_norm"] for s in samples])

        layer_summary[str(l)] = {
            "avg_decode_ratio": round(avg_decode_ratio, 4),
            "avg_cos_target": round((avg_cos_w1 + avg_cos_w2) / 2, 4),
            "avg_cos_inject": round(avg_cos_inject, 4),
            "avg_delta_norm": round(avg_delta_norm, 4),
            "n_samples": len(samples),
        }

    results["layer_summary"] = layer_summary

    # 打印关键结果
    print(f"\n  === P435 Key Results for {model_name} ===")
    print(f"  W_U PR = {PR_wu:.1f}, decode_basis_dim = {n_basis}")
    print(f"  Layer | decode_ratio | cos_target | cos_inject | delta_norm")
    for l in [0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]:
        ls = layer_summary.get(str(l), {})
        if not ls:
            continue
        print(f"  L{l:3d} | {ls.get('avg_decode_ratio',0):12.4f} | {ls.get('avg_cos_target',0):10.4f} | {ls.get('avg_cos_inject',0):10.4f} | {ls.get('avg_delta_norm',0):10.4f}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxix_p435_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P436: W_U解码器分析 ==========

def run_p436(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P436: W_U Decoder Analysis - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    W_U = model.lm_head.weight.detach().cpu().float().numpy()

    results = {"model": model_name, "exp": "p436", "n_layers": n_layers, "d_model": d_model}

    # ---- 1. W_U全面分析 ----
    print("  Full W_U analysis...")
    from scipy.sparse.linalg import svds
    # 只计算前k个奇异值
    k_svd = min(500, min(W_U.shape) - 1)
    try:
        U_wu, S_wu_full, Vt_wu = svds(W_U.astype(np.float32), k=k_svd)
        S_wu_full = S_wu_full[::-1]
        U_wu = U_wu[:, ::-1]
        Vt_wu = Vt_wu[::-1, :]
    except Exception as e:
        print(f"  svds failed: {e}, using truncated approach")
        # 使用W_U的子矩阵近似
        sub = W_U[:5000].astype(np.float32)
        U_sub, S_wu_full, Vt_wu = np.linalg.svd(sub, full_matrices=False)
        U_wu = U_sub
    vocab_size = W_U.shape[0]
    S_wu = S_wu_full  # 只取了前k个

    # PR (基于已知的奇异值)
    PR = (S_wu.sum() ** 2) / (S_wu ** 2).sum()
    results["W_U_shape"] = list(W_U.shape)
    results["W_U_PR"] = round(float(PR), 4)
    results["W_U_svd_top20"] = [round(float(x), 4) for x in S_wu[:20]]
    results["W_U_svd_computed_k"] = len(S_wu)

    # 谱分布统计
    results["W_U_spectral_stats"] = {
        "s1": round(float(S_wu[0]), 4),
        "s10": round(float(S_wu[9]), 4),
        "s100": round(float(S_wu[99]), 4) if len(S_wu) > 99 else None,
        "s500": round(float(S_wu[min(499, len(S_wu)-1)]), 4),
        "s_tail_computed": round(float(S_wu[-1]), 6),
        "energy_top10_pct": round(float((S_wu[:10]**2).sum() / (S_wu**2).sum()), 4),
        "energy_top100_pct": round(float((S_wu[:100]**2).sum() / (S_wu**2).sum()), 4),
        "energy_top500_pct": round(float((S_wu[:min(500, len(S_wu))]**2).sum() / (S_wu**2).sum()), 4),
    }

    # ---- 2. 每层hidden state在W_U解码空间中的投影 ----
    # decode_basis = Vt_wu的前PR行
    n_basis = min(int(PR) * 3, 500, d_model)
    decode_basis = Vt_wu[:n_basis, :]  # [n_basis, d_model]

    prompts = PROMPTS[:10]

    layer_projection = {str(l): [] for l in range(n_layers)}

    for pi, prompt in enumerate(prompts):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids

        # 收集每层hidden state
        hs_dict = {}

        def make_hs_hook(store, layer_idx):
            def hook_fn(module, args):
                x = args[0] if isinstance(args, tuple) else args
                store[layer_idx] = x[0, -1, :].detach().cpu().float().numpy()
                return args
            return hook_fn

        hooks = []
        for l in range(n_layers):
            hooks.append(layers[l].register_forward_pre_hook(make_hs_hook(hs_dict, l)))

        with torch.no_grad():
            _ = model(input_ids)
        for h in hooks:
            h.remove()

        # 每层: hidden state在decode_basis中的投影比例
        for l in range(n_layers):
            if l not in hs_dict:
                continue
            h = hs_dict[l]
            proj = decode_basis @ h  # [n_basis]
            energy_in_decode = float(np.sum(proj ** 2))
            energy_total = float(np.sum(h ** 2))
            proj_ratio = energy_in_decode / energy_total if energy_total > 1e-20 else 0

            # 前10个分量(最主方向)的能量
            energy_top10 = float(np.sum(proj[:10] ** 2)) / energy_total if energy_total > 1e-20 else 0
            energy_top100 = float(np.sum(proj[:min(100, n_basis)] ** 2)) / energy_total if energy_total > 1e-20 else 0

            layer_projection[str(l)].append({
                "decode_proj_ratio": round(proj_ratio, 6),
                "top10_ratio": round(energy_top10, 6),
                "top100_ratio": round(energy_top100, 6),
            })

        del hs_dict

    # 汇总
    layer_summary = {}
    for l in range(n_layers):
        samples = layer_projection[str(l)]
        if not samples:
            continue
        layer_summary[str(l)] = {
            "decode_proj_ratio": round(float(np.mean([s["decode_proj_ratio"] for s in samples])), 4),
            "top10_ratio": round(float(np.mean([s["top10_ratio"] for s in samples])), 4),
            "top100_ratio": round(float(np.mean([s["top100_ratio"] for s in samples])), 4),
            "n_samples": len(samples),
        }

    results["layer_projection"] = layer_summary

    # ---- 3. W_U行向量的聚类结构 ----
    # 用SVD的前k个分量, 看词汇在decode空间中的分布
    # 取1000个高频词, 在前3个PC上的分布
    n_sample_words = min(1000, vocab_size, U_wu.shape[0])
    if n_sample_words >= 3:
        coords = U_wu[:n_sample_words, :3] * S_wu[:3]  # [n_sample_words, 3]
        results["word_coord_stats"] = {
            "PC0_range": [round(float(coords[:, 0].min()), 4), round(float(coords[:, 0].max()), 4)],
            "PC1_range": [round(float(coords[:, 1].min()), 4), round(float(coords[:, 1].max()), 4)],
            "PC2_range": [round(float(coords[:, 2].min()), 4), round(float(coords[:, 2].max()), 4)],
            "PC0_std": round(float(coords[:, 0].std()), 4),
            "PC1_std": round(float(coords[:, 1].std()), 4),
            "PC2_std": round(float(coords[:, 2].std()), 4),
        }

    # 打印关键结果
    print(f"\n  === P436 Key Results for {model_name} ===")
    print(f"  W_U shape: {W_U.shape}, PR: {PR:.1f}")
    s100_str = f"{S_wu[99]:.4f}" if len(S_wu) > 99 else "N/A"
    print(f"  S1={S_wu[0]:.2f}, S10={S_wu[9]:.2f}, S100={s100_str}, S_tail={S_wu[-1]:.6f}")
    print(f"  Energy: top10={results['W_U_spectral_stats']['energy_top10_pct']:.2%}, top100={results['W_U_spectral_stats']['energy_top100_pct']:.2%}")
    print(f"  Layer | decode_proj | top10 | top100")
    for l in [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        ls = layer_summary.get(str(l), {})
        if not ls:
            continue
        print(f"  L{l:3d} | {ls.get('decode_proj_ratio',0):11.4f} | {ls.get('top10_ratio',0):5.4f} | {ls.get('top100_ratio',0):6.4f}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxix_p436_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P437: 非线性信号传播数学模型 ==========

def run_p437(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P437: Nonlinear Signal Propagation Model - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]

    results = {"model": model_name, "exp": "p437", "n_layers": n_layers, "d_model": d_model}

    # ---- 核心方法: 逐层雅可比矩阵的有限差分近似 ----
    # 对N个随机方向d_i, 计算 J_l @ d_i = (f_l(x + eps*d_i) - f_l(x - eps*d_i)) / (2*eps)
    # 然后从N个输出中近似J_l的奇异值分布

    n_directions = 50  # 随机方向数
    eps = 0.01  # 微小扰动

    prompts = PROMPTS[:3]  # 少量prompt, 因为每个方向需要2次forward
    sample_layers = [0, 1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    sample_layers = sorted(set(sample_layers))

    layer_jacobian = {}

    for pi, prompt in enumerate(prompts):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids

        # 收集baseline每层hidden state
        baseline_hs = {}

        def make_hs_hook(store, layer_idx):
            def hook_fn(module, args):
                x = args[0] if isinstance(args, tuple) else args
                store[layer_idx] = x[0, -1, :].detach().cpu().float()
                return args
            return hook_fn

        hooks = []
        for l in sample_layers:
            hooks.append(layers[l].register_forward_pre_hook(make_hs_hook(baseline_hs, l)))

        with torch.no_grad():
            _ = model(input_ids)
        for h in hooks:
            h.remove()

        # 对每个随机方向, 计算+eps和-eps的输出
        jacobian_outputs = {str(l): [] for l in sample_layers}

        for di in range(n_directions):
            rand_dir = torch.randn(d_model, dtype=torch.float32)
            rand_dir = rand_dir / rand_dir.norm() * eps

            # +eps
            plus_hs = {}
            hooks_p = []
            for l in sample_layers:
                hooks_p.append(layers[l].register_forward_pre_hook(make_hs_hook(plus_hs, l)))

            def plus_hook(module, input, output, d=rand_dir):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(plus_hook)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_p:
                h.remove()

            # -eps
            minus_hs = {}
            hooks_m = []
            for l in sample_layers:
                hooks_m.append(layers[l].register_forward_pre_hook(make_hs_hook(minus_hs, l)))

            def minus_hook(module, input, output, d=rand_dir):
                modified = output.clone()
                modified[0, -1, :] -= d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(minus_hook)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_m:
                h.remove()

            # 计算 J @ d = (plus - minus) / (2*eps)
            for l in sample_layers:
                if l in plus_hs and l in minus_hs and l in baseline_hs:
                    jd = (plus_hs[l] - minus_hs[l]) / (2 * eps)  # d_model维
                    jacobian_outputs[str(l)].append(jd.numpy())

            del plus_hs, minus_hs

        # 从J@d的集合中近似J的奇异值分布
        for l in sample_layers:
            jd_list = jacobian_outputs[str(l)]
            if len(jd_list) < 5:
                continue

            # 构造 [n_directions, d_model] 矩阵
            J_approx = np.array(jd_list)  # [n_dirs, d_model]

            # SVD of J_approx: 这给出J的右奇异向量近似
            # 但J_approx = J @ D, 其中D是[n_dirs, d_model]的随机方向矩阵
            # 所以J_approx的奇异值分布近似J的奇异值分布(但维度受限)

            # 更简单: 计算 ||J@d|| / ||d|| = ||J@d|| / eps 的分布
            gains = np.array([np.linalg.norm(jd) for jd in jd_list])

            # 方向选择性: gains的变异系数
            mean_gain = float(gains.mean())
            std_gain = float(gains.std())
            cv_gain = std_gain / mean_gain if mean_gain > 1e-10 else 0

            # 最大/最小增益比 → 各向异性程度
            max_gain = float(gains.max())
            min_gain = float(gains.min())
            anisotropy = max_gain / min_gain if min_gain > 1e-10 else float('inf')

            # J@d的方向分布: 余弦相似度矩阵
            cos_matrix = np.zeros((min(10, len(jd_list)), min(10, len(jd_list))))
            for i in range(min(10, len(jd_list))):
                for j in range(min(10, len(jd_list))):
                    ni = np.linalg.norm(jd_list[i])
                    nj = np.linalg.norm(jd_list[j])
                    if ni > 1e-10 and nj > 1e-10:
                        cos_matrix[i, j] = float(np.dot(jd_list[i], jd_list[j]) / (ni * nj))

            # 平均off-diagonal cos → 方向散射度
            off_diag = cos_matrix[np.triu_indices(min(10, len(jd_list)), k=1)]
            avg_off_cos = float(off_diag.mean()) if len(off_diag) > 0 else 0

            key = str(l)
            if key not in layer_jacobian:
                layer_jacobian[key] = {
                    "gains": [], "cv_gains": [], "anisotropies": [],
                    "avg_off_cos": [], "mean_gains": [], "max_gains": [], "min_gains": [],
                }
            layer_jacobian[key]["gains"].extend([round(float(g), 4) for g in gains])
            layer_jacobian[key]["cv_gains"].append(round(cv_gain, 4))
            layer_jacobian[key]["anisotropies"].append(round(anisotropy, 4))
            layer_jacobian[key]["avg_off_cos"].append(round(avg_off_cos, 4))
            layer_jacobian[key]["mean_gains"].append(round(mean_gain, 4))
            layer_jacobian[key]["max_gains"].append(round(max_gain, 4))
            layer_jacobian[key]["min_gains"].append(round(min_gain, 4))

    # 汇总
    jacobian_summary = {}
    for l in sample_layers:
        lj = layer_jacobian.get(str(l), {})
        if not lj:
            continue
        jacobian_summary[str(l)] = {
            "mean_gain": round(float(np.mean(lj["mean_gains"])), 4),
            "max_gain": round(float(np.mean(lj["max_gains"])), 4),
            "min_gain": round(float(np.mean(lj["min_gains"])), 4),
            "cv_gain": round(float(np.mean(lj["cv_gains"])), 4),
            "anisotropy": round(float(np.mean(lj["anisotropies"])), 4),
            "avg_off_cos": round(float(np.mean(lj["avg_off_cos"])), 4),
            "n_prompts": len(lj["mean_gains"]),
        }

    results["jacobian_summary"] = jacobian_summary

    # 打印关键结果
    print(f"\n  === P437 Key Results for {model_name} ===")
    print(f"  Layer | mean_gain | max_gain | min_gain | anisotropy | off_cos | cv_gain")
    for l in sample_layers:
        js = jacobian_summary.get(str(l), {})
        if not js:
            continue
        print(f"  L{l:3d} | {js.get('mean_gain',0):9.2f} | {js.get('max_gain',0):8.2f} | {js.get('min_gain',0):8.2f} | {js.get('anisotropy',0):10.2f} | {js.get('avg_off_cos',0):7.4f} | {js.get('cv_gain',0):7.4f}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxix_p437_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P438: 信息重编码量化 ==========

def run_p438(model, tokenizer, device, model_name, p435_results, p436_results):
    print(f"\n{'='*60}")
    print(f"P438: Information Recoding Quantification - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]

    results = {"model": model_name, "exp": "p438", "n_layers": n_layers, "d_model": d_model}

    # ---- 1. 构造V_lang子空间 ----
    # 用DIM_PAIRS中的所有方向向量构造正交基
    print("  Building V_lang subspace...")
    all_dirs = []
    functional_names = []
    for dim_name, (w1, w2) in DIM_PAIRS.items():
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        # 测试这个方向是否有效(目标词logit差>0.5)
        direction_t = torch.tensor(direction * 8.0, dtype=torch.float32)
        prompt = PROMPTS[0]
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids

        with torch.no_grad():
            baseline_logits = model(input_ids).logits[0, -1].detach().cpu().float()

        def inj_hook_fn(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified

        h_embed = embed.register_forward_hook(inj_hook_fn)
        with torch.no_grad():
            intervened_logits = model(input_ids).logits[0, -1].detach().cpu().float()
        h_embed.remove()

        t1 = tokenizer.encode(w1, add_special_tokens=False)
        t2 = tokenizer.encode(w2, add_special_tokens=False)
        target_dlogit = 0
        if t1 and t2:
            target_dlogit = abs((intervened_logits - baseline_logits)[t1[0]].item() - (intervened_logits - baseline_logits)[t2[0]].item())

        if target_dlogit > 0.5:
            all_dirs.append(direction)
            functional_names.append(dim_name)

    # QR分解构造正交基
    if len(all_dirs) > 0:
        D = np.array(all_dirs)  # [n_dirs, d_model]
        Q, R = np.linalg.qr(D.T)  # Q: [d_model, n_dirs], 正交基
        vlang_basis = Q[:, :min(len(all_dirs), d_model)].T  # [n_basis, d_model]
        n_vlang = vlang_basis.shape[0]
    else:
        vlang_basis = np.eye(d_model)[:1]
        n_vlang = 0

    results["n_functional_dims"] = len(all_dirs)
    results["n_vlang_basis"] = n_vlang
    results["functional_names"] = functional_names
    print(f"  V_lang: {len(all_dirs)} functional dims, {n_vlang} basis vectors")

    # ---- 2. 每层hidden state在V_lang中的投影 ----
    # 以及信号(delta_h)在V_lang中的投影

    prompts = PROMPTS[:10]
    test_dims = ["logic", "sentiment", "style"]
    beta = 1.0

    layer_vlang_data = {str(l): {"hs_proj": [], "delta_proj": []} for l in range(n_layers)}

    for dim_name in test_dims:
        w1, w2 = DIM_PAIRS[dim_name]
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)

        for pi, prompt in enumerate(prompts):
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            # Baseline
            baseline_hs = {}

            def make_hs_hook(store, layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    store[layer_idx] = x[0, -1, :].detach().cpu().float().numpy()
                    return args
                return hook_fn

            hooks_b = []
            for l in range(n_layers):
                hooks_b.append(layers[l].register_forward_pre_hook(make_hs_hook(baseline_hs, l)))
            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks_b:
                h.remove()

            # Intervened
            intervened_hs = {}
            hooks_i = []
            for l in range(n_layers):
                hooks_i.append(layers[l].register_forward_pre_hook(make_hs_hook(intervened_hs, l)))

            def inj_hook_fn2(module, input, output, d=direction_t):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(inj_hook_fn2)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_i:
                h.remove()

            # 逐层分析
            for l in range(n_layers):
                if l not in baseline_hs or l not in intervened_hs:
                    continue

                h = baseline_hs[l]
                delta_h = intervened_hs[l] - baseline_hs[l]

                # hidden state在V_lang中的投影比例
                proj_h = vlang_basis @ h
                energy_h_vlang = float(np.sum(proj_h ** 2))
                energy_h_total = float(np.sum(h ** 2))
                hs_proj_ratio = energy_h_vlang / energy_h_total if energy_h_total > 1e-20 else 0

                # delta_h在V_lang中的投影比例
                proj_d = vlang_basis @ delta_h
                energy_d_vlang = float(np.sum(proj_d ** 2))
                energy_d_total = float(np.sum(delta_h ** 2))
                delta_proj_ratio = energy_d_vlang / energy_d_total if energy_d_total > 1e-20 else 0

                layer_vlang_data[str(l)]["hs_proj"].append(hs_proj_ratio)
                layer_vlang_data[str(l)]["delta_proj"].append(delta_proj_ratio)

            del baseline_hs, intervened_hs

    # 汇总
    vlang_summary = {}
    for l in range(n_layers):
        ld = layer_vlang_data[str(l)]
        if ld["hs_proj"]:
            vlang_summary[str(l)] = {
                "hs_in_vlang_ratio": round(float(np.mean(ld["hs_proj"])), 4),
                "delta_in_vlang_ratio": round(float(np.mean(ld["delta_proj"])), 4),
                "hs_vlang_std": round(float(np.std(ld["hs_proj"])), 4),
                "delta_vlang_std": round(float(np.std(ld["delta_proj"])), 4),
                "n_samples": len(ld["hs_proj"]),
            }

    results["vlang_summary"] = vlang_summary

    # ---- 3. V_lang维度的逐层投影能量分布 ----
    # 对L0和L_last, 看每个V_lang基向量上的投影能量
    # 这能揭示: 信号的哪些维度被保留, 哪些被丢弃

    for target_l in [0, n_layers//2, n_layers-1]:
        prompt = PROMPTS[0]
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids

        hs_dict = {}
        hooks = []
        for l in [target_l]:
            hooks.append(layers[l].register_forward_pre_hook(make_hs_hook(hs_dict, l)))
        with torch.no_grad():
            _ = model(input_ids)
        for h in hooks:
            h.remove()

        if target_l in hs_dict:
            h = hs_dict[target_l]
            proj = vlang_basis @ h  # [n_vlang]
            # Top-10 V_lang分量的能量
            energies = proj ** 2
            top10_idx = np.argsort(energies)[-10:][::-1]
            results[f"vlang_energy_L{target_l}"] = {
                "top10_indices": [int(x) for x in top10_idx],
                "top10_names": [functional_names[i] if i < len(functional_names) else f"dim_{i}" for i in top10_idx],
                "top10_energies": [round(float(energies[i]), 6) for i in top10_idx],
                "total_energy_in_vlang": round(float(energies.sum()), 4),
                "total_energy": round(float(np.sum(h ** 2)), 4),
            }

    # 打印关键结果
    print(f"\n  === P438 Key Results for {model_name} ===")
    print(f"  V_lang: {n_vlang} basis vectors from {len(all_dirs)} functional dims")
    print(f"  Layer | hs_in_vlang | delta_in_vlang | n_samples")
    for l in [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        vs = vlang_summary.get(str(l), {})
        if not vs:
            continue
        print(f"  L{l:3d} | {vs.get('hs_in_vlang_ratio',0):11.4f} | {vs.get('delta_in_vlang_ratio',0):14.4f} | {vs.get('n_samples',0)}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxix_p438_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== 主程序 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, required=True, choices=["p435", "p436", "p437", "p438", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    def load_latest_result(exp_name, model_name):
        import glob
        pattern = str(OUT_DIR / f"phase_lxxxix_{exp_name}_{model_name}_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]
            print(f"  Loading {exp_name} results from {latest}")
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    if args.exp == "p435" or args.exp == "all":
        p435_results = run_p435(model, tokenizer, device, args.model)
    else:
        p435_results = load_latest_result("p435", args.model)

    if args.exp == "p436" or args.exp == "all":
        p436_results = run_p436(model, tokenizer, device, args.model)
    else:
        p436_results = load_latest_result("p436", args.model)

    if args.exp == "p437" or args.exp == "all":
        p437_results = run_p437(model, tokenizer, device, args.model)
    else:
        p437_results = {}

    if args.exp == "p438" or args.exp == "all":
        p438_results = run_p438(model, tokenizer, device, args.model, p435_results, p436_results)
    else:
        p438_results = {}

    # 释放GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nDone! GPU memory freed.")


if __name__ == "__main__":
    main()
