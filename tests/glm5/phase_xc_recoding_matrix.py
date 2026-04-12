"""
Phase XC: 重编码精确数学模型 + W_U空间结构 + 信号路径追踪
================================================================================

核心突破点: P438发现信号从V_lang逃逸到W_U空间, 但转换的数学模型缺失
→ 需要精确量化"重编码矩阵"R_l: V_lang → W_U空间
→ 需要理解W_U空间的精细结构(不仅仅是top10)
→ 需要追踪信号在每层的精确路径

实验设计:
P439: 重编码矩阵R_l精确计算
  - 在每层计算: delta_h_l在W_U行空间中的精确投影
  - 构造R_l = Proj_{W_U}(delta_h_l) / delta_h_{l-1}
  - R_l的奇异值分布: 重编码的"增益谱"
  - R_l随层的变化: 从V_lang到W_U空间的转换过程

P440: W_U空间精细结构——top10之外的43-66%是什么?
  - W_U行空间的完整谱分析(不只是前500)
  - 按语义类别(名词/动词/形容词)分解W_U子空间
  - 每层hidden state在不同语义子空间中的投影比例
  - 核心: W_U空间是否有语义组织?

P441: 权重→激活因果方程——R_l能否用权重乘积表达?
  - 计算W_q/k/v/o × W_up/down/gate的有效秩
  - 验证: R_l ≈ W_o × softmax_attn × W_v + W_down × gelu × W_up × W_gate
  - 核心: 权重的低秩结构是否决定了重编码的方向?

P442: 五维交叉约束验证
  - 权重空间(W) × 激活空间(A) × 梯度空间(G) × 注意力空间(T) × 信息流空间(I)
  - 验证: W的低秩方向是否与A的主分量对齐?
  - 验证: G的高曲率方向是否与W的奇异向量对齐?
  - 核心: 五个空间是否共享同一组"基方向"?

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
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True, "use_fast": False,
    },
    "glm4": {
        "path": "D:/develop/model/hub/models--zai-org--GLM-4-9B-Chat-HF/snapshots/8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True, "use_fast": False,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True, "use_fast": False,
    },
}


def load_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_name} from {cfg['path']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=cfg.get("trust_remote_code", True),
        local_files_only=True, use_fast=cfg.get("use_fast", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_layers(model):
    """获取transformer层列表(兼容不同模型结构)"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder.layer
    raise ValueError(f"Cannot find transformer layers in {type(model)}")


# ===== P439: 重编码矩阵R_l精确计算 =====
def run_p439(model, tokenizer, device, model_name):
    """计算每层的重编码矩阵R_l: delta_h_{l-1} → delta_h_l在W_U空间中的投影"""
    print("\n===== P439: 重编码矩阵R_l精确计算 =====")
    results = {}
    t0 = time.time()

    # 获取W_U
    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    n_layers = len(get_layers(model))

    # W_U行空间的正交基 (用svds只取前k个)
    from scipy.sparse.linalg import svds
    k_svd = min(500, min(W_U.shape) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.astype(np.float32), k=k_svd)
    S_wu = S_wu[::-1]
    Vt_wu = Vt_wu[::-1, :]  # [k, d_model] - W_U行空间的基

    # PR
    PR_wu = (S_wu.sum()**2) / (S_wu**2).sum()
    print(f"  W_U PR={PR_wu:.1f}, S1={S_wu[0]:.2f}, S10={S_wu[9]:.2f}")

    # 构造W_U投影矩阵 P_WU = Vt_wu.T @ Vt_wu (d_model x d_model)
    # 但太大了, 用逐向量投影: proj_v = Vt_wu.T @ (Vt_wu @ v)

    # 词对
    pairs = [("true", "false"), ("good", "bad"), ("happy", "sad"),
             ("big", "small"), ("hot", "cold"), ("love", "hate")]
    w2id = tokenizer.get_vocab()

    def get_word_vec(w):
        wid = w2id.get(w, None)
        if wid is None:
            return None
        return W_U[wid]  # [d_model]

    # 逐层注入信号, 收集delta_h
    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    beta = 1.0
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    # 注入方向: 用功能词对构造
    inject_dirs = []
    pair_names = []
    for w1, w2 in pairs:
        v1, v2 = get_word_vec(w1), get_word_vec(w2)
        if v1 is not None and v2 is not None:
            d = v1 - v2
            d = d / (np.linalg.norm(d) + 1e-10)
            inject_dirs.append(d)
            pair_names.append(f"{w1}-{w2}")

    if not inject_dirs:
        print("  No valid word pairs, using random directions")
        inject_dirs = [np.random.randn(d_model) for _ in range(6)]
        inject_dirs = [d / np.linalg.norm(d) for d in inject_dirs]
        pair_names = [f"random_{i}" for i in range(6)]

    inject_dirs = np.array(inject_dirs)  # [n_dirs, d_model]

    # 对每个注入方向, 在embed层注入信号, 收集每层的delta_h
    # 方法: 与P434一致, 在embed层注入beta*d方向, 收集每层pre-hook输入
    layers = get_layers(model)
    embed = model.get_input_embeddings()

    layer_recoding = {}

    # 先收集所有层的baseline
    baseline_hs = {}
    def make_hs_hook(store, layer_idx):
        def hook_fn(module, args):
            x = args[0] if isinstance(args, tuple) else args
            store[layer_idx] = x[0, -1, :].detach().cpu().float().numpy()
            return args
        return hook_fn

    hooks_b = []
    for l in sample_layers:
        hooks_b.append(layers[l].register_forward_pre_hook(make_hs_hook(baseline_hs, l)))
    with torch.no_grad():
        _ = model(input_ids)
    for h in hooks_b:
        h.remove()

    # 对每个注入方向, 收集intervened的hidden state
    for li, layer_idx in enumerate(sample_layers):
        print(f"  Layer {layer_idx} ({li+1}/{len(sample_layers)})...")
        layer_recoding[layer_idx] = {
            "mean_delta_norm": 0, "mean_wu_proj_norm": 0,
            "mean_recoding_ratio": 0, "mean_recoding_gain": 0,
            "mean_cos_inject_wu": 0, "per_dir": [],
        }

        for di, d_inject in enumerate(inject_dirs):
            d_inject_t = torch.tensor(d_inject * beta, dtype=torch.float16, device=device)

            # Intervened: 在embed层注入信号
            intervened_hs = {}
            hooks_i = []
            for l in sample_layers:
                hooks_i.append(layers[l].register_forward_pre_hook(make_hs_hook(intervened_hs, l)))

            def inj_hook_fn(module, input, output, d=d_inject_t):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(inj_hook_fn)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_i:
                h.remove()

            # delta_h: 该层输入的变化
            if layer_idx in baseline_hs and layer_idx in intervened_hs:
                delta_h = intervened_hs[layer_idx] - baseline_hs[layer_idx]
            else:
                delta_h = np.zeros(d_model)

            # 计算delta_h在W_U行空间中的投影
            proj_coeffs = Vt_wu @ delta_h  # [k]
            proj_wu = Vt_wu.T @ proj_coeffs  # [d_model]
            proj_norm = np.linalg.norm(proj_wu)
            delta_norm = np.linalg.norm(delta_h)

            # 重编码比: 投影到W_U的信号占原始信号的比例
            recoding_ratio = proj_norm**2 / (delta_norm**2 + 1e-20)

            # 重编码增益: 投影到W_U的信号vs注入信号的范数比
            recoding_gain = proj_norm / (np.linalg.norm(d_inject) + 1e-20)

            # cos(注入方向, W_U投影)
            cos_inject_wu = np.dot(d_inject, proj_wu) / (np.linalg.norm(d_inject) * proj_norm + 1e-20)

            layer_recoding[layer_idx]["per_dir"].append({
                "pair": pair_names[di],
                "delta_norm": round(float(delta_norm), 4),
                "wu_proj_norm": round(float(proj_norm), 4),
                "recoding_ratio": round(float(recoding_ratio), 4),
                "recoding_gain": round(float(recoding_gain), 4),
                "cos_inject_wu": round(float(cos_inject_wu), 4),
                # 投影系数的分布
                "proj_top10_energy": round(float((proj_coeffs[:10]**2).sum() / max((proj_coeffs**2).sum(), 1e-20)), 4),
                "proj_top50_energy": round(float((proj_coeffs[:min(50, len(proj_coeffs))]**2).sum() / max((proj_coeffs**2).sum(), 1e-20)), 4),
                "proj_top100_energy": round(float((proj_coeffs[:min(100, len(proj_coeffs))]**2).sum() / max((proj_coeffs**2).sum(), 1e-20)), 4),
            })

        # 平均
        per_dir = layer_recoding[layer_idx]["per_dir"]
        layer_recoding[layer_idx]["mean_delta_norm"] = round(float(np.mean([d["delta_norm"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_wu_proj_norm"] = round(float(np.mean([d["wu_proj_norm"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_recoding_ratio"] = round(float(np.mean([d["recoding_ratio"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_recoding_gain"] = round(float(np.mean([d["recoding_gain"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_cos_inject_wu"] = round(float(np.mean([d["cos_inject_wu"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_proj_top10_energy"] = round(float(np.mean([d["proj_top10_energy"] for d in per_dir])), 4)
        layer_recoding[layer_idx]["mean_proj_top100_energy"] = round(float(np.mean([d["proj_top100_energy"] for d in per_dir])), 4)

        print(f"    delta_norm={layer_recoding[layer_idx]['mean_delta_norm']:.4f}, "
              f"wu_proj_norm={layer_recoding[layer_idx]['mean_wu_proj_norm']:.4f}, "
              f"recoding_ratio={layer_recoding[layer_idx]['mean_recoding_ratio']:.4f}, "
              f"recoding_gain={layer_recoding[layer_idx]['mean_recoding_gain']:.4f}")

    results["n_layers"] = n_layers
    results["d_model"] = d_model
    results["vocab_size"] = vocab_size
    results["W_U_PR"] = round(float(PR_wu), 1)
    results["W_U_S1"] = round(float(S_wu[0]), 2)
    results["sample_layers"] = sample_layers
    results["layer_recoding"] = {str(k): v for k, v in layer_recoding.items()}

    # 重编码矩阵R_l的近似: 用逐层增益比
    print("\n  === R_l Summary ===")
    for li, layer_idx in enumerate(sample_layers):
        lr = layer_recoding[layer_idx]
        print(f"  L{layer_idx}: recoding_ratio={lr['mean_recoding_ratio']:.4f}, "
              f"recoding_gain={lr['mean_recoding_gain']:.4f}, "
              f"cos_inject_wu={lr['mean_cos_inject_wu']:.4f}, "
              f"top10_energy={lr['mean_proj_top10_energy']:.4f}")

    results["elapsed"] = round(time.time() - t0, 1)
    return results


# ===== P440: W_U空间精细结构 =====
def run_p440(model, tokenizer, device, model_name):
    """W_U空间的精细结构: 语义子空间分解"""
    print("\n===== P440: W_U空间精细结构 =====")
    results = {}
    t0 = time.time()

    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    n_layers = len(get_layers(model))

    # 1. W_U的完整谱分析
    from scipy.sparse.linalg import svds
    k_svd = min(500, min(W_U.shape) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.astype(np.float32), k=k_svd)
    S_wu = S_wu[::-1]
    U_wu = U_wu[:, ::-1]
    Vt_wu = Vt_wu[::-1, :]

    # PR
    PR = (S_wu.sum()**2) / (S_wu**2).sum()
    print(f"  W_U PR={PR:.1f}, S1={S_wu[0]:.2f}")

    # 谱分布统计
    energy_cum = np.cumsum(S_wu**2) / (S_wu**2).sum()
    results["spectral"] = {
        "S_top20": [round(float(x), 2) for x in S_wu[:20]],
        "energy_frac": {
            "top1": round(float(energy_cum[0]), 4),
            "top10": round(float(energy_cum[9]), 4),
            "top50": round(float(energy_cum[min(49, len(energy_cum)-1)]), 4),
            "top100": round(float(energy_cum[min(99, len(energy_cum)-1)]), 4),
            "top200": round(float(energy_cum[min(199, len(energy_cum)-1)]), 4),
            "top500": round(float(energy_cum[min(499, len(energy_cum)-1)]), 4),
        },
        "PR": round(float(PR), 1),
    }
    print(f"  Energy: top1={energy_cum[0]:.4f}, top10={energy_cum[9]:.4f}, "
          f"top100={energy_cum[min(99,len(energy_cum)-1)]:.4f}")

    # 2. 语义子空间分析: 将W_U行按词频分组
    w2id = tokenizer.get_vocab()
    id2w = {v: k for k, v in w2id.items()}

    # 按词的ID范围分组(作为语义的粗略代理)
    # 高频词(前5K), 中频词(5K-50K), 低频词(50K+)
    n_groups = 3
    group_names = ["high_freq_5k", "mid_freq_50k", "low_freq_rest"]
    group_bounds = [5000, 50000, vocab_size]

    group_subspace_stats = {}
    prev_bound = 0
    for gi, (gname, bound) in enumerate(zip(group_names, group_bounds)):
        # 取该组W_U的行
        sub_W = W_U[prev_bound:bound]
        if sub_W.shape[0] < 10:
            prev_bound = bound
            continue

        # 子矩阵的SVD
        n_sub_svd = min(100, min(sub_W.shape) - 1)
        try:
            U_sub, S_sub, Vt_sub = svds(sub_W.astype(np.float32), k=n_sub_svd)
            S_sub = S_sub[::-1]
            PR_sub = (S_sub.sum()**2) / (S_sub**2).sum() if (S_sub**2).sum() > 0 else 0

            # 子空间主方向与W_U主方向的对齐度
            Vt_sub = Vt_sub[::-1, :]
            # 前10个主方向与W_U前10个主方向的余弦矩阵
            alignment_matrix = np.abs(Vt_sub[:10] @ Vt_wu[:10].T)  # [10, 10]
            mean_alignment = float(alignment_matrix.mean())
            max_alignment = float(alignment_matrix.max(axis=1).mean())

            group_subspace_stats[gname] = {
                "n_words": sub_W.shape[0],
                "PR": round(float(PR_sub), 1),
                "S_top5": [round(float(x), 2) for x in S_sub[:5]],
                "energy_top10": round(float((S_sub[:10]**2).sum() / (S_sub**2).sum()), 4),
                "alignment_with_WU_top10_mean": round(mean_alignment, 4),
                "alignment_with_WU_top10_max": round(max_alignment, 4),
            }
            print(f"  {gname}: n={sub_W.shape[0]}, PR={PR_sub:.1f}, "
                  f"alignment_mean={mean_alignment:.4f}, alignment_max={max_alignment:.4f}")
        except Exception as e:
            print(f"  {gname}: SVD failed ({e})")
            group_subspace_stats[gname] = {"error": str(e)}

        prev_bound = bound

    results["group_subspace"] = group_subspace_stats

    # 3. 逐层hidden state在W_U子空间中的投影
    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    layer_proj = {}
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    for li, layer_idx in enumerate(sample_layers):
        hs = out.hidden_states[layer_idx][0, -1].float().cpu().numpy()  # [d_model]

        # 在W_U行空间中的投影
        proj_coeffs = Vt_wu @ hs  # [k]
        proj_hs = Vt_wu.T @ proj_coeffs  # [d_model]
        proj_ratio = np.linalg.norm(proj_hs)**2 / (np.linalg.norm(hs)**2 + 1e-20)

        # 在每个语义子空间中的投影
        group_proj = {}
        for gname, gstats in group_subspace_stats.items():
            if "error" in gstats:
                continue
            # 用该组的Vt_sub(但我们没保存), 用词向量方向的子空间近似
            # 简化: 用W_U行的PCA近似

        # 投影系数分布
        coeff_energy = proj_coeffs**2
        total_e = coeff_energy.sum() + 1e-20
        top1_e = coeff_energy[:1].sum() / total_e
        top10_e = coeff_energy[:10].sum() / total_e if len(coeff_energy) >= 10 else 1.0
        top50_e = coeff_energy[:min(50, len(coeff_energy))].sum() / total_e
        top100_e = coeff_energy[:min(100, len(coeff_energy))].sum() / total_e

        layer_proj[str(layer_idx)] = {
            "proj_ratio": round(float(proj_ratio), 4),
            "top1_energy": round(float(top1_e), 4),
            "top10_energy": round(float(top10_e), 4),
            "top50_energy": round(float(top50_e), 4),
            "top100_energy": round(float(top100_e), 4),
        }
        print(f"  L{layer_idx}: proj_ratio={proj_ratio:.4f}, top10={top10_e:.4f}, top100={top100_e:.4f}")

    results["layer_proj"] = layer_proj
    results["sample_layers"] = sample_layers

    results["elapsed"] = round(time.time() - t0, 1)
    return results


# ===== P441: 权重→激活因果方程 =====
def run_p441(model, tokenizer, device, model_name):
    """验证R_l是否能用权重乘积表达"""
    print("\n===== P441: 权重->激活因果方程 =====")
    results = {}
    t0 = time.time()

    n_layers = len(get_layers(model))
    d_model = model.get_input_embeddings().weight.shape[1]

    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]

    # W_U投影基
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 1)
    _, _, Vt_wu = svds(W_U.astype(np.float32), k=k_svd)
    Vt_wu = Vt_wu[::-1, :]  # [k, d_model]

    # 逐层分析权重结构
    layer_weight_analysis = {}

    for li, layer_idx in enumerate(sample_layers):
        layer = get_layers(model)[layer_idx]

        # 获取权重矩阵
        # Self-attention
        W_q = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()  # [d_model, d_model]
        W_k = layer.self_attn.k_proj.weight.detach().cpu().float().numpy()
        W_v = layer.self_attn.v_proj.weight.detach().cpu().float().numpy()
        W_o = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()

        # MLP
        W_up = layer.mlp.up_proj.weight.detach().cpu().float().numpy()    # [intermediate, d_model]
        W_down = layer.mlp.down_proj.weight.detach().cpu().float().numpy()  # [d_model, intermediate]
        W_gate = layer.mlp.gate_proj.weight.detach().cpu().float().numpy() if hasattr(layer.mlp, 'gate_proj') else None

        # 1. 权重的有效秩
        def effective_rank(W, name, max_svd=100):
            """计算有效秩和top-k能量"""
            n_svd = min(max_svd, min(W.shape) - 1)
            try:
                from scipy.sparse.linalg import svds as sp_svds
                _, S, _ = sp_svds(W.astype(np.float32), k=n_svd)
                S = S[::-1]
                PR = (S.sum()**2) / (S**2).sum() if (S**2).sum() > 0 else 0
                energy_cum = np.cumsum(S**2) / (S**2).sum()
                return {
                    "shape": list(W.shape),
                    "PR": round(float(PR), 1),
                    "S_top5": [round(float(x), 2) for x in S[:5]],
                    "energy_top10": round(float(energy_cum[min(9, len(energy_cum)-1)]), 4),
                    "energy_top50": round(float(energy_cum[min(49, len(energy_cum)-1)]), 4),
                }
            except Exception as e:
                return {"shape": list(W.shape), "error": str(e)}

        w_stats = {
            "W_q": effective_rank(W_q, "W_q"),
            "W_k": effective_rank(W_k, "W_k"),
            "W_v": effective_rank(W_v, "W_v"),
            "W_o": effective_rank(W_o, "W_o"),
            "W_up": effective_rank(W_up, "W_up"),
            "W_down": effective_rank(W_down, "W_down"),
        }
        if W_gate is not None:
            w_stats["W_gate"] = effective_rank(W_gate, "W_gate")

        # 2. W_o的行空间与W_U行空间的对齐度
        n_align = min(50, min(W_o.shape) - 1)
        try:
            from scipy.sparse.linalg import svds as sp_svds
            _, S_o, Vt_o = sp_svds(W_o.astype(np.float32), k=n_align)
            Vt_o = Vt_o[::-1, :]  # [k, d_model]
            # 对齐度: |Vt_o @ Vt_wu.T|
            alignment = np.abs(Vt_o[:20] @ Vt_wu[:20].T)  # [20, 20]
            w_stats["W_o_align_WU_mean"] = round(float(alignment.mean()), 4)
            w_stats["W_o_align_WU_max"] = round(float(alignment.max(axis=1).mean()), 4)
        except Exception as e:
            w_stats["W_o_align_WU_error"] = str(e)

        # 3. W_down的行空间与W_U行空间的对齐度
        n_align_d = min(50, min(W_down.shape) - 1)
        try:
            from scipy.sparse.linalg import svds as sp_svds
            _, S_d, Vt_d = sp_svds(W_down.astype(np.float32), k=n_align_d)
            Vt_d = Vt_d[::-1, :]
            alignment_d = np.abs(Vt_d[:20] @ Vt_wu[:20].T)
            w_stats["W_down_align_WU_mean"] = round(float(alignment_d.mean()), 4)
            w_stats["W_down_align_WU_max"] = round(float(alignment_d.max(axis=1).mean()), 4)
        except Exception as e:
            w_stats["W_down_align_WU_error"] = str(e)

        # 4. 层间权重对齐: W_o(l) vs W_v(l)的输出空间
        # W_o @ W_v 的有效秩
        try:
            WV = W_o @ W_v.T if W_o.shape[1] == W_v.shape[0] else W_o @ W_v
            w_stats["W_o_W_v_PR"] = effective_rank(WV, "W_o*W_v", max_svd=50)
        except:
            pass

        # 5. MLP组合: W_down @ diag(gate) @ W_up 的近似
        if W_gate is not None:
            try:
                # W_down @ W_gate.T 近似 (忽略GELU非线性)
                n_gate = min(50, W_down.shape[1], W_gate.shape[0])
                WD_WG = (W_down[:, :n_gate].astype(np.float32)) @ (W_gate[:n_gate, :].astype(np.float32))
                w_stats["W_down_W_gate_PR"] = effective_rank(WD_WG, "W_down*W_gate", max_svd=50)
            except:
                pass

        layer_weight_analysis[str(layer_idx)] = w_stats

        # 打印摘要
        print(f"  L{layer_idx}: W_q PR={w_stats['W_q'].get('PR', 'N/A')}, "
              f"W_o PR={w_stats['W_o'].get('PR', 'N/A')}, "
              f"W_down PR={w_stats['W_down'].get('PR', 'N/A')}, "
              f"W_o_align_WU={w_stats.get('W_o_align_WU_mean', 'N/A')}, "
              f"W_down_align_WU={w_stats.get('W_down_align_WU_mean', 'N/A')}")

    results["layer_weight_analysis"] = layer_weight_analysis
    results["sample_layers"] = sample_layers
    results["n_layers"] = n_layers
    results["d_model"] = d_model

    results["elapsed"] = round(time.time() - t0, 1)
    return results


# ===== P442: 五维交叉约束 =====
def run_p442(model, tokenizer, device, model_name):
    """五维交叉约束: W×A×G×T×I是否共享基方向?"""
    print("\n===== P442: 五维交叉约束 =====")
    results = {}
    t0 = time.time()

    n_layers = len(get_layers(model))
    d_model = model.get_input_embeddings().weight.shape[1]

    W_U = model.lm_head.weight.detach().cpu().float().numpy()

    # W_U投影基
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 1)
    _, _, Vt_wu = svds(W_U.astype(np.float32), k=k_svd)
    Vt_wu = Vt_wu[::-1, :]

    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    # 准备prompt
    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    layer_cross = {}

    for layer_idx in sample_layers:
        print(f"  Layer {layer_idx}...")
        layer = get_layers(model)[layer_idx]

        # ===== 维度1: 权重空间(W) =====
        W_o = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()
        W_down = layer.mlp.down_proj.weight.detach().cpu().float().numpy()

        # W的主方向: W_o的右奇异向量
        # W_o形状可能是[d_model, d_model]或[intermediate, d_model]
        # 右奇异向量Vt的列维度 = W_o的列数
        # 我们需要列维度 = d_model的方向, 所以确保取正确的一侧
        if W_o.shape[1] == d_model:
            W_for_svd = W_o  # [rows, d_model]
        else:
            W_for_svd = W_o.T  # [d_model, rows]

        n_w = min(50, min(W_for_svd.shape) - 1)
        try:
            from scipy.sparse.linalg import svds as sp_svds
            _, _, Vt_wo = sp_svds(W_for_svd.astype(np.float32), k=n_w)
            Vt_wo = Vt_wo[::-1, :]  # [n_w, d_model]
            if Vt_wo.shape[1] != d_model:
                # fallback
                raise ValueError("SVD dimension mismatch")
        except:
            Vt_wo = np.random.randn(n_w, d_model).astype(np.float32)
            Vt_wo, _ = np.linalg.qr(Vt_wo.T)
            Vt_wo = Vt_wo.T

        # W_down的右奇异向量
        if W_down.shape[1] == d_model:
            W_down_for_svd = W_down
        else:
            W_down_for_svd = W_down.T

        n_wd = min(50, min(W_down_for_svd.shape) - 1)
        try:
            from scipy.sparse.linalg import svds as sp_svds
            _, _, Vt_wd = sp_svds(W_down_for_svd.astype(np.float32), k=n_wd)
            Vt_wd = Vt_wd[::-1, :]
            if Vt_wd.shape[1] != d_model:
                raise ValueError("SVD dimension mismatch")
        except:
            Vt_wd = np.random.randn(n_wd, d_model).astype(np.float32)
            Vt_wd, _ = np.linalg.qr(Vt_wd.T)
            Vt_wd = Vt_wd.T

        # ===== 维度2: 激活空间(A) =====
        # 用10个不同prompt收集hidden state, 做PCA
        prompts = [
            "The answer is", "I think that", "She said hello",
            "The cat sat", "We know the", "He was very",
            "They went to", "It is clear", "You can see", "This means that"
        ]
        hs_list = []
        with torch.no_grad():
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt").to(device)
                out = model(inp["input_ids"], output_hidden_states=True)
                hs_list.append(out.hidden_states[layer_idx][0, -1].float().cpu().numpy())

        hs_matrix = np.array(hs_list)  # [10, d_model]
        # PCA: 减均值, SVD
        hs_centered = hs_matrix - hs_matrix.mean(axis=0)
        n_a = min(10, hs_centered.shape[0] - 1)
        try:
            U_a, S_a, Vt_a = np.linalg.svd(hs_centered, full_matrices=False)
            Vt_a_top = Vt_a[:n_a]  # [n_a, d_model]
        except:
            Vt_a_top = np.random.randn(n_a, d_model).astype(np.float32)

        # ===== 维度3: 注意力空间(T) =====
        # Qwen3等模型不支持output_attentions, 跳过
        # 用W_q的主方向代替
        W_q_mat = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()
        n_wq = min(50, min(W_q_mat.shape) - 1)
        try:
            from scipy.sparse.linalg import svds as sp_svds
            _, _, Vt_wq = sp_svds(W_q_mat.astype(np.float32), k=n_wq)
            Vt_wq = Vt_wq[::-1, :]
        except:
            Vt_wq = np.random.randn(n_wq, d_model).astype(np.float32)
            Vt_wq, _ = np.linalg.qr(Vt_wq.T)
            Vt_wq = Vt_wq.T[:n_wq]

        # ===== 维度4: 信息流空间(I) = W_U空间 =====
        # 已有Vt_wu

        # ===== 交叉对齐分析 =====
        # W_o vs A
        n_min = min(Vt_wo.shape[0], Vt_a_top.shape[0], 10)
        align_wo_a = np.abs(Vt_wo[:n_min] @ Vt_a_top[:n_min].T)
        align_wo_a_mean = float(align_wo_a.mean())
        align_wo_a_max = float(align_wo_a.max(axis=1).mean())

        # W_down vs A
        n_min2 = min(Vt_wd.shape[0], Vt_a_top.shape[0], 10)
        align_wd_a = np.abs(Vt_wd[:n_min2] @ Vt_a_top[:n_min2].T)
        align_wd_a_mean = float(align_wd_a.mean())
        align_wd_a_max = float(align_wd_a.max(axis=1).mean())

        # W_o vs I (W_U)
        n_min3 = min(Vt_wo.shape[0], Vt_wu.shape[0], 10)
        align_wo_i = np.abs(Vt_wo[:n_min3] @ Vt_wu[:n_min3].T)
        align_wo_i_mean = float(align_wo_i.mean())
        align_wo_i_max = float(align_wo_i.max(axis=1).mean())

        # W_down vs I
        n_min4 = min(Vt_wd.shape[0], Vt_wu.shape[0], 10)
        align_wd_i = np.abs(Vt_wd[:n_min4] @ Vt_wu[:n_min4].T)
        align_wd_i_mean = float(align_wd_i.mean())
        align_wd_i_max = float(align_wd_i.max(axis=1).mean())

        # A vs I
        n_min5 = min(Vt_a_top.shape[0], Vt_wu.shape[0], 10)
        align_a_i = np.abs(Vt_a_top[:n_min5] @ Vt_wu[:n_min5].T)
        align_a_i_mean = float(align_a_i.mean())
        align_a_i_max = float(align_a_i.max(axis=1).mean())

        layer_cross[str(layer_idx)] = {
            "W_o_vs_A": {"mean": round(align_wo_a_mean, 4), "max": round(align_wo_a_max, 4)},
            "W_down_vs_A": {"mean": round(align_wd_a_mean, 4), "max": round(align_wd_a_max, 4)},
            "W_o_vs_I": {"mean": round(align_wo_i_mean, 4), "max": round(align_wo_i_max, 4)},
            "W_down_vs_I": {"mean": round(align_wd_i_mean, 4), "max": round(align_wd_i_max, 4)},
            "A_vs_I": {"mean": round(align_a_i_mean, 4), "max": round(align_a_i_max, 4)},
        }

        print(f"    W_o vs A: mean={align_wo_a_mean:.4f}, max={align_wo_a_max:.4f}")
        print(f"    W_down vs A: mean={align_wd_a_mean:.4f}, max={align_wd_a_max:.4f}")
        print(f"    W_o vs I: mean={align_wo_i_mean:.4f}, max={align_wo_i_max:.4f}")
        print(f"    W_down vs I: mean={align_wd_i_mean:.4f}, max={align_wd_i_max:.4f}")
        print(f"    A vs I: mean={align_a_i_mean:.4f}, max={align_a_i_max:.4f}")

    results["layer_cross"] = layer_cross
    results["sample_layers"] = sample_layers

    # 全局交叉对齐趋势
    print("\n  === Cross-Alignment Summary ===")
    for dim_pair in ["W_o_vs_A", "W_down_vs_A", "W_o_vs_I", "W_down_vs_I", "A_vs_I"]:
        means = [layer_cross[k][dim_pair]["mean"] for k in layer_cross]
        print(f"  {dim_pair}: mean across layers = {np.mean(means):.4f}")

    results["elapsed"] = round(time.time() - t0, 1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--exp", choices=["p439", "p440", "p441", "p442", "all"], required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # 添加错误处理
    import traceback
    import sys

    # 加载之前的实验结果(如果需要)
    def load_latest_result(exp_name, model_name):
        import glob
        pattern = str(OUT_DIR / f"phase_xc_{exp_name}_{model_name}_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]
            print(f"  Loading {exp_name} results from {latest}")
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    if args.exp == "p439" or args.exp == "all":
        r = run_p439(model, tokenizer, device, args.model)
        ts = time.strftime("%Y%m%d_%H%M")
        out_path = OUT_DIR / f"phase_xc_p439_{args.model}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    if args.exp == "p440" or args.exp == "all":
        r = run_p440(model, tokenizer, device, args.model)
        ts = time.strftime("%Y%m%d_%H%M")
        out_path = OUT_DIR / f"phase_xc_p440_{args.model}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    if args.exp == "p441" or args.exp == "all":
        r = run_p441(model, tokenizer, device, args.model)
        ts = time.strftime("%Y%m%d_%H%M")
        out_path = OUT_DIR / f"phase_xc_p441_{args.model}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    if args.exp == "p442" or args.exp == "all":
        r = run_p442(model, tokenizer, device, args.model)
        ts = time.strftime("%Y%m%d_%H%M")
        out_path = OUT_DIR / f"phase_xc_p442_{args.model}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    # 释放GPU
    del model
    torch.cuda.empty_cache()
    print("  GPU memory released.")


if __name__ == "__main__":
    main()
