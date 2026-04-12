"""
Phase LXXXI-P403/404/405/406: W_lm精确结构分析 + 信号分配机制 + 高维搜索
================================================================================

阶段D核心任务 - 从信号分配效率极低(1e-6)到精确数学解释:

P403: W_lm的SVD精确几何 - 151K行的行空间结构
P404: 信号分配效率的数学上限 - JL引理 + 随机投影理论
P405: 高维正交空间搜索 - 自动发现第8、9...维
P406: DS7B的"信号吸收"路径追踪 - proj≈0但dlogit≠0的机制

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
    "quantity": [("many", "few")],
}

EXTRA_DIM_PAIRS = {
    "complexity": [("complex", "simple")],
    "formality": [("professional", "casual")],
    "politeness": [("please", "now")],
    "specificity": [("specifically", "generally")],
    "speed": [("quickly", "slowly")],
    "size": [("large", "small")],
    "age": [("old", "new")],
    "distance": [("near", "far")],
    "brightness": [("bright", "dark")],
    "temperature": [("hot", "cold")],
    "weight": [("heavy", "light")],
    "sound": [("loud", "quiet")],
    "frequency": [("often", "rarely")],
    "importance": [("important", "trivial")],
    "position": [("above", "below")],
    "direction": [("forward", "backward")],
    "completeness": [("complete", "partial")],
    "value": [("valuable", "worthless")],
}

PROMPTS = ["The apple is", "In the future, people will", "The scientist explained that", "When the rain stopped,"]

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


def get_w_lm(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm(model, tokenizer, word_neg)
    diff = w_pos - w_neg
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        return w_pos, 0.0
    return diff / norm, norm


def get_layers(model, max_layers=None):
    if hasattr(model.model, "layers"):
        all_layers = list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        all_layers = list(model.model.encoder.layers)
    else:
        raise ValueError("Cannot find layers")
    if max_layers is None:
        return all_layers
    return all_layers[:max_layers]


# ========== P403: W_lm的SVD精确几何 ==========

def run_p403(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P403: W_lm SVD precise geometry - {model_name}")
    print(f"{'='*60}")

    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape
    print(f"  W_lm shape: [{vocab_size}, {hidden_dim}]")

    # 随机SVD近似
    from sklearn.decomposition import TruncatedSVD
    n_components = min(hidden_dim, 256)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_lm)
    singular_values = svd.singular_values_
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    rank_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)
    rank_999 = int(np.searchsorted(cumulative_variance, 0.999) + 1)

    print(f"\n  Top 20 singular values: {[f'{s:.1f}' for s in singular_values[:20]]}")
    print(f"  Rank 95%: {rank_95}, 99%: {rank_99}, 99.9%: {rank_999}")
    print(f"  CumVar@20: {cumulative_variance[19]:.6f}, @50: {cumulative_variance[min(49,n_components-1)]:.6f}, @100: {cumulative_variance[min(99,n_components-1)]:.6f}")

    # 幂律拟合
    log_s = np.log(singular_values[:100] + 1e-10)
    log_k = np.log(np.arange(1, 101))
    try:
        slope, _ = np.polyfit(log_k, log_s, 1)
        power_law_exp = -slope
    except:
        power_law_exp = 0
    print(f"  Power law exponent (top 100): {power_law_exp:.3f}")

    # 维度方向在SVD空间的投影
    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty", "quantity"]
    dim_directions = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = direction

    V_right = svd.components_
    dim_spectrum = {}
    for name, direction in dim_directions.items():
        proj_on_sing = direction @ V_right.T
        proj_sq = proj_on_sing ** 2
        total_proj = np.sum(proj_sq)
        top10_frac = np.sum(proj_sq[:10]) / max(total_proj, 1e-10)
        top50_frac = np.sum(proj_sq[:50]) / max(total_proj, 1e-10)
        top100_frac = np.sum(proj_sq[:100]) / max(total_proj, 1e-10)
        dim_spectrum[name] = {
            "top10_frac": float(top10_frac),
            "top50_frac": float(top50_frac),
            "top100_frac": float(top100_frac),
        }
        print(f"  {name}: top10={top10_frac:.4f}, top50={top50_frac:.4f}, top100={top100_frac:.4f}")

    frobenius_sq = float(np.sum(W_lm ** 2))
    mean_row_norm_sq = float(np.mean(np.sum(W_lm ** 2, axis=1)))

    results = {
        "W_lm_shape": [vocab_size, hidden_dim],
        "singular_values_top20": [float(s) for s in singular_values[:20]],
        "rank_95": rank_95, "rank_99": rank_99, "rank_999": rank_999,
        "power_law_exponent": float(power_law_exp),
        "frobenius_sq": frobenius_sq,
        "mean_row_norm_sq": mean_row_norm_sq,
        "random_efficiency": float(1.0 / vocab_size),
        "dim_spectrum": dim_spectrum,
    }
    return results


# ========== P404: 信号分配效率的数学上限 ==========

def run_p404(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P404: Signal allocation efficiency upper bound - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape
    W_lm_F_sq = float(np.sum(W_lm ** 2))
    row_norms = np.sqrt(np.sum(W_lm ** 2, axis=1))
    mean_row_norm = float(np.mean(row_norms))

    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0
    layers = get_layers(model)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {"direction": direction, "w_diff": w_diff, "pos_id": pos_id, "neg_id": neg_id}

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handle = layers[-1].register_forward_hook(make_hook(captured_base, "last"))
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
    handle.remove()
    h_base = captured_base["last"][0, -1, :].cpu().numpy()

    # JL引理
    jl_epsilon = np.sqrt(np.log(vocab_size) / hidden_dim)
    signal_retention = 1 - jl_epsilon
    print(f"  JL epsilon={jl_epsilon:.4f}, retention={signal_retention:.4f}")

    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handle = layers[-1].register_forward_hook(make_hook(captured_int, "last"))
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        handle.remove()

        h_int = captured_int["last"][0, -1, :].cpu().numpy()
        dh = h_int - h_base
        dh_norm = float(np.linalg.norm(dh))

        actual_delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        actual_delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        actual_dlogit = actual_delta_pos - actual_delta_neg

        delta_logits = W_lm @ dh
        total_energy_L1 = float(np.sum(np.abs(delta_logits)))
        total_energy_L2 = float(np.sum(delta_logits ** 2))

        eff_L1 = abs(actual_dlogit) / max(total_energy_L1, 1e-10)
        eff_L2 = actual_dlogit**2 / max(total_energy_L2, 1e-10)

        w_diff_norm_sq = float(np.sum(w_diff ** 2))
        theory_lower = w_diff_norm_sq / max(W_lm_F_sq, 1e-10)
        theory_mid = float(np.linalg.norm(w_diff)) / max(vocab_size * mean_row_norm, 1e-10)

        proj_on_dh = delta_logits
        proj_mean = float(np.mean(proj_on_dh))
        proj_std = float(np.std(proj_on_dh))
        x_target = actual_dlogit / max(proj_std, 1e-10)
        gaussian_eff = np.exp(-x_target**2 / 2) / max(vocab_size * np.sqrt(2 * np.pi), 1)

        results_per_dim[dim_name] = {
            "actual_dlogit": actual_dlogit,
            "eff_L1": eff_L1, "eff_L2": eff_L2,
            "theory_lower": theory_lower, "theory_mid": theory_mid,
            "total_energy_L1": total_energy_L1, "total_energy_L2": total_energy_L2,
            "dh_norm": dh_norm,
            "proj_mean": proj_mean, "proj_std": proj_std,
            "x_target": float(x_target), "gaussian_eff": float(gaussian_eff),
        }

        print(f"  {dim_name}: eff_L1={eff_L1:.6e}, theory_lower={theory_lower:.6e}, "
              f"theory_mid={theory_mid:.6e}, gaussian={gaussian_eff:.6e}")
        print(f"    x_target={x_target:.3f}, ratio_actual/random={eff_L1/max(1/vocab_size,1e-15):.1f}x")

    results = {
        "jl_epsilon": float(jl_epsilon), "signal_retention": float(signal_retention),
        "W_lm_F_sq": W_lm_F_sq, "mean_row_norm": mean_row_norm,
        "random_efficiency": float(1/vocab_size),
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P405: 高维正交空间搜索 ==========

def run_p405(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P405: High-dimensional orthogonal space search - {model_name}")
    print(f"{'='*60}")

    known_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty", "quantity"]
    all_dims = dict(DIM_PAIRS)
    all_dims.update(EXTRA_DIM_PAIRS)

    dim_directions = {}
    for name, pairs in all_dims.items():
        pos, neg = pairs[0]
        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm > 1e-8:
                dim_directions[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}
        except Exception as e:
            print(f"  Skip {name}: {e}")

    print(f"  Total candidate dimensions: {len(dim_directions)}")

    # 正交性检验
    known_dirs = np.array([dim_directions[d]["direction"] for d in known_dims if d in dim_directions])
    Q, R = np.linalg.qr(known_dirs.T)

    new_dims_found = []
    for name, info in dim_directions.items():
        if name in known_dims:
            continue
        direction = info["direction"]
        cos_with_known = known_dirs @ direction
        max_cos_known = float(np.max(np.abs(cos_with_known)))
        proj_on_known = Q.T @ direction
        proj_frac = float(np.sum(proj_on_known ** 2))
        residual = direction - Q @ proj_on_known
        residual_norm = float(np.linalg.norm(residual))

        is_orthogonal = max_cos_known < 0.15
        if is_orthogonal:
            new_dims_found.append({
                "name": name, "pos": info["pos"], "neg": info["neg"],
                "max_cos_with_known": max_cos_known,
                "proj_frac_on_known": proj_frac,
                "residual_norm": residual_norm,
            })

        if max_cos_known < 0.3:
            orth_flag = " *** ORTHOGONAL ***" if is_orthogonal else ""
            print(f"  {name} ({info['pos']}/{info['neg']}): max_cos={max_cos_known:.4f}, "
                  f"proj_frac={proj_frac:.4f}{orth_flag}")

    # SVD分析
    all_dirs = [dim_directions[n]["direction"] for n in dim_directions]
    all_names = list(dim_directions.keys())
    D_matrix = np.array(all_dirs)
    U, S, Vt = np.linalg.svd(D_matrix, full_matrices=False)
    eff_rank_10 = int(np.sum(S > 0.1 * S[0]))
    eff_rank_05 = int(np.sum(S > 0.05 * S[0]))

    print(f"\n  SVD: total={len(all_names)}, eff_rank(10%)={eff_rank_10}, eff_rank(5%)={eff_rank_05}")
    print(f"  Top 15 singular values: {[f'{s:.3f}' for s in S[:15]]}")

    # 贪心搜索
    selected = [d for d in known_dims if d in dim_directions]
    selected_dirs = [dim_directions[d]["direction"] for d in selected]
    remaining = [n for n in dim_directions if n not in selected]

    for step in range(10):
        if not remaining:
            break
        best_name = None
        best_max_cos = 1.0
        for name in remaining:
            d = dim_directions[name]["direction"]
            max_cos = 0.0
            for sd in selected_dirs:
                c = abs(float(np.dot(d, sd)))
                max_cos = max(max_cos, c)
            if max_cos < best_max_cos:
                best_max_cos = max_cos
                best_name = name

        if best_name is None or best_max_cos > 0.2:
            break
        selected.append(best_name)
        selected_dirs.append(dim_directions[best_name]["direction"])
        remaining.remove(best_name)
        print(f"  Greedy step {step+1}: added {best_name} "
              f"({dim_directions[best_name]['pos']}/{dim_directions[best_name]['neg']}), "
              f"max_cos={best_max_cos:.4f}")

    # 更新后重新计算有效秩
    if len(selected) > 7:
        sel_matrix = np.array([dim_directions[n]["direction"] for n in selected])
        U2, S2, Vt2 = np.linalg.svd(sel_matrix, full_matrices=False)
        new_eff_rank = int(np.sum(S2 > 0.1 * S2[0]))
        print(f"\n  Updated SVD: {len(selected)} dims, effective_rank={new_eff_rank}")
        print(f"  Updated singular values: {[f'{s:.3f}' for s in S2[:min(15,len(S2))]]}")

    results = {
        "new_dims_found": new_dims_found,
        "total_candidates": len(dim_directions),
        "eff_rank_10": eff_rank_10,
        "eff_rank_05": eff_rank_05,
        "singular_values_top15": [float(s) for s in S[:15]],
        "greedy_selected": selected,
        "greedy_n": len(selected),
    }
    return results


# ========== P406: DS7B的"信号吸收"路径追踪 ==========

def run_p406(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P406: Signal absorption path tracking - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()

    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0
    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 2))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    layers = get_layers(model)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw
        w_diff_norm = float(np.linalg.norm(w_diff))
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {
            "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles = []
    for l in scan_layers:
        if l < len(layers):
            handles.append(layers[l].register_forward_hook(make_hook(captured_base, f"L{l}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles:
        h.remove()

    h_base_per_layer = {}
    for l in scan_layers:
        key = f"L{l}"
        if key in captured_base:
            h_base_per_layer[l] = captured_base[key][0, -1, :].cpu().numpy()

    # 对每个维度, 逐层追踪dh
    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handles = []
        for l in scan_layers:
            if l < len(layers):
                handles.append(layers[l].register_forward_hook(make_hook(captured_int, f"L{l}")))
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles:
            h.remove()

        # 逐层分析
        dh_per_layer = {}
        cos_dh_wdiff_per_layer = {}
        proj_dh_wdiff_per_layer = {}
        dh_norm_per_layer = {}
        # ★★★ 关键新指标: Δlogit通过W_lm的理论预测 ★★★
        # 如果dh被旋转到W_diff正交方向, proj(dh, W_diff)≈0
        # 但dh可能在W_lm的行空间中仍有投影→Δlogit≠0
        # 所以关键是: dh在W_lm的哪些右奇异向量方向有投影?
        
        delta_logit_pred_per_layer = {}  # 用dh预测Δlogit
        
        for l in scan_layers:
            key = f"L{l}"
            if key in captured_int and key in captured_base:
                dh = (captured_int[key][0, -1, :] - captured_base[key][0, -1, :]).cpu().numpy()
                dh_per_layer[l] = dh
                dh_norm = float(np.linalg.norm(dh))
                dh_norm_per_layer[l] = dh_norm
                
                cos_wdiff = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0.0
                proj_wdiff = float(np.dot(dh, w_diff))
                cos_dh_wdiff_per_layer[l] = cos_wdiff
                proj_dh_wdiff_per_layer[l] = proj_wdiff
                
                # Δlogit预测
                delta_logits_theory = W_lm @ dh
                delta_pos_theory = float(delta_logits_theory[pos_id])
                delta_neg_theory = float(delta_logits_theory[neg_id])
                dlogit_theory = delta_pos_theory - delta_neg_theory
                delta_logit_pred_per_layer[l] = dlogit_theory

        # 最后层的实际Δlogit
        actual_delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu()) if 'logits_base' in dir() else 0
        actual_delta_neg = float(logits_int[neg_id].cpu() - logits_base[pos_id].cpu()) if 'logits_base' in dir() else 0

        # ★★★ 核心: 信号被"吸收"到哪个子空间? ★★★
        # 如果proj(dh, W_diff)≈0 但 Δlogit≠0
        # 则信号存在于W_lm行空间中, 但不在W_diff方向
        # 关键: dh分解 = proj_on_Wdiff方向 + residual方向
        # Δlogit = W_lm @ dh = W_lm @ (proj_on_Wdiff + residual)
        #        = proj(dh, W_diff) + W_lm @ residual
        # 如果residual对Δlogit的贡献 > proj的贡献 → "信号吸收"

        # 用最后层的dh分析
        last_l = scan_layers[-1]
        if last_l in dh_per_layer:
            dh_last = dh_per_layer[last_l]
            dh_norm_last = dh_norm_per_layer[last_l]
            
            # dh在W_diff方向的投影
            w_diff_unit = w_diff / max(w_diff_norm, 1e-10)
            proj_component = np.dot(dh_last, w_diff_unit) * w_diff_unit  # W_diff方向的分量
            residual_component = dh_last - proj_component  # 正交于W_diff的分量
            
            proj_norm = float(np.linalg.norm(proj_component))
            residual_norm = float(np.linalg.norm(residual_component))
            
            # 两个分量对Δlogit的贡献
            dlogit_from_proj = float(np.dot(proj_component, w_diff))
            dlogit_from_residual = float(np.dot(residual_component, w_diff))
            
            # residual在W_lm行空间中的投影 → 对所有词的Δlogit
            delta_logits_from_residual = W_lm @ residual_component
            total_energy_residual = float(np.sum(np.abs(delta_logits_from_residual)))
            dlogit_target_from_residual = float(delta_logits_from_residual[pos_id] - delta_logits_from_residual[neg_id])
            
            print(f"\n  {dim_name}:")
            print(f"    proj_norm={proj_norm:.3f}, residual_norm={residual_norm:.3f}")
            print(f"    dlogit_from_proj={dlogit_from_proj:.3f}, dlogit_from_residual={dlogit_from_residual:.3f}")
            print(f"    total_energy_residual={total_energy_residual:.1f}, dlogit_target_residual={dlogit_target_from_residual:.3f}")
            print(f"    cos(dh, W_diff)={cos_dh_wdiff_per_layer.get(last_l, 0):.4f}")

            results_per_dim[dim_name] = {
                "dh_norm_last": dh_norm_last,
                "proj_norm": proj_norm,
                "residual_norm": residual_norm,
                "dlogit_from_proj": dlogit_from_proj,
                "dlogit_from_residual": dlogit_from_residual,
                "total_energy_residual": total_energy_residual,
                "dlogit_target_from_residual": dlogit_target_from_residual,
                "cos_dh_wdiff_per_layer": {str(k): v for k, v in cos_dh_wdiff_per_layer.items()},
                "proj_dh_wdiff_per_layer": {str(k): v for k, v in proj_dh_wdiff_per_layer.items()},
                "dh_norm_per_layer": {str(k): v for k, v in dh_norm_per_layer.items()},
                "delta_logit_pred_per_layer": {str(k): v for k, v in delta_logit_pred_per_layer.items()},
            }

    # 获取基线logits(如果还没获取)
    with torch.no_grad():
        logits_base_ref = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
    
    results = {
        "scan_layers": scan_layers,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p403", "p404", "p405", "p406", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p403", "p404", "p405", "p406"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p403":
                results["p403"] = run_p403(model, tokenizer, device, args.model)
            elif exp == "p404":
                results["p404"] = run_p404(model, tokenizer, device, args.model)
            elif exp == "p405":
                results["p405"] = run_p405(model, tokenizer, device, args.model)
            elif exp == "p406":
                results["p406"] = run_p406(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    out_file = OUT_DIR / f"phase_lxxxi_p403_406_{args.model}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
