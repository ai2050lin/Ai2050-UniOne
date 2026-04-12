"""
Phase LXXX-P399/400/401/402: LCS精确化 + DS7B非线性机制 + 预测力验证
================================================================================

阶段C核心任务 - 从LCS参数化描述到精确数学形式:

P399: proj(dh_final, W_diff)的完整投影分布
  - P396发现: Qwen3的proj(dh_final, W_diff)是actual Δlogit的5-7倍
  - 核心问题: proj(dh, W_diff)中, 多少分配给目标词pos, 多少分配给neg, 多少分配给其他词?
  - 定义: Δlogit_i = W_lm[i] · dh 对所有i的分布
  - 预期: 只有~15-20%的proj分配给(pos-neg)差值, 其余分配给其他token

P400: DS7B非线性映射机制
  - P396发现: DS7B的Δlogit不能由任何单一指标预测(R²<0.7)
  - 假设: Δlogit = f(proj_dh_wdiff, dh_norm, cos_dh_wdiff, ...) 的非线性函数
  - 方法: 用多项式回归/符号回归搜索最佳映射
  - 预期: DS7B需要高阶交叉项(proj × cos或norm × cos)

P401: LCS的精确矩阵估计
  - P398只有参数化描述(γ, 交互范数), 需要精确矩阵形式
  - R_rotate: 用多层dh数据估计旋转矩阵 R_l (dh_{l+1} = R_l @ dh_l + noise)
  - C_compete: 用联合注入数据估计竞争矩阵
  - M_map: 用W_lm SVD分析映射的精确结构

P402: LCS预测力验证
  - 用LCS模型预测新维度(quantity维度)的干预效果
  - 对比LCS预测值 vs 实际干预值
  - 这是LCS是否有效的终极检验

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
    "quantity": [("many", "few")],  # P402新维度
}

PROMPTS = [
    "The apple is",
    "In the future, people will",
    "The scientist explained that",
    "When the rain stopped,",
]

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True,
        "use_fast": False,
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


# ========== P399: proj(dh, W_diff)的完整投影分布 ==========

def run_p399(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P399: Full projection distribution of proj(dh, W_diff) - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()  # [vocab_size, hidden_dim]
    vocab_size, hidden_dim = W_lm.shape

    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
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
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "w_pos_raw": w_pos_raw, "w_neg_raw": w_neg_raw,
            "w_diff": w_diff,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线logits和hidden states
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    # 捕获末层hidden state
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
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    handle.remove()

    h_base = captured_base["last"][0, -1, :].cpu().numpy()  # 末层基线hidden

    results = {}

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]
        w_diff = dim_info[dim_name]["w_diff"]

        # L0注入
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

        # ★★★ 核心分析: Δlogit_i = W_lm[i] · dh 的完整分布 ★★★
        # 计算所有token的Δlogit
        delta_logits = W_lm @ dh  # [vocab_size] — 这是理论值(logit变化)
        # 但实际logits经过softmax等, 需要与实际对比

        actual_delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        actual_delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        actual_dlogit = actual_delta_pos - actual_delta_neg

        # 理论值: proj(dh, W_diff) = dot(dh, W_pos - W_neg)
        proj_dh_wdiff = float(np.dot(dh, w_diff))

        # ★★★ 关键: proj(dh, W_diff)中各部分的比例 ★★★
        # proj(dh, W_diff) = dot(dh, W_pos) - dot(dh, W_neg)
        proj_on_pos = float(np.dot(dh, dim_info[dim_name]["w_pos_raw"]))
        proj_on_neg = float(np.dot(dh, dim_info[dim_name]["w_neg_raw"]))

        # Δlogit的统计分布
        delta_logits_abs = np.abs(delta_logits)
        total_energy = float(np.sum(delta_logits_abs))
        energy_on_pos = float(np.abs(delta_logits[pos_id]))
        energy_on_neg = float(np.abs(delta_logits[neg_id]))
        energy_on_dlogit = float(np.abs(delta_logits[pos_id] - delta_logits[neg_id]))

        # Top-K token分析: 哪些token获得了最多的Δlogit?
        top_k = 20
        top_indices = np.argsort(np.abs(delta_logits))[::-1][:top_k]
        top_tokens = []
        for idx in top_indices:
            tok_str = tokenizer.decode([int(idx)])
            top_tokens.append({
                "token_id": int(idx),
                "token_str": tok_str[:20],
                "delta_logit": float(delta_logits[idx]),
                "abs_delta_logit": float(abs(delta_logits[idx])),
                "fraction_of_total": float(abs(delta_logits[idx]) / max(total_energy, 1e-10)),
            })

        # Δlogit分布的统计量
        pos_count = int(np.sum(delta_logits > 0))
        neg_count = int(np.sum(delta_logits < 0))
        mean_abs = float(np.mean(delta_logits_abs))
        std_dl = float(np.std(delta_logits))
        max_dl = float(np.max(delta_logits_abs))
        percentile_95 = float(np.percentile(delta_logits_abs, 95))
        percentile_99 = float(np.percentile(delta_logits_abs, 99))

        # ★★★ 关键比值 ★★★
        ratio_actual_proj = actual_dlogit / max(abs(proj_dh_wdiff), 1e-10)
        ratio_proj_on_pos = proj_on_pos / max(abs(proj_dh_wdiff), 1e-10)
        ratio_proj_on_neg = proj_on_neg / max(abs(proj_dh_wdiff), 1e-10)

        # Δlogit(pos) - Δlogit(neg) vs sum(|Δlogit_i|) for all i
        # 这告诉我们"信号分配效率": 有多少总Δlogit能量分配给了目标维度
        signal_efficiency = abs(actual_dlogit) / max(total_energy, 1e-10)

        results[dim_name] = {
            "actual_dlogit": actual_dlogit,
            "proj_dh_wdiff": proj_dh_wdiff,
            "proj_on_pos": proj_on_pos,
            "proj_on_neg": proj_on_neg,
            "ratio_actual_proj": ratio_actual_proj,
            "signal_efficiency": signal_efficiency,
            "total_energy": total_energy,
            "energy_on_pos": energy_on_pos,
            "energy_on_neg": energy_on_neg,
            "dh_norm": dh_norm,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "mean_abs_dl": mean_abs,
            "std_dl": std_dl,
            "max_dl": max_dl,
            "p95": percentile_95,
            "p99": percentile_99,
            "top_tokens": top_tokens,
        }

        print(f"\n  {dim_name}: actual_dlogit={actual_dlogit:.3f}, proj_wdiff={proj_dh_wdiff:.3f}")
        print(f"    proj_on_pos={proj_on_pos:.3f}, proj_on_neg={proj_on_neg:.3f}")
        print(f"    ratio_actual/proj={ratio_actual_proj:.4f}")
        print(f"    signal_efficiency={signal_efficiency:.6f} (|dlogit|/total_energy)")
        print(f"    total_energy={total_energy:.1f}, pos_tokens={pos_count}, neg_tokens={neg_count}")
        top_strs = [(t["token_str"][:20].encode("ascii", "replace").decode(), round(t["delta_logit"], 3)) for t in top_tokens[:5]]
        print(f"    top tokens: {top_strs}")

    return results


# ========== P400: DS7B非线性映射机制 ==========

def run_p400(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P400: Nonlinear mapping mechanism - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()

    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    betas = [4.0, 8.0, 12.0, 16.0]  # 多个注入强度

    n_layers_total = len(get_layers(model))
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
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
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

    # 收集所有(dim, beta)组合的数据
    all_data = []

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        for beta in betas:
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

            proj_dh_wdiff = float(np.dot(dh, w_diff))
            cos_dh_wdiff = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0.0

            all_data.append({
                "dim": dim_name,
                "beta": beta,
                "actual_dlogit": actual_dlogit,
                "proj_dh_wdiff": proj_dh_wdiff,
                "cos_dh_wdiff": cos_dh_wdiff,
                "dh_norm": dh_norm,
                "delta_pos": actual_delta_pos,
                "delta_neg": actual_delta_neg,
            })

            print(f"  {dim_name} beta={beta}: dlogit={actual_dlogit:.3f}, proj={proj_dh_wdiff:.3f}, "
                  f"cos={cos_dh_wdiff:.4f}, dh_norm={dh_norm:.1f}")

    # ★★★ 非线性回归: 搜索最佳映射 ★★★
    print(f"\n  === Nonlinear regression: search for best mapping ===")

    actual_arr = np.array([d["actual_dlogit"] for d in all_data])
    proj_arr = np.array([d["proj_dh_wdiff"] for d in all_data])
    cos_arr = np.array([d["cos_dh_wdiff"] for d in all_data])
    norm_arr = np.array([d["dh_norm"] for d in all_data])

    # 特征集
    features = {
        "proj": proj_arr,
        "cos": cos_arr,
        "norm": norm_arr,
        "proj_x_cos": proj_arr * cos_arr,
        "proj_x_norm": proj_arr * norm_arr,
        "cos_x_norm": cos_arr * norm_arr,
        "proj_sq": proj_arr ** 2,
        "norm_sq": norm_arr ** 2,
        "cos_sq": cos_arr ** 2,
        "proj_cubed": proj_arr ** 3,
        "sign_cos_x_proj": np.sign(cos_arr) * proj_arr,
        "abs_cos_x_proj": np.abs(cos_arr) * proj_arr,
    }

    def linear_r2(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return -999, 0, 0
        slope = np.dot(x - x.mean(), y - y.mean()) / max(np.sum((x - x.mean())**2), 1e-10)
        intercept = y.mean() - slope * x.mean()
        r2 = 1 - np.sum((y - slope * x - intercept)**2) / max(np.sum((y - y.mean())**2), 1e-10)
        return r2, slope, intercept

    # 单特征回归
    single_r2 = {}
    for fname, farr in features.items():
        r2, slope, intercept = linear_r2(farr, actual_arr)
        single_r2[fname] = {"r2": float(r2), "slope": float(slope)}
        print(f"  {fname}: R2={r2:.4f}")

    # 多特征回归 (前3个最佳特征)
    sorted_features = sorted(single_r2.items(), key=lambda x: x[1]["r2"], reverse=True)
    top3 = [f[0] for f in sorted_features[:3]]

    X_multi = np.column_stack([features[f] for f in top3] + [np.ones(len(actual_arr))])
    try:
        beta_multi = np.linalg.lstsq(X_multi, actual_arr, rcond=None)[0]
        pred_multi = X_multi @ beta_multi
        ss_res = np.sum((actual_arr - pred_multi)**2)
        ss_tot = np.sum((actual_arr - actual_arr.mean())**2)
        r2_multi = 1 - ss_res / max(ss_tot, 1e-10)
    except:
        r2_multi = -999
        beta_multi = [0] * len(top3)

    print(f"\n  Multi-feature R2 (top3: {top3}): {r2_multi:.4f}")
    for i, f in enumerate(top3):
        print(f"    {f}: coefficient={beta_multi[i]:.6f}")

    # ★★★ 关键分析: Δlogit vs β (注入强度) 的非线性 ★★★
    print(f"\n  === Delta_logit vs beta (injection strength) ===")
    for dim_name in test_dims:
        dim_data = [d for d in all_data if d["dim"] == dim_name]
        betas_dim = [d["beta"] for d in dim_data]
        dlogits = [d["actual_dlogit"] for d in dim_data]
        projs = [d["proj_dh_wdiff"] for d in dim_data]
        print(f"  {dim_name}: beta={betas_dim}, dlogit={[f'{d:.3f}' for d in dlogits]}, "
              f"proj={[f'{p:.3f}' for p in projs]}")

        # 线性度检测: dlogit / beta 是否常数?
        ratios = [d / b for d, b in zip(dlogits, betas_dim)]
        print(f"    dlogit/beta: {[f'{r:.4f}' for r in ratios]}")

    results = {
        "all_data": all_data,
        "single_r2": single_r2,
        "top3_features": top3,
        "r2_multi": float(r2_multi),
        "beta_multi": [float(b) for b in beta_multi],
    }
    return results


# ========== P401: LCS的精确矩阵估计 ==========

def run_p401(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P401: Precise LCS matrix estimation - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 4))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_info[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线hidden states
    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles = []
    for i, layer in enumerate(layers):
        if i in scan_layers:
            handles.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles:
        h.remove()

    # 收集所有维度的dh_per_layer
    dh_per_dim_per_layer = {}
    dim_names = list(dim_info.keys())

    for dim_name in dim_names:
        direction = dim_info[dim_name]["direction"]
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handles_int = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
        for h in handles_int:
            h.remove()

        dh_per_layer = {}
        for l in scan_layers:
            key = f"L{l}"
            if key in captured_int and key in captured_base:
                dh = (captured_int[key][0, -1, :] - captured_base[key][0, -1, :]).cpu().numpy()
                dh_per_layer[l] = dh
        dh_per_dim_per_layer[dim_name] = dh_per_layer

    # ★★★ R_rotate: 逐层旋转矩阵估计 ★★★
    # 对于每对相邻扫描层, 估计线性映射: dh_{l+1} ≈ A_l @ dh_l
    # 由于dh维度=hidden_dim(2560-4096)而样本数=4(维度数), 用低秩近似

    print(f"\n  === R_rotate: Layer-to-layer rotation matrix ===")

    rotation_results = {}
    for i in range(len(scan_layers) - 1):
        l1 = scan_layers[i]
        l2 = scan_layers[i + 1]

        # 构建 dh_l1 和 dh_l2 矩阵
        dh_l1_list = [dh_per_dim_per_layer[d][l1] for d in dim_names if l1 in dh_per_dim_per_layer[d]]
        dh_l2_list = [dh_per_dim_per_layer[d][l2] for d in dim_names if l2 in dh_per_dim_per_layer[d]]

        if len(dh_l1_list) < 2 or len(dh_l2_list) < 2:
            continue

        # 使用PCA降维到有效维度
        dh_l1_arr = np.array(dh_l1_list)  # [n_dims, hidden_dim]
        dh_l2_arr = np.array(dh_l2_list)

        # 计算cos(dh_l1, dh_l2)矩阵 — 维度间旋转
        n_d = len(dh_l1_list)
        cos_matrix = np.zeros((n_d, n_d))
        for j in range(n_d):
            for k in range(n_d):
                n1 = np.linalg.norm(dh_l1_arr[j])
                n2 = np.linalg.norm(dh_l2_arr[k])
                if n1 > 1e-8 and n2 > 1e-8:
                    cos_matrix[j, k] = float(np.dot(dh_l1_arr[j], dh_l2_arr[k]) / (n1 * n2))

        # 对角线元素 = cos(dh_l1_dim, dh_l2_dim) — 自身方向保持
        diag_cos = np.diag(cos_matrix)

        # 非对角线最大元素 = 交叉旋转最大值
        off_diag = cos_matrix.copy()
        np.fill_diagonal(off_diag, 0)
        max_cross = float(np.max(np.abs(off_diag)))

        rotation_results[f"L{l1}-L{l2}"] = {
            "diag_cos": [float(c) for c in diag_cos],
            "mean_diag_cos": float(np.mean(diag_cos)),
            "max_cross_rotation": max_cross,
            "layer_gap": l2 - l1,
        }

        if l1 % 8 == 0 or l1 == scan_layers[-2]:
            print(f"  L{l1}->L{l2}: mean_diag_cos={np.mean(diag_cos):.4f}, "
                  f"max_cross={max_cross:.4f}, diag={[f'{c:.3f}' for c in diag_cos]}")

    # ★★★ C_compete: 精确竞争矩阵 ★★★
    # 用联合注入数据估计: Δlogit_ij - Δlogit_i - Δlogit_j = interaction(i,j)

    print(f"\n  === C_compete: Precise competition matrix ===")

    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    compete_matrix = np.zeros((len(dim_names), len(dim_names)))

    for i, d1_name in enumerate(dim_names):
        for j, d2_name in enumerate(dim_names):
            if i >= j:
                continue

            d1_dir = dim_info[d1_name]["direction"]
            d2_dir = dim_info[d2_name]["direction"]

            # 联合注入
            w12_tensor = torch.tensor((d1_dir + d2_dir) * beta, dtype=torch.float32, device=device)
            inputs_embeds_12 = inputs_embeds_base.clone()
            inputs_embeds_12[0, -1, :] += w12_tensor.to(model.dtype)

            # 单独注入1
            w1_tensor = torch.tensor(d1_dir * beta, dtype=torch.float32, device=device)
            inputs_embeds_1 = inputs_embeds_base.clone()
            inputs_embeds_1[0, -1, :] += w1_tensor.to(model.dtype)

            # 单独注入2
            w2_tensor = torch.tensor(d2_dir * beta, dtype=torch.float32, device=device)
            inputs_embeds_2 = inputs_embeds_base.clone()
            inputs_embeds_2[0, -1, :] += w2_tensor.to(model.dtype)

            with torch.no_grad():
                logits_12 = model(inputs_embeds=inputs_embeds_12, position_ids=position_ids).logits[0, -1, :].float()
                logits_1 = model(inputs_embeds=inputs_embeds_1, position_ids=position_ids).logits[0, -1, :].float()
                logits_2 = model(inputs_embeds=inputs_embeds_2, position_ids=position_ids).logits[0, -1, :].float()

            # 交互 = logits_12 - logits_1 - logits_2 + logits_base
            interaction = (logits_12 - logits_1 - logits_2 + logits_base).cpu().numpy()
            interact_norm = float(np.linalg.norm(interaction))

            # 对dim1/dim2目标词的交互
            pos1_id = tokenizer.encode(dim_info[d1_name]["pos"], add_special_tokens=False)[0]
            neg1_id = tokenizer.encode(dim_info[d1_name]["neg"], add_special_tokens=False)[0]
            pos2_id = tokenizer.encode(dim_info[d2_name]["pos"], add_special_tokens=False)[0]
            neg2_id = tokenizer.encode(dim_info[d2_name]["neg"], add_special_tokens=False)[0]

            interact_d1 = float(interaction[pos1_id] - interaction[neg1_id])
            interact_d2 = float(interaction[pos2_id] - interaction[neg2_id])

            compete_matrix[i, j] = interact_d1
            compete_matrix[j, i] = interact_d2

            print(f"  {d1_name}-{d2_name}: interact_norm={interact_norm:.1f}, "
                  f"d1_interact={interact_d1:.3f}, d2_interact={interact_d2:.3f}")

    results = {
        "rotation_results": rotation_results,
        "compete_matrix": compete_matrix.tolist(),
        "dim_names": dim_names,
        "scan_layers": scan_layers,
    }
    return results


# ========== P402: LCS预测力验证 ==========

def run_p402(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P402: LCS predictive power validation - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()

    # 用quantity(many/few)作为测试维度 — P396-P398中没有用过
    test_dim = "quantity"
    pos, neg = DIM_PAIRS[test_dim][0]
    direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
    beta = 8.0

    w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
    w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
    w_diff = w_pos_raw - w_neg_raw
    w_diff_norm = float(np.linalg.norm(w_diff))
    pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]

    n_layers_total = len(get_layers(model))
    layers = get_layers(model)

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
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

    # L0注入获取实际Δlogit
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

    # ★★★ LCS预测 ★★★
    proj_dh_wdiff = float(np.dot(dh, w_diff))
    cos_dh_wdiff = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0.0

    # 模型1: Δlogit = k × proj(dh_final, W_diff) — P396发现的线性模型
    # k值从P396数据获取 (Qwen3=0.17, GLM4=1.0, DS7B需要非线性)
    # 这里用实验中的实际比例
    predicted_linear = proj_dh_wdiff  # 默认k=1

    # 模型2: Δlogit = k × cos × dh_norm × w_diff_norm
    predicted_cos_norm = cos_dh_wdiff * dh_norm * w_diff_norm

    # 验证: quantity维度与已知6维的正交性
    known_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    orthogonality = {}
    for known in known_dims:
        k_pos, k_neg = DIM_PAIRS[known][0]
        k_dir, k_norm = get_dimension_direction(model, tokenizer, k_pos, k_neg)
        cos_val = float(np.dot(direction, k_dir))
        orthogonality[known] = cos_val

    print(f"\n  === P402: LCS prediction for '{test_dim}' ({pos}/{neg}) ===")
    print(f"  actual_dlogit={actual_dlogit:.3f}")
    print(f"  proj_dh_wdiff={proj_dh_wdiff:.3f}")
    print(f"  cos_dh_wdiff={cos_dh_wdiff:.4f}")
    print(f"  dh_norm={dh_norm:.1f}")
    print(f"  predicted_linear(k=1)={predicted_linear:.3f}")
    print(f"  predicted_cos_norm={predicted_cos_norm:.3f}")
    print(f"  ratio_actual/proj={actual_dlogit/max(abs(proj_dh_wdiff),1e-6):.4f}")
    print(f"\n  Orthogonality with known dims:")
    for k, v in orthogonality.items():
        print(f"    cos(quantity, {k})={v:.4f}")

    # V_lang扩展: quantity是否构成第7维?
    all_dirs = []
    all_names = known_dims + [test_dim]
    for name in all_names:
        p, n = DIM_PAIRS[name][0]
        d, _ = get_dimension_direction(model, tokenizer, p, n)
        all_dirs.append(d)

    D_matrix = np.array(all_dirs)  # [7, hidden_dim]
    cos_all = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            cos_all[i, j] = float(np.dot(D_matrix[i], D_matrix[j]))

    U, S, Vt = np.linalg.svd(D_matrix, full_matrices=False)
    effective_rank_7 = int(np.sum(S > 0.1 * S[0]))

    print(f"\n  7-dim SVD: effective_rank={effective_rank_7}, singular_values={[f'{s:.3f}' for s in S]}")
    print(f"  7-dim cos matrix (quantity row):")
    for j in range(7):
        print(f"    cos(quantity, {all_names[j]})={cos_all[6, j]:.4f}")

    results = {
        "test_dim": test_dim,
        "actual_dlogit": actual_dlogit,
        "proj_dh_wdiff": proj_dh_wdiff,
        "cos_dh_wdiff": cos_dh_wdiff,
        "dh_norm": dh_norm,
        "ratio_actual_proj": actual_dlogit / max(abs(proj_dh_wdiff), 1e-10),
        "orthogonality": orthogonality,
        "effective_rank_7": effective_rank_7,
        "singular_values_7": [float(s) for s in S],
        "cos_matrix_7": cos_all.tolist(),
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p399", "p400", "p401", "p402", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p399", "p400", "p401", "p402"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p399":
                results["p399"] = run_p399(model, tokenizer, device, args.model)
            elif exp == "p400":
                results["p400"] = run_p400(model, tokenizer, device, args.model)
            elif exp == "p401":
                results["p401"] = run_p401(model, tokenizer, device, args.model)
            elif exp == "p402":
                results["p402"] = run_p402(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    # 保存结果
    out_file = OUT_DIR / f"phase_lxxx_p399_402_{args.model}_{timestamp}.json"

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

    # 清理GPU内存
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
