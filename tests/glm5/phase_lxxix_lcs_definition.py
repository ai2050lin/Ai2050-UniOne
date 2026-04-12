"""
Phase LXXIX-P396/397/398: 层间积分信号效率 + 修正激活修补 + 语言计算结构(LCS)定义
================================================================================

阶段C核心任务 - 定义语言计算结构(LCS)的数学形式:

P396: 层间积分信号效率指标 S_int
  - P393发现: 单层cos(dh, W_target)不预测delta_logit (R^2=0.04-0.24)
  - 核心假设: delta_logit是所有层累积效果的结果, 不是任何单层指标
  - 定义: S_int = sum_l [ proj(dh_l, W_diff) * alpha_l ]
    其中 alpha_l = ||dh_l|| / ||dh_0|| (信号增长因子)
    proj(dh_l, W_diff) = dot(dh_l, W_pos - W_neg) / ||W_pos - W_neg||
  - 预期: S_int应该高精度预测delta_logit!

P397: 修正激活修补 — 只修改最后一个token位置的delta_h
  - P394的缺陷: 替换了整个层输出, 效果等同于L0注入
  - 修正: 只在特定层添加delta_h到最后一个token位置, 其他位置保持基线
  - 这才能真正回答: "哪一层对哪个维度的信号最关键"

P398: 语言计算结构(LCS)的数学定义
  - 整合所有发现, 定义LCS为一个统一的数学结构
  - LCS = (V_lang, R_rotate, C_compete, M_map)
    V_lang: 8维正交语言空间 (P392)
    R_rotate: 层间信号旋转算子 (P393: cos从0.7→0.03)
    C_compete: 末层维度竞争算子 (P395: 交互范数在末层最大)
    M_map: 语言空间→词空间的映射 (lm_head)
  - 目标: 用LCS预测任意维度干预的输出效果

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


# ========== P396: 层间积分信号效率指标 ==========

def run_p396(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P396: Inter-layer integral signal efficiency S_int - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 2))
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    # 维度方向 + 目标词W_lm
    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw  # 不归一化, 保留原始尺度
        w_diff_norm = np.linalg.norm(w_diff)
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "w_pos_normed": w_pos_normed, "w_neg_normed": w_neg_normed,
            "w_diff": w_diff, "w_diff_norm": w_diff_norm,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线logits
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    # 基线hidden states
    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles_base = []
    for i, layer in enumerate(layers):
        if i in scan_layers:
            handles_base.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles_base:
        h.remove()

    # 对每个维度在L0注入, 跟踪层间积分
    integral_results = {}

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        pos_word = dim_info[dim_name]["pos"]
        neg_word = dim_info[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]

        # L0注入
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        # 捕获注入后各层hidden state
        captured_int = {}
        handles_int = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_int:
            h.remove()

        # 实际delta_logit
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        actual_delta_logit = delta_pos - delta_neg

        # 逐层分析: 计算积分信号效率
        layer_data = []
        cumsum_proj = 0.0
        cumsum_proj_weighted = 0.0

        dh_0 = None  # L0的delta_h作为基准

        for l in scan_layers:
            key = f"L{l}"
            if key not in captured_int or key not in captured_base:
                continue

            h_int = captured_int[key][0, -1, :].cpu().numpy()  # 最后一个token
            h_base = captured_base[key][0, -1, :].cpu().numpy()
            dh = h_int - h_base
            dh_norm = float(np.linalg.norm(dh))

            if l == scan_layers[0]:
                dh_0 = dh.copy()
                dh_0_norm = dh_norm
            else:
                dh_0_norm = dh_0_norm if dh_0 is not None else 1.0

            # proj(dh, W_diff) = dot(dh, W_diff) — 直接正比于delta_logit的理论预测
            proj_dh_wdiff = float(np.dot(dh, w_diff))

            # alpha_l = dh_norm / dh_0_norm (信号增长因子)
            alpha_l = dh_norm / max(dh_0_norm, 1e-8)

            # cos(dh, W_diff) — 方向对齐度
            if dh_norm > 1e-8 and w_diff_norm > 1e-8:
                cos_dh_wdiff = float(np.dot(dh, w_diff) / (dh_norm * w_diff_norm))
            else:
                cos_dh_wdiff = 0.0

            # 累积投影
            cumsum_proj += proj_dh_wdiff
            # 加权累积: 用alpha_l加权(信号增长越大的层权重越高)
            cumsum_proj_weighted += proj_dh_wdiff * alpha_l

            # 另一个积分: sum of cos(dh_l, W_diff) * dh_norm
            # 这衡量"每层有多少信号强度指向目标方向"
            signal_toward_target = cos_dh_wdiff * dh_norm

            # 残差连接贡献估计: 每层的delta_h中, 有多少来自残差连接(即前一层直接传递)
            if l > scan_layers[0] and dh_0 is not None:
                # 残差连接意味着 dh_l ≈ dh_{l-1} + mlp_contribution
                # proj(dh_l, dh_0) 衡量残差连接保留了多少L0信号
                cos_dh_dh0 = float(np.dot(dh, dh_0) / max(dh_norm * dh_0_norm, 1e-8))
                residual_preservation = cos_dh_dh0 * dh_norm / max(dh_0_norm, 1e-8)
            else:
                cos_dh_dh0 = 1.0
                residual_preservation = 1.0

            layer_data.append({
                "layer": l,
                "dh_norm": dh_norm,
                "proj_dh_wdiff": proj_dh_wdiff,
                "cos_dh_wdiff": cos_dh_wdiff,
                "alpha_l": alpha_l,
                "signal_toward_target": signal_toward_target,
                "cumsum_proj": cumsum_proj,
                "cumsum_proj_weighted": cumsum_proj_weighted,
                "cos_dh_dh0": cos_dh_dh0,
                "residual_preservation": residual_preservation,
            })

            if l % 10 == 0 or l == scan_layers[-1]:
                print(f"    L{l}: dh_norm={dh_norm:.1f}, proj={proj_dh_wdiff:.3f}, "
                      f"cos_dh_wdiff={cos_dh_wdiff:.4f}, alpha={alpha_l:.2f}, "
                      f"signal_toward={signal_toward_target:.2f}, "
                      f"cumsum={cumsum_proj:.2f}, cos_dh0={cos_dh_dh0:.3f}")

        # ===== 核心分析: 哪个指标最好地预测actual_delta_logit? =====
        # 1. 简单累积投影 cumsum_proj
        # 2. 加权累积投影 cumsum_proj_weighted
        # 3. 末层投影 proj_dh_wdiff (末层)
        # 4. 末层 proj + alpha (末层投影 × 增长因子)
        last_layer_data = layer_data[-1] if layer_data else {}

        integral_results[dim_name] = {
            "actual_delta_logit": actual_delta_logit,
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "layer_data": layer_data,
            "cumsum_proj_final": cumsum_proj,
            "cumsum_proj_weighted_final": cumsum_proj_weighted,
            "last_proj": last_layer_data.get("proj_dh_wdiff", 0),
            "last_cos_dh_wdiff": last_layer_data.get("cos_dh_wdiff", 0),
            "last_dh_norm": last_layer_data.get("dh_norm", 0),
            "last_alpha": last_layer_data.get("alpha_l", 0),
        }

        print(f"  {dim_name}: actual_dlogit={actual_delta_logit:.3f}, "
              f"cumsum_proj={cumsum_proj:.2f}, cumsum_w={cumsum_proj_weighted:.2f}, "
              f"last_proj={last_layer_data.get('proj_dh_wdiff', 0):.2f}")

    # ===== 跨维度回归: 哪个指标最准确预测delta_logit? =====
    print(f"\n  === Cross-dimension regression: Which metric best predicts delta_logit? ===")

    actual = np.array([integral_results[d]["actual_delta_logit"] for d in dim_info])
    cumsum_proj_arr = np.array([integral_results[d]["cumsum_proj_final"] for d in dim_info])
    cumsum_w_arr = np.array([integral_results[d]["cumsum_proj_weighted_final"] for d in dim_info])
    last_proj_arr = np.array([integral_results[d]["last_proj"] for d in dim_info])
    last_cos_arr = np.array([integral_results[d]["last_cos_dh_wdiff"] for d in dim_info])

    def linear_r2(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return -999, 0, 0
        slope = np.dot(x - x.mean(), y - y.mean()) / max(np.sum((x - x.mean())**2), 1e-10)
        intercept = y.mean() - slope * x.mean()
        r2 = 1 - np.sum((y - slope * x - intercept)**2) / max(np.sum((y - y.mean())**2), 1e-10)
        return r2, slope, intercept

    metrics = {
        "cumsum_proj": cumsum_proj_arr,
        "cumsum_proj_weighted": cumsum_w_arr,
        "last_proj": last_proj_arr,
        "last_cos_dh_wdiff": last_cos_arr,
    }

    metric_r2 = {}
    for mname, marr in metrics.items():
        r2, slope, intercept = linear_r2(marr, actual)
        metric_r2[mname] = {"r2": float(r2), "slope": float(slope), "intercept": float(intercept)}
        print(f"  {mname}: R2={r2:.4f}, slope={slope:.4f}")

    # ===== 理论预测: delta_logit = proj(dh_final, W_diff) =====
    # 这是线性近似: delta_logit = (logits_int - logits_base) = W_lm * (h_int - h_base)
    # 但实际上 lm_head 可能不是线性映射, 或者 h_int - h_base 不是最后一个token
    print(f"\n  === Theoretical check: delta_logit vs proj(dh_final, W_diff) ===")
    for d in dim_info:
        r = integral_results[d]
        print(f"  {d}: actual={r['actual_delta_logit']:.3f}, "
              f"last_proj={r['last_proj']:.3f}, "
              f"ratio={r['actual_delta_logit']/max(abs(r['last_proj']), 1e-6):.3f}")

    results = {
        "integral_results": integral_results,
        "metric_r2": metric_r2,
        "scan_layers": scan_layers,
        "beta": beta,
    }
    return results


# ========== P397: 修正激活修补 ==========

def run_p397(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P397: Corrected activation patching (delta_h only) - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 4))  # 每4层扫描一次
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    # 维度方向
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

    # L0注入的完整delta_h (用于在各层添加)
    # 先跑一次L0注入, 获取每层的delta_h
    captured_base_all = {}
    captured_int_all = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    # 基线所有层
    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(captured_base_all, f"L{i}")))
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
    for h in handles:
        h.remove()

    # 对每个维度, 计算delta_h在每层的值
    patching_results = {}

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        pos_word = dim_info[dim_name]["pos"]
        neg_word = dim_info[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        # L0注入
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        # 捕获注入后所有层
        captured_int_all = {}
        handles_int = []
        for i, layer in enumerate(layers):
            handles_int.append(layer.register_forward_hook(make_hook(captured_int_all, f"L{i}")))
        with torch.no_grad():
            logits_int_l0 = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_int:
            h.remove()

        # L0注入的delta_logit
        delta_pos_l0 = float(logits_int_l0[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg_l0 = float(logits_int_l0[neg_id].cpu() - logits_base[neg_id].cpu())
        dlogit_l0 = delta_pos_l0 - delta_neg_l0

        # 计算每层的delta_h (只在最后一个token位置)
        delta_h_per_layer = {}
        for i in range(len(layers)):
            key = f"L{i}"
            if key in captured_int_all and key in captured_base_all:
                h_int = captured_int_all[key]  # [seq_len, hidden_dim]
                h_base = captured_base_all[key]
                # 只取最后一个token的delta_h
                dh = (h_int[0, -1, :] - h_base[0, -1, :]).cpu().numpy()
                delta_h_per_layer[i] = dh

        # ===== 修正激活修补: 在patch_layer添加delta_h, 而不是替换整个层输出 =====
        print(f"\n  --- Dimension: {dim_name} (L0 dlogit={dlogit_l0:.3f}) ---")

        layer_effects = []

        for patch_layer in scan_layers:
            if patch_layer not in delta_h_per_layer:
                continue

            dh_to_add = delta_h_per_layer[patch_layer]
            dh_tensor = torch.tensor(dh_to_add, dtype=torch.float32, device=device)

            # 用hook在patch_layer添加delta_h到最后一个token
            def add_dh_hook(module, input, output, dh=dh_tensor, pl=patch_layer):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    # 只修改最后一个token位置
                    h[0, -1, :] += dh.to(h.dtype)
                    return (h.to(output[0].dtype),) + output[1:]
                else:
                    h = output.clone()
                    h[0, -1, :] += dh.to(h.dtype)
                    return h.to(output.dtype)

            handle = layers[patch_layer].register_forward_hook(add_dh_hook)
            with torch.no_grad():
                logits_patched = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
            handle.remove()

            delta_pos = float(logits_patched[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_patched[neg_id].cpu() - logits_base[neg_id].cpu())
            dlogit_patched = delta_pos - delta_neg

            # 与L0注入效果的比值
            ratio = dlogit_patched / max(abs(dlogit_l0), 1e-6)

            layer_effects.append({
                "patch_layer": patch_layer,
                "dlogit_patched": dlogit_patched,
                "dlogit_l0": dlogit_l0,
                "ratio_to_l0": ratio,
            })

            if patch_layer % 10 == 0 or patch_layer == scan_layers[-1]:
                print(f"    Patch@L{patch_layer}: dlogit={dlogit_patched:.3f} "
                      f"(ratio to L0={ratio:.3f})")

        # 找到"关键层": patch效果最大的层
        if layer_effects:
            max_effect = max(layer_effects, key=lambda x: abs(x["dlogit_patched"]))
            # 找到ratio>0.5的最早层 (信号超过L0效果50%的最早层)
            significant_layers = [e for e in layer_effects if abs(e["ratio_to_l0"]) > 0.5]
            first_significant = significant_layers[0]["patch_layer"] if significant_layers else -1
        else:
            max_effect = {"patch_layer": -1, "dlogit_patched": 0, "ratio_to_l0": 0}
            first_significant = -1

        patching_results[dim_name] = {
            "layer_effects": layer_effects,
            "dlogit_l0": dlogit_l0,
            "max_effect_layer": max_effect["patch_layer"],
            "max_effect_dlogit": max_effect["dlogit_patched"],
            "first_significant_layer": first_significant,
        }

        print(f"  -> Key layer: L{max_effect['patch_layer']} (dlogit={max_effect['dlogit_patched']:.3f}), "
              f"first >50% at L{first_significant}")

    results = {
        "patching_results": patching_results,
        "scan_layers": scan_layers,
        "beta": beta,
    }
    return results


# ========== P398: 语言计算结构(LCS)的数学定义 ==========

def run_p398(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P398: Language Computation Structure (LCS) definition - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 4))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    # 维度方向
    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_normed, w_pos_raw = get_w_lm(model, tokenizer, pos)
        w_neg_normed, w_neg_raw = get_w_lm(model, tokenizer, neg)
        w_diff = w_pos_raw - w_neg_raw
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "w_pos_normed": w_pos_normed, "w_neg_normed": w_neg_normed,
            "w_diff": w_diff,
        }

    # ===== V_lang: 8维正交语言空间验证 =====
    # 计算所有维度在W_lm空间中的cos矩阵
    dim_names = list(dim_info.keys())
    n_dims = len(dim_names)

    # 获取维度方向矩阵 D (n_dims x hidden_dim)
    D = np.array([dim_info[d]["direction"] for d in dim_names])

    # 获取W_lm矩阵 W (n_dims x hidden_dim)
    W = np.array([dim_info[d]["w_diff"] / max(np.linalg.norm(dim_info[d]["w_diff"]), 1e-8) for d in dim_names])

    # cos矩阵: cos(D_i, D_j) — 维度方向间的cos
    cos_DD = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            cos_DD[i, j] = float(np.dot(D[i], D[j]) / max(np.linalg.norm(D[i]) * np.linalg.norm(D[j]), 1e-8))

    # cos矩阵: cos(W_i, W_j) — W_lm方向间的cos
    cos_WW = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            cos_WW[i, j] = float(np.dot(W[i], W[j]))

    # SVD of W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    effective_rank = int(np.sum(S > 0.1 * S[0]))

    print(f"\n  === V_lang: Orthogonal language space ===")
    print(f"  cos(D,D) matrix (dim direction cos):")
    for i in range(n_dims):
        row = " ".join([f"{cos_DD[i,j]:.3f}" for j in range(n_dims)])
        print(f"    {dim_names[i]}: [{row}]")
    print(f"  cos(W,W) matrix (W_lm direction cos):")
    for i in range(n_dims):
        row = " ".join([f"{cos_WW[i,j]:.3f}" for j in range(n_dims)])
        print(f"    {dim_names[i]}: [{row}]")
    print(f"  SVD of W: singular values = {[f'{s:.3f}' for s in S]}")
    print(f"  Effective rank (S > 0.1*S[0]): {effective_rank}")

    # ===== R_rotate: 层间信号旋转算子 =====
    # 测量每层的旋转矩阵 R_l, 使得 h_{l+1} = R_l @ h_l + ...
    # 但直接估计R_l太困难, 改用cos衰减曲线的参数化模型

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

    handles_base = []
    for i, layer in enumerate(layers):
        if i in scan_layers:
            handles_base.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles_base:
        h.remove()

    # 对2个核心维度(style, grammar)做L0注入, 跟踪cos(dh, dh_0)衰减
    rotation_data = {}
    for dim_name in ["style", "grammar"]:
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

        # 计算每层cos(dh_l, dh_0) — 信号方向随层数的旋转
        cos_decay = []
        dh_0 = None
        dh_0_norm = 0
        for l in scan_layers:
            key = f"L{l}"
            if key not in captured_int or key not in captured_base:
                continue
            dh = (captured_int[key][0, -1, :] - captured_base[key][0, -1, :]).cpu().numpy()
            dh_norm = float(np.linalg.norm(dh))
            if dh_0 is None:
                dh_0 = dh
                dh_0_norm = dh_norm
                cos_decay.append(1.0)
            else:
                if dh_norm > 1e-8 and dh_0_norm > 1e-8:
                    cos_val = float(np.dot(dh, dh_0) / (dh_norm * dh_0_norm))
                else:
                    cos_val = 0.0
                cos_decay.append(cos_val)

        # 拟合指数衰减: cos(l) = exp(-gamma * l)
        layers_arr = np.array(scan_layers[:len(cos_decay)], dtype=float)
        cos_arr = np.clip(np.array(cos_decay), 0.001, 1.0)
        log_cos = np.log(cos_arr)
        if np.std(layers_arr) > 1e-8:
            gamma = -np.dot(layers_arr - layers_arr.mean(), log_cos - log_cos.mean()) / max(np.sum((layers_arr - layers_arr.mean())**2), 1e-10)
        else:
            gamma = 0

        rotation_data[dim_name] = {
            "cos_decay": dict(zip([str(l) for l in scan_layers[:len(cos_decay)]], [float(c) for c in cos_decay])),
            "gamma": float(gamma),
            "half_life": float(np.log(2) / max(abs(gamma), 1e-10)),
        }

        print(f"  {dim_name}: gamma={gamma:.4f}, half_life={np.log(2)/max(abs(gamma), 1e-10):.1f} layers, "
              f"cos_decay=[{', '.join([f'{c:.3f}' for c in cos_decay[:6]])}...]")

    # ===== C_compete: 末层维度竞争算子 =====
    # 用2个维度对(style-grammar, logic-sentiment)做竞争分析
    # 先计算基线logits
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    compete_data = {}
    for dim1_name, dim2_name in [("style", "grammar"), ("logic", "sentiment")]:
        d1 = dim_info[dim1_name]["direction"]
        d2 = dim_info[dim2_name]["direction"]

        # 联合注入
        w12_tensor = torch.tensor((d1 + d2) * beta, dtype=torch.float32, device=device)
        inputs_embeds_12 = inputs_embeds_base.clone()
        inputs_embeds_12[0, -1, :] += w12_tensor.to(model.dtype)

        with torch.no_grad():
            logits_12 = model(inputs_embeds=inputs_embeds_12, position_ids=position_ids).logits[0, -1, :].float()

        # 单独注入1
        w1_tensor = torch.tensor(d1 * beta, dtype=torch.float32, device=device)
        inputs_embeds_1 = inputs_embeds_base.clone()
        inputs_embeds_1[0, -1, :] += w1_tensor.to(model.dtype)
        with torch.no_grad():
            logits_1 = model(inputs_embeds=inputs_embeds_1, position_ids=position_ids).logits[0, -1, :].float()

        # 单独注入2
        w2_tensor = torch.tensor(d2 * beta, dtype=torch.float32, device=device)
        inputs_embeds_2 = inputs_embeds_base.clone()
        inputs_embeds_2[0, -1, :] += w2_tensor.to(model.dtype)
        with torch.no_grad():
            logits_2 = model(inputs_embeds=inputs_embeds_2, position_ids=position_ids).logits[0, -1, :].float()

        # 计算交互: (logits_12 - logits_base) - (logits_1 - logits_base) - (logits_2 - logits_base)
        # = logits_12 - logits_1 - logits_2 + logits_base
        interaction = (logits_12 - logits_1 - logits_2 + logits_base).cpu().numpy()
        interaction_norm = float(np.linalg.norm(interaction))

        # 对dim1/dim2的目标词交互
        pos1_id = tokenizer.encode(dim_info[dim1_name]["pos"], add_special_tokens=False)[0]
        neg1_id = tokenizer.encode(dim_info[dim1_name]["neg"], add_special_tokens=False)[0]
        pos2_id = tokenizer.encode(dim_info[dim2_name]["pos"], add_special_tokens=False)[0]
        neg2_id = tokenizer.encode(dim_info[dim2_name]["neg"], add_special_tokens=False)[0]

        interact_d1 = float(interaction[pos1_id] - interaction[neg1_id])
        interact_d2 = float(interaction[pos2_id] - interaction[neg2_id])

        compete_data[f"{dim1_name}-{dim2_name}"] = {
            "interaction_norm": interaction_norm,
            "interact_dim1": interact_d1,
            "interact_dim2": interact_d2,
            "is_suppressive_d1": interact_d1 < 0,
            "is_suppressive_d2": interact_d2 < 0,
        }

        print(f"  {dim1_name}-{dim2_name}: interaction_norm={interaction_norm:.2f}, "
              f"interact_d1={interact_d1:.3f}({'suppress' if interact_d1 < 0 else 'enhance'}), "
              f"interact_d2={interact_d2:.3f}({'suppress' if interact_d2 < 0 else 'enhance'})")

    # ===== M_map: 语言空间→词空间的映射 (lm_head) =====
    # 分析lm_head的W矩阵特性
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()  # [vocab_size, hidden_dim]
    vocab_size, hidden_dim = W_lm.shape

    # SVD of W_lm (取前100个奇异值)
    print(f"\n  === M_map: lm_head mapping analysis ===")
    print(f"  W_lm shape: {vocab_size} x {hidden_dim}")

    # 采样1000个高频词的W_lm行做SVD (避免全词表太大)
    sample_size = min(1000, vocab_size)
    W_sample = W_lm[:sample_size]
    U_lm, S_lm, Vt_lm = np.linalg.svd(W_sample, full_matrices=False)

    print(f"  Top 20 singular values of W_lm: {[f'{s:.2f}' for s in S_lm[:20]]}")
    print(f"  Condition number (S[0]/S[-1]): {S_lm[0]/max(S_lm[-1], 1e-10):.1f}")

    # 维度方向在W_lm SVD空间中的投影
    print(f"\n  Dim directions in W_lm SVD space:")
    for d in dim_names:
        w_d = dim_info[d]["w_diff"]
        proj = np.array([float(np.dot(w_d, Vt_lm[k])) for k in range(min(10, len(Vt_lm)))])
        top_k = int(np.argmax(np.abs(proj)))
        print(f"    {d}: top SVD comp={top_k}, projection={proj[top_k]:.3f}, "
              f"energy in top 10={float(np.sum(proj**2)/max(np.sum(w_d**2), 1e-10)):.3f}")

    # ===== LCS 统一定义 =====
    print(f"\n  === LCS = (V_lang, R_rotate, C_compete, M_map) ===")

    lcs = {
        "V_lang": {
            "n_dims": n_dims,
            "effective_rank": int(effective_rank),
            "singular_values": [float(s) for s in S],
            "avg_cos_between_dims": float(np.mean(np.abs(cos_WW[np.triu_indices(n_dims, k=1)]))),
            "max_cos_between_dims": float(np.max(np.abs(cos_WW[np.triu_indices(n_dims, k=1)]))),
            "dim_names": dim_names,
        },
        "R_rotate": {
            dim_name: {
                "gamma": rotation_data[dim_name]["gamma"],
                "half_life_layers": rotation_data[dim_name]["half_life"],
                "cos_at_mid_layer": rotation_data[dim_name]["cos_decay"].get(str(scan_layers[len(scan_layers)//2]), 0),
                "cos_at_last_layer": rotation_data[dim_name]["cos_decay"].get(str(scan_layers[-1]), 0),
            }
            for dim_name in rotation_data
        },
        "C_compete": compete_data,
        "M_map": {
            "W_lm_shape": [int(vocab_size), int(hidden_dim)],
            "top_singular_values": [float(s) for s in S_lm[:20]],
            "condition_number": float(S_lm[0] / max(S_lm[-1], 1e-10)),
        },
        "model": model_name,
        "n_layers": n_layers_total,
    }

    print(f"  V_lang: {n_dims} dims, effective_rank={effective_rank}, avg_cos={lcs['V_lang']['avg_cos_between_dims']:.4f}")
    print(f"  R_rotate: gamma={[f'{rotation_data[d]['gamma']:.4f}' for d in rotation_data]}, "
          f"half_life={[f'{rotation_data[d]['half_life']:.1f}' for d in rotation_data]}")
    compete_strs = [f"{k}={v['interaction_norm']:.1f}" for k, v in compete_data.items()]
    print(f"  C_compete: {compete_strs}")
    print(f"  M_map: condition_number={lcs['M_map']['condition_number']:.1f}")

    results = {
        "lcs": lcs,
        "cos_DD": cos_DD.tolist(),
        "cos_WW": cos_WW.tolist(),
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p396", "p397", "p398", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p396", "p397", "p398"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p396":
                results["p396"] = run_p396(model, tokenizer, device, args.model)
            elif exp == "p397":
                results["p397"] = run_p397(model, tokenizer, device, args.model)
            elif exp == "p398":
                results["p398"] = run_p398(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    # 保存结果
    out_file = OUT_DIR / f"phase_lxxix_p396_398_{args.model}_{timestamp}.json"

    # 转换numpy为python类型
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
