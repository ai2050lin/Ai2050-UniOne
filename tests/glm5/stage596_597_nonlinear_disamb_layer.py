#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage596_597: hidden→logit非线性映射机制解析 + 消歧信息逐层生成追踪
合并运行以减少GPU加载次数。

Stage596目标：
  1. 精确定位hidden→logit非线性来源（RMSNorm/bias/其他）
  2. 逐层对比 h@U.T vs actual_logits，寻找cosine转折点
  3. 测试手动施加RMSNorm后线性度是否恢复

Stage597目标：
  1. 对歧义词对，逐层计算消歧度 1-cos(ctx1, ctx2)
  2. 检测"涌现"层——消歧度从0开始显著增长的层
  3. 测量消歧信息的逐层增长率（信息论角度）
"""

from __future__ import annotations
import sys, json, time, gc, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, move_batch_to_model_device, get_model_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_unembed_matrix(model):
    """获取unembedding矩阵（确保float32）"""
    w = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        w = model.lm_head.weight.data
    elif hasattr(model, 'get_output_embeddings'):
        oe = model.get_output_embeddings()
        if oe is not None:
            w = oe.weight.data
    if w is not None:
        return w.float().cpu()
    return None


def get_final_norm(model):
    """获取模型最终的normalization层（RMSNorm/LayerNorm）"""
    # 尝试常见的位置
    for attr_name in ['model', 'transformer']:
        base = getattr(model, attr_name, None)
        if base is None:
            continue
        for norm_name in ['norm', 'final_norm', 'final_layernorm', 'ln_f']:
            norm = getattr(base, norm_name, None)
            if norm is not None:
                return norm
    # Gemma4可能在顶层
    for norm_name in ['norm', 'final_norm', 'final_layernorm', 'ln_f']:
        norm = getattr(model, norm_name, None)
        if norm is not None:
            return norm
    return None


def rms_norm_manual(x, weight, eps=1e-6):
    """手动实现RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms) * weight.float()


def layer_norm_manual(x, weight, bias, eps=1e-5):
    """手动实现LayerNorm"""
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    xn = (x.float() - mean) / torch.sqrt(var + eps)
    return xn * weight.float() + bias.float()


def run_stage596(model, tokenizer, model_key):
    """Stage596: hidden→logit非线性映射机制解析"""
    print(f"\n  --- Stage596: hidden→logit非线性映射 ---")
    t0 = time.time()

    unembed = get_unembed_matrix(model)  # already float32
    final_norm = get_final_norm(model)
    # unembed在CPU上，所有计算在CPU

    print(f"  unembed: {unembed.shape}, dtype: {unembed.dtype}")
    print(f"  final_norm: {type(final_norm).__name__ if final_norm else 'NOT FOUND'}")
    if final_norm is not None:
        print(f"  norm attrs: {[(a, type(getattr(final_norm, a)).__name__) for a in dir(final_norm) if not a.startswith('_') and hasattr(getattr(final_norm, a), 'shape')]}")

    sentences = [
        "The cat sat on the mat.",
        "Mathematics is the language of science.",
        "The capital of France is Paris.",
    ]

    results = {
        "norm_type": type(final_norm).__name__ if final_norm else "none",
        "per_layer": {},
        "norm_intervention": {},
    }

    # === 逐层分析hidden→logit线性度 ===
    for sent in sentences:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            actual_logits = out.logits[0, -1, :].float().cpu()

        for li, h in enumerate(out.hidden_states):
            hv = h[0, -1, :].float().cpu()

            # 原始: h @ U.T
            raw_logits = hv @ unembed.T
            raw_cos = cos(actual_logits, raw_logits)

            # 手动RMSNorm后: norm(h) @ U.T
            if final_norm is not None:
                try:
                    normed = apply_norm(final_norm, hv)
                    normed_logits = normed @ unembed.T
                    normed_cos = cos(actual_logits, normed_logits)
                except Exception as e:
                    normed_cos = float('nan')
            else:
                normed_cos = float('nan')

            li_str = str(li)
            if li_str not in results["per_layer"]:
                results["per_layer"][li_str] = {"raw_cos": [], "normed_cos": []}
            results["per_layer"][li_str]["raw_cos"].append(raw_cos)
            results["per_layer"][li_str]["normed_cos"].append(normed_cos)

    # 汇总逐层数据
    layer_summary = {}
    for li_str in sorted(results["per_layer"].keys(), key=int):
        raw_vals = results["per_layer"][li_str]["raw_cos"]
        norm_vals = results["per_layer"][li_str]["normed_cos"]
        layer_summary[li_str] = {
            "raw_cos_mean": round(np.mean(raw_vals), 6),
            "raw_cos_min": round(np.min(raw_vals), 6),
            "normed_cos_mean": round(np.nanmean(norm_vals), 6) if any(not np.isnan(v) for v in norm_vals) else "N/A",
            "normed_cos_min": round(np.nanmin(norm_vals), 6) if any(not np.isnan(v) for v in norm_vals) else "N/A",
        }

    # === 精确干预实验：定位非线性来源 ===
    print(f"  [干预实验] 定位非线性来源:")
    test_sent = "The cat sat on the mat."
    enc = tokenizer(test_sent, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        last_h = out.hidden_states[-1][0, -1, :].float()
        actual_logits = out.logits[0, -1, :].float().cpu()

    # 测试各种变换的线性度
    interventions = {}

    # 1. 原始hidden state
    hv = last_h.cpu()
    logits_raw = hv @ unembed.T
    interventions["raw_h@U"] = cos(actual_logits, logits_raw)

    # 2. 模型内置norm(hidden) @ U
    if final_norm is not None:
        with torch.no_grad():
            try:
                normed_builtin = final_norm(last_h.unsqueeze(0).to(get_model_device(model))).squeeze(0)
                logits_builtin_norm = normed_builtin.float().cpu() @ unembed.T
                interventions["model_norm(h)@U"] = cos(actual_logits, logits_builtin_norm)
            except Exception as e:
                interventions["model_norm(h)@U"] = f"error: {e}"

    # 3. 手动RMSNorm(hidden) @ U
    if final_norm is not None:
        try:
            normed_manual = apply_norm(final_norm, hv)
            logits_manual_norm = normed_manual @ unembed.T
            interventions["manual_rmsnorm(h)@U"] = cos(actual_logits, logits_manual_norm)
        except Exception as e:
            interventions["manual_rmsnorm(h)@U"] = f"error: {e}"

    # 4. 检查unembed是否有bias
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is not None:
        has_bias = hasattr(lm_head, 'bias') and lm_head.bias is not None
        interventions["lm_head_has_bias"] = has_bias
        if has_bias:
            logits_with_bias = hv @ unembed.T + lm_head.bias.float().cpu()
            interventions["raw_h@U+bias"] = cos(actual_logits, logits_with_bias)

    # 5. norm(h) @ U + bias
    if final_norm is not None and lm_head is not None and hasattr(lm_head, 'bias') and lm_head.bias is not None:
        try:
            normed = apply_norm(final_norm, hv)
            logits_full = normed @ unembed.T + lm_head.bias.float().cpu()
            interventions["norm(h)@U+bias"] = cos(actual_logits, logits_full)
        except Exception as e:
            interventions["norm(h)@U+bias"] = f"error: {e}"

    # 6. 检查 hidden_states[-1] vs model内部norm后的差异
    # 可能hidden_states已经过了norm
    with torch.no_grad():
        out2 = model(**enc, output_hidden_states=True)
        h_last = out2.hidden_states[-1][0, -1, :].float()
        # 检查h_last是否已经被norm了（通过比较RMS）
        h_rms = torch.sqrt(torch.mean(h_last ** 2)).item()
        interventions["last_h_RMS"] = round(h_rms, 4)

    results["norm_intervention"] = {k: round(v, 8) if isinstance(v, float) else v for k, v in interventions.items()}
    results["layer_summary"] = layer_summary

    for k, v in interventions.items():
        if isinstance(v, float):
            print(f"    {k}: cos={v:.8f}")
        else:
            print(f"    {k}: {v}")

    elapsed = time.time() - t0
    print(f"  Stage596 time: {elapsed:.1f}s")
    results["elapsed_s"] = round(elapsed, 1)
    return results


def apply_norm(norm_layer, x):
    """对输入x应用normalization层（兼容不同类型）"""
    norm_type = type(norm_layer).__name__.lower()

    if 'rmsnorm' in norm_type:
        eps = getattr(norm_layer, 'eps', 1e-6)
        weight = norm_layer.weight
        return rms_norm_manual(x, weight, eps)
    elif 'layernorm' in norm_type:
        eps = getattr(norm_layer, 'eps', 1e-5) if hasattr(norm_layer, 'eps') else getattr(norm_layer, 'elementwise_affine', True)
        weight = norm_layer.weight
        bias = norm_layer.bias if hasattr(norm_layer, 'bias') else torch.zeros_like(weight)
        return layer_norm_manual(x, weight, bias)
    else:
        # 尝试直接调用
        with torch.no_grad():
            return norm_layer(x.unsqueeze(0)).squeeze(0).float()


def run_stage597(model, tokenizer, model_key):
    """Stage597: 消歧信息逐层生成追踪"""
    print(f"\n  --- Stage597: 消歧信息逐层生成 ---")
    t0 = time.time()

    layers = discover_layers(model)
    n_layers = len(layers)

    # 歧义词对
    disamb_pairs = [
        ("The river bank was muddy.", "The bank approved the loan.", "bank"),
        ("She ate a red apple.", "Apple released the iPhone.", "apple"),
        ("The factory plant employs workers.", "She watered the plant.", "plant"),
        ("The hot spring resort.", "Spring is beautiful.", "spring"),
        ("He hit the nail with a hammer.", "She painted her fingernail.", "nail"),
    ]

    results = {"per_word": {}, "summary": {}}

    all_layer_disamb = {i: [] for i in range(n_layers)}

    for s1, s2, word in disamb_pairs:
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        enc1 = move_batch_to_model_device(model, enc1)
        enc2 = move_batch_to_model_device(model, enc2)

        with torch.no_grad():
            out1 = model(**enc1, output_hidden_states=True)
            out2 = model(**enc2, output_hidden_states=True)

        per_layer_disamb = []
        for li in range(n_layers):
            h1 = out1.hidden_states[li][0, -1, :].float().cpu()
            h2 = out2.hidden_states[li][0, -1, :].float().cpu()
            disamb = 1 - cos(h1, h2)
            per_layer_disamb.append(disamb)
            all_layer_disamb[li].append(disamb)

        results["per_word"][word] = [round(d, 6) for d in per_layer_disamb]
        print(f"    {word}: peak_L{np.argmax(per_layer_disamb)}, peak={max(per_layer_disamb):.4f}, "
              f"L0={per_layer_disamb[0]:.6f}, Lf={per_layer_disamb[-1]:.4f}")

    # 汇总统计
    mean_disamb = []
    for li in range(n_layers):
        mean_d = np.mean(all_layer_disamb[li])
        mean_disamb.append(mean_d)

    # 1. 涌现层检测：消歧度首次超过0.01的层
    emergence_layers = []
    for li in range(n_layers):
        if mean_disamb[li] > 0.005:
            emergence_layers.append(li)
            break
    if not emergence_layers:
        for li in range(n_layers):
            if mean_disamb[li] > 0.001:
                emergence_layers.append(li)
                break
    first_emergence = emergence_layers[0] if emergence_layers else -1

    # 2. 最大增长率层
    growth_rates = []
    for li in range(1, n_layers):
        growth_rates.append(mean_disamb[li] - mean_disamb[li - 1])
    max_growth_layer = np.argmax(growth_rates) + 1 if growth_rates else 0
    max_growth_val = growth_rates[max_growth_layer - 1] if growth_rates else 0

    # 3. 累积消歧信息
    cumulative = np.cumsum(growth_rates)

    # 4. 信息论分析：每层的信息增益（用消歧度变化近似）
    info_gains = [max(0, g) for g in growth_rates]  # 只计正增益
    total_info = sum(info_gains) if sum(info_gains) > 0 else 1e-8

    # 5. 前N层贡献的消歧比例
    contrib_25pct = cumulative[int(n_layers * 0.25) - 1] / (cumulative[-1] + 1e-8)
    contrib_50pct = cumulative[int(n_layers * 0.5) - 1] / (cumulative[-1] + 1e-8)
    contrib_75pct = cumulative[int(n_layers * 0.75) - 1] / (cumulative[-1] + 1e-8)

    results["summary"] = {
        "n_layers": n_layers,
        "first_emergence_layer": first_emergence,
        "first_emergence_disamb": round(mean_disamb[first_emergence], 6) if first_emergence >= 0 else 0,
        "max_growth_layer": int(max_growth_layer),
        "max_growth_value": round(max_growth_val, 6),
        "peak_disamb_layer": int(np.argmax(mean_disamb)),
        "peak_disamb_value": round(max(mean_disamb), 6),
        "final_disamb": round(mean_disamb[-1], 6),
        "contrib_25pct_layers": round(contrib_25pct, 4),
        "contrib_50pct_layers": round(contrib_50pct, 4),
        "contrib_75pct_layers": round(contrib_75pct, 4),
        "per_layer_mean_disamb": [round(d, 6) for d in mean_disamb],
        "per_layer_growth_rate": [round(g, 6) for g in growth_rates],
        "positive_info_layers": sum(1 for g in growth_rates if g > 0.001),
        "negative_info_layers": sum(1 for g in growth_rates if g < -0.001),
    }

    print(f"\n    汇总:")
    print(f"      首次涌现层: L{first_emergence} (disamb={mean_disamb[first_emergence]:.6f})")
    print(f"      最大增长率层: L{max_growth_layer} (growth={max_growth_val:.6f})")
    print(f"      峰值消歧层: L{np.argmax(mean_disamb)} (disamb={max(mean_disamb):.4f})")
    print(f"      末层消歧度: {mean_disamb[-1]:.4f}")
    print(f"      前25%层贡献: {contrib_25pct*100:.1f}%")
    print(f"      前50%层贡献: {contrib_50pct*100:.1f}%")
    print(f"      正增益层数: {results['summary']['positive_info_layers']}")
    print(f"      负增益层数: {results['summary']['negative_info_layers']}")

    # 消歧度曲线
    print(f"      逐层消歧度: ", end="")
    for d in mean_disamb:
        bar = int(d * 200)
        print(f"{bar * '#'}", end=" ")
    print()

    elapsed = time.time() - t0
    print(f"    Stage597 time: {elapsed:.1f}s")
    results["elapsed_s"] = round(elapsed, 1)
    return results


def run_single_model(model_key):
    """对单个模型同时运行596和597"""
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — Stage596+597")
    print(f"{'='*60}")

    t0 = time.time()
    bundle = load_model_bundle(model_key)
    if bundle is None:
        return {"error": f"Cannot load {model_key}"}
    model, tokenizer = bundle

    s596 = run_stage596(model, tokenizer, model_key)
    s597 = run_stage597(model, tokenizer, model_key)

    free_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Total time for {model_key}: {elapsed:.1f}s")

    return {
        "stage596": s596,
        "stage597": s597,
        "total_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("  Stage596+597: 非线性映射机制 + 消歧逐层生成")
    print("=" * 60)

    all_results = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        all_results[mk] = run_single_model(mk)

    # === 跨模型对比 ===
    print(f"\n{'='*60}")
    print("  CROSS-MODEL SUMMARY")
    print(f"{'='*60}")

    # Stage596对比
    print(f"\n  --- Stage596: hidden→logit非线性 ---")
    print(f"  {'Metric':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")

    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        s = all_results.get(mk, {}).get("stage596", {})
        ls = s.get("layer_summary", {})
        ni = s.get("norm_intervention", {})
        # 找最差层
        worst_li = -1
        worst_cos = 999
        for li_str, lv in ls.items():
            if isinstance(lv.get("raw_cos_min"), (int, float)) and lv["raw_cos_min"] < worst_cos:
                worst_cos = lv["raw_cos_min"]
                worst_li = int(li_str)
        # 找最佳norm干预
        best_norm_key = "N/A"
        best_norm_val = -999
        for k, v in ni.items():
            if isinstance(v, (int, float)) and 'cos' not in k and v > best_norm_val:
                best_norm_val = v
                best_norm_key = k
        print(f"  {mk:<25} norm={s.get('norm_type','?'):<8} worst_L{worst_li}={worst_cos:.4f} best_norm={best_norm_key}: {best_norm_val:.4f}" if isinstance(best_norm_val, float) else f"  {mk:<25} norm={s.get('norm_type','?'):<8} worst_L{worst_li}={worst_cos:.4f}")

    # 更详细的对比
    print(f"\n  干预实验详情:")
    print(f"  {'Intervention':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")
    for key in ["raw_h@U", "model_norm(h)@U", "manual_rmsnorm(h)@U", "lm_head_has_bias", "raw_h@U+bias", "norm(h)@U+bias"]:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = all_results.get(mk, {}).get("stage596", {}).get("norm_intervention", {}).get(key, "N/A")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        print(f"  {key:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # Stage597对比
    print(f"\n  --- Stage597: 消歧逐层生成 ---")
    print(f"  {'Metric':<25} {'Qwen3':>10} {'DS7B':>10} {'GLM4':>10} {'Gemma4':>10}")
    print(f"  {'-'*60}")

    s597_keys = [
        ("first_emergence_layer", "first_emergence"),
        ("max_growth_layer", "max_growth_L"),
        ("peak_disamb_layer", "peak_L"),
        ("peak_disamb_value", "peak_val"),
        ("final_disamb", "final"),
        ("contrib_25pct_layers", "25%contrib"),
        ("contrib_50pct_layers", "50%contrib"),
        ("positive_info_layers", "pos_layers"),
        ("negative_info_layers", "neg_layers"),
    ]

    for label, key in s597_keys:
        vals = []
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            v = all_results.get(mk, {}).get("stage597", {}).get("summary", {}).get(key, "N/A")
            vals.append(f"{v}" if isinstance(v, int) else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        print(f"  {label:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # 逐层消歧度曲线
    print(f"\n  逐层消歧度曲线:")
    header = f"  {'Layer':<8}"
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        header += f" {mk:>8}"
    print(header)
    print(f"  {'-'*45}")

    max_layers = 0
    per_layer_data = {}
    for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
        pd = all_results.get(mk, {}).get("stage597", {}).get("summary", {}).get("per_layer_mean_disamb", [])
        per_layer_data[mk] = pd
        max_layers = max(max_layers, len(pd))

    for li in range(max_layers):
        row = f"  L{li:<6}"
        for mk in ["qwen3", "deepseek7b", "glm4", "gemma4"]:
            pd = per_layer_data.get(mk, [])
            if li < len(pd):
                row += f" {pd[li]:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # 保存结果
    out_path = OUTPUT_DIR / f"stage596_597_combined_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": TIMESTAMP, "models": all_results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
