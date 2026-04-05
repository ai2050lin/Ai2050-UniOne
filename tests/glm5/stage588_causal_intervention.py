#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage588: 四模型attention消歧因果干预实验
目标：
  1. L0 attention消融 → 测量对消歧的因果影响（验证stage567 bank delta=-0.55）
  2. 末层MLP消融 → 测量对消歧的因果影响
  3. 单head消融 → 验证head协同假说（stage565: 0/32有效）
  4. 峰值层attention→MLP路径强制切换 → 验证路径可替代性
  5. 跨模型对比所有因果效应
模型：Qwen3 / DeepSeek7B / GLM4 / Gemma4
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
    load_model_bundle, free_model, get_model_device,
    discover_layers, ZeroModule, move_batch_to_model_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 精选8个歧义词（覆盖不同消歧强度和路径类型）
POLYSEMY_WORDS = [
    {"word": "bank",    "ctx1": "The river bank was muddy after the heavy rain.", "ctx2": "The bank approved the loan for the new business."},
    {"word": "plant",   "ctx1": "The factory plant employs over five hundred workers.", "ctx2": "She watered the plant in the garden every morning."},
    {"word": "spring",  "ctx1": "The hot spring resort attracts many tourists each year.", "ctx2": "Spring is the most beautiful season of the year."},
    {"word": "apple",   "ctx1": "She ate a sweet red apple from the orchard.", "ctx2": "Apple released the new iPhone at the conference."},
    {"word": "pool",    "ctx1": "They swam in the swimming pool all afternoon.", "ctx2": "The car pool arrangement saved everyone money."},
    {"word": "seal",    "ctx1": "The seal balanced a ball on its nose at the zoo.", "ctx2": "Please seal the envelope before mailing it."},
    {"word": "match",   "ctx1": "He struck the match to light the candle.", "ctx2": "The football match was exciting to watch."},
    {"word": "light",   "ctx1": "Please turn on the light in the dark room.", "ctx2": "The package was very light and easy to carry."},
]


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def layer_wise_disamb(model, tokenizer, s1, s2):
    enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    return [cos(h1[0, -1, :].float().cpu(), h2[0, -1, :].float().cpu())
            for h1, h2 in zip(out1.hidden_states, out2.hidden_states)]


def find_peak(scores):
    scores_no_emb = scores[1:]
    min_c = min(scores_no_emb)
    return scores_no_emb.index(min_c) + 1


def get_last_hidden(model, tokenizer, sentence):
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    with torch.no_grad():
        return model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()


def ablate_component(model, tokenizer, sentence, layer_idx, component="attn"):
    """消融指定层的attn或mlp，返回末层hidden state"""
    layers = discover_layers(model)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    def zero_out(module, inp, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    target = layers[layer_idx].self_attn if component == "attn" else layers[layer_idx].mlp
    handle = target.register_forward_hook(zero_out)
    with torch.no_grad():
        h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
    handle.remove()
    return h


def ablate_single_head(model, tokenizer, sentence, layer_idx, head_idx):
    """消融指定层指定head，返回末层hidden state"""
    layers = discover_layers(model)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    orig_attn = layers[layer_idx].self_attn

    def zero_head_hook(module, inp, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn_out = output[0]  # (batch, seq, d_model)
            # 对于大多数模型，head维度被嵌入到输出投影中
            # 简化方法：直接零化整个head不太可行，用全零attn output代替
            # 更精确的做法需要模型特定的head分解
            return (torch.zeros_like(attn_out),) + output[1:]
        return output

    handle = orig_attn.register_forward_hook(zero_head_hook)
    with torch.no_grad():
        h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
    handle.remove()
    return h


def measure_disamb_impact(model, tokenizer, pw, intervention_fn, intervention_name):
    """
    测量某个干预对消歧度的影响
    返回：baseline_disamb, intervened_disamb, delta
    """
    # baseline
    scores = layer_wise_disamb(model, tokenizer, pw["ctx1"], pw["ctx2"])
    baseline_disamb = 1 - min(scores[1:])  # 跳过embedding

    # intervened
    h1_int = intervention_fn(model, tokenizer, pw["ctx1"])
    h2_int = intervention_fn(model, tokenizer, pw["ctx2"])
    intervened_disamb = 1 - cos(h1_int, h2_int)

    delta = intervened_disamb - baseline_disamb
    return {
        "baseline_disamb": round(baseline_disamb, 4),
        "intervened_disamb": round(intervened_disamb, 4),
        "delta": round(delta, 4),
        "intervention": intervention_name,
    }


def _make_zero_hook():
    """创建消融hook"""
    def zero_out(mod, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    return zero_out


def _ablate_layers(model, tokenizer, sentence, layer_indices, component):
    """消融多个层的指定组件"""
    layers = discover_layers(model)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)
    handles = []
    for li in layer_indices:
        target = layers[li].self_attn if component == "attn" else layers[li].mlp
        handles.append(target.register_forward_hook(_make_zero_hook()))
    with torch.no_grad():
        h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
    for hnd in handles:
        hnd.remove()
    return h


def run_causal_suite(model, tokenizer, n_layers, pw):
    """对单个词运行完整因果干预套件"""
    results = {}

    # 先计算峰值层
    scores = layer_wise_disamb(model, tokenizer, pw["ctx1"], pw["ctx2"])
    peak = min(find_peak(scores), n_layers - 1)

    # 1. L0 attention消融
    results["L0_attn_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s: _ablate_layers(m, t, s, [0], "attn"),
        "L0_attn_zero"
    )

    # 2. 末层MLP消融
    results["last_mlp_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s: _ablate_layers(m, t, s, [n_layers - 1], "mlp"),
        f"L{n_layers-1}_mlp_zero"
    )

    # 3. 峰值层attention消融
    results[f"L{peak}_attn_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s, _p=peak: _ablate_layers(m, t, s, [_p], "attn"),
        f"L{peak}_attn_zero"
    )

    # 4. 峰值层MLP消融
    results[f"L{peak}_mlp_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s, _p=peak: _ablate_layers(m, t, s, [_p], "mlp"),
        f"L{peak}_mlp_zero"
    )

    # 5. L0 + 峰值层联合attention消融
    results["L0+peak_attn_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s, _p=peak: _ablate_layers(m, t, s, [0, _p], "attn"),
        f"L0+L{peak}_attn_zero"
    )

    # 6. 入口+出口联合消融（L0_attn + last_mlp）
    results["entry_exit_ablation"] = measure_disamb_impact(
        model, tokenizer, pw,
        lambda m, t, s: _ablate_layers(m, t, s, [0], "attn") if True else _ablate_layers(m, t, s, [0], "attn"),
        f"L0_attn+L{n_layers-1}_mlp_zero"
    )
    # 正确的入口+出口联合消融
    def _entry_exit(m, t, s):
        return _ablate_layers(m, t, s, [0], "attn")  # placeholder, overwritten below
    # 用自定义方式同时消融attn和mlp
    def _entry_exit_real(m, t, s):
        layers = discover_layers(m)
        enc = t(s, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(m, enc)
        handles = []
        handles.append(layers[0].self_attn.register_forward_hook(_make_zero_hook()))
        handles.append(layers[n_layers - 1].mlp.register_forward_hook(_make_zero_hook()))
        with torch.no_grad():
            h = m(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
        for hnd in handles:
            hnd.remove()
        return h
    results["entry_exit_ablation"] = measure_disamb_impact(
        model, tokenizer, pw, _entry_exit_real, f"L0_attn+L{n_layers-1}_mlp_zero"
    )

    # 7. 逐层attention消融（关键层采样）
    layer_attn_deltas = {}
    scan_li = sorted(set(
        [0, 1, 2, 3, 4, peak, peak + 1, max(0, peak - 1),
         n_layers // 2, n_layers - 1]
    ))
    for li in scan_li:
        li = max(0, min(li, n_layers - 1))
        if str(li) in layer_attn_deltas:
            continue
        r = measure_disamb_impact(
            model, tokenizer, pw,
            lambda m, t, s, _li=li: _ablate_layers(m, t, s, [_li], "attn"),
            f"L{li}_attn_zero"
        )
        layer_attn_deltas[str(li)] = r["delta"]
    results["layer_attn_deltas"] = layer_attn_deltas

    # 8. 逐层MLP消融（关键层采样）
    layer_mlp_deltas = {}
    for li in sorted(set([0, peak, n_layers // 2, n_layers - 2, n_layers - 1])):
        li = max(0, min(li, n_layers - 1))
        if str(li) in layer_mlp_deltas:
            continue
        r = measure_disamb_impact(
            model, tokenizer, pw,
            lambda m, t, s, _li=li: _ablate_layers(m, t, s, [_li], "mlp"),
            f"L{li}_mlp_zero"
        )
        layer_mlp_deltas[str(li)] = r["delta"]
    results["layer_mlp_deltas"] = layer_mlp_deltas

    return results


def run_single_model(model_key):
    print(f"\n{'='*60}")
    print(f"Stage588 - 模型: {model_key}")
    print(f"{'='*60}")
    t0 = time.time()

    model, tokenizer = load_model_bundle(model_key)
    layers = discover_layers(model)
    n_layers = len(layers)
    print(f"  层数={n_layers}")

    word_results = {}
    for i, pw in enumerate(POLYSEMY_WORDS):
        w = pw["word"]
        print(f"  [{i+1}/8] {w}...", end="", flush=True)
        wt0 = time.time()
        try:
            res = run_causal_suite(model, tokenizer, n_layers, pw)
            word_results[w] = res
            # 打印关键结果
            l0_delta = res["L0_attn_ablation"]["delta"]
            last_mlp_delta = res["last_mlp_ablation"]["delta"]
            peak_key = [k for k in res if k.endswith("_attn_ablation") and k.startswith("L") and "L0" not in k and "+" not in k][0]
            peak_delta = res[peak_key]["delta"]
            ee_delta = res["entry_exit_ablation"]["delta"]
            print(f" L0={l0_delta:+.4f}, peak_attn={peak_delta:+.4f}, "
                  f"last_mlp={last_mlp_delta:+.4f}, entry_exit={ee_delta:+.4f} "
                  f"({time.time()-wt0:.1f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            word_results[w] = {"error": str(e)}

    total = time.time() - t0

    # 汇总
    l0_deltas = []
    last_mlp_deltas = []
    ee_deltas = []
    for w, r in word_results.items():
        if "error" in r:
            continue
        l0_deltas.append(r["L0_attn_ablation"]["delta"])
        last_mlp_deltas.append(r["last_mlp_ablation"]["delta"])
        ee_deltas.append(r["entry_exit_ablation"]["delta"])

    summary = {
        "model": model_key,
        "n_layers": n_layers,
        "total_time_s": round(total, 1),
        "mean_L0_attn_delta": round(np.mean(l0_deltas), 4) if l0_deltas else None,
        "mean_last_mlp_delta": round(np.mean(last_mlp_deltas), 4) if last_mlp_deltas else None,
        "mean_entry_exit_delta": round(np.mean(ee_deltas), 4) if ee_deltas else None,
        "max_L0_attn_delta": round(max(l0_deltas), 4) if l0_deltas else None,
        "max_last_mlp_delta": round(max(last_mlp_deltas), 4) if last_mlp_deltas else None,
        "word_results": word_results,
    }

    free_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)

    return summary


def main():
    print("=" * 60)
    print("Stage 588: 四模型attention消歧因果干预实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    all_results = {}

    for mk in model_keys:
        try:
            summary = run_single_model(mk)
            all_results[mk] = summary
        except Exception as e:
            print(f"\n  {mk} ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[mk] = {"error": str(e)}

    # 跨模型对比
    cross = {}
    for mk, s in all_results.items():
        if "error" in s:
            continue
        cross[mk] = {
            "mean_L0_attn_delta": s["mean_L0_attn_delta"],
            "mean_last_mlp_delta": s["mean_last_mlp_delta"],
            "mean_entry_exit_delta": s["mean_entry_exit_delta"],
            "max_L0_attn_delta": s["max_L0_attn_delta"],
            "max_last_mlp_delta": s["max_last_mlp_delta"],
        }

    final = {
        "timestamp": TIMESTAMP,
        "stage": "588",
        "title": "四模型attention消歧因果干预实验",
        "cross_model_summary": cross,
        "models": all_results,
    }

    out_path = OUTPUT_DIR / f"stage588_causal_intervention_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印跨模型总结
    print("\n" + "=" * 60)
    print("跨模型因果对比")
    print("=" * 60)
    for mk, c in cross.items():
        print(f"\n  {mk}:")
        v = c['mean_L0_attn_delta']
        print(f"    L0 attn消融(均值): {v:+.4f}" if v is not None else "    L0 attn消融(均值): N/A")
        v = c['max_L0_attn_delta']
        print(f"    L0 attn消融(最大): {v:+.4f}" if v is not None else "    L0 attn消融(最大): N/A")
        v = c['mean_last_mlp_delta']
        print(f"    末层MLP消融(均值): {v:+.4f}" if v is not None else "    末层MLP消融(均值): N/A")
        v = c['max_last_mlp_delta']
        print(f"    末层MLP消融(最大): {v:+.4f}" if v is not None else "    末层MLP消融(最大): N/A")
        v = c['mean_entry_exit_delta']
        print(f"    入口+出口联合(均值): {v:+.4f}" if v is not None else "    入口+出口联合(均值): N/A")

    # 不变量验证
    print("\n" + "=" * 60)
    print("不变量验证")
    print("=" * 60)
    l0_vals = [c["mean_L0_attn_delta"] for c in cross.values() if c["mean_L0_attn_delta"] is not None]
    mlp_vals = [c["mean_last_mlp_delta"] for c in cross.values() if c["mean_last_mlp_delta"] is not None]

    print(f"\n  L0 attention因果效力(负值=消歧减弱): {l0_vals}")
    print(f"    全部显著? {all(abs(v) > 0.05 for v in l0_vals)}")
    print(f"    跨模型一致方向? {all(v < 0 for v in l0_vals) or all(v > 0 for v in l0_vals)}")

    print(f"\n  末层MLP因果效力: {mlp_vals}")
    print(f"    全部显著? {all(abs(v) > 0.05 for v in mlp_vals)}")
    print(f"    跨模型一致方向? {all(v < 0 for v in mlp_vals) or all(v > 0 for v in mlp_vals)}")


if __name__ == "__main__":
    main()
