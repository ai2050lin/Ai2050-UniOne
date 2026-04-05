#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage577-580合并: 维度坍缩因果机制
目标：理解3/4模型在15-17%层发生维度坍缩的相变机制
模型：Qwen3-4B
"""

from __future__ import annotations
import sys, time, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from qwen3_language_shared import (
    load_qwen3_model, discover_layers, qwen_hidden_dim,
    remove_hooks, move_batch_to_model_device
)


def effective_rank(tensor, threshold=0.01):
    """计算有效秩（奇异值大于threshold*max的比例）"""
    if tensor.dim() == 1:
        return 1.0
    _, S, _ = torch.linalg.svd(tensor.float(), full_matrices=False)
    return (S > threshold * S.max()).sum().item()


def layer_wise_effective_rank(model, tokenizer, sentences, sample_layers=None):
    """逐层计算多个句子的平均有效秩"""
    layers = discover_layers(model)
    n_layers = len(layers)
    if sample_layers is None:
        sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
        if (n_layers - 1) not in sample_layers:
            sample_layers.append(n_layers - 1)

    results = {}
    for label, sent in sentences.items():
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for l in sample_layers:
            h = out.hidden_states[l][0, -1, :].float().cpu()
            key = f"L{l}"
            if key not in results:
                results[key] = []
            results[key].append(effective_rank(h.unsqueeze(0)))
    return {k: np.mean(v) for k, v in results.items()}


def ablation_effective_rank(model, tokenizer, sentence, layer_idx, component="attn"):
    """消融指定层的attn或MLP后计算下游有效秩"""
    layers = discover_layers(model)
    n_layers = len(layers)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    # baseline
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    baseline = {l: effective_rank(out.hidden_states[l][0, -1, :].unsqueeze(0).float().cpu())
                for l in range(layer_idx, min(layer_idx + 6, n_layers))}

    # ablation
    handles = []
    zero_fn = lambda _m, _i, _o: (torch.zeros_like(_o[0]),) + _o[1:] if isinstance(_o, tuple) else torch.zeros_like(_o)
    if component == "attn":
        handles.append(layers[layer_idx].self_attn.register_forward_hook(zero_fn))
    else:
        handles.append(layers[layer_idx].mlp.register_forward_hook(zero_fn))

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    ablated = {l: effective_rank(out.hidden_states[l][0, -1, :].unsqueeze(0).float().cpu())
               for l in range(layer_idx, min(layer_idx + 6, n_layers))}
    remove_hooks(handles)

    return {"baseline": baseline, "ablated": ablated}


def inject_high_dim(model, tokenizer, sentence, layer_idx, dim=256):
    """在坍缩后的层注入高维信息，看网络能否利用"""
    layers = discover_layers(model)
    n_layers = len(layers)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    # 基线
    with torch.no_grad():
        out_base = model(**enc, output_hidden_states=True)
    base_rank = effective_rank(out_base.hidden_states[-1][0, -1, :].unsqueeze(0).float().cpu())

    # 注入随机高维信号
    h = out_base.hidden_states[layer_idx][0, -1, :].clone()
    # 添加高维噪声
    noise = torch.randn_like(h) * h.std() * 0.5
    # 做正交化确保新增维度正交
    noise = noise - (noise @ h) / (h @ h + 1e-8) * h

    buffers = {}
    def inject_hook(_m, _i, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs[0, -1, :] = (hs[0, -1, :] + noise.to(hs.device, hs.dtype))
            return (hs,) + output[1:]
        return output

    handles = [layers[layer_idx].register_forward_hook(inject_hook)]
    with torch.no_grad():
        out_inject = model(**enc, output_hidden_states=True)
    remove_hooks(handles)

    inject_rank = effective_rank(out_inject.hidden_states[-1][0, -1, :].unsqueeze(0).float().cpu())

    # logit差异
    U = model.lm_head.weight.float().cpu() if hasattr(model, 'lm_head') else model.model.embed_tokens.weight.float().cpu()
    base_logits = out_base.hidden_states[-1][0, -1, :].float().cpu() @ U.T
    inject_logits = out_inject.hidden_states[-1][0, -1, :].float().cpu() @ U.T
    logit_cos = cos(base_logits, inject_logits)

    return {"base_rank": round(base_rank, 1), "inject_rank": round(inject_rank, 1),
            "logit_cos": round(logit_cos, 6)}


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    print("=" * 70)
    print("stage577-580: 维度坍缩因果机制")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    # 坍缩预测发生在约15-17%层 = L5-L6(36层)
    collapse_layers = [4, 5, 6, 7, 8]
    post_collapse = [10, 12, 16, 20]

    sentences = {
        "apple": "apple",
        "red apple": "red apple",
        "The red apple": "The red apple is sweet.",
        "bank": "bank",
        "river bank": "river bank",
    }

    # ── 实验1: 逐层有效秩 ────────────────────────
    print("\n[2] 逐层有效秩（验证坍缩位置）:")
    all_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if (n_layers - 1) not in all_layers:
        all_layers.append(n_layers - 1)

    for label, sent in sentences.items():
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        ranks = []
        for l in all_layers:
            h = out.hidden_states[l][0, -1, :].unsqueeze(0).float().cpu()
            ranks.append(effective_rank(h))

        min_l = all_layers[ranks.index(min(ranks))]
        print(f"  {label:20s}: min_rank={min(ranks):.1f}@L{min_l}, L0={ranks[0]:.1f}, "
              f"L35={ranks[-1]:.1f}")

    # ── 实验2: 坍缩层消融attn/MLP ────────────────
    print("\n[3] 坍缩层消融attn/MLP对下游有效秩的影响:")
    for cl in collapse_layers:
        for comp in ["attn", "mlp"]:
            result = ablation_effective_rank(model, tokenizer, "apple", cl, comp)
            print(f"  消融L{cl}.{comp}: ", end="")
            for l in sorted(result["baseline"].keys()):
                delta = result["ablated"][l] - result["baseline"][l]
                print(f"L{l}={result['baseline'][l]:.0f}→{result['ablated'][l]:.0f}({delta:+.0f}) ", end="")
            print()

    # ── 实验3: 反向坍缩注入实验 ──────────────────
    print("\n[4] 反向坍缩注入实验(在坍缩后注入高维噪声):")
    for label, sent in sentences.items():
        for cl in collapse_layers + post_collapse:
            if cl >= n_layers:
                continue
            r = inject_high_dim(model, tokenizer, sent, cl)
            print(f"  {label:20s} @L{cl:2d}: base_rank={r['base_rank']:6.1f} "
                  f"inject_rank={r['inject_rank']:6.1f} logit_cos={r['logit_cos']:.6f}")

    # ── 实验4: 坍缩前后信息内容对比 ──────────────
    print("\n[5] 坍缩前后信息内容对比(PCA):")
    for label, sent in sentences.items():
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        pre_collapse = out.hidden_states[3][0, -1, :].float().cpu()
        at_collapse = out.hidden_states[6][0, -1, :].float().cpu()
        post_collapse = out.hidden_states[12][0, -1, :].float().cpu()

        for name, h in [("L3(pre)", pre_collapse), ("L6(at)", at_collapse), ("L12(post)", post_collapse)]:
            _, S, _ = torch.linalg.svd(h.unsqueeze(0), full_matrices=False)
            top1_ratio = (S[0] ** 2 / (S ** 2).sum()).item()
            top5_ratio = (S[:5] ** 2).sum() / (S ** 2).sum()
            print(f"  {label:20s} {name:10s}: top1={top1_ratio:.4f} top5={top5_ratio:.4f} eff_rank={effective_rank(h.unsqueeze(0)):.1f}")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
