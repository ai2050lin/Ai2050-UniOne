#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage572: 跨模型unembedding矩阵结构对比
目标：比较Qwen3/DeepSeek7B/GLM4/Gemma4的unembedding矩阵结构
"""

from __future__ import annotations
import sys, time, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from qwen3_language_shared import (
    load_qwen3_model, load_deepseek7b_model, load_glm4_model, load_gemma4_model,
    discover_layers, qwen_hidden_dim, remove_hooks
)


def get_unembed(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().float().cpu()
    return model.model.embed_tokens.weight.detach().float().cpu()


def analyze_unembed(U, name):
    """分析unembedding矩阵的统计特性"""
    vocab_size, hidden_dim = U.shape
    U_c = U - U.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(U_c, full_matrices=False)

    eff_rank = (S ** 2).sum() / (S.max() ** 2)
    cum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()

    norms = U.norm(dim=1)
    return {
        "name": name,
        "shape": f"{vocab_size}x{hidden_dim}",
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "eff_rank": round(eff_rank.item(), 1),
        "top10_var": round(cum[9].item(), 4),
        "top50_var": round(cum[min(49,len(cum)-1)].item(), 4),
        "top100_var": round(cum[min(99,len(cum)-1)].item(), 4),
        "top256_var": round(cum[min(255,len(cum)-1)].item(), 4),
        "top512_var": round(cum[min(511,len(cum)-1)].item(), 4),
        "top1024_var": round(cum[min(1023,len(cum)-1)].item(), 4),
        "sv_range": f"[{S[-1]:.2f}, {S[0]:.2f}]",
        "token_norm_mean": round(norms.mean().item(), 2),
        "token_norm_std": round(norms.std().item(), 2),
        "row_cos_mean": round(_avg_pairwise_cos(U, 500), 4),  # 采样500个token计算
    }


def _avg_pairwise_cos(U, sample_n=500):
    """采样计算token embedding之间的平均cosine"""
    idx = torch.randperm(U.shape[0])[:sample_n]
    sub = U[idx]
    norms = sub / sub.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_matrix = norms @ norms.T
    # 取上三角（不含对角线）
    mask = torch.triu(torch.ones(sample_n, sample_n), diagonal=1).bool()
    return cos_matrix[mask].mean().item()


def hidden_to_logit_linearity(model, tokenizer, hidden_dim):
    """测试hidden->logit映射的线性度"""
    sents = ["The cat sat on the mat.", "The dog ran in the park."]
    U = get_unembed(model)
    results = []

    for s in sents:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
        results.append(h)

    h1, h2 = results
    h_mix = 0.5 * h1 + 0.5 * h2
    logit_mix = h_mix @ U.T
    logit_exp = 0.5 * (h1 @ U.T) + 0.5 * (h2 @ U.T)
    return round(F.cosine_similarity(logit_mix.unsqueeze(0), logit_exp.unsqueeze(0)).item(), 8)


def main():
    print("=" * 70)
    print("stage572: 跨模型Unembedding矩阵结构对比")
    print("=" * 70)

    results = []

    # Qwen3
    t0 = time.time()
    print("\n[1] Qwen3-4B...")
    try:
        m, t = load_qwen3_model()
        U = get_unembed(m)
        r = analyze_unembed(U, "Qwen3-4B")
        r["linearity"] = hidden_to_logit_linearity(m, t, r["hidden_dim"])
        results.append(r)
        print(f"  完成: {time.time()-t0:.1f}s")
        del m
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败: {e}")

    # DeepSeek7B
    t0 = time.time()
    print("\n[2] DeepSeek7B...")
    try:
        m, t = load_deepseek7b_model()
        U = get_unembed(m)
        r = analyze_unembed(U, "DeepSeek7B")
        r["linearity"] = hidden_to_logit_linearity(m, t, r["hidden_dim"])
        results.append(r)
        print(f"  完成: {time.time()-t0:.1f}s")
        del m
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败: {e}")

    # GLM4
    t0 = time.time()
    print("\n[3] GLM4-9B...")
    try:
        m, t = load_glm4_model()
        U = get_unembed(m)
        r = analyze_unembed(U, "GLM4-9B")
        r["linearity"] = hidden_to_logit_linearity(m, t, r["hidden_dim"])
        results.append(r)
        print(f"  完成: {time.time()-t0:.1f}s")
        del m
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败: {e}")

    # Gemma4
    t0 = time.time()
    print("\n[4] Gemma4-2B...")
    try:
        m, t = load_gemma4_model()
        U = get_unembed(m)
        r = analyze_unembed(U, "Gemma4-2B")
        r["linearity"] = hidden_to_logit_linearity(m, t, r["hidden_dim"])
        results.append(r)
        print(f"  完成: {time.time()-t0:.1f}s")
        del m
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败: {e}")

    # ── 汇总 ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("跨模型对比汇总:")
    print(f"{'属性':>16} " + " ".join(f"{r['name']:>12}" for r in results))
    keys = ["hidden_dim", "vocab_size", "eff_rank", "top100_var", "top512_var", "top1024_var",
            "linearity", "token_norm_mean", "row_cos_mean"]
    for k in keys:
        vals = " ".join(f"{str(r.get(k, 'N/A')):>12}" for r in results)
        print(f"{k:>16} {vals}")


if __name__ == "__main__":
    main()
