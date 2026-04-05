#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage571: 末层hidden state子空间因果分解
目标：用forward hook逐步替换末层hidden state的子空间，看哪个子空间对logit影响最大
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

BANK_RIVER = "The river bank was muddy after the flood."
BANK_FINANCE = "The bank approved the loan for the new business."
APPLE_FRUIT = "She ate a sweet red apple from the orchard."
APPLE_COMPANY = "Apple released the new iPhone yesterday."


def safe_tok(tokenizer, tid, maxlen=25):
    return repr(tokenizer.decode([tid]))[:maxlen].encode('ascii', errors='replace').decode('ascii')


def get_top_tokens(logits, tokenizer, k=5):
    topk_v, topk_i = torch.topk(logits, k)
    return [(safe_tok(tokenizer, t.item()), round(v.item(), 2)) for t, v in zip(topk_i, topk_v)]


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def subspace_intervention(model, tokenizer, sentences, k_list=[10, 50, 100, 500, 1000]):
    """对hidden state做子空间消融/保留，分析对logit的影响"""
    U = model.lm_head.weight.float().cpu() if hasattr(model, 'lm_head') else model.model.embed_tokens.weight.float().cpu()

    # 对U做SVD获取基底
    U_c = U - U.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(U_c, full_matrices=False)

    results = {}
    for label, sent in sentences.items():
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        enc = move_batch_to_model_device(model, enc)

        with torch.no_grad():
            full_logits = model(**enc).logits[0, -1, :].float().cpu()
            full_hidden = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()

        r = {"full_top": get_top_tokens(full_logits, tokenizer)}
        for k in k_list:
            if k >= len(S):
                continue
            # 保留top-k子空间
            basis = Vt[:k]
            proj = full_hidden @ basis.T @ basis
            proj_logits = proj @ U.T
            # 消融top-k子空间（用剩余部分）
            residual = full_hidden - proj
            resid_logits = residual @ U.T

            full_cos = cos(full_logits, proj_logits)
            resid_cos = cos(full_logits, resid_logits)

            r[f"keep_top{k}"] = {
                "cos_to_full": round(full_cos, 4),
                "top_tokens": get_top_tokens(proj_logits, tokenizer, 3),
            }
            r[f"remove_top{k}"] = {
                "cos_to_full": round(resid_cos, 4),
                "top_tokens": get_top_tokens(resid_logits, tokenizer, 3),
            }
        results[label] = r

    return results


def random_direction_intervention(model, tokenizer, sent, n_random=10):
    """用随机方向替换hidden state的各部分，看logit的敏感度"""
    U = model.lm_head.weight.float().cpu() if hasattr(model, 'lm_head') else model.model.embed_tokens.weight.float().cpu()

    enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    with torch.no_grad():
        full_logits = model(**enc).logits[0, -1, :].float().cpu()
        full_hidden = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()

    # 将hidden state分成n_random个等宽频带
    dim = full_hidden.shape[0]
    band_width = dim // n_random

    sensitivities = []
    for i in range(n_random):
        start = i * band_width
        end = start + band_width if i < n_random - 1 else dim
        # 用零替换这个频带
        modified = full_hidden.clone()
        modified[start:end] = 0
        mod_logits = modified @ U.T
        delta = cos(full_logits, mod_logits)
        sensitivities.append({"band": i, "range": f"[{start},{end})", "cos": round(delta, 4)})

    return sensitivities


def main():
    print("=" * 70)
    print("stage571: 末层hidden state子空间因果分解")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    sentences = {
        "bank-river": BANK_RIVER,
        "bank-finance": BANK_FINANCE,
        "apple-fruit": APPLE_FRUIT,
        "apple-company": APPLE_COMPANY,
    }

    # ── 实验1: 子空间保留/消融 ──────────────────
    print("\n[2] 子空间保留/消融对logit的影响:")
    results = subspace_intervention(model, tokenizer, sentences, [10, 50, 100, 256, 512, 1024, 2048])
    for label in sentences:
        print(f"\n  {label}:")
        r = results[label]
        print(f"    full top-5: {r['full_top']}")
        for k in [10, 100, 512, 2048]:
            kk = f"keep_top{k}"
            if kk in r:
                d = r[kk]
                print(f"    keep top-{k}: cos_full={d['cos_to_full']:.4f}, top={d['top_tokens']}")
            kk = f"remove_top{k}"
            if kk in r:
                d = r[kk]
                print(f"    remove top-{k}: cos_full={d['cos_to_full']:.4f}, top={d['top_tokens']}")

    # ── 实验2: 频带敏感度 ────────────────────────
    print("\n[3] hidden state频带敏感度(零化各频带对logit的影响):")
    for label, sent in sentences.items():
        sens = random_direction_intervention(model, tokenizer, sent, n_random=16)
        # 找到最敏感和最不敏感的频带
        sorted_sens = sorted(sens, key=lambda x: x["cos"])
        print(f"\n  {label}:")
        print(f"    最敏感频带: {sorted_sens[:3]}")
        print(f"    最不敏感频带: {sorted_sens[-3:]}")
        # 检查是否有明显的频带偏好
        avg_cos = np.mean([s["cos"] for s in sens])
        print(f"    平均cos: {avg_cos:.4f}")

    # ── 实验3: bank vs apple的logit差异来源 ─────
    print("\n[4] bank vs apple在logit空间的差异:")
    U = model.lm_head.weight.float().cpu() if hasattr(model, 'lm_head') else model.model.embed_tokens.weight.float().cpu()

    for pair_name, (s1, s2) in [("bank", (BANK_RIVER, BANK_FINANCE)), ("apple", (APPLE_FRUIT, APPLE_COMPANY))]:
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        enc1 = move_batch_to_model_device(model, enc1)
        enc2 = move_batch_to_model_device(model, enc2)
        with torch.no_grad():
            h1 = model(**enc1, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
            h2 = model(**enc2, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()

        logit_diff = (h1 - h2) @ U.T
        top_diff_v, top_diff_i = torch.topk(logit_diff.abs(), 10)
        print(f"\n  {pair_name} logit差异最大的10个token:")
        for v, i in zip(top_diff_v, top_diff_i):
            actual_diff = logit_diff[i].item()
            print(f"    {safe_tok(tokenizer, i.item())}: diff={actual_diff:+.2f} (abs={v.item():.2f})")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
