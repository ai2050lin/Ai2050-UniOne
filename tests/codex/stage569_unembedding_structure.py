#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage569: 末层hidden state -> logit映射矩阵结构分析
目标：分析unembedding矩阵的数学结构，理解它如何把2560维编码翻译成词概率
模型：Qwen3-4B
"""

from __future__ import annotations
import sys, json, time, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from qwen3_language_shared import (
    load_qwen3_model, discover_layers, qwen_hidden_dim,
    get_model_device, remove_hooks, move_batch_to_model_device
)

BANK_RIVER = "The river bank was muddy after the flood."
BANK_FINANCE = "The bank approved the loan for the new business."
APPLE_FRUIT = "She ate a sweet red apple from the orchard."
APPLE_COMPANY = "Apple released the new iPhone yesterday."


def get_unembed_matrix(model):
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head.weight.detach().float().cpu()
    return model.model.embed_tokens.weight.detach().float().cpu()


def safe_decode(tokenizer, token_id, max_len=30):
    return repr(tokenizer.decode([token_id]))[:max_len].encode('ascii', errors='replace').decode('ascii')


def subspace_reconstruction(hidden, U, top_k_list, tokenizer):
    full_logits = hidden @ U.T
    topk_full, topk_ids_full = torch.topk(full_logits, 5)
    full_tokens = [safe_decode(tokenizer, t.item()) for t in topk_ids_full]

    U_centered = U - U.mean(dim=0, keepdim=True)
    # 用randomized SVD避免OOM
    try:
        _, S, Vt = torch.linalg.svd(U_centered, full_matrices=False)
    except RuntimeError:
        import numpy as np
        U_np = U_centered.numpy()
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(U_np.T @ rng.randn(2560, min(2560, U_np.shape[1])))
        B = Q.T @ U_np.T
        _, S_np, Vt_np = np.linalg.svd(B, full_matrices=False)
        S = torch.from_numpy(S_np.astype(np.float32))
        Vt = torch.from_numpy(Vt_np.astype(np.float32))

    results = {"full_tokens": full_tokens, "full_top_logits": [round(v.item(), 2) for v in topk_full]}
    for k in top_k_list:
        if k >= S.shape[0]:
            continue
        basis = Vt[:k]
        proj_hidden = hidden @ basis.T @ basis
        proj_logits = proj_hidden @ U.T
        topk_proj, _ = torch.topk(proj_logits, 5)
        proj_tokens = [safe_decode(tokenizer, t.item()) for t in torch.topk(proj_logits, 5)[1]]
        corr = F.cosine_similarity(full_logits.unsqueeze(0), proj_logits.unsqueeze(0)).item()
        results[f"k={k}"] = {
            "tokens": proj_tokens,
            "cos_to_full": round(corr, 4),
            "top_logit_ratio": round(topk_proj[0].item() / topk_full[0].item(), 4),
        }
    return results


def main():
    print("=" * 70)
    print("stage569: Unembedding矩阵结构分析")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    vocab_size = tokenizer.vocab_size
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}, 词表大小={vocab_size}")

    # ── 实验1: Unembedding矩阵全局结构 ────────────
    print("\n[2] Unembedding矩阵全局结构...")
    U = get_unembed_matrix(model)
    print(f"  形状: {U.shape}")

    U_mean = U.mean(dim=0, keepdim=True)
    U_centered = U - U_mean
    # 使用随机SVD避免内存溢出（矩阵太大无法做full_matrices=True）
    # 对U_centered做行采样+随机投影，只取前512个奇异值
    try:
        _, S, Vt = torch.linalg.svd(U_centered, full_matrices=False)
    except RuntimeError:
        # 如果仍然OOM，用随机化方法
        import numpy as np
        U_np = U_centered.numpy()
        # 随机投影到2560维
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(U_np.T @ rng.randn(2560, 2560))
        B = Q.T @ U_np.T  # (2560, 151936)
        _, S_np, Vt_np = np.linalg.svd(B, full_matrices=False)
        S = torch.from_numpy(S_np[:2560].astype(np.float32))
        Vt = torch.from_numpy(Vt_np[:2560, :].astype(np.float32))

    eff_rank = (S ** 2).sum() / (S.max() ** 2)
    cumsum_ratio = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()

    print(f"  奇异值范围: [{S[-1]:.2f}, {S[0]:.2f}]")
    print(f"  有效秩: {eff_rank:.1f}")
    for k in [10, 50, 100, 256, 512, 1024]:
        if k < len(S):
            print(f"  Top-{k} 解释方差比: {cumsum_ratio[k-1]:.4f}")

    print(f"\n  前5个奇异向量方向的top-5 token:")
    for i in range(5):
        direction = Vt[i]
        direction_logits = direction @ U.T
        top5_v, top5_i = torch.topk(direction_logits, 5)
        tokens = [repr(tokenizer.decode([t.item()]))[:30] for t in top5_i]
        safe_tokens = [t.encode('ascii', errors='replace').decode('ascii') for t in tokens]
        print(f"    SV{i}(val={S[i]:.1f}): {list(zip(safe_tokens, [f'{v:.1f}' for v in top5_v.tolist()]))}")

    # ── 实验2: hidden state结构分析 ──────────────
    print("\n[3] hidden state结构...")
    sentences = {
        "bank-river": BANK_RIVER,
        "bank-finance": BANK_FINANCE,
        "apple-fruit": APPLE_FRUIT,
        "apple-company": APPLE_COMPANY,
    }

    for label, sent in sentences.items():
        encoded = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        h = hidden.float()
        print(f"  {label}: norm={h.norm():.2f}, std={h.std():.2f}, "
              f"kurtosis={(((h-h.mean())**4).mean()/(h.std()**4).item()-3):.2f}, "
              f"sparsity(0.01)={(h.abs() < 0.01*h.abs().max()).float().mean():.4f}")

    # ── 实验3: 子空间重建 ────────────────────────
    print("\n[4] 子空间重建——多少维度足以重建logit?")
    for label, sent in sentences.items():
        encoded = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        result = subspace_reconstruction(hidden, U, [10, 50, 100, 500, 1000, 2048], tokenizer)
        print(f"\n  {label}:")
        print(f"    全量top-5: {result['full_tokens']}")
        for k_key in sorted(result.keys()):
            if k_key.startswith("k="):
                d = result[k_key]
                print(f"    {k_key}: cos_full={d['cos_to_full']:.4f}, ratio={d['top_logit_ratio']:.4f}, "
                      f"tokens={d['tokens'][:3]}")

    # ── 实验4: bank/apple在unembedding空间的分离 ──
    print("\n[5] 歧义词在unembedding空间的分离度...")
    bank_words = ["river", "water", "stream", "money", "loan", "deposit"]
    apple_words = ["fruit", "tree", "orchard", "iPhone", "company", "MacBook"]

    bank_embs, apple_embs = [], []
    for w in bank_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            bank_embs.append(U[ids[0]])
    for w in apple_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            apple_embs.append(U[ids[0]])

    if bank_embs and apple_embs:
        bm = torch.stack(bank_embs).mean(dim=0)
        am = torch.stack(apple_embs).mean(dim=0)
        cross_cos = F.cosine_similarity(bm.unsqueeze(0), am.unsqueeze(0)).item()

        bank_intra = []
        for i in range(len(bank_embs)):
            for j in range(i+1, len(bank_embs)):
                bank_intra.append(F.cosine_similarity(bank_embs[i].unsqueeze(0), bank_embs[j].unsqueeze(0)).item())
        apple_intra = []
        for i in range(len(apple_embs)):
            for j in range(i+1, len(apple_embs)):
                apple_intra.append(F.cosine_similarity(apple_embs[i].unsqueeze(0), apple_embs[j].unsqueeze(0)).item())

        print(f"  bank内部avg cos: {np.mean(bank_intra):.4f}")
        print(f"  apple内部avg cos: {np.mean(apple_intra):.4f}")
        print(f"  bank-apple交叉cos: {cross_cos:.4f}")

    # ── 实验5: hidden->logit线性度 ──────────────
    print("\n[6] hidden->logit线性度检验...")
    enc1 = tokenizer(BANK_RIVER, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(BANK_FINANCE, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)

    with torch.no_grad():
        h1 = model(**enc1, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
        h2 = model(**enc2, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()

    h_mix = 0.5 * h1 + 0.5 * h2
    logit_mix = h_mix @ U.T
    logit_exp = 0.5 * (h1 @ U.T) + 0.5 * (h2 @ U.T)
    linear_cos = F.cosine_similarity(logit_mix.unsqueeze(0), logit_exp.unsqueeze(0)).item()
    print(f"  线性度(cosine): {linear_cos:.8f} (1.0=完美线性)")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总结: 有效秩={eff_rank:.1f}, Top-100方差比={cumsum_ratio[min(99,len(cumsum_ratio)-1)]:.4f}, "
          f"线性度={linear_cos:.8f}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
