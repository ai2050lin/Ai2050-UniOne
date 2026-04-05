#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage568: L35 MLP子层消融 —— 精确消融L35 MLP的三个子层(up/gate/down_proj)
目标：找到L35 MLP中哪个子层在做"读出"（因果关键子层）
模型：Qwen3-0.6B（实际使用Qwen3-4B，36层）
"""

from __future__ import annotations
import sys, json, time, torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from qwen3_language_shared import (
    load_qwen3_model, load_qwen3_tokenizer, discover_layers,
    qwen_hidden_dim, qwen_neuron_dim, get_model_device,
    remove_hooks, move_batch_to_model_device
)

# ── 配置 ──────────────────────────────────────────
BANK_RIVER = "The river bank was muddy after the flood."
BANK_FINANCE = "The bank approved the loan for the new business."
APPLE_FRUIT = "She ate a sweet red apple from the orchard."
APPLE_COMPANY = "Apple released the new iPhone yesterday."

SENTENCES = [BANK_RIVER, BANK_FINANCE, APPLE_FRUIT, APPLE_COMPANY]
LABELS = ["bank-river", "bank-finance", "apple-fruit", "apple-company"]

ABLATION_LAYERS = [35]  # 末层
TARGET_TOKEN_POS = -1   # 最后一个token


def get_unembed_matrix(model):
    """获取unembedding矩阵 (lm_head.weight)"""
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().float()
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        # Qwen3风格：lm_head可能和embed_tokens共享
        if hasattr(model, "lm_head"):
            return model.lm_head.weight.detach().float()
        else:
            return model.model.embed_tokens.weight.detach().float()
    raise RuntimeError("无法获取unembedding矩阵")


def extract_last_hidden(model, tokenizer, sentence):
    """提取最后一个token的hidden state（经过Layernorm后）"""
    encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    encoded = move_batch_to_model_device(model, encoded)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        # 取最后一个token在最后一层的hidden state
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
    return hidden


def ablate_mlp_sublayer(model, layer_idx, sublayer_name):
    """消融指定层的MLP子层，返回handles"""
    layers = discover_layers(model)
    layer = layers[layer_idx]
    handles = []

    if sublayer_name == "up_proj":
        module = layer.mlp.up_proj
    elif sublayer_name == "gate_proj":
        module = layer.mlp.gate_proj
    elif sublayer_name == "down_proj":
        module = layer.mlp.down_proj
    elif sublayer_name == "full_mlp":
        module = layer.mlp
    else:
        raise ValueError(f"未知子层: {sublayer_name}")

    def zero_hook(_module, _inputs, _output):
        return torch.zeros_like(_output)

    handles.append(module.register_forward_hook(zero_hook))
    return handles


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    print("=" * 70)
    print("stage568: L35 MLP子层消融 —— 精确定位读出关键子层")
    print("=" * 70)

    # ── 加载模型 ─────────────────────────────────
    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    layers = discover_layers(model)
    n_layers = len(layers)
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    # ── 实验1：基线hidden states ──────────────────
    print("\n[2] 提取基线hidden states...")
    baselines = {}
    for sent, label in zip(SENTENCES, LABELS):
        h = extract_last_hidden(model, tokenizer, sent)
        baselines[label] = h
        print(f"  {label}: norm={h.norm():.4f}")

    # 基线距离矩阵
    print("\n  基线距离矩阵(cosine):")
    print(f"  {'':>14}", end="")
    for l in LABELS:
        print(f"  {l:>14}", end="")
    print()
    for l1 in LABELS:
        print(f"  {l1:>14}", end="")
        for l2 in LABELS:
            cos = cosine_sim(baselines[l1], baselines[l2])
            print(f"  {cos:>14.4f}", end="")
        print()

    # ── 实验2：逐子层消融 ────────────────────────
    sublayers = ["up_proj", "gate_proj", "down_proj", "full_mlp"]
    results = {}

    for sublayer in sublayers:
        print(f"\n[3] 消融L35.{sublayer}...")
        handles = ablate_mlp_sublayer(model, 35, sublayer)

        try:
            ablated = {}
            for sent, label in zip(SENTENCES, LABELS):
                h = extract_last_hidden(model, tokenizer, sent)
                ablated[label] = h

            # 计算delta
            deltas = {}
            for label in LABELS:
                delta = 1.0 - cosine_sim(baselines[label], ablated[label])
                deltas[label] = delta

            # 计算消歧能力变化
            bank_cos_ablated = cosine_sim(ablated["bank-river"], ablated["bank-finance"])
            apple_cos_ablated = cosine_sim(ablated["apple-fruit"], ablated["apple-company"])
            bank_cos_baseline = cosine_sim(baselines["bank-river"], baselines["bank-finance"])
            apple_cos_baseline = cosine_sim(baselines["apple-fruit"], baselines["apple-company"])

            bank_disamb_delta = bank_cos_ablated - bank_cos_baseline
            apple_disamb_delta = apple_cos_ablated - apple_cos_baseline

            results[sublayer] = {
                "deltas": {k: round(v, 6) for k, v in deltas.items()},
                "bank_cos_ablated": round(bank_cos_ablated, 4),
                "apple_cos_ablated": round(apple_cos_ablated, 4),
                "bank_cos_baseline": round(bank_cos_baseline, 4),
                "apple_cos_baseline": round(apple_cos_baseline, 4),
                "bank_disamb_delta": round(bank_disamb_delta, 4),
                "apple_disamb_delta": round(apple_disamb_delta, 4),
            }

            print(f"  bank-river delta: {deltas['bank-river']:.4f}")
            print(f"  bank-finance delta: {deltas['bank-finance']:.4f}")
            print(f"  apple-fruit delta: {deltas['apple-fruit']:.4f}")
            print(f"  apple-company delta: {deltas['apple-company']:.4f}")
            print(f"  bank消歧cos变化: {bank_cos_baseline:.4f} -> {bank_cos_ablated:.4f} (delta={bank_disamb_delta:+.4f})")
            print(f"  apple消歧cos变化: {apple_cos_baseline:.4f} -> {apple_cos_ablated:.4f} (delta={apple_disamb_delta:+.4f})")
        finally:
            remove_hooks(handles)

    # ── 实验3：多层MLP消融比较（L0/L8/L20/L35）──
    print("\n[4] 多层MLP消融比较...")
    compare_layers = [0, 8, 20, 35]
    layer_results = {}

    for lidx in compare_layers:
        if lidx >= n_layers:
            continue
        handles = ablate_mlp_sublayer(model, lidx, "full_mlp")
        try:
            avg_delta = 0.0
            for sent, label in zip(SENTENCES, LABELS):
                h = extract_last_hidden(model, tokenizer, sent)
                delta = 1.0 - cosine_sim(baselines[label], h)
                avg_delta += delta
            avg_delta /= len(LABELS)

            bank_cos = cosine_sim(
                extract_last_hidden(model, tokenizer, BANK_RIVER),
                extract_last_hidden(model, tokenizer, BANK_FINANCE)
            )
            apple_cos = cosine_sim(
                extract_last_hidden(model, tokenizer, APPLE_FRUIT),
                extract_last_hidden(model, tokenizer, APPLE_COMPANY)
            )
            layer_results[f"L{lidx}"] = {
                "avg_delta": round(avg_delta, 4),
                "bank_cos": round(bank_cos, 4),
                "apple_cos": round(apple_cos, 4),
            }
            print(f"  L{lidx}: avg_delta={avg_delta:.4f}, bank_cos={bank_cos:.4f}, apple_cos={apple_cos:.4f}")
        finally:
            remove_hooks(handles)

    # ── 实验4：Unembedding矩阵结构 ──────────────
    print("\n[5] Unembedding矩阵结构分析...")
    try:
        U = get_unembed_matrix(model)
        print(f"  unembedding矩阵形状: {U.shape}")

        # SVD分析
        U_centered = U - U.mean(dim=0, keepdim=True)
        _, S, Vt = torch.linalg.svd(U_centered, full_matrices=False)
        top5_sv = S[:10].tolist()
        total_sv = S.sum().item()
        top5_ratio = sum(S[:5]) / total_sv
        print(f"  Top-5奇异值: {[f'{v:.1f}' for v in top5_sv]}")
        print(f"  Top-5奇异值占比: {top5_ratio:.4f}")

        # 有效秩
        eff_rank = (S ** 2).sum() / (S ** 2).max()
        print(f"  有效秩: {eff_rank:.1f}")
    except Exception as e:
        print(f"  unembedding矩阵分析失败: {e}")

    # ── 实验5：hidden state -> logit因果分析 ──────
    print("\n[6] hidden state -> logit因果分析...")
    probe_words_bank = ["river", "water", "money", "loan", "deposit"]
    probe_words_apple = ["fruit", "tree", "orchard", "iPhone", "company", "technology"]

    U_matrix = get_unembed_matrix(model).cpu()

    for label, hidden in baselines.items():
        logits = hidden @ U_matrix.T  # (vocab_size,)
        topk_vals, topk_ids = torch.topk(logits, 5)
        topk_tokens = [repr(tokenizer.decode([tid.item()])) for tid in topk_ids]
        print(f"  {label} top-5 logits: {list(zip(topk_tokens, [f'{v:.2f}' for v in topk_vals.tolist()]))}")

    # ── 总结 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("总结:")
    print("-" * 70)

    # 找到delta最大的子层
    max_sublayer = max(results.keys(), key=lambda k: max(results[k]["deltas"].values()))
    print(f"  因果效力最大的子层: {max_sublayer}")
    for sublayer, data in results.items():
        avg_d = sum(data["deltas"].values()) / len(data["deltas"])
        print(f"    {sublayer}: avg_delta={avg_d:.4f}")

    # 层比较
    print(f"\n  层比较(avg_delta):")
    for l, data in layer_results.items():
        print(f"    {l}: {data['avg_delta']:.4f}")

    print(f"\n  消歧影响:")
    for sublayer, data in results.items():
        print(f"    {sublayer}: bank_delta={data['bank_disamb_delta']:+.4f}, apple_delta={data['apple_disamb_delta']:+.4f}")

    total_time = time.time() - t0
    print(f"\n  总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
