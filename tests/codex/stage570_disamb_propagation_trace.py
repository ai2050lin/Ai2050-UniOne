#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage570: 消歧信号从L8到logit的完整传播路径追踪
目标：在bank/apple消歧实验中，追踪消歧信号从L8(峰值层)到最终logit的完整传播
模型：Qwen3-4B
"""

from __future__ import annotations
import sys, time, torch
import torch.nn.functional as F
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


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_disamb_score(model, tokenizer, sent1, sent2, token_pos=-1):
    """获取两个句子在指定位置的hidden state的消歧cosine距离"""
    enc1 = tokenizer(sent1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(sent2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)
    with torch.no_grad():
        h1 = model(**enc1, output_hidden_states=True).hidden_states[-1][0, token_pos, :].float().cpu()
        h2 = model(**enc2, output_hidden_states=True).hidden_states[-1][0, token_pos, :].float().cpu()
    return cos(h1, h2)


def layer_wise_disamb(model, tokenizer, sent1, sent2):
    """逐层提取消歧cosine距离"""
    enc1 = tokenizer(sent1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(sent2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    scores = []
    for l, (h1, h2) in enumerate(zip(out1.hidden_states, out2.hidden_states)):
        c = cos(h1[0, -1, :].float().cpu(), h2[0, -1, :].float().cpu())
        scores.append(c)
    return scores


def per_component_contribution(model, tokenizer, sent, layer_idx):
    """分析attn vs MLP各自的贡献"""
    layers = discover_layers(model)
    enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    # baseline
    with torch.no_grad():
        baseline = model(**enc, output_hidden_states=True).hidden_states[layer_idx][0, -1, :].float().cpu()

    # 消融attn
    handles = []
    def zero_attn(_m, _i, _o):
        if isinstance(_o, tuple):
            return (torch.zeros_like(_o[0]),) + _o[1:]
        return torch.zeros_like(_o)
    handles.append(layers[layer_idx].self_attn.register_forward_hook(zero_attn))
    with torch.no_grad():
        no_attn = model(**enc, output_hidden_states=True).hidden_states[layer_idx][0, -1, :].float().cpu()
    remove_hooks(handles)

    # 消融MLP
    handles = []
    def zero_mlp(_m, _i, _o):
        if isinstance(_o, tuple):
            return (torch.zeros_like(_o[0]),) + _o[1:]
        return torch.zeros_like(_o)
    handles.append(layers[layer_idx].mlp.register_forward_hook(zero_mlp))
    with torch.no_grad():
        no_mlp = model(**enc, output_hidden_states=True).hidden_states[layer_idx][0, -1, :].float().cpu()
    remove_hooks(handles)

    attn_contrib = cos(baseline, no_attn)  # attn被移除后的偏差
    mlp_contrib = cos(baseline, no_mlp)

    # 各自的信息量
    attn_delta = 1 - attn_contrib
    mlp_delta = 1 - mlp_contrib

    return {"attn_delta": round(attn_delta, 6), "mlp_delta": round(mlp_delta, 6),
            "attn_cos": round(attn_contrib, 4), "mlp_cos": round(mlp_contrib, 4)}


def logit_space_disamb(model, tokenizer, sent1, sent2):
    """在logit空间计算消歧度"""
    enc1 = tokenizer(sent1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(sent2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)

    U = model.lm_head.weight.float().cpu() if hasattr(model, 'lm_head') else model.model.embed_tokens.weight.float().cpu()

    with torch.no_grad():
        h1 = model(**enc1, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()
        h2 = model(**enc2, output_hidden_states=True).hidden_states[-1][0, -1, :].float().cpu()

    logit1 = h1 @ U.T
    logit2 = h2 @ U.T
    return cos(logit1, logit2)


def main():
    print("=" * 70)
    print("stage570: 消歧信号传播路径追踪 L8 -> logit")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    layers = discover_layers(model)
    n_layers = len(layers)
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    # ── 实验1: 逐层消歧度变化 ──────────────────
    print("\n[2] 逐层消歧度变化(1-cos=消歧距离):")
    pairs = [
        ("bank", BANK_RIVER, BANK_FINANCE),
        ("apple", APPLE_FRUIT, APPLE_COMPANY),
    ]

    for name, s1, s2 in pairs:
        scores = layer_wise_disamb(model, tokenizer, s1, s2)
        # 找到峰值层
        peak_layer = min(range(len(scores)), key=lambda i: scores[i])
        print(f"\n  {name}消歧距离(1-cos):")
        key_layers = list(range(0, n_layers, max(1, n_layers // 12)))
        if (n_layers - 1) not in key_layers:
            key_layers.append(n_layers - 1)
        for l in key_layers:
            disamb = 1 - scores[l]
            marker = " <-- PEAK" if l == peak_layer else ""
            print(f"    L{l:2d}: disamb={disamb:.4f} (cos={scores[l]:.4f}){marker}")

    # ── 实验2: 各层attn vs MLP对消歧的贡献 ────
    print("\n[3] 各层attn vs MLP对消歧的贡献(delta):")
    scan_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if (n_layers - 1) not in scan_layers:
        scan_layers.append(n_layers - 1)

    for l in scan_layers:
        for name, s1, s2 in pairs:
            # bank: river语境下该层的attn/mlp贡献
            c1 = per_component_contribution(model, tokenizer, s1, l)
            c2 = per_component_contribution(model, tokenizer, s2, l)
            diff = abs(c1["attn_delta"] - c2["attn_delta"]) + abs(c1["mlp_delta"] - c2["mlp_delta"])
            if diff > 0.01:
                print(f"  L{l:2d} {name:6s}: attn_d=[{c1['attn_delta']:.4f},{c2['attn_delta']:.4f}] "
                      f"mlp_d=[{c1['mlp_delta']:.4f},{c2['mlp_delta']:.4f}] diff={diff:.4f}")

    # ── 实验3: logit空间消歧度 ──────────────────
    print("\n[4] logit空间消歧度:")
    for name, s1, s2 in pairs:
        lc = logit_space_disamb(model, tokenizer, s1, s2)
        print(f"  {name}: logit_cos={lc:.4f}, logit_disamb={1-lc:.4f}")

    # ── 实验4: 消歧信号传播——逐层注入实验 ──────
    print("\n[5] 消歧信号跨层传播追踪:")
    # 在L8用forward hook替换hidden state为target语境，看下游层如何传播
    for name, src_sent, tgt_sent in pairs:
        enc_src = tokenizer(src_sent, return_tensors="pt", truncation=True, max_length=64)
        enc_tgt = tokenizer(tgt_sent, return_tensors="pt", truncation=True, max_length=64)
        enc_src = move_batch_to_model_device(model, enc_src)
        enc_tgt = move_batch_to_model_device(model, enc_tgt)

        # target hidden state at L8
        with torch.no_grad():
            tgt_h8 = model(**enc_tgt, output_hidden_states=True).hidden_states[8][0, -1, :].clone()

        # hook L8 to inject target hidden
        buffers = {}
        def make_hook(h_val):
            def hook(_m, _i, output):
                # output is (hidden_states, ...) tuple for attention or just tensor for MLP
                if isinstance(output, tuple):
                    hs = output[0]
                    hs[0, -1, :] = h_val.to(hs.device, hs.dtype)
                    return (hs,) + output[1:]
                else:
                    return output
            return hook

        handles = [layers[8].register_forward_hook(make_hook(tgt_h8))]

        with torch.no_grad():
            hooked_out = model(**enc_src, output_hidden_states=True)
        remove_hooks(handles)

        # 比较hooked和target在后续层的cosine
        with torch.no_grad():
            tgt_full = model(**enc_tgt, output_hidden_states=True)

        print(f"\n  {name} hook@L8传播(cos hooked vs target):")
        for l in [8, 10, 12, 16, 20, 24, 28, 32, 35]:
            if l >= n_layers:
                continue
            h_hooked = hooked_out.hidden_states[l][0, -1, :].float().cpu()
            h_target = tgt_full.hidden_states[l][0, -1, :].float().cpu()
            c = cos(h_hooked, h_target)
            print(f"    L{l:2d}: cos={c:.4f}")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
