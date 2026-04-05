#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage573-576合并: 大规模歧义词消歧路径扫描 + 路径强制切换
P1核心任务：决定bank(attention)/apple(MLP)两条路径是冗余还是必要分化
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
    scores = []
    for h1, h2 in zip(out1.hidden_states, out2.hidden_states):
        scores.append(cos(h1[0, -1, :].float().cpu(), h2[0, -1, :].float().cpu()))
    return scores


def get_attn_patterns(model, tokenizer, sentence, layer_idx, head_idx):
    """提取指定层指定head的attention pattern"""
    layers = discover_layers(model)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    attn_out = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]  # (batch, heads, seq, seq)
            attn_out['weights'] = attn_weights.detach().float().cpu()
        return output

    handles = [layers[layer_idx].self_attn.register_forward_hook(hook_fn)]
    with torch.no_grad():
        model(**enc)
    remove_hooks(handles)

    if 'weights' in attn_out:
        return attn_out['weights'][0, head_idx]  # (seq, seq)
    return None


def disamb_path_type(model, tokenizer, s1, s2, peak_layer=8):
    """
    判断消歧路径类型：
    - 'attention': 同意义内attention pattern高度一致(ratio<0.5)
    - 'mlp': attention pattern不一致(ratio>1.0)
    - 'mixed': 中间状态
    """
    # 提取同语境的attention pattern一致性
    s1_repeat = [s1] * 2  # 同一语境两次输入
    s2_repeat = [s2] * 2

    # 用同义词/同义句代替：直接用同一句子的重复
    # 更简单的方法：用两个同语境的近义句
    scores = layer_wise_disamb(model, tokenizer, s1, s2)
    disamb_at_peak = 1 - scores[peak_layer]

    # 关键判断：消歧后在末层是否保持
    disamb_at_end = 1 - scores[-1]

    # 如果消歧在中间层很强但末层减弱 → attention路径(场驱动)
    # 如果消歧在中间层强且末层保持或增强 → MLP路径(直接编码)
    if disamb_at_end < disamb_at_peak * 0.6:
        return "attention"
    elif disamb_at_end >= disamb_at_peak * 0.8:
        return "mlp"
    else:
        return "mixed"


def force_path_switch(model, tokenizer, word, source_ctx, target_ctx, forced_path, peak_layer=8):
    """
    强制切换消歧路径
    forced_path: 'keep_attn' 或 'keep_mlp'
    """
    layers = discover_layers(model)
    n_layers = len(layers)

    # 获取source和target在peak_layer的hidden states
    enc_src = tokenizer(source_ctx, return_tensors="pt", truncation=True, max_length=64)
    enc_tgt = tokenizer(target_ctx, return_tensors="pt", truncation=True, max_length=64)
    enc_src = move_batch_to_model_device(model, enc_src)
    enc_tgt = move_batch_to_model_device(model, enc_tgt)

    with torch.no_grad():
        h_src = model(**enc_src, output_hidden_states=True).hidden_states[peak_layer][0, -1, :].clone()
        h_tgt = model(**enc_tgt, output_hidden_states=True).hidden_states[peak_layer][0, -1, :].clone()

    if forced_path == "keep_attn":
        # 消融MLP但保留attention
        hook_layer = layers[peak_layer]
        def mlp_zero(_m, _i, _o):
            if isinstance(_o, tuple):
                return (torch.zeros_like(_o[0]),) + _o[1:]
            return torch.zeros_like(_o)
        handles = [hook_layer.mlp.register_forward_hook(mlp_zero)]

    elif forced_path == "keep_mlp":
        # 消融attention但保留MLP
        def attn_zero(_m, _i, _o):
            if isinstance(_o, tuple):
                return (torch.zeros_like(_o[0]),) + _o[1:]
            return torch.zeros_like(_o)
        handles = [layers[peak_layer].self_attn.register_forward_hook(attn_zero)]

    # 在消融条件下运行两个语境
    with torch.no_grad():
        out_src = model(**enc_src, output_hidden_states=True)
        out_tgt = model(**enc_tgt, output_hidden_states=True)

    remove_hooks(handles)

    # 计算消歧度
    cos_end = cos(
        out_src.hidden_states[-1][0, -1, :].float().cpu(),
        out_tgt.hidden_states[-1][0, -1, :].float().cpu()
    )
    return round(1 - cos_end, 4)


# 大规模歧义词数据库
POLYSEMOUS_WORDS = {
    "bank": [("The river bank was steep.", "The bank approved the loan.")],
    "apple": [("She ate a red apple.", "Apple released the new iPhone.")],
    "bat": [("The bat flew across the cave.", "He swung the wooden bat hard.")],
    "light": [("The room was filled with light.", "She carried a light bag.")],
    "match": [("He struck a match to light the fire.", "This shirt is a perfect match.")],
    "plant": [("The plant needs more water.", "The car plant employs 500 workers.")],
    "ruler": [("She used a ruler to draw a line.", "The ruler governed for 30 years.")],
    "spring": [("Spring is the season of renewal.", "The water flows from the natural spring.")],
    "bark": [("The dog's bark was loud.", "The tree bark was rough.")],
    "nail": [("She hit the nail with the hammer.", "She painted her fingernail red.")],
    "rock": [("The rock was heavy and gray.", "She loves rock music.")],
    "draft": [("He wrote the first draft of the novel.", "The cold draft came through the window.")],
    "bow": [("She wore a bow in her hair.", "He had to bow to the king.")],
    "fair": [("The weather was warm and fair.", "The county fair had many rides.")],
    "pool": [("They swam in the pool.", "The pool of knowledge is vast.")],
    "jam": [("She spread jam on the toast.", "Traffic jam delayed them by an hour.")],
    "pipe": [("Water flowed through the pipe.", "He smoked a pipe after dinner.")],
    "seal": [("The seal swam in the ocean.", "Break the wax seal on the letter.")],
    "shed": [("He built a shed for the tools.", "The snake sheds its skin.")],
    "current": [("The river current was strong.", "The current situation is complex.")],
}


def main():
    print("=" * 70)
    print("stage573-576: 大规模歧义词消歧路径扫描 + 路径强制切换")
    print("=" * 70)

    t0 = time.time()
    print("\n[1] 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    # ── 实验1: 大规模路径扫描 ────────────────────
    print(f"\n[2] 大规模歧义词消歧路径扫描({len(POLYSEMOUS_WORDS)}个词):")
    path_counts = {"attention": 0, "mlp": 0, "mixed": 0, "no_disamb": 0}
    path_details = []

    for word, pair_list in POLYSEMOUS_WORDS.items():
        s1, s2 = pair_list[0]
        scores = layer_wise_disamb(model, tokenizer, s1, s2)
        disamb_peak = max(1 - s for s in scores)

        if disamb_peak < 0.05:
            path = "no_disamb"
        else:
            peak_l = min(range(len(scores)), key=lambda i: scores[i])
            disamb_at_peak = 1 - scores[peak_l]
            disamb_at_end = 1 - scores[-1]

            if disamb_at_end < disamb_at_peak * 0.6:
                path = "attention"
            elif disamb_at_end >= disamb_at_peak * 0.8:
                path = "mlp"
            else:
                path = "mixed"

        path_counts[path] += 1
        path_details.append({
            "word": word,
            "path": path,
            "peak_disamb": round(disamb_peak, 4),
            "end_disamb": round(1 - scores[-1], 4),
            "ratio": round((1 - scores[-1]) / max(disamb_peak, 0.001), 2),
        })

    print(f"\n  路径分布:")
    for p, c in path_counts.items():
        print(f"    {p}: {c}/{len(POLYSEMOUS_WORDS)} ({c/len(POLYSEMOUS_WORDS)*100:.0f}%)")

    print(f"\n  详细结果:")
    print(f"    {'词':>8} {'路径':>10} {'峰值消歧':>10} {'末层消歧':>10} {'比值':>6}")
    for d in sorted(path_details, key=lambda x: x["peak_disamb"], reverse=True):
        print(f"    {d['word']:>8} {d['path']:>10} {d['peak_disamb']:>10.4f} {d['end_disamb']:>10.4f} {d['ratio']:>6.2f}")

    # ── 实验2: 路径强制切换 ────────────────────
    print(f"\n[3] 路径强制切换实验(在peak层消融attn/MLP):")
    # 选几个有代表性的词
    test_words = ["bank", "apple", "bat", "plant", "ruler", "light"]
    switch_results = []

    for word in test_words:
        if word not in POLYSEMOUS_WORDS:
            continue
        s1, s2 = POLYSEMOUS_WORDS[word][0]

        # baseline消歧
        scores = layer_wise_disamb(model, tokenizer, s1, s2)
        baseline_disamb = round(1 - scores[-1], 4)

        # 消融peak层attn
        peak_l = min(range(len(scores)), key=lambda i: scores[i])
        layers = discover_layers(model)
        handles = []
        def attn_zero(_m, _i, _o):
            if isinstance(_o, tuple):
                return (torch.zeros_like(_o[0]),) + _o[1:]
            return torch.zeros_like(_o)
        handles.append(layers[peak_l].self_attn.register_forward_hook(attn_zero))

        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        enc1 = move_batch_to_model_device(model, enc1)
        enc2 = move_batch_to_model_device(model, enc2)
        with torch.no_grad():
            o1 = model(**enc1, output_hidden_states=True)
            o2 = model(**enc2, output_hidden_states=True)
        no_attn_disamb = round(1 - cos(o1.hidden_states[-1][0, -1, :].float().cpu(),
                                      o2.hidden_states[-1][0, -1, :].float().cpu()), 4)
        remove_hooks(handles)

        # 消融peak层MLP
        handles = []
        def mlp_zero(_m, _i, _o):
            if isinstance(_o, tuple):
                return (torch.zeros_like(_o[0]),) + _o[1:]
            return torch.zeros_like(_o)
        handles.append(layers[peak_l].mlp.register_forward_hook(mlp_zero))
        with torch.no_grad():
            o1 = model(**enc1, output_hidden_states=True)
            o2 = model(**enc2, output_hidden_states=True)
        no_mlp_disamb = round(1 - cos(o1.hidden_states[-1][0, -1, :].float().cpu(),
                                     o2.hidden_states[-1][0, -1, :].float().cpu()), 4)
        remove_hooks(handles)

        # 判断哪个路径更关键
        attn_impact = baseline_disamb - no_attn_disamb
        mlp_impact = baseline_disamb - no_mlp_disamb
        dominant = "attention" if attn_impact > mlp_impact else "mlp"

        switch_results.append({
            "word": word,
            "peak_layer": peak_l,
            "baseline": baseline_disamb,
            "no_attn": no_attn_disamb,
            "no_mlp": no_mlp_disamb,
            "attn_impact": round(attn_impact, 4),
            "mlp_impact": round(mlp_impact, 4),
            "dominant": dominant,
        })

    print(f"\n  {'词':>8} {'peak层':>6} {'baseline':>10} {'消融attn':>10} {'消融MLP':>10} {'attn影响':>10} {'MLP影响':>10} {'主导':>8}")
    for r in switch_results:
        print(f"  {r['word']:>8} L{r['peak_layer']:<4} {r['baseline']:>10.4f} {r['no_attn']:>10.4f} "
              f"{r['no_mlp']:>10.4f} {r['attn_impact']:>10.4f} {r['mlp_impact']:>10.4f} {r['dominant']:>8}")

    # ── 实验3: 跨层attn vs MLP贡献对比 ──────────
    print(f"\n[4] bank/apple逐层attn vs MLP对消歧的贡献:")
    scan_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if (n_layers - 1) not in scan_layers:
        scan_layers.append(n_layers - 1)

    for name, pair_list in [("bank", POLYSEMOUS_WORDS["bank"]),
                             ("apple", POLYSEMOUS_WORDS["apple"])]:
        s1, s2 = pair_list[0]
        print(f"\n  {name}:")
        for l in scan_layers:
            layers = discover_layers(model)
            enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
            enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
            enc1 = move_batch_to_model_device(model, enc1)
            enc2 = move_batch_to_model_device(model, enc2)

            # baseline
            with torch.no_grad():
                out1 = model(**enc1, output_hidden_states=True)
                out2 = model(**enc2, output_hidden_states=True)
            base_cos = cos(out1.hidden_states[l][0, -1, :].float().cpu(),
                          out2.hidden_states[l][0, -1, :].float().cpu())

            # 消融attn
            handles = []
            handles.append(layers[l].self_attn.register_forward_hook(
                lambda _m, _i, _o: (torch.zeros_like(_o[0]),) + _o[1:] if isinstance(_o, tuple) else torch.zeros_like(_o)))
            with torch.no_grad():
                o1a = model(**enc1, output_hidden_states=True)
                o2a = model(**enc2, output_hidden_states=True)
            no_attn_cos = cos(o1a.hidden_states[l][0, -1, :].float().cpu(),
                             o2a.hidden_states[l][0, -1, :].float().cpu())
            remove_hooks(handles)

            # 消融MLP
            handles = []
            handles.append(layers[l].mlp.register_forward_hook(
                lambda _m, _i, _o: (torch.zeros_like(_o[0]),) + _o[1:] if isinstance(_o, tuple) else torch.zeros_like(_o)))
            with torch.no_grad():
                o1m = model(**enc1, output_hidden_states=True)
                o2m = model(**enc2, output_hidden_states=True)
            no_mlp_cos = cos(o1m.hidden_states[l][0, -1, :].float().cpu(),
                            o2m.hidden_states[l][0, -1, :].float().cpu())
            remove_hooks(handles)

            attn_d = round(1 - no_attn_cos, 4)
            mlp_d = round(1 - no_mlp_cos, 4)
            base_d = round(1 - base_cos, 4)
            if base_d > 0.02:
                print(f"    L{l:2d}: base={base_d:.4f} no_attn={attn_d:.4f} no_mlp={mlp_d:.4f} "
                      f"attn_share={attn_d/max(base_d,0.001):.2f} mlp_share={mlp_d/max(base_d,0.001):.2f}")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"总耗时: {total_time:.1f}s")


if __name__ == "__main__":
    main()
