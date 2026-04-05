#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage506: 翻译机制逐层分析
分析翻译（如 "apple" -> "苹果"）在各层中的运行机制

核心问题：
1. 英文词和中文翻译在哪些层最相似？（对齐层）
2. 翻译对 vs 非翻译对的注意力模式差异
3. 翻译过程中信息是如何从一种语言转换到另一种语言的？
4. 是否存在专门的"翻译子空间"？

分析维度：
- T1: 跨语言余弦相似度——翻译对在各层的相似度
- T2: 翻译注意力指纹——生成中文时对英文源词的注意力
- T3: 翻译子空间检测——找到区分翻译/非翻译对的子空间
- T4: 逐层信息流——从英文输入到中文输出的信息传递路径
- T5: 跨模型翻译对齐一致性
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device,
    qwen_hidden_dim,
)


# ============================================================
# 翻译对和非翻译对
# ============================================================
TRANSLATION_PAIRS = [
    ("apple", "苹果"), ("fruit", "水果"), ("city", "城市"),
    ("water", "水"), ("sun", "太阳"), ("book", "书"),
    ("mountain", "山"), ("river", "河流"), ("fire", "火"),
    ("tree", "树"), ("flower", "花"), ("earth", "地球"),
]

# 非翻译但语义相关
NON_TRANSLATION_PAIRS = [
    ("apple", "水果"), ("city", "建筑"), ("water", "雨"),
    ("sun", "月亮"), ("book", "图书馆"), ("mountain", "石头"),
    ("river", "船"), ("fire", "烟"), ("tree", "森林"),
    ("flower", "草"), ("earth", "天空"), ("apple", "梨"),
]

# 语义无关对
UNRELATED_PAIRS = [
    ("apple", "城市"), ("city", "水"), ("water", "书"),
    ("sun", "山"), ("book", "河流"), ("mountain", "火"),
]


def get_hidden_at_last_token(model, tokenizer, text, layer_indices):
    """获取文本最后一个token在各层的隐藏状态"""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    device = get_model_device(model)
    input_ids = encoded["input_ids"].to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    result = {}
    for li in layer_indices:
        h = outputs.hidden_states[li][0, -1].float().cpu()
        result[li] = h
    return result


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ============================================================
# T1: 跨语言余弦相似度
# ============================================================
def analyze_T1_cross_lingual_cosine(model, tokenizer, layer_indices):
    """T1: 翻译对在各层的余弦相似度"""
    results = {}

    for pair_type, pairs in [("translation", TRANSLATION_PAIRS),
                               ("related", NON_TRANSLATION_PAIRS),
                               ("unrelated", UNRELATED_PAIRS)]:
        results[pair_type] = {}
        for li in layer_indices:
            sims = []
            for en, zh in pairs:
                h_en = get_hidden_at_last_token(model, tokenizer, en, [li])
                h_zh = get_hidden_at_last_token(model, tokenizer, zh, [li])
                sim = cosine_sim(h_en[li], h_zh[li])
                sims.append(sim)
            results[pair_type][f"L{li}"] = {
                "mean": round(np.mean(sims), 4),
                "std": round(np.std(sims), 4),
                "min": round(np.min(sims), 4),
                "max": round(np.max(sims), 4),
            }

    # 关键指标：翻译对 vs 非翻译对的差异
    for li in layer_indices:
        trans_mean = results["translation"][f"L{li}"]["mean"]
        related_mean = results["related"][f"L{li}"]["mean"]
        unrelated_mean = results["unrelated"][f"L{li}"]["mean"]
        results[f"L{li}_discrimination"] = {
            "trans_vs_related": round(trans_mean - related_mean, 4),
            "trans_vs_unrelated": round(trans_mean - unrelated_mean, 4),
        }

    # 找到翻译对齐最强的层
    best_layer = None
    best_signal = -999
    for li in layer_indices:
        signal = results[f"L{li}_discrimination"]["trans_vs_unrelated"]
        if signal > best_signal:
            best_signal = signal
            best_layer = f"L{li}"
    results["_best_alignment_layer"] = best_layer
    results["_best_alignment_signal"] = round(best_signal, 4)

    return {"T1_cross_lingual_cosine": results}


# ============================================================
# T2: 翻译注意力指纹
# ============================================================
def analyze_T2_translation_attention(model, tokenizer, layer_indices):
    """T2: 生成中文时对英文源词的注意力"""
    results = {}

    # 构造翻译提示
    translation_prompts = [
        ("Translate apple to Chinese:", "苹果"),
        ("Translate fruit to Chinese:", "水果"),
        ("Translate city to Chinese:", "城市"),
        ("The apple is", "苹果"),
    ]

    for prompt, expected_zh in translation_prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        device = get_model_device(model)
        input_ids = encoded["input_ids"].to(device)

        with torch.inference_mode():
            try:
                outputs = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
            except Exception:
                results[prompt] = {"error": "model does not support output_attentions"}
                continue

        prompt_key = prompt[:30]
        results[prompt_key] = {}
        for li in layer_indices:
            if outputs.attentions is None:
                results[prompt_key][f"L{li}"] = {"attention_available": False}
                continue
            attn = outputs.attentions[li][0].float().cpu()  # [heads, seq, seq]
            # 平均注意力
            avg_attn = attn.mean(dim=0)  # [seq, seq]
            # 最后一个token对前面所有token的注意力分布
            last_row = avg_attn[-1]
            results[prompt_key][f"L{li}"] = {
                "attention_entropy": round(float(-torch.softmax(last_row, dim=0).dot(
                    torch.log_softmax(last_row, dim=0) + 1e-10)), 4),
                "max_attention_pos": int(last_row.argmax().item()),
                "attention_max": round(float(last_row.max()), 4),
                "attention_min": round(float(last_row.min()), 4),
            }

    return {"T2_translation_attention": results}


# ============================================================
# T3: 翻译子空间检测
# ============================================================
def analyze_T3_translation_subspace(model, tokenizer, layer_indices):
    """T3: 用PCA找到翻译对齐的子空间"""
    results = {}

    # 选择中间层和深层分析
    target_layers = [layer_indices[len(layer_indices) // 3],
                     layer_indices[2 * len(layer_indices) // 3]]

    for li in target_layers:
        # 收集翻译对和非翻译对的差异向量
        trans_diffs = []
        nontrans_diffs = []

        for en, zh in TRANSLATION_PAIRS:
            h_en = get_hidden_at_last_token(model, tokenizer, en, [li])
            h_zh = get_hidden_at_last_token(model, tokenizer, zh, [li])
            diff = (h_zh[li] - h_en[li]).unsqueeze(0)
            trans_diffs.append(diff)

        for en, zh in NON_TRANSLATION_PAIRS:
            h_en = get_hidden_at_last_token(model, tokenizer, en, [li])
            h_zh = get_hidden_at_last_token(model, tokenizer, zh, [li])
            diff = (h_zh[li] - h_en[li]).unsqueeze(0)
            nontrans_diffs.append(diff)

        # PCA on 差异向量
        all_diffs = torch.cat(trans_diffs + nontrans_diffs, dim=0).numpy()
        mean_diff = all_diffs.mean(axis=0, keepdims=True)
        centered = all_diffs - mean_diff
        n_components = min(5, centered.shape[0] - 1, centered.shape[1])
        if n_components < 1:
            continue
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        total_var = (S ** 2).sum()
        explained = [(S[i] ** 2) / total_var for i in range(min(n_components, len(S)))]

        # 翻译差异 vs 非翻译差异在第一主成分上的投影
        trans_proj = centered[:len(trans_diffs)] @ Vt[0]
        nontrans_proj = centered[len(trans_diffs):] @ Vt[0]

        # 用第一主成分能否区分翻译/非翻译？
        all_proj = np.concatenate([trans_proj, nontrans_proj])
        labels = [1] * len(trans_diffs) + [0] * len(nontrans_diffs)
        threshold = np.median(all_proj)
        correct = sum(1 for p, l in zip(all_proj, labels) if (p > threshold) == (l == 1))

        results[f"L{li}"] = {
            "top5_explained_variance": [round(e, 4) for e in explained[:5]],
            "trans_proj_mean": round(float(np.mean(trans_proj)), 4),
            "nontrans_proj_mean": round(float(np.mean(nontrans_proj)), 4),
            "pc1_separability": round(correct / len(all_proj), 4),
            "n_trans_pairs": len(trans_diffs),
            "n_nontrans_pairs": len(nontrans_diffs),
        }

    return {"T3_translation_subspace": results}


# ============================================================
# T4: 逐层信息流——英文输入到中文输出的传递路径
# ============================================================
def analyze_T4_information_flow(model, tokenizer, layer_indices):
    """T4: 分析信息如何从英文token传递到中文token"""
    results = {}

    # 构造包含英中的序列
    sequences = [
        "Apple is a fruit. 苹果是一种水果。",
        "The city is beautiful. 这座城市很美。",
        "I like water. 我喜欢水。",
    ]

    for seq in sequences:
        seq_key = seq[:20] + "..."
        encoded = tokenizer(seq, return_tensors="pt", truncation=True, max_length=64)
        device = get_model_device(model)
        input_ids = encoded["input_ids"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        # 获取token位置
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        results[seq_key] = {
            "tokens": tokens[:20],  # 限制长度
            "token_count": len(tokens),
        }

        for li in layer_indices:
            hidden = outputs.hidden_states[li][0].float().cpu()  # [seq_len, hidden_dim]
            # 计算每个token位置与前一个token的信息变化量
            changes = []
            for t in range(1, hidden.shape[0]):
                change = (hidden[t] - hidden[t - 1]).norm().item()
                changes.append(round(change, 2))
            results[seq_key][f"L{li}_token_changes"] = changes

            # 英文部分到中文部分的过渡：找到中英文边界
            # 用token变化的最大跳跃点作为语言切换点
            if changes:
                max_change_idx = changes.index(max(changes))
                results[seq_key][f"L{li}_language_switch_pos"] = max_change_idx

    return {"T4_information_flow": results}


# ============================================================
# T5: 同义词 vs 翻译对——区分两种语义关系
# ============================================================
def analyze_T5_synonym_vs_translation(model, tokenizer, layer_indices):
    """T5: 同义词对 vs 翻译对在向量空间中的模式差异"""
    results = {}

    # 中文同义词对
    synonym_pairs = [
        ("苹果", "苹果"),  # 相同词
        ("美丽", "漂亮"), ("高兴", "快乐"), ("大", "巨大"),
        ("小", "微小"), ("快", "迅速"), ("好", "优秀"),
    ]

    for li in layer_indices:
        trans_sims = []
        syn_sims = []
        unrelated_sims = []

        for en, zh in TRANSLATION_PAIRS[:8]:
            h_en = get_hidden_at_last_token(model, tokenizer, en, [li])
            h_zh = get_hidden_at_last_token(model, tokenizer, zh, [li])
            trans_sims.append(cosine_sim(h_en[li], h_zh[li]))

        for w1, w2 in synonym_pairs:
            if w1 == w2:
                continue
            h1 = get_hidden_at_last_token(model, tokenizer, w1, [li])
            h2 = get_hidden_at_last_token(model, tokenizer, w2, [li])
            syn_sims.append(cosine_sim(h1[li], h2[li]))

        for en, zh in UNRELATED_PAIRS[:6]:
            h_en = get_hidden_at_last_token(model, tokenizer, en, [li])
            h_zh = get_hidden_at_last_token(model, tokenizer, zh, [li])
            unrelated_sims.append(cosine_sim(h_en[li], h_zh[li]))

        results[f"L{li}"] = {
            "translation_pair_avg": round(np.mean(trans_sims), 4) if trans_sims else 0,
            "synonym_pair_avg": round(np.mean(syn_sims), 4) if syn_sims else 0,
            "unrelated_pair_avg": round(np.mean(unrelated_sims), 4) if unrelated_sims else 0,
            "trans_vs_synonym_gap": round(
                (np.mean(trans_sims) if trans_sims else 0) - (np.mean(syn_sims) if syn_sims else 0), 4),
        }

    return {"T5_synonym_vs_translation": results}


# ============================================================
# 主流程
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python stage506_translation_mechanism.py <model_name>")
        print("  model_name: qwen3 | deepseek7b | gemma4")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"[Stage506] 翻译机制逐层分析 - {model_name}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载模型...")

    if model_name == "qwen3":
        model, tokenizer = load_qwen3_model()
    elif model_name == "deepseek7b":
        model, tokenizer = load_deepseek7b_model()
    elif model_name == "gemma4":
        model, tokenizer = load_gemma4_model()
    else:
        print(f"未知模型: {model_name}")
        sys.exit(1)

    n_layers = len(discover_layers(model))
    hidden_dim = qwen_hidden_dim(model)
    print(f"  层数={n_layers}, 隐藏维度={hidden_dim}")

    n_samples = min(8, n_layers)
    layer_indices = [round(i * (n_layers - 1) / (n_samples - 1)) for i in range(n_samples)]

    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_indices": layer_indices,
        "timestamp": datetime.now().isoformat(),
    }

    # T1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] T1: 跨语言余弦相似度...")
    t0 = time.time()
    t1 = analyze_T1_cross_lingual_cosine(model, tokenizer, layer_indices)
    summary.update(t1)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # T2
    print(f"[{datetime.now().strftime('%H:%M:%S')}] T2: 翻译注意力指纹...")
    t0 = time.time()
    t2 = analyze_T2_translation_attention(model, tokenizer, layer_indices)
    summary.update(t2)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # T3
    print(f"[{datetime.now().strftime('%H:%M:%S')}] T3: 翻译子空间检测...")
    t0 = time.time()
    t3 = analyze_T3_translation_subspace(model, tokenizer, layer_indices)
    summary.update(t3)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # T4
    print(f"[{datetime.now().strftime('%H:%M:%S')}] T4: 逐层信息流...")
    t0 = time.time()
    t4 = analyze_T4_information_flow(model, tokenizer, layer_indices)
    summary.update(t4)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # T5
    print(f"[{datetime.now().strftime('%H:%M:%S')}] T5: 同义词 vs 翻译对...")
    t0 = time.time()
    t5 = analyze_T5_synonym_vs_translation(model, tokenizer, layer_indices)
    summary.update(t5)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # 保存
    out_dir = Path("tests/codex_temp") / f"stage506_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{model_name}.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印关键发现
    print("\n" + "=" * 60)
    print("关键发现摘要")
    print("=" * 60)

    t1_data = summary["T1_cross_lingual_cosine"]
    print(f"\n[T1] 翻译对齐最佳层: {t1_data['_best_alignment_layer']}, "
          f"信号强度: {t1_data['_best_alignment_signal']:.3f}")
    print("  各层翻译/相关/无关平均相似度:")
    for li in layer_indices:
        trans = t1_data["translation"][f"L{li}"]["mean"]
        related = t1_data["related"][f"L{li}"]["mean"]
        unrelated = t1_data["unrelated"][f"L{li}"]["mean"]
        disc = t1_data[f"L{li}_discrimination"]["trans_vs_unrelated"]
        print(f"    L{li}: 翻译={trans:.3f}, 相关={related:.3f}, 无关={unrelated:.3f}, 区分度={disc:.3f}")

    t3_data = summary["T3_translation_subspace"]
    print("\n[T3] 翻译子空间检测:")
    for li_str, data in t3_data.items():
        print(f"  {li_str}: PC1分离度={data['pc1_separability']:.3f}, "
              f"前5PC方差比={data['top5_explained_variance']}")

    t5_data = summary["T5_synonym_vs_translation"]
    print("\n[T5] 同义词 vs 翻译对:")
    for li in layer_indices:
        data = t5_data[f"L{li}"]
        print(f"  L{li}: 翻译对={data['translation_pair_avg']:.3f}, "
              f"同义词={data['synonym_pair_avg']:.3f}, 差距={data['trans_vs_synonym_gap']:.3f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
