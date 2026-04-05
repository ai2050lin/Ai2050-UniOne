#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage507: 多token上下文验证
核心问题：深层cos=1.0是否是单token分析的假象？用完整句子是否恢复层级/翻译结构？

对照实验：
- A1: 单token vs 句子上下文——层级相似度对比
- A2: 单token vs 句子上下文——翻译对齐对比
- A3: 上下文长度对层级信号的影响
- A4: 不同句子模板的稳定性
- A5: 深层残差流方差分析——cos=1.0是因为方差归零？
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device, qwen_hidden_dim,
)


# ============================================================
# 数据定义
# ============================================================
CONCEPT_PAIRS = [
    # (具体, 类别, 句子模板)
    ("苹果", "水果", "我喜欢吃苹果，它是一种{cat}"),
    ("猫", "动物", "我养了一只猫，它是一种{cat}"),
    ("北京", "城市", "北京是一个美丽的{cat}"),
    ("太阳", "恒星", "太阳是一颗{cat}"),
]

TRANSLATION_PAIRS = [
    ("apple", "苹果", "The apple is red."),
    ("water", "水", "I drink water every day."),
    ("city", "城市", "This is a big city."),
]

HIERARCHY_WORDS = {
    "L1": ["苹果", "香蕉", "葡萄"],
    "L2": ["水果", "蔬菜", "肉类"],
    "L3": ["食物", "饮料", "衣服"],
}


def get_hidden_states(model, tokenizer, text, layer_indices, target_pos=-1):
    """获取指定位置的隐藏状态"""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    device = get_model_device(model)
    input_ids = encoded["input_ids"].to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    result = {}
    for li in layer_indices:
        h = outputs.hidden_states[li][0, target_pos].float().cpu()
        result[li] = h
    return result


def find_token_position(tokenizer, text, target_word):
    """找到目标词在tokenized序列中的位置"""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    # 找最匹配的token
    target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    for i, t in enumerate(tokens):
        if t in [tokenizer.convert_ids_to_tokens([tid])[0] for tid in target_tokens]:
            return i
    return -1


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ============================================================
# A1: 单token vs 句子上下文——层级相似度
# ============================================================
def analyze_A1_context_hierarchy(model, tokenizer, layer_indices):
    """A1: 对比单token和句子上下文中的层级信号"""
    results = {"single_token": {}, "sentence_context": {}}

    # 单token
    for level_name, words in HIERARCHY_WORDS.items():
        for w in words:
            h_map = get_hidden_states(model, tokenizer, w, layer_indices)
            for li in layer_indices:
                key = f"{level_name}_{w}"
                if key not in results["single_token"]:
                    results["single_token"][key] = {}
                results["single_token"][key][f"L{li}"] = h_map[li]

    # 句子上下文
    templates = [
        "这是{word}，它是{level}的一种。",
        "{word}很好，是一种常见的{level}。",
        "我看到了{word}，它属于{level}。",
    ]
    level_map = {"L1": "水果/蔬菜/肉类", "L2": "食物/饮料", "L3": "日用品"}
    for level_name, words in HIERARCHY_WORDS.items():
        for w in words:
            # 选一个模板
            tpl = templates[0].format(word=w, level=level_name)
            pos = find_token_position(tokenizer, tpl, w)
            if pos < 0:
                pos = -1  # fallback to last
            h_map = get_hidden_states(model, tokenizer, tpl, layer_indices, target_pos=pos)
            key = f"{level_name}_{w}"
            if key not in results["sentence_context"]:
                results["sentence_context"][key] = {}
            for li in layer_indices:
                results["sentence_context"][key][f"L{li}"] = h_map[li]

    # 计算层级信号：同层级 vs 不同层级
    for mode in ["single_token", "sentence_context"]:
        layer_signals = {}
        for li in layer_indices:
            intra_sims = []
            cross_sims = []
            level_keys = {ln: [f"{ln}_{w}" for w in HIERARCHY_WORDS[ln]] for ln in HIERARCHY_WORDS}
            for ln, keys in level_keys.items():
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        if keys[i] in results[mode] and keys[j] in results[mode]:
                            sim = cosine_sim(results[mode][keys[i]][f"L{li}"],
                                            results[mode][keys[j]][f"L{li}"])
                            intra_sims.append(sim)
            for ln1 in level_keys:
                for ln2 in level_keys:
                    if ln1 >= ln2:
                        continue
                    for k1 in level_keys[ln1][:2]:
                        for k2 in level_keys[ln2][:2]:
                            if k1 in results[mode] and k2 in results[mode]:
                                sim = cosine_sim(results[mode][k1][f"L{li}"],
                                                results[mode][k2][f"L{li}"])
                                cross_sims.append(sim)
            intra_avg = np.mean(intra_sims) if intra_sims else 0
            cross_avg = np.mean(cross_sims) if cross_sims else 0
            layer_signals[f"L{li}"] = {
                "intra_avg": round(intra_avg, 4),
                "cross_avg": round(cross_avg, 4),
                "hierarchy_signal": round(intra_avg - cross_avg, 4),
            }
        results[f"{mode}_signals"] = layer_signals

    return {"A1_context_hierarchy": results}


# ============================================================
# A2: 单token vs 句子上下文——翻译对齐
# ============================================================
def analyze_A2_context_translation(model, tokenizer, layer_indices):
    """A2: 对比单token和句子上下文中的翻译对齐"""
    results = {"single_token": {}, "sentence_context": {}}

    for en, zh, sentence in TRANSLATION_PAIRS:
        # 单token
        h_en = get_hidden_states(model, tokenizer, en, layer_indices)
        h_zh = get_hidden_states(model, tokenizer, zh, layer_indices)
        results["single_token"][f"{en}_{zh}"] = {}
        for li in layer_indices:
            results["single_token"][f"{en}_{zh}"][f"L{li}"] = cosine_sim(h_en[li], h_zh[li])

        # 句子上下文
        en_pos = find_token_position(tokenizer, sentence, en)
        h_en_ctx = get_hidden_states(model, tokenizer, sentence, layer_indices,
                                     target_pos=max(0, en_pos))
        zh_sentence = sentence.replace(en, zh)
        zh_pos = find_token_position(tokenizer, zh_sentence, zh)
        h_zh_ctx = get_hidden_states(model, tokenizer, zh_sentence, layer_indices,
                                     target_pos=max(0, zh_pos))
        results["sentence_context"][f"{en}_{zh}"] = {}
        for li in layer_indices:
            results["sentence_context"][f"{en}_{zh}"][f"L{li}"] = cosine_sim(h_en_ctx[li], h_zh_ctx[li])

    # 汇总
    for mode in ["single_token", "sentence_context"]:
        layer_avgs = {}
        for li in layer_indices:
            sims = [results[mode][pair][f"L{li}"] for pair in results[mode]]
            layer_avgs[f"L{li}"] = round(np.mean(sims), 4)
        results[f"{mode}_avg"] = layer_avgs

    return {"A2_context_translation": results}


# ============================================================
# A3: 上下文长度对层级信号的影响
# ============================================================
def analyze_A3_context_length(model, tokenizer, layer_indices):
    """A3: 不同上下文长度下的层级信号"""
    results = {}

    # 用固定词对
    specific_words = ["苹果", "香蕉"]
    general_words = ["水果", "蔬菜"]

    context_lengths = {
        "no_context": lambda w: w,
        "short": lambda w: f"这是{w}",
        "medium": lambda w: f"我昨天买了一些{w}，很好吃",
        "long": lambda w: f"我昨天去了超市买了一些新鲜的{w}，回家后做了一道美味的菜，全家人都很喜欢吃",
    }

    for length_name, tpl_fn in context_lengths.items():
        results[length_name] = {}
        for li in layer_indices:
            # 具体vs具体
            h1 = get_hidden_states(model, tokenizer, tpl_fn(specific_words[0]), layer_indices)
            h2 = get_hidden_states(model, tokenizer, tpl_fn(specific_words[1]), layer_indices)
            specific_sim = cosine_sim(h1[li], h2[li])

            # 一般vs一般
            h3 = get_hidden_states(model, tokenizer, tpl_fn(general_words[0]), layer_indices)
            h4 = get_hidden_states(model, tokenizer, tpl_fn(general_words[1]), layer_indices)
            general_sim = cosine_sim(h3[li], h4[li])

            # 具体vs一般
            cross_sims = []
            for sw in specific_words:
                for gw in general_words:
                    hs = get_hidden_states(model, tokenizer, tpl_fn(sw), layer_indices)
                    hg = get_hidden_states(model, tokenizer, tpl_fn(gw), layer_indices)
                    cross_sims.append(cosine_sim(hs[li], hg[li]))

            results[length_name][f"L{li}"] = {
                "specific_sim": round(specific_sim, 4),
                "general_sim": round(general_sim, 4),
                "cross_sim_avg": round(np.mean(cross_sims), 4),
                "hierarchy_signal": round(specific_sim - np.mean(cross_sims), 4),
            }

    return {"A3_context_length": results}


# ============================================================
# A4: 句子模板稳定性
# ============================================================
def analyze_A4_template_stability(model, tokenizer, layer_indices):
    """A4: 不同句子模板下概念表示的稳定性"""
    results = {}

    templates = [
        "这是{w}。",
        "{w}很好。",
        "我喜欢{w}。",
        "你知道{w}吗？",
        "{w}是什么意思？",
    ]
    target_word = "苹果"

    for li in [layer_indices[0], layer_indices[-1]]:
        hidden_list = []
        for tpl in templates:
            text = tpl.format(w=target_word)
            pos = find_token_position(tokenizer, text, target_word)
            h = get_hidden_states(model, tokenizer, text, [li], target_pos=max(0, pos))
            hidden_list.append(h[li])

        # 计算不同模板间的一致性
        pair_sims = []
        for i in range(len(hidden_list)):
            for j in range(i + 1, len(hidden_list)):
                pair_sims.append(cosine_sim(hidden_list[i], hidden_list[j]))

        results[f"L{li}"] = {
            "template_count": len(templates),
            "mean_consistency": round(np.mean(pair_sims), 4),
            "min_consistency": round(np.min(pair_sims), 4),
            "max_consistency": round(np.max(pair_sims), 4),
            "std_consistency": round(np.std(pair_sims), 4),
        }

    return {"A4_template_stability": results}


# ============================================================
# A5: 深层残差流方差分析
# ============================================================
def analyze_A5_deep_variance(model, tokenizer, layer_indices):
    """A5: 各层的隐藏状态范数和方差"""
    results = {}

    test_words = ["苹果", "水果", "物体", "猫", "动物", "城市"]

    for li in layer_indices:
        norms = []
        variances = []
        for w in test_words:
            h = get_hidden_states(model, tokenizer, w, [li])
            norms.append(float(h[li].norm()))
            variances.append(float(h[li].var()))

        # 也计算所有词表示之间的方差
        all_vecs = []
        for w in test_words:
            h = get_hidden_states(model, tokenizer, w, [li])
            all_vecs.append(h[li])
        stacked = torch.stack(all_vecs)
        inter_word_std = float(stacked.std(dim=0).mean())

        results[f"L{li}"] = {
            "mean_norm": round(np.mean(norms), 4),
            "std_norm": round(np.std(norms), 4),
            "mean_variance": round(np.mean(variances), 4),
            "inter_word_std": round(inter_word_std, 4),
            "norm_range": [round(np.min(norms), 4), round(np.max(norms), 4)],
        }

    return {"A5_deep_variance": results}


# ============================================================
# 主流程
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python stage507_multi_token_context.py <model_name>")
        print("  model_name: qwen3 | deepseek7b | gemma4")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"[Stage507] 多token上下文验证 - {model_name}")
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

    # A5 先运行（最快，最有诊断价值）
    print(f"[{datetime.now().strftime('%H:%M:%S')}] A5: 深层方差分析...")
    t0 = time.time()
    a5 = analyze_A5_deep_variance(model, tokenizer, layer_indices)
    summary.update(a5)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # A4
    print(f"[{datetime.now().strftime('%H:%M:%S')}] A4: 句子模板稳定性...")
    t0 = time.time()
    a4 = analyze_A4_template_stability(model, tokenizer, layer_indices)
    summary.update(a4)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # A1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] A1: 单token vs 句子上下文层级...")
    t0 = time.time()
    a1 = analyze_A1_context_hierarchy(model, tokenizer, layer_indices)
    summary.update(a1)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # A2
    print(f"[{datetime.now().strftime('%H:%M:%S')}] A2: 单token vs 句子上下文翻译...")
    t0 = time.time()
    a2 = analyze_A2_context_translation(model, tokenizer, layer_indices)
    summary.update(a2)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # A3
    print(f"[{datetime.now().strftime('%H:%M:%S')}] A3: 上下文长度影响...")
    t0 = time.time()
    a3 = analyze_A3_context_length(model, tokenizer, layer_indices)
    summary.update(a3)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # 保存
    # 保存前清理：移除不可序列化的tensor数据
    a1_data = summary.get("A1_context_hierarchy", {})
    for mode in ["single_token", "sentence_context"]:
        if mode in a1_data:
            del a1_data[mode]

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

    out_dir = Path("tests/codex_temp") / f"stage507_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{model_name}.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(convert(summary), f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印关键发现
    print("\n" + "=" * 60)
    print("关键发现摘要")
    print("=" * 60)

    a5_data = summary["A5_deep_variance"]
    print("\n[A5] 各层norm和方差:")
    for li in layer_indices:
        d = a5_data[f"L{li}"]
        print(f"  L{li}: norm={d['mean_norm']:.1f}({d['std_norm']:.1f}), "
              f"var={d['mean_variance']:.4f}, 词间std={d['inter_word_std']:.4f}")

    a1_data = summary["A1_context_hierarchy"]
    print("\n[A1] 层级信号: 单token vs 句子上下文")
    for mode in ["single_token_signals", "sentence_context_signals"]:
        sigs = a1_data[mode]
        print(f"  {mode}:")
        for k in sorted(sigs.keys()):
            print(f"    {k}: 信号={sigs[k]['hierarchy_signal']:.3f}")

    a2_data = summary["A2_context_translation"]
    print("\n[A2] 翻译对齐: 单token vs 句子上下文")
    for mode in ["single_token_avg", "sentence_context_avg"]:
        avgs = a2_data[mode]
        vals = [f"{k}={avgs[k]:.3f}" for k in sorted(avgs.keys())]
        print(f"  {mode}: {', '.join(vals)}")

    a4_data = summary["A4_template_stability"]
    print("\n[A4] 句子模板稳定性:")
    for k, d in a4_data.items():
        print(f"  {k}: 一致性={d['mean_consistency']:.3f} (std={d['std_consistency']:.3f})")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
