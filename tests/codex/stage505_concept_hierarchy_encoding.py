#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage505: 概念层级神经元编码分析
分析 "苹果-水果-食物-物体" 这种多层级的神经元编码机制

核心问题：
1. 在每一层中，"苹果"/"水果"/"物体" 的隐藏表示是什么样的？
2. 是否存在层级结构：苹果 ⊂ 水果 ⊂ 物体（子集关系在激活模式上体现）？
3. 哪些神经元是层级特异的，哪些是共享的？
4. "苹果"这个词在哪些层表现为"具体水果"，哪些层表现为"更抽象概念"？

分析维度：
- H1: 隐藏状态余弦相似度层级
- H2: 神经元激活子集关系（苹果的活跃神经元 ⊂ 水果的活跃神经元？）
- H3: 层间层级演化（层级关系在哪一层最强/最弱）
- H4: 线性探针分类（能否用某层表示预测层级关系）
- H5: 维度贡献分析（哪些维度对层级区分贡献最大）
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex.qwen3_language_shared import (
    load_qwen3_model, load_gemma4_model, load_deepseek7b_model,
    discover_layers, get_model_device, capture_qwen_mlp_payloads,
    remove_hooks, qwen_hidden_dim, qwen_neuron_dim,
)


# ============================================================
# 概念层级体系
# ============================================================
CONCEPT_HIERARCHIES = {
    "食物链": {
        "L1_具体": ["苹果", "香蕉", "葡萄"],
        "L2_类别": ["水果", "蔬菜", "肉类"],
        "L3_抽象": ["食物", "饮料", "调料"],
        "L4_顶层": ["物体", "概念", "事物"],
    },
    "动物链": {
        "L1_具体": ["猫", "狗", "鸟"],
        "L2_类别": ["哺乳动物", "鸟类", "鱼类"],
        "L3_抽象": ["动物", "植物", "矿物"],
        "L4_顶层": ["生物", "生命", "自然"],
    },
    "空间链": {
        "L1_具体": ["北京", "上海", "广州"],
        "L2_类别": ["城市", "村庄", "港口"],
        "L3_抽象": ["地点", "区域", "地带"],
        "L4_顶层": ["空间", "位置", "方位"],
    },
}

# 也加入跨层级对
CROSS_LEVEL_PAIRS = [
    # (具体, 类别) — 应该是 包含关系
    ("苹果", "水果"), ("香蕉", "水果"),
    ("猫", "哺乳动物"), ("狗", "哺乳动物"),
    ("北京", "城市"), ("上海", "城市"),
    # (类别, 抽象)
    ("水果", "食物"), ("蔬菜", "食物"),
    ("哺乳动物", "动物"), ("鸟类", "动物"),
    ("城市", "地点"), ("村庄", "地点"),
    # (抽象, 顶层)
    ("食物", "物体"), ("动物", "生物"),
    ("地点", "空间"),
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


def get_active_dims(hidden, threshold_percentile=90):
    """获取活跃维度索引（激活值超过阈值的维度）"""
    threshold = torch.quantile(hidden.abs(), threshold_percentile / 100.0)
    active = (hidden.abs() >= threshold).nonzero(as_tuple=True)[0]
    return set(active.tolist())


def jaccard(set_a, set_b):
    """Jaccard相似度"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def subset_ratio(set_sub, set_super):
    """子集比例：set_sub中有多少比例属于set_super"""
    if not set_sub:
        return 0.0
    return len(set_sub & set_super) / len(set_sub)


# ============================================================
# H1: 层级余弦相似度矩阵
# ============================================================
def analyze_H1_hierarchy_cosine(model, tokenizer, layer_indices):
    """H1: 层级概念间的余弦相似度矩阵"""
    # 选"食物链"层级作为主要分析对象
    hierarchy = CONCEPT_HIERARCHIES["食物链"]
    all_concepts = []
    level_map = {}
    for level_name, words in hierarchy.items():
        for w in words:
            all_concepts.append(w)
            level_map[w] = level_name

    # 获取所有概念在所有层的隐藏状态
    concept_hidden = {}  # {concept: {layer_idx: hidden_vec}}
    for concept in all_concepts:
        concept_hidden[concept] = get_hidden_at_last_token(model, tokenizer, concept, layer_indices)

    results = {}
    for li in layer_indices:
        matrix = {}
        for c1 in all_concepts:
            for c2 in all_concepts:
                if c1 >= c2:
                    continue
                sim = cosine_sim(concept_hidden[c1][li], concept_hidden[c2][li])
                key = f"{c1}_{c2}"
                matrix[key] = round(sim, 4)
        results[f"L{li}"] = matrix

    # 关键指标：层级内 vs 跨层级 的相似度差异
    for li in layer_indices:
        intra = []  # 同层级
        cross = []  # 相邻层级
        far = []  # 远层级
        levels = list(hierarchy.keys())
        for c1 in all_concepts:
            for c2 in all_concepts:
                if c1 >= c2:
                    continue
                l1_idx = levels.index(level_map[c1])
                l2_idx = levels.index(level_map[c2])
                sim = cosine_sim(concept_hidden[c1][li], concept_hidden[c2][li])
                if l1_idx == l2_idx:
                    intra.append(sim)
                elif abs(l1_idx - l2_idx) == 1:
                    cross.append(sim)
                else:
                    far.append(sim)
        avg_intra = np.mean(intra) if intra else 0
        avg_cross = np.mean(cross) if cross else 0
        avg_far = np.mean(far) if far else 0
        results[f"L{li}"]["_stats"] = {
            "intra_level_avg": round(avg_intra, 4),
            "adjacent_level_avg": round(avg_cross, 4),
            "far_level_avg": round(avg_far, 4),
            "hierarchy_signal": round(avg_intra - avg_far, 4),  # 正值=层级结构存在
        }

    return {"H1_hierarchy_cosine": results}


# ============================================================
# H2: 神经元激活子集关系
# ============================================================
def analyze_H2_neuron_subset(model, tokenizer, layer_indices):
    """H2: 检查 '苹果'的活跃神经元 是否是 '水果'活跃神经元的子集"""
    results = {}

    for pair in CROSS_LEVEL_PAIRS:
        specific, general = pair
        h_specific = get_hidden_at_last_token(model, tokenizer, specific, layer_indices)
        h_general = get_hidden_at_last_token(model, tokenizer, general, layer_indices)

        pair_key = f"{specific}_{general}"
        results[pair_key] = {}
        for li in layer_indices:
            active_specific = get_active_dims(h_specific[li], threshold_percentile=75)
            active_general = get_active_dims(h_general[li], threshold_percentile=75)

            # 核心指标：具体概念的活跃维度中，有多少在一般概念的活跃维度中？
            containment = subset_ratio(active_specific, active_general)
            # 反向：一般概念的活跃维度中，有多少在具体概念的活跃维度中？
            reverse = subset_ratio(active_general, active_specific)
            # Jaccard重叠
            overlap = jaccard(active_specific, active_general)

            results[pair_key][f"L{li}"] = {
                "containment": round(containment, 4),
                "reverse_containment": round(reverse, 4),
                "jaccard": round(overlap, 4),
                "active_specific": len(active_specific),
                "active_general": len(active_general),
            }

    return {"H2_neuron_subset": results}


# ============================================================
# H3: 层间层级演化
# ============================================================
def analyze_H3_layer_evolution(model, tokenizer, layer_indices):
    """H3: 层级关系在不同层的强度变化"""
    results = {"layer_signals": {}}

    for li in layer_indices:
        signals = []
        for specific, general in CROSS_LEVEL_PAIRS[:9]:  # 前9对
            h1 = get_hidden_at_last_token(model, tokenizer, specific, layer_indices)
            h2 = get_hidden_at_last_token(model, tokenizer, general, layer_indices)
            sim = cosine_sim(h1[li], h2[li])
            signals.append(sim)
        avg_signal = np.mean(signals)
        results["layer_signals"][f"L{li}"] = round(avg_signal, 4)

    # 找到层级信号最强的层和最弱的层
    signals_list = [(k, v) for k, v in results["layer_signals"].items()]
    signals_list.sort(key=lambda x: x[1])
    results["strongest_layer"] = signals_list[-1]
    results["weakest_layer"] = signals_list[0]
    results["trend"] = "increasing" if signals_list[-1][1] > signals_list[0][1] else "decreasing"

    # 使用三个层级体系做更全面的分析
    for chain_name, hierarchy in CONCEPT_HIERARCHIES.items():
        chain_results = {}
        for li in layer_indices:
            # 同层级概念的平均相似度
            level_avgs = []
            for level_name, words in hierarchy.items():
                sims = []
                for i in range(len(words)):
                    for j in range(i + 1, len(words)):
                        h1 = get_hidden_at_last_token(model, tokenizer, words[i], [li])
                        h2 = get_hidden_at_last_token(model, tokenizer, words[j], [li])
                        sims.append(cosine_sim(h1[li], h2[li]))
                if sims:
                    level_avgs.append(np.mean(sims))
            chain_results[f"L{li}"] = round(np.mean(level_avgs) if level_avgs else 0, 4)
        results[f"{chain_name}_intra_avg"] = chain_results

    return {"H3_layer_evolution": results}


# ============================================================
# H4: 线性探针——用隐藏状态预测层级关系
# ============================================================
def analyze_H4_linear_probe(model, tokenizer, layer_indices):
    """H4: 用最近邻分类器判断层级关系能否被隐藏状态区分"""
    hierarchy = CONCEPT_HIERARCHIES["食物链"]
    level_names = list(hierarchy.keys())

    # 收集训练数据
    X_by_layer = {li: [] for li in layer_indices}
    y = []

    for level_idx, (level_name, words) in enumerate(hierarchy.items()):
        for w in words:
            h_map = get_hidden_at_last_token(model, tokenizer, w, layer_indices)
            for li in layer_indices:
                X_by_layer[li].append(h_map[li].numpy())
            y.append(level_idx)

    results = {}
    for li in layer_indices:
        X = np.array(X_by_layer[li])
        y_arr = np.array(y)

        # Leave-one-out 最近邻分类
        correct = 0
        total = 0
        for i in range(len(y_arr)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y_arr, i)
            X_test = X[i]

            # 计算到所有训练样本的余弦相似度
            train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
            test_norm = np.linalg.norm(X_test)
            if test_norm < 1e-10 or np.any(train_norms.squeeze() < 1e-10):
                continue
            cos_sims = (X_train @ X_test) / (train_norms.squeeze() * test_norm)
            nearest_idx = np.argmax(cos_sims)
            pred = y_train[nearest_idx]
            if pred == y_arr[i]:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[f"L{li}"] = {
            "leave_one_out_accuracy": round(float(accuracy), 4),
            "n_classes": len(level_names),
            "n_samples": len(y_arr),
        }

    # 找最佳层
    best_layer = max(results.items(), key=lambda x: x[1]["leave_one_out_accuracy"])
    results["_best_layer"] = best_layer

    return {"H4_linear_probe": results}


# ============================================================
# H5: 维度贡献分析——PCA找层级区分方向
# ============================================================
def analyze_H5_dimension_contribution(model, tokenizer, layer_indices):
    """H5: 用PCA找到对层级区分贡献最大的维度"""
    hierarchy = CONCEPT_HIERARCHIES["食物链"]
    level_names = list(hierarchy.keys())

    results = {}
    for li in [layer_indices[len(layer_indices) // 4], layer_indices[len(layer_indices) // 2],
                layer_indices[3 * len(layer_indices) // 4]]:
        # 收集该层所有概念的隐藏状态
        all_vecs = []
        labels = []
        for level_idx, (level_name, words) in enumerate(hierarchy.items()):
            for w in words:
                h = get_hidden_at_last_token(model, tokenizer, w, [li])
                all_vecs.append(h[li].unsqueeze(0))
                labels.append(level_idx)

        X = torch.cat(all_vecs, dim=0).numpy()  # [12, hidden_dim]
        y = np.array(labels)

        # PCA
        mean = X.mean(axis=0, keepdims=True)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 前几个主成分的解释方差比
        total_var = (S ** 2).sum()
        explained = [(S[i] ** 2) / total_var for i in range(min(5, len(S)))]

        # 每个概念在前2个PC上的投影
        projections = (X_centered @ Vt[:2].T).tolist()

        results[f"L{li}"] = {
            "top5_explained_variance_ratio": [round(e, 4) for e in explained],
            "cumulative_variance_pc2": round(sum(explained[:2]), 4),
            "projections_pc1_pc2": projections,
            "labels": labels,
            "level_names": level_names,
        }

    return {"H5_dimension_contribution": results}


# ============================================================
# 主流程
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python stage505_concept_hierarchy_encoding.py <model_name>")
        print("  model_name: qwen3 | deepseek7b | gemma4")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"[Stage505] 概念层级神经元编码分析 - {model_name}")
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

    # 均匀采样层索引
    n_samples = min(8, n_layers)
    layer_indices = [round(i * (n_layers - 1) / (n_samples - 1)) for i in range(n_samples)]

    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_indices": layer_indices,
        "timestamp": datetime.now().isoformat(),
    }

    # H1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] H1: 层级余弦相似度矩阵...")
    t0 = time.time()
    h1 = analyze_H1_hierarchy_cosine(model, tokenizer, layer_indices)
    summary.update(h1)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # H2
    print(f"[{datetime.now().strftime('%H:%M:%S')}] H2: 神经元激活子集关系...")
    t0 = time.time()
    h2 = analyze_H2_neuron_subset(model, tokenizer, layer_indices)
    summary.update(h2)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # H3
    print(f"[{datetime.now().strftime('%H:%M:%S')}] H3: 层间层级演化...")
    t0 = time.time()
    h3 = analyze_H3_layer_evolution(model, tokenizer, layer_indices)
    summary.update(h3)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # H4
    print(f"[{datetime.now().strftime('%H:%M:%S')}] H4: 线性探针分类...")
    t0 = time.time()
    h4 = analyze_H4_linear_probe(model, tokenizer, layer_indices)
    summary.update(h4)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # H5
    print(f"[{datetime.now().strftime('%H:%M:%S')}] H5: 维度贡献PCA...")
    t0 = time.time()
    h5 = analyze_H5_dimension_contribution(model, tokenizer, layer_indices)
    summary.update(h5)
    print(f"  完成 ({time.time() - t0:.1f}s)")

    # 保存
    out_dir = Path("tests/codex_temp") / f"stage505_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{model_name}.json"

    # 递归转换numpy类型为Python原生类型
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

    h1_data = summary["H1_hierarchy_cosine"]
    print("\n[H1] 层级余弦相似度:")
    for li_str in [f"L{li}" for li in layer_indices]:
        stats = h1_data[li_str].get("_stats", {})
        if stats:
            print(f"  {li_str}: 层内={stats['intra_level_avg']:.3f}, 相邻层={stats['adjacent_level_avg']:.3f}, "
                  f"远层={stats['far_level_avg']:.3f}, 层级信号={stats['hierarchy_signal']:.3f}")

    h2_data = summary["H2_neuron_subset"]
    print("\n[H2] 神经元子集关系 (containment=具体包含于一般的比例):")
    for pair_key in list(h2_data.keys())[:6]:
        mid_layer = f"L{layer_indices[len(layer_indices) // 2]}"
        if mid_layer in h2_data[pair_key]:
            d = h2_data[pair_key][mid_layer]
            print(f"  {pair_key}: 包含率={d['containment']:.3f}, 反向={d['reverse_containment']:.3f}, "
                  f"Jaccard={d['jaccard']:.3f}")

    h3_data = summary["H3_layer_evolution"]
    print(f"\n[H3] 层间演化: 趋势={h3_data['trend']}, "
          f"最强层={h3_data['strongest_layer']}, 最弱层={h3_data['weakest_layer']}")

    h4_data = summary["H4_linear_probe"]
    best = h4_data["_best_layer"]
    print(f"\n[H4] 层级分类最佳层: {best[0]}, LOO准确率={best[1]['leave_one_out_accuracy']:.3f}")

    h5_data = summary["H5_dimension_contribution"]
    print("\n[H5] PCA维度贡献:")
    for li_str, data in h5_data.items():
        cum_var = data["cumulative_variance_pc2"]
        print(f"  {li_str}: 前2PC累计方差={cum_var:.3f}")

    # 释放GPU
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
