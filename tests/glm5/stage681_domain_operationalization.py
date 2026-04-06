#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage681: P35 信息域操作化——用数学工具定义"信息域"

目标：用hidden state空间中的几何/拓扑特征来操作化定义"信息域"，
     替代之前模糊的"能力类别"概念。

方法：
  1. 对大量样本计算hidden state，在最终层构建点云
  2. 用PCA降维后观察聚类结构——信息域应表现为流形上的簇
  3. 测量：
     a. Silhouette Score（轮廓系数）——聚类紧密度
     b. Davies-Bouldin Index（DB指数）——簇间分离度
     c. 同能力样本的cos一致性 vs 跨能力cos
     d. PCA前N个主成分解释的方差比例
  4. 验证：如果信息域SEPARATED → 应有清晰的聚类结构
           如果信息域DENSE → 应无聚类结构（均匀分布）

INV-324: SEPARATED模型（Qwen3）的Silhouette Score > DENSE模型（DS7B）
INV-325: 信息域的"厚度"（PCA90所需维度）在SEPARATED模型中更低
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    MODEL_SPECS,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


# ============================================================
# 多样本测试数据——每种能力15个样本
# ============================================================
DOMAIN_SAMPLES = {
    "disamb": [
        "The river bank was muddy and wet.",
        "The bank approved the loan today.",
        "The bat flew across the dark cave.",
        "He swung the wooden bat hard.",
        "The plant needs more sunlight.",
        "The plant closed its doors permanently.",
        "The mouse ran across the floor.",
        "She clicked the computer mouse.",
        "The match was struck and lit.",
        "The football match was exciting.",
        "The bow was tied with ribbon.",
        "He aimed the bow at the target.",
        "The file was saved on disk.",
        "She filed the papers carefully.",
        "The nail was hammered into wood.",
        "She painted her fingernails red.",
    ],
    "relation": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "London is the capital of England.",
        "Beijing is the capital of China.",
        "Moscow is the capital of Russia.",
        "The Pacific is the largest ocean.",
        "The Nile is the longest river.",
        "Mount Everest is the tallest peak.",
        "The Sahara is the largest desert.",
        "Mercury is the closest planet to the Sun.",
        "Jupiter is the largest planet.",
        "The Amazon has the most biodiversity.",
        "Antarctica is the coldest continent.",
        "The Dead Sea is the lowest point.",
    ],
    "syntax": [
        "She quickly ran to the store.",
        "The boy slowly walked home.",
        "They carefully opened the box.",
        "He loudly shouted at the crowd.",
        "She quietly read her favorite book.",
        "The dog happily chased the ball.",
        "We eagerly awaited the results.",
        "He gently placed the glass down.",
        "They nervously approached the door.",
        "She proudly displayed her work.",
        "The wind softly blew through trees.",
        "He angrily slammed the door shut.",
        "We patiently waited in line.",
        "She gracefully danced across stage.",
        "The rain heavily fell all night.",
    ],
    "style": [
        "The meeting was extremely productive.",
        "That get-together was quite fruitful.",
        "The presentation was remarkably clear.",
        "This discussion was incredibly valuable.",
        "The workshop was highly informative.",
        "That seminar was particularly useful.",
        "The conference was deeply inspiring.",
        "This dialogue was exceptionally engaging.",
        "The review was thoroughly comprehensive.",
        "That analysis was surprisingly insightful.",
        "The debate was profoundly thought-provoking.",
        "This proposal was fundamentally sound.",
        "The briefing was notably concise.",
        "That session was immensely practical.",
        "The consultation was broadly beneficial.",
    ],
    "spatial": [
        "The cat is under the table.",
        "The bird is above the tree.",
        "The book is on the shelf.",
        "The keys are inside the drawer.",
        "The children played behind the house.",
        "The dog slept beside the fireplace.",
        "The painting hung above the sofa.",
        "The flowers grew between the stones.",
        "The bridge stretched across the river.",
        "The path led through the forest.",
        "The store is next to the bank.",
        "The park is near the library.",
        "The car parked behind the building.",
        "The star shone above the mountain.",
        "The boat sailed around the island.",
    ],
    "temporal": [
        "Yesterday it rained heavily all day.",
        "Tomorrow it will snow in the north.",
        "Last week she finished her project.",
        "Next month they launch the product.",
        "Last year revenue doubled significantly.",
        "In the morning she drinks coffee.",
        "At night the stars appear clearly.",
        "During summer temperatures rise above thirty.",
        "Before dinner they went for a walk.",
        "After the movie they discussed it.",
        "Previously he worked at a bank.",
        "Eventually they found the solution.",
        "Meanwhile she prepared the report.",
        "Subsequently the results were published.",
        "Initially the plan seemed impossible.",
    ],
}


def collect_hidden_states(model, tokenizer, texts, layer_idx=-1):
    """收集所有文本在指定层的hidden state"""
    model_device = next(model.parameters()).device
    all_hs = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
        h = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()
        all_hs.append(h.numpy())
    return np.array(all_hs)  # (N, d)


def analyze_domain_structure(hidden_states, labels):
    """分析信息域结构"""
    n_samples, dim = hidden_states.shape

    results = {}

    # 1. PCA分析
    pca = PCA()
    pca.fit(hidden_states)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # PCA90: 解释90%方差需要的维度
    pca90 = int(np.searchsorted(cumvar, 0.90)) + 1
    # PCA50
    pca50 = int(np.searchsorted(cumvar, 0.50)) + 1
    # Top1主成分解释比例
    top1_var = pca.explained_variance_ratio_[0]

    results["pca90"] = pca90
    results["pca50"] = pca50
    results["top1_var"] = top1_var
    results["effective_dim"] = dim

    # 2. Silhouette Score（轮廓系数）——值越高聚类越好
    if len(set(labels)) >= 2 and n_samples > len(set(labels)):
        try:
            sil = silhouette_score(hidden_states, labels)
            results["silhouette"] = sil
        except Exception:
            results["silhouette"] = -1
    else:
        results["silhouette"] = -1

    # 3. Davies-Bouldin Index——值越低聚类越好
    if len(set(labels)) >= 2 and n_samples > len(set(labels)):
        try:
            db = davies_bouldin_score(hidden_states, labels)
            results["db_index"] = db
        except Exception:
            results["db_index"] = -1
    else:
        results["db_index"] = -1

    # 4. 域内cos一致性 vs 域间cos
    domain_cos_within = {}
    domain_cos_between = {}

    domain_indices = defaultdict(list)
    for i, label in enumerate(labels):
        domain_indices[label].append(i)

    # 域内cos
    for domain, indices in domain_indices.items():
        if len(indices) >= 2:
            cos_pairs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    h1 = hidden_states[indices[i]]
                    h2 = hidden_states[indices[j]]
                    cos_v = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                    cos_pairs.append(cos_v)
            domain_cos_within[domain] = {
                "mean": float(np.mean(cos_pairs)),
                "std": float(np.std(cos_pairs)),
                "n": len(cos_pairs),
            }

    # 域间cos
    domain_names = list(domain_indices.keys())
    between_pairs = []
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            if i < j:
                for idx1 in domain_indices[d1][:3]:  # 取前3个样本加速
                    for idx2 in domain_indices[d2][:3]:
                        h1 = hidden_states[idx1]
                        h2 = hidden_states[idx2]
                        cos_v = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                        between_pairs.append(abs(cos_v))

    results["within_cos"] = domain_cos_within
    results["between_cos_mean"] = float(np.mean(between_pairs)) if between_pairs else 0
    results["between_cos_std"] = float(np.std(between_pairs)) if len(between_pairs) > 1 else 0

    # 5. 域中心之间的距离
    domain_centers = {}
    for domain, indices in domain_indices.items():
        center = np.mean(hidden_states[indices], axis=0)
        domain_centers[domain] = center

    center_dists = []
    domain_names = list(domain_centers.keys())
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            if i < j:
                c1 = domain_centers[d1]
                c2 = domain_centers[d2]
                cos_v = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                center_dists.append(cos_v)

    results["center_cos_mean"] = float(np.mean([abs(d) for d in center_dists])) if center_dists else 0

    return results


def run_domain_experiment(model_arg):
    """主函数：信息域操作化"""
    print("=" * 65)
    print(f"  P35 信息域操作化——hidden state空间的聚类结构")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    # 收集所有样本的hidden state
    all_texts = []
    all_labels = []
    for domain, texts in DOMAIN_SAMPLES.items():
        all_texts.extend(texts)
        all_labels.extend([domain] * len(texts))

    print(f"\n  收集 {len(all_texts)} 个样本的hidden state...")
    hidden_states = collect_hidden_states(model, tokenizer, all_texts, layer_idx=-1)
    print(f"  hidden state形状: {hidden_states.shape}")

    # 分析信息域结构
    print(f"\n  --- PCA分析 ---")
    results = analyze_domain_structure(hidden_states, all_labels)
    print(f"  PCA50: {results['pca50']} 维 (解释50%方差)")
    print(f"  PCA90: {results['pca90']} 维 (解释90%方差)")
    print(f"  Top1主成分: {results['top1_var']:.4f} ({results['top1_var']*100:.1f}%)")

    print(f"\n  --- 聚类质量 ---")
    print(f"  Silhouette Score: {results['silhouette']:.4f} (>0:有聚类结构, <0:无)")
    print(f"  Davies-Bouldin Index: {results['db_index']:.4f} (<1:好的聚类)")

    print(f"\n  --- 域内cos一致性 ---")
    for domain in sorted(results["within_cos"].keys()):
        info = results["within_cos"][domain]
        print(f"    {domain:>10}: mean_cos={info['mean']:.4f} ± {info['std']:.4f}")

    print(f"\n  --- 域间cos ---")
    print(f"    跨域cos均值: {results['between_cos_mean']:.4f} ± {results['between_cos_std']:.4f}")
    print(f"    域中心cos均值: {results['center_cos_mean']:.4f}")

    # INV验证
    print(f"\n  --- 不变量验证 ---")
    # SEPARATED vs DENSE的判别
    if results['silhouette'] > 0.1:
        status = "SEPARATED（有聚类结构）"
    elif results['silhouette'] > 0:
        status = "半SEPARATED（弱聚类结构）"
    else:
        status = "DENSE（无聚类结构）"

    print(f"  信息域状态: {status}")
    print(f"  INV-324: silhouette={results['silhouette']:.4f} {'>0.1→SEPARATED' if results['silhouette'] > 0.1 else '<0.1→DENSE'}")

    # 域内vs域间cos比
    within_means = [v["mean"] for v in results["within_cos"].values()]
    within_avg = statistics.mean(within_means) if within_means else 0
    ratio = within_avg / max(results["between_cos_mean"], 1e-8)
    print(f"  域内/域间cos比: {ratio:.2f}x (>1.5→SEPARATED, <1.5→DENSE)")

    # 信息域容量估计
    est_capacity = results['effective_dim'] / (2 * math.log(results['effective_dim']))
    print(f"\n  信息域容量估计: {est_capacity:.1f} (d / 2ln(d))")
    print(f"  实际域数量: {len(set(all_labels))}")

    free_model(model)

    return {"model": model_name, "results": results, "status": status}


if __name__ == "__main__":
    import math
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_domain_experiment(model_arg)
