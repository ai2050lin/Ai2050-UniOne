#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage683: P37 DS7B信息瓶颈分析——PCA90=13维中编码了什么？

目标：深入分析DS7B的PCA结构，回答：
  1. 前13个主成分分别编码了什么信息？
  2. 不同语言能力在PCA空间中的分布模式
  3. 与Qwen3的PCA结构对比——为什么Qwen3需要45维？

方法：
  1. 收集91个样本的hidden state（6域×15样本）
  2. PCA降维后，分析各主成分与域标签的关联
  3. 对每个主成分，找到投影最大和最小的样本——解读语义
  4. 对比DS7B vs Qwen3 vs GLM4 的PCA结构

INV-329: DS7B的Top1主成分编码"语言通用特征"（所有文本共享的方向）
INV-330: Qwen3的PCA结构更"均匀"——各主成分解释的方差更分散
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

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


DOMAIN_SAMPLES = {
    "disamb": [
        "The river bank was muddy.", "The bank approved the loan.",
        "The bat flew across the cave.", "He swung the bat hard.",
        "The plant needs sunlight.", "The plant closed permanently.",
        "The mouse ran across the floor.", "She clicked the computer mouse.",
        "The match was struck and lit.", "The football match was exciting.",
        "The bow was tied with ribbon.", "He aimed the bow at the target.",
        "The file was saved on disk.", "She filed the papers carefully.",
        "The nail was hammered into wood.", "She painted her fingernails red.",
    ],
    "relation": [
        "Paris is the capital of France.", "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.", "London is the capital of England.",
        "The Pacific is the largest ocean.", "The Nile is the longest river.",
        "Mount Everest is the tallest peak.", "The Sahara is the largest desert.",
        "Mercury is the closest planet to the Sun.", "Jupiter is the largest planet.",
        "The Amazon has the most biodiversity.", "Antarctica is the coldest continent.",
        "The Dead Sea is the lowest point.", "The Mariana Trench is the deepest ocean.",
    ],
    "syntax": [
        "She quickly ran to the store.", "The boy slowly walked home.",
        "They carefully opened the box.", "He loudly shouted at the crowd.",
        "She quietly read her favorite book.", "The dog happily chased the ball.",
        "We eagerly awaited the results.", "He gently placed the glass down.",
        "They nervously approached the door.", "She proudly displayed her work.",
        "The wind softly blew through trees.", "He angrily slammed the door shut.",
        "We patiently waited in line.", "She gracefully danced across stage.",
    ],
    "style": [
        "The meeting was extremely productive.", "That get-together was quite fruitful.",
        "The presentation was remarkably clear.", "This discussion was incredibly valuable.",
        "The workshop was highly informative.", "That seminar was particularly useful.",
        "The conference was deeply inspiring.", "This dialogue was exceptionally engaging.",
        "The review was thoroughly comprehensive.", "That analysis was surprisingly insightful.",
        "The debate was profoundly thought-provoking.", "This proposal was fundamentally sound.",
        "The briefing was notably concise.", "That session was immensely practical.",
    ],
    "spatial": [
        "The cat is under the table.", "The bird is above the tree.",
        "The book is on the shelf.", "The keys are inside the drawer.",
        "The children played behind the house.", "The dog slept beside the fireplace.",
        "The painting hung above the sofa.", "The flowers grew between the stones.",
        "The bridge stretched across the river.", "The path led through the forest.",
        "The store is next to the bank.", "The park is near the library.",
        "The car parked behind the building.", "The star shone above the mountain.",
    ],
    "temporal": [
        "Yesterday it rained heavily all day.", "Tomorrow it will snow in the north.",
        "Last week she finished her project.", "Next month they launch the product.",
        "In the morning she drinks coffee.", "At night the stars appear clearly.",
        "During summer temperatures rise.", "Before dinner they went for a walk.",
        "Previously he worked at a bank.", "Eventually they found the solution.",
        "Meanwhile she prepared the report.", "Subsequently the results were published.",
        "Initially the plan seemed impossible.", "Finally they agreed on terms.",
    ],
}


def collect_all_hs(model, tokenizer, domain_samples):
    """收集所有域的hidden state"""
    model_device = next(model.parameters()).device
    all_hs = []
    all_labels = []
    all_texts = []

    for domain, texts in domain_samples.items():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
            with torch.no_grad():
                outputs = model(tokens, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
            all_hs.append(h)
            all_labels.append(domain)
            all_texts.append(text)

    return np.array(all_hs), all_labels, all_texts


def analyze_pca_structure(hidden_states, labels, texts, model_name, n_components=20):
    """分析PCA结构"""
    n_samples, dim = hidden_states.shape

    pca = PCA(n_components=min(n_components, dim))
    transformed = pca.fit_transform(hidden_states)

    print(f"\n  === {model_name} PCA分析 ===")
    print(f"  原始维度: {dim}, 样本数: {n_samples}")
    print(f"  前20个主成分的方差解释比:")

    cumvar = 0
    pca90_idx = 0
    pca50_idx = 0

    print(f"  {'PC':>4} | {'方差%':>7} | {'累积%':>7} | {'域关联'}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+{'-'*40}")

    domain_indices = defaultdict(list)
    for i, label in enumerate(labels):
        domain_indices[label].append(i)

    pc_domain_info = []
    for i in range(min(n_components, len(pca.explained_variance_ratio_))):
        var_pct = pca.explained_variance_ratio_[i] * 100
        cumvar += var_pct
        if cumvar >= 50 and pca50_idx == 0:
            pca50_idx = i + 1
        if cumvar >= 90 and pca90_idx == 0:
            pca90_idx = i + 1

        # 哪个域在这个主成分上的投影最大？
        domain_proj = {}
        for domain, indices in domain_indices.items():
            mean_proj = np.mean(transformed[indices, i])
            std_proj = np.std(transformed[indices, i])
            domain_proj[domain] = (mean_proj, std_proj)

        # 找投影差异最大的域对
        max_range = 0
        max_pair = ("", "")
        domains = sorted(domain_proj.keys())
        for j, d1 in enumerate(domains):
            for k, d2 in enumerate(domains):
                if j < k:
                    range_val = abs(domain_proj[d1][0] - domain_proj[d2][0])
                    if range_val > max_range:
                        max_range = range_val
                        max_pair = (d1, d2)

        # 找投影最大和最小的样本
        max_idx = np.argmax(transformed[:, i])
        min_idx = np.argmin(transformed[:, i])

        marker = ""
        if i == 0:
            marker = " ← Top1"
        elif i == pca90_idx - 1:
            marker = " ← PCA90"

        if i < 10:
            print(f"  PC{i+1:>2} | {var_pct:>6.2f}% | {cumvar:>6.2f}% | {max_pair[0]}↔{max_pair[1]} (Δ={max_range:.1f}){marker}")

        pc_domain_info.append({
            "pc": i + 1,
            "var_pct": var_pct,
            "cumvar": cumvar,
            "max_pair": max_pair,
            "max_range": max_range,
            "max_sample": texts[max_idx][:40],
            "min_sample": texts[min_idx][:40],
        })

    print(f"\n  PCA50 = PC{pca50_idx}, PCA90 = PC{pca90_idx}")

    # 方差分布的"均匀度"——用熵衡量
    var_ratios = pca.explained_variance_ratio_[:n_components]
    var_probs = var_ratios / var_ratios.sum()
    entropy = -np.sum(var_probs * np.log2(var_probs + 1e-10))
    max_entropy = np.log2(n_components)
    uniformity = entropy / max_entropy  # 1.0 = 完全均匀

    print(f"  方差分布均匀度: {uniformity:.4f} (1.0=完全均匀)")

    # 每个域在Top5主成分上的投影模式
    print(f"\n  各域在Top5主成分上的投影:")
    header = f"  {'域':>10}"
    for i in range(5):
        header += f" | PC{i+1:>7}"
    print(header)
    print(f"  {'-'*10}" + "+-" + "--------" * 5)

    for domain in sorted(domain_indices.keys()):
        indices = domain_indices[domain]
        row = f"  {domain:>10}"
        for i in range(5):
            mean_proj = np.mean(transformed[indices, i])
            row += f" | {mean_proj:>+7.1f}"
        print(row)

    return {
        "model": model_name,
        "pca50": pca50_idx,
        "pca90": pca90_idx,
        "uniformity": uniformity,
        "top1_var": pca.explained_variance_ratio_[0] * 100,
        "pc_info": pc_domain_info,
    }


def run_bottleneck_analysis(model_arg):
    """主函数：信息瓶颈分析"""
    print("=" * 65)
    print(f"  P37 信息瓶颈分析——PCA结构深度解读")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    all_texts = []
    all_labels = []
    for domain, texts in DOMAIN_SAMPLES.items():
        all_texts.extend(texts)
        all_labels.extend([domain] * len(texts))

    hidden_states, labels, texts = collect_all_hs(model, tokenizer, DOMAIN_SAMPLES)
    result = analyze_pca_structure(hidden_states, labels, texts, model_name)

    free_model(model)
    return result


if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_bottleneck_analysis(model_arg)
