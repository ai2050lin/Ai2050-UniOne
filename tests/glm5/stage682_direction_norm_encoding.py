#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage682: P36 方向+norm协同编码——语义如何通过方向和模长共同编码

目标：分解hidden state为方向(θ)和模长(||h||)两个分量，
     分析各自携带的信息量，建立协同编码的数学描述。

核心问题：
  P35发现所有模型域内cos>0.87，域间cos≈0.89。
  如果方向如此相似，语义信息一定也编码在norm中。

假说：
  H1: 方向编码"类型/范畴"信息（是什么能力）
  H2: Norm编码"强度/置信度"信息（多确定）
  H3: 语义差异主要由norm差异贡献，而非方向差异

实验方法：
  1. 对同一能力的不同文本，分解为方向和norm
  2. 测量：
     a. 方向的域内一致性cos（已有）
     b. Norm的域内变异系数(CV)
     c. 方向vs Norm对区分不同域的贡献度
     d. 跨层：方向和norm的演化模式

INV-326: 方向cos的区分能力 < norm的区分能力
INV-327: Norm在不同能力间有系统性差异（某些能力 systematically larger/smaller）
INV-328: 方向+norm联合的区分能力 > 任一单独分量
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import statistics
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score

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


# 多域多样本
DOMAIN_SAMPLES = {
    "disamb": [
        "The river bank was muddy.", "The bank approved the loan.",
        "The bat flew across the cave.", "He swung the bat hard.",
        "The plant needs sunlight.", "The plant closed permanently.",
        "The mouse ran across the floor.", "She clicked the computer mouse.",
    ],
    "relation": [
        "Paris is the capital of France.", "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.", "London is the capital of England.",
        "The Pacific is the largest ocean.", "The Nile is the longest river.",
        "Mount Everest is the tallest peak.", "The Sahara is the largest desert.",
    ],
    "syntax": [
        "She quickly ran to the store.", "The boy slowly walked home.",
        "They carefully opened the box.", "He loudly shouted at the crowd.",
        "She quietly read her favorite book.", "The dog happily chased the ball.",
        "We eagerly awaited the results.", "He gently placed the glass down.",
    ],
    "style": [
        "The meeting was extremely productive.", "That get-together was quite fruitful.",
        "The presentation was remarkably clear.", "This discussion was incredibly valuable.",
        "The workshop was highly informative.", "That seminar was particularly useful.",
        "The conference was deeply inspiring.", "This dialogue was exceptionally engaging.",
    ],
    "spatial": [
        "The cat is under the table.", "The bird is above the tree.",
        "The book is on the shelf.", "The keys are inside the drawer.",
        "The children played behind the house.", "The dog slept beside the fireplace.",
        "The painting hung above the sofa.", "The flowers grew between the stones.",
    ],
    "temporal": [
        "Yesterday it rained heavily all day.", "Tomorrow it will snow in the north.",
        "Last week she finished her project.", "Next month they launch the product.",
        "In the morning she drinks coffee.", "At night the stars appear clearly.",
        "During summer temperatures rise.", "Before dinner they went for a walk.",
    ],
}


def collect_hs_with_norm(model, tokenizer, texts, layer_idx=-1):
    """收集hidden state并分解为方向和norm"""
    model_device = next(model.parameters()).device
    directions = []
    norms = []
    raw_hs = []

    for text in texts:
        tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
        h = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()
        norm = h.norm().item()
        direction = (h / max(norm, 1e-8)).numpy()
        raw_hs.append(h.numpy())
        directions.append(direction)
        norms.append(norm)

    return {
        "directions": np.array(directions),
        "norms": np.array(norms),
        "raw": np.array(raw_hs),
    }


def compute_direction_separability(directions, labels):
    """方向分量的域间分离度（Silhouette）"""
    try:
        sil = silhouette_score(directions, labels)
        return sil
    except Exception:
        return -1


def compute_norm_separability(norms, labels):
    """Norm分量的域间分离度"""
    norm_2d = norms.reshape(-1, 1)
    try:
        sil = silhouette_score(norm_2d, labels)
        return sil
    except Exception:
        return -1


def compute_joint_separability(raw_hs, labels):
    """原始hidden state的域间分离度"""
    try:
        sil = silhouette_score(raw_hs, labels)
        return sil
    except Exception:
        return -1


def compute_norm_discriminability(norms, labels):
    """Norm在不同域之间的区分能力（Fisher判别比）"""
    domain_norms = defaultdict(list)
    for norm, label in zip(norms, labels):
        domain_norms[label].append(norm)

    grand_mean = np.mean(norms)
    between_var = 0
    within_var = 0
    n_total = len(norms)

    for domain, ns in domain_norms.items():
        domain_mean = np.mean(ns)
        between_var += len(ns) * (domain_mean - grand_mean) ** 2
        within_var += sum((x - domain_mean) ** 2 for x in ns)

    between_var /= n_total
    within_var /= n_total
    fisher = between_var / max(within_var, 1e-8)
    return fisher, between_var, within_var


def run_direction_norm_experiment(model_arg):
    """主函数：方向+norm协同编码分析"""
    print("=" * 65)
    print(f"  P36 方向+norm协同编码——语义的双分量分析")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    # 收集数据
    all_texts = []
    all_labels = []
    for domain, texts in DOMAIN_SAMPLES.items():
        all_texts.extend(texts)
        all_labels.extend([domain] * len(texts))

    print(f"\n  收集 {len(all_texts)} 个样本...")
    data = collect_hs_with_norm(model, tokenizer, all_texts)
    dirs = data["directions"]
    norms = data["norms"]
    raw = data["raw"]

    # === 1. 方向分量分析 ===
    print(f"\n{'='*50}")
    print("  1. 方向分量（Direction）分析")
    print(f"{'='*50}")

    dir_sil = compute_direction_separability(dirs, all_labels)
    print(f"  方向Silhouette Score: {dir_sil:.4f}")

    # 域内方向cos
    domain_dirs = defaultdict(list)
    for i, label in enumerate(all_labels):
        domain_dirs[label].append(i)

    print(f"\n  域内方向一致性:")
    for domain in sorted(domain_dirs.keys()):
        indices = domain_dirs[domain]
        cos_pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                cos_v = np.dot(dirs[indices[i]], dirs[indices[j]])
                cos_pairs.append(cos_v)
        mean_cos = np.mean(cos_pairs)
        std_cos = np.std(cos_pairs)
        print(f"    {domain:>10}: cos={mean_cos:.4f} ± {std_cos:.4f}")

    # 域间方向cos
    domain_names = sorted(domain_dirs.keys())
    between_cos = []
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            if i < j:
                c1 = np.mean(dirs[domain_dirs[d1]], axis=0)
                c2 = np.mean(dirs[domain_dirs[d2]], axis=0)
                cos_v = abs(np.dot(c1, c2))
                between_cos.append(cos_v)
    print(f"  域中心cos均值: {np.mean(between_cos):.4f}")

    # === 2. Norm分量分析 ===
    print(f"\n{'='*50}")
    print("  2. 模长分量（Norm）分析")
    print(f"{'='*50}")

    norm_sil = compute_norm_separability(norms, all_labels)
    print(f"  Norm Silhouette Score: {norm_sil:.4f}")

    fisher, between_var, within_var = compute_norm_discriminability(norms, all_labels)
    print(f"  Fisher判别比: {fisher:.2f} (组间方差={between_var:.2f}, 组内方差={within_var:.2f})")

    print(f"\n  各域Norm统计:")
    for domain in sorted(domain_dirs.keys()):
        indices = domain_dirs[domain]
        domain_norms = [norms[i] for i in indices]
        mean_n = np.mean(domain_norms)
        std_n = np.std(domain_norms)
        cv = std_n / max(mean_n, 1e-8)
        print(f"    {domain:>10}: mean={mean_n:.1f} ± {std_n:.1f} (CV={cv:.4f})")

    # === 3. 联合分析 ===
    print(f"\n{'='*50}")
    print("  3. 方向+Norm联合分析")
    print(f"{'='*50}")

    joint_sil = compute_joint_separability(raw, all_labels)
    print(f"  原始hidden state Silhouette: {joint_sil:.4f}")
    print(f"  方向Silhouette:               {dir_sil:.4f}")
    print(f"  Norm Silhouette:              {norm_sil:.4f}")

    # INV-326验证
    norm_better = abs(norm_sil) > abs(dir_sil)
    inv326 = "✅确认(Norm区分力更强)" if norm_better else "❌未确认(方向区分力更强)"
    print(f"\n  INV-326 Norm区分能力 > 方向区分能力: {inv326}")

    # INV-327验证——Norm是否有系统性域差异
    domain_means = {domain: np.mean([norms[i] for i in domain_dirs[domain]]) for domain in domain_dirs}
    norm_range = max(domain_means.values()) - min(domain_means.values())
    mean_norm = np.mean(norms)
    norm_range_pct = norm_range / max(mean_norm, 1e-8) * 100
    inv327 = "✅确认" if norm_range_pct > 5 else "❌未确认"
    print(f"  INV-327 Norm系统性域差异: {inv327} (范围={norm_range:.1f}, {norm_range_pct:.1f}%)")

    # INV-328验证——联合区分力
    inv328 = "✅确认" if abs(joint_sil) > max(abs(dir_sil), abs(norm_sil)) else "❌未确认"
    print(f"  INV-328 联合区分力 > 单独分量: {inv328}")

    # === 4. 方向+Norm的信息贡献分解 ===
    print(f"\n{'='*50}")
    print("  4. 信息贡献分解")
    print(f"{'='*50}")

    # 用cos²衡量方向的信息量，用norm²衡量norm的信息量
    # 每个样本: ||h||² = norm² → 总能量
    # 方向信息: 每个样本到全局均值的cos²
    global_mean_dir = np.mean(dirs, axis=0)
    global_mean_dir = global_mean_dir / max(np.linalg.norm(global_mean_dir), 1e-8)

    dir_info = []
    norm_info = []
    for i in range(len(all_texts)):
        cos_to_mean = np.dot(dirs[i], global_mean_dir)
        dir_info.append(cos_to_mean ** 2)
        norm_info.append((norms[i] - mean_norm) ** 2)

    dir_info_ratio = np.mean(dir_info)
    norm_info_ratio = np.mean(norm_info) / max(mean_norm ** 2, 1e-8)
    print(f"  方向信息量(到均值cos²): {dir_info_ratio:.4f}")
    print(f"  Norm信息量(到均值偏差²): {norm_info_ratio:.4f}")
    print(f"  Norm/方向 信息比: {norm_info_ratio/max(dir_info_ratio,1e-8):.2f}x")

    free_model(model)

    return {
        "model": model_name,
        "dir_sil": dir_sil,
        "norm_sil": norm_sil,
        "joint_sil": joint_sil,
        "fisher": fisher,
        "norm_range_pct": norm_range_pct,
        "inv326": inv326,
        "inv327": inv327,
        "inv328": inv328,
    }


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_direction_norm_experiment(model_arg)
