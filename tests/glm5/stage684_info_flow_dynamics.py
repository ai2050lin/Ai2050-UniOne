#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage684: P38 跨层信息流动态追踪——信息从输入到输出的完整流动图

目标：追踪hidden state在不同层的演化过程，建立信息流动的定量模型。

核心问题：
  1. 不同类型的语义信息在哪一层出现？
  2. 信息是如何跨层传递的？（逐层累积 vs 突然涌现）
  3. 不同模型的层动力学模式有何差异？

方法：
  1. 对同一文本，提取每一层的hidden state
  2. 在每一层计算：
     a. 域方向的一致性（域内cos）
     b. 域间分离度（Silhouette）
     c. PCA结构（PCA90/Top1方差）
     d. 信号强度（||h||的统计量）
  3. 绘制"信息演化图"

INV-331: 域区分力在中间层最低（信息最"混乱"），在首尾层最高
INV-332: 信号强度(||h||)在后期层最高
INV-333: PCA90在早期层最低（信息最压缩），后期层最高（信息最分散）
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

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score
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


def collect_multilayer_hs(model, tokenizer, domain_samples):
    """收集每一层的hidden state"""
    model_device = next(model.parameters()).device
    all_texts = []
    all_labels = []

    # 收集所有文本
    for domain, texts in domain_samples.items():
        all_texts.extend(texts)
        all_labels.extend([domain] * len(texts))

    # 第一次前向传播确定层数
    tokens0 = tokenizer.encode(all_texts[0], return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs0 = model(tokens0, output_hidden_states=True)
    n_layers = len(outputs0.hidden_states)

    # 收集每层的hidden state
    layer_hs = {i: [] for i in range(n_layers)}
    layer_norms = {i: [] for i in range(n_layers)}

    for text in all_texts:
        tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)

        for layer_idx in range(n_layers):
            h = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()
            layer_hs[layer_idx].append(h.numpy())
            layer_norms[layer_idx].append(h.norm().item())

    # 转为numpy
    for i in range(n_layers):
        layer_hs[i] = np.array(layer_hs[i])

    return layer_hs, layer_norms, all_labels, n_layers


def analyze_layer(layer_hs, layer_norms, labels, layer_idx, n_layers):
    """分析单层"""
    hs = layer_hs[layer_idx]
    norms = layer_norms[layer_idx]

    # 1. 域内cos一致性
    domain_indices = defaultdict(list)
    for i, label in enumerate(labels):
        domain_indices[label].append(i)

    within_cos = []
    for domain, indices in domain_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    h1 = hs[indices[i]]
                    h2 = hs[indices[j]]
                    cos_v = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                    within_cos.append(cos_v)
    mean_within = np.mean(within_cos) if within_cos else 0

    # 2. Silhouette
    try:
        sil = silhouette_score(hs, labels)
    except Exception:
        sil = 0

    # 3. PCA结构
    try:
        pca = PCA()
        pca.fit(hs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        pca90 = int(np.searchsorted(cumvar, 0.90)) + 1
        top1 = pca.explained_variance_ratio_[0] * 100
    except Exception:
        pca90 = -1
        top1 = -1

    # 4. 信号强度
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    return {
        "layer": layer_idx,
        "within_cos": mean_within,
        "silhouette": sil,
        "pca90": pca90,
        "top1_var": top1,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
    }


def run_flow_analysis(model_arg):
    """主函数：信息流动态追踪"""
    print("=" * 65)
    print(f"  P38 跨层信息流动态追踪")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    print(f"\n  收集多层数据...")
    layer_hs, layer_norms, labels, n_layers = collect_multilayer_hs(model, tokenizer, DOMAIN_SAMPLES)
    print(f"  总层数: {n_layers-1} (+ 1 embedding层)")

    # 分析每一层
    layer_results = []
    for i in range(n_layers):
        result = analyze_layer(layer_hs, layer_norms, labels, i, n_layers)
        layer_results.append(result)

    # 输出演化表
    print(f"\n  {'Layer':>5} | {'域内cos':>8} | {'Silhouette':>10} | {'PCA90':>6} | {'Top1%':>6} | {'Norm均值':>8} | {'Norm_std':>8}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for r in layer_results:
        print(f"  L{r['layer']:>3} | {r['within_cos']:>8.4f} | {r['silhouette']:>10.4f} | {r['pca90']:>6} | {r['top1_var']:>5.1f}% | {r['mean_norm']:>8.1f} | {r['std_norm']:>8.2f}")

    # 找关键层
    sil_values = [r['silhouette'] for r in layer_results]
    max_sil_layer = np.argmax(sil_values)
    min_sil_layer = np.argmin(sil_values)

    norm_values = [r['mean_norm'] for r in layer_results]
    max_norm_layer = np.argmax(norm_values)
    min_norm_layer = np.argmin(norm_values)

    print(f"\n  --- 关键层 ---")
    print(f"  域分离最强层: L{max_sil_layer} (Silhouette={sil_values[max_sil_layer]:.4f})")
    print(f"  域分离最弱层: L{min_sil_layer} (Silhouette={sil_values[min_sil_layer]:.4f})")
    print(f"  信号最强层:   L{max_norm_layer} (Norm={norm_values[max_norm_layer]:.1f})")
    print(f"  信号最弱层:   L{min_norm_layer} (Norm={norm_values[min_norm_layer]:.1f})")

    # INV验证
    # 只看Transformer层（跳过embedding层L0）
    trans_results = layer_results[1:]  # 跳过embedding
    trans_sils = [r['silhouette'] for r in trans_results]
    if len(trans_sils) > 2:
        # 找中间层（25%-75%范围）
        n_trans = len(trans_sils)
        mid_start = n_trans // 4
        mid_end = 3 * n_trans // 4
        mid_sils = trans_sils[mid_start:mid_end]
        early_sils = trans_sils[:mid_start]
        late_sils = trans_sils[mid_end:]

        if mid_sils and early_sils and late_sils:
            mid_mean = np.mean(mid_sils)
            edge_mean = (np.mean(early_sils) + np.mean(late_sils)) / 2
            inv331 = "✅确认" if mid_mean < edge_mean else "❌未确认"
            print(f"\n  INV-331 中间层域分离最低: {inv331} (中间={mid_mean:.4f}, 边缘={edge_mean:.4f})")

    # INV-332: 信号强度后期层最高
    trans_norms = [r['mean_norm'] for r in trans_results]
    if len(trans_norms) > 2:
        first_half = np.mean(trans_norms[:len(trans_norms)//2])
        second_half = np.mean(trans_norms[len(trans_norms)//2:])
        inv332 = "✅确认" if second_half > first_half else "❌未确认"
        print(f"  INV-332 后期信号更强: {inv332} (前半={first_half:.1f}, 后半={second_half:.1f})")

    # 信息流总结
    print(f"\n  --- 信息流总结 ---")
    for i, r in enumerate(layer_results):
        if i == 0:
            phase = "嵌入层"
        elif i < len(layer_results) * 0.25:
            phase = "早期层(编码)"
        elif i < len(layer_results) * 0.75:
            phase = "中间层(处理)"
        else:
            phase = "后期层(聚焦)"

        marker = ""
        if i == max_sil_layer:
            marker = " ← 域分离最强"
        elif i == min_sil_layer:
            marker = " ← 域分离最弱"

        bar_len = int(max(r['silhouette'], 0) * 50)
        bar = "█" * max(bar_len, 0)
        print(f"  L{i:>2} [{phase:>12}] sil={r['silhouette']:.4f} {bar}{marker}")

    free_model(model)
    return {"model": model_name, "layers": layer_results, "n_layers": n_layers}


if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_flow_analysis(model_arg)
