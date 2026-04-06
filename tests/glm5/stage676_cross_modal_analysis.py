#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage676: P30 跨模态信息流分析——Gemma4视觉→语言信息传递路径

目标：分析Gemma4在处理纯文本 vs 视觉文本时，编码结构的差异，
     验证多模态信息如何影响语言编码。

核心问题：
  1. 纯文本输入时，Gemma4是否仍然表现DENSE策略？
  2. 视觉相关词汇 vs 纯语言词汇的编码维度分布有何不同？
  3. 归一化如何实现模态隔离？

INV-311预测：移除多模态输入后，INV-293/285应更接近成立（更SEPARATED）
INV-309预测：视觉token比语言token的PCA90更高
INV-310预测：跨模态干扰 < 跨能力干扰
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

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


@dataclass(frozen=True)
class TestCase:
    capability: str
    prompt_a: str
    prompt_b: str
    label: str
    has_visual_concept: bool  # 是否包含视觉相关概念

CASES = [
    TestCase("disamb", "The river bank was muddy.", "The bank approved the loan.", "消歧(无视觉)", False),
    TestCase("disamb_v", "The red car stopped at the traffic light.", "The blue car stopped at the sign.", "消歧(有视觉)", True),
    TestCase("syntax", "She quickly ran home.", "Home she ran quickly.", "语法(无视觉)", False),
    TestCase("syntax_v", "The tall building reflected sunlight beautifully.", "Beautifully the building reflected tall sunlight.", "语法(有视觉)", True),
    TestCase("relation", "Paris is the capital of France.", "Berlin is the capital of Germany.", "关系(无视觉)", False),
    TestCase("relation_v", "The Eiffel Tower is located in Paris.", "Big Ben is located in London.", "关系(有视觉)", True),
    TestCase("spatial", "The cat is under the table.", "The bird is above the tree.", "空间(视觉)", True),
    TestCase("color", "The apple is red.", "The sky is blue.", "颜色(纯视觉)", True),
    TestCase("texture", "The surface was rough and uneven.", "The fabric was smooth and silky.", "纹理(纯视觉)", True),
    TestCase("size", "The elephant is very large.", "The mouse is extremely small.", "大小(纯视觉)", True),
    TestCase("style", "The meeting was extremely productive.", "That get-together was quite fruitful.", "风格(无视觉)", False),
    TestCase("temporal", "Yesterday it rained heavily.", "Tomorrow it will snow.", "时序(无视觉)", False),
]


def get_all_hiddens(model, tokenizer, text: str) -> List[torch.Tensor]:
    """获取所有层的hidden state (最后一个token)"""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        return [h[0, -1, :].float().cpu() for h in outputs.hidden_states]


def get_last_hidden(model, tokenizer, text: str) -> torch.Tensor:
    hiddens = get_all_hiddens(model, tokenizer, text)
    return hiddens[-1]


def analyze_dimension_distribution(direction: torch.Tensor, name: str) -> Dict:
    """分析方向向量的维度分布"""
    d_abs = torch.abs(direction)
    sorted_vals, _ = torch.sort(d_abs, descending=True)
    total_energy = float(torch.sum(direction ** 2))

    if total_energy < 1e-10:
        return {"name": name, "total_energy": 0}

    # 累积能量分布
    cumulative = torch.cumsum(sorted_vals ** 2, dim=0) / total_energy
    pca90 = int((cumulative < 0.90).sum().item()) + 1
    pca95 = int((cumulative < 0.95).sum().item()) + 1
    pca99 = int((cumulative < 0.99).sum().item()) + 1

    # top-k能量占比
    dim = len(direction)
    top1pct = int(dim * 0.01)
    top10pct = int(dim * 0.10)
    top1_energy = float(torch.sum(sorted_vals[:max(top1pct, 1)] ** 2)) / total_energy
    top10_energy = float(torch.sum(sorted_vals[:max(top10pct, 1)] ** 2)) / total_energy

    # 有效维度
    p = sorted_vals ** 2 / total_energy
    p = p[p > 1e-10]
    entropy = -float(torch.sum(p * torch.log(p + 1e-10)))
    eff_dim = float(torch.exp(torch.tensor(entropy)))

    return {
        "name": name,
        "pca90": pca90,
        "pca95": pca95,
        "pca99": pca99,
        "top1pct_energy": top1_energy,
        "top10pct_energy": top10_energy,
        "eff_dim": eff_dim,
        "norm": float(direction.norm()),
        "total_energy": total_energy,
    }


def analyze_cross_layer_propagation(model, tokenizer, text: str) -> Dict:
    """分析信息在各层的传播：cos与最终方向的变化"""
    hiddens = get_all_hiddens(model, tokenizer, text)
    n = len(hiddens)
    final_dir = hiddens[-1]  # 最终方向

    layer_cos = []
    for i, h in enumerate(hiddens):
        c = F.cosine_similarity(h.unsqueeze(0), final_dir.unsqueeze(0)).item()
        layer_cos.append(c)

    # 逐层变化量
    layer_changes = []
    for i in range(1, n):
        cos_diff = F.cosine_similarity(hiddens[i-1].unsqueeze(0), hiddens[i].unsqueeze(0)).item()
        layer_changes.append(cos_diff)

    return {
        "layer_cos_to_final": layer_cos,
        "layer_to_layer_cos": layer_changes,
        "n_layers": n,
        "early_cos": statistics.mean([abs(c) for c in layer_cos[:n//2]]),
        "late_cos": statistics.mean([abs(c) for c in layer_cos[n//2:]]),
    }


def run_cross_modal_analysis(model, tokenizer, model_name: str) -> Dict:
    """跨模态信息流分析"""
    print(f"\n{'='*70}")
    print(f"  P30 跨模态信息流分析: {model_name}")
    print(f"{'='*70}")

    # 1. 区分视觉 vs 非视觉能力的编码差异
    print("\n[1] 视觉 vs 非视觉能力的维度分布:")
    print(f"  {'能力':>15} {'PCA90':>6} {'EffDim':>8} {'Top1%':>7} {'Top10%':>7} {'||d||':>8} {'类型':>6}")
    print(f"  {'-'*65}")

    visual_dims = []
    nonvisual_dims = []

    for case in CASES:
        ha = get_last_hidden(model, tokenizer, case.prompt_a)
        hb = get_last_hidden(model, tokenizer, case.prompt_b)
        d = ha - hb
        dims = analyze_dimension_distribution(d, case.label)

        tag = "视觉" if case.has_visual_concept else "语言"
        print(f"  {case.label:>15} {dims['pca90']:>6} {dims['eff_dim']:>8.1f} "
              f"{dims['top1pct_energy']:>6.1%} {dims['top10pct_energy']:>6.1%} "
              f"{dims['norm']:>8.2f} {tag:>6}")

        if case.has_visual_concept:
            visual_dims.append(dims)
        else:
            nonvisual_dims.append(dims)

    # 汇总对比
    if visual_dims and nonvisual_dims:
        v_pca = statistics.mean([d["pca90"] for d in visual_dims])
        nv_pca = statistics.mean([d["pca90"] for d in nonvisual_dims])
        v_eff = statistics.mean([d["eff_dim"] for d in visual_dims])
        nv_eff = statistics.mean([d["eff_dim"] for d in nonvisual_dims])
        v_top1 = statistics.mean([d["top1pct_energy"] for d in visual_dims])
        nv_top1 = statistics.mean([d["top1pct_energy"] for d in nonvisual_dims])

        print(f"\n  汇总对比:")
        print(f"    视觉能力:  PCA90={v_pca:.0f}, EffDim={v_eff:.1f}, Top1%={v_top1:.1%}")
        print(f"    非视觉能力: PCA90={nv_pca:.0f}, EffDim={nv_eff:.1f}, Top1%={nv_top1:.1%}")
        print(f"    差异: PCA90 diff={v_pca - nv_pca:.0f}, EffDim diff={v_eff - nv_eff:.1f}")

        # INV-309验证：视觉token的PCA90更高
        inv309 = v_pca > nv_pca
        print(f"\n  INV-309验证(视觉PCA90 > 非视觉PCA90): {'✅ 确认' if inv309 else '❌ 未确认'}")
        print(f"    视觉PCA90({v_pca:.0f}) {'>' if inv309 else '<='} 非视觉PCA90({nv_pca:.0f})")

    # 2. 跨层信息传播分析
    print(f"\n[2] 跨层信息传播分析(视觉 vs 非视觉文本):")

    visual_text = "The red car stopped at the traffic light near the tall building."
    nonvisual_text = "The meeting was extremely productive and all participants agreed."
    visual_prop = analyze_cross_layer_propagation(model, tokenizer, visual_text)
    nonvisual_prop = analyze_cross_layer_propagation(model, tokenizer, nonvisual_text)

    print(f"  层   | 视觉文本cos  | 非视觉文本cos  | 层间变化(视觉) | 层间变化(非视觉)")
    print(f"  {'-'*75}")

    for i in range(visual_prop["n_layers"]):
        v_cos = visual_prop["layer_cos_to_final"][i]
        nv_cos = nonvisual_prop["layer_cos_to_final"][i]
        v_change = visual_prop["layer_to_layer_cos"][i-1] if i > 0 else 1.0
        nv_change = nonvisual_prop["layer_to_layer_cos"][i-1] if i > 0 else 1.0
        print(f"  L{i:>2}  | {v_cos:>12.4f} | {nv_cos:>13.4f} | {v_change:>13.4f} | {nv_change:>15.4f}")

    # INV-310: 跨模态干扰 vs 跨能力干扰
    print(f"\n[3] INV-310验证(跨模态干扰 vs 跨能力干扰):")

    # 计算跨模态干扰：两个视觉能力之间的cos
    visual_directions = []
    nonvisual_directions = []

    for case in CASES:
        ha = get_last_hidden(model, tokenizer, case.prompt_a)
        hb = get_last_hidden(model, tokenizer, case.prompt_b)
        d = ha - hb
        if case.has_visual_concept:
            visual_directions.append(d)
        else:
            nonvisual_directions.append(d)

    # 跨模态干扰（视觉↔视觉）
    if len(visual_directions) >= 2:
        cross_visual = []
        for i in range(len(visual_directions)):
            for j in range(i+1, len(visual_directions)):
                c = abs(F.cosine_similarity(visual_directions[i].unsqueeze(0), visual_directions[j].unsqueeze(0)).item())
                cross_visual.append(c)
        avg_cross_visual = statistics.mean(cross_visual)
    else:
        avg_cross_visual = 0

    # 跨能力干扰（非视觉↔非视觉）
    if len(nonvisual_directions) >= 2:
        cross_nonvisual = []
        for i in range(len(nonvisual_directions)):
            for j in range(i+1, len(nonvisual_directions)):
                c = abs(F.cosine_similarity(nonvisual_directions[i].unsqueeze(0), nonvisual_directions[j].unsqueeze(0)).item())
                cross_nonvisual.append(c)
        avg_cross_nonvisual = statistics.mean(cross_nonvisual)
    else:
        avg_cross_nonvisual = 0

    # 跨域干扰（视觉↔非视觉）
    cross_domain = []
    for vd in visual_directions:
        for nvd in nonvisual_directions:
            c = abs(F.cosine_similarity(vd.unsqueeze(0), nvd.unsqueeze(0)).item())
            cross_domain.append(c)
    avg_cross_domain = statistics.mean(cross_domain) if cross_domain else 0

    print(f"    视觉↔视觉(跨能力): {avg_cross_visual:.4f}")
    print(f"    非视觉↔非视觉(跨能力): {avg_cross_nonvisual:.4f}")
    print(f"    视觉↔非视觉(跨域): {avg_cross_domain:.4f}")
    print(f"    INV-310(跨域 < 跨能力): {'✅' if avg_cross_domain < avg_cross_visual else '❌'}")

    # 3. INV-311验证：纯文本输入时的编码策略
    print(f"\n[4] INV-311验证(纯文本输入的编码策略):")
    # 使用纯非视觉case计算纠缠度
    if len(nonvisual_directions) >= 2:
        pure_text_entanglement = avg_cross_nonvisual
        print(f"    纯文本能力纠缠度: {pure_text_entanglement:.4f}")
        print(f"    全部能力纠缠度(含视觉): {avg_cross_domain:.4f}")
        inv311 = pure_text_entanglement < avg_cross_domain
        print(f"    INV-311(纯文本更SEPARATED): {'✅ 确认' if inv311 else '❌ 未确认'}")

    return {
        "visual_avg_pca90": v_pca if visual_dims else 0,
        "nonvisual_avg_pca90": nv_pca if nonvisual_dims else 0,
        "visual_avg_effdim": v_eff if visual_dims else 0,
        "nonvisual_avg_effdim": nv_eff if nonvisual_dims else 0,
        "cross_visual_interference": avg_cross_visual,
        "cross_nonvisual_interference": avg_cross_nonvisual,
        "cross_domain_interference": avg_cross_domain,
        "inv309": inv309 if visual_dims and nonvisual_dims else None,
        "inv310": avg_cross_domain < avg_cross_visual if visual_directions else None,
    }


def main():
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
    print(f"模型参数: {model_arg}")

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)
    try:
        result = run_cross_modal_analysis(model, tokenizer, model_name)
    finally:
        free_model(model)

    output_file = OUTPUT_DIR / f"stage676_cross_modal_{model_arg}_{TIMESTAMP}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
