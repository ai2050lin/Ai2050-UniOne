#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage665: P18 抵消率的因果干预 — 反事实实验

P17发现（预期）：抵消可能与噪声过滤有关
P18目标：通过反事实干预，强制改变抵消率，观察对模型输出的因果效应

核心思路：
1. 在前向传播时，对特定层的增量差进行"反抵消"操作：
   - 正常：increment_diff_k 自然累加，56-78%被抵消
   - 干预：放大increment_diff_k中被"对消"的分量（反对齐d_final方向的分量被翻转）
   - 效果：抵消率降低（如从60%降到30%）
2. 测量干预后的效果：
   - margin(score函数)如何变化？
   - 模型是否仍然选择正确的token？
   - 其他能力的编码是否受干扰？

实验设计：
- 实验1：全层反抵消 — 在所有层同时降低抵消率
- 实验2：逐层反抵消 — 只在特定层降低抵消率，找到"关键层"
- 实验3：定向反抵消 — 只翻转反对齐d_final的分量，保留对齐分量
- 实验4：抵消放大 — 反向操作，人为增加抵消率

预注册判伪：
INV-288: "降低抵消率会降低margin（抵消是功能性的）"
  如果降低抵消率后margin不变或增加 → 抵消是副产品
INV-289: "存在关键层，该层的抵消率对margin影响最大"
"""

from __future__ import annotations

import sys
import io
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import copy
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    free_model,
    load_model_bundle,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    prompt_b: str
    positive_a: str
    negative_a: str


CASES = [
    TestCase(
        name="bank_financial",
        prompt_a="The bank approved the loan with favorable terms.",
        prompt_b="The river bank was covered with wild flowers.",
        positive_a="financial",
        negative_a="river",
    ),
    TestCase(
        name="grammar_subj",
        prompt_a="The tall building stands proudly.",
        prompt_b="The tall buildings stand proudly.",
        positive_a="stands",
        negative_a="stand",
    ),
    TestCase(
        name="coreference",
        prompt_a="Mary gave the book to John because she wanted to help.",
        prompt_b="Mary gave the book to John because he wanted to help.",
        positive_a="Mary",
        negative_a="John",
    ),
]


class ResidualBiasModule(nn.Module):
    """
    在指定层注入偏置到残差流中
    实现方法：替换该层的forward，在输出后加上bias向量
    """
    def __init__(self, original_layer, bias_vector: torch.Tensor):
        super().__init__()
        self.original_layer = original_layer
        self.register_buffer('bias', bias_vector.unsqueeze(0).unsqueeze(0))
        self._bias_applied = False
    
    def forward(self, *args, **kwargs):
        output = self.original_layer(*args, **kwargs)
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.bias
            return (hidden,) + output[1:]
        return output + self.bias


def compute_increment_diffs(model, tokenizer, prompt_a: str, prompt_b: str,
                             num_layers: int) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算逐层增量差和最终d_L"""
    device = next(model.parameters()).device
    
    def get_all_hidden(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = {}
        hooks = []
        layers = discover_layers(model)
        
        def make_hook(li):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden[li] = output[0][:, -1, :].detach().cpu().float()
                else:
                    hidden[li] = output[:, -1, :].detach().cpu().float()
            return hook_fn
        
        for li in range(len(layers)):
            hooks.append(layers[li].register_forward_hook(make_hook(li)))
        
        with torch.no_grad():
            model(**inputs)
        
        for h in hooks:
            h.remove()
        return hidden
    
    hidden_a = get_all_hidden(prompt_a)
    hidden_b = get_all_hidden(prompt_b)
    
    increment_diffs = {}
    prev_a, prev_b = None, None
    for li in range(num_layers):
        h_a = hidden_a.get(li)
        h_b = hidden_b.get(li)
        if h_a is None or h_b is None:
            increment_diffs[li] = None
            continue
        if prev_a is None:
            increment_diffs[li] = h_a - h_b
        else:
            increment_diffs[li] = (h_a - prev_a) - (h_b - prev_b)
        prev_a, prev_b = h_a, h_b
    
    d_final = hidden_a[num_layers - 1].flatten() - hidden_b[num_layers - 1].flatten()
    return increment_diffs, d_final, hidden_a[0].flatten(), hidden_b[0].flatten()


def run_causal_intervention(model, tokenizer, case: TestCase,
                             intervention_type: str, **kwargs) -> Dict:
    """
    执行因果干预实验
    
    intervention_type:
    - "full_anti_cancel": 全层反抵消（翻转所有反对齐分量）
    - "layer_anti_cancel": 特定层反抵消
    - "anti_align_flip": 定向翻转（只翻转反对齐d_final的分量）
    - "amplify_cancel": 放大抵消（增强抵消效应）
    """
    num_layers = len(discover_layers(model))
    device = next(model.parameters()).device

    
    # Step 1: 计算baseline
    margin_a_baseline = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    margin_b_baseline = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_a) - \
                         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_a)
    
    # Step 2: 计算增量差
    increment_diffs, d_final, h0_a, h0_b = compute_increment_diffs(
        model, tokenizer, case.prompt_a, case.prompt_b, num_layers)
    
    d_unit = d_final / (d_final.norm() + 1e-10)
    
    # Step 3: 计算反抵消偏置
    layers = discover_layers(model)
    biases = {}  # layer_idx -> bias_vector (on CPU)
    
    if intervention_type == "full_anti_cancel":
        strength = kwargs.get("strength", 1.0)
        for li in range(num_layers):
            diff_k = increment_diffs.get(li)
            if diff_k is None or diff_k.flatten().norm() < 1e-10:
                continue
            dk = diff_k.flatten()
            proj = float(torch.dot(dk, d_unit))
            if proj < 0:
                bias = -2 * proj * d_unit * strength
                biases[li] = bias
    
    elif intervention_type == "layer_anti_cancel":
        target_layer = kwargs.get("target_layer", 0)
        strength = kwargs.get("strength", 1.0)
        diff_k = increment_diffs.get(target_layer)
        if diff_k is not None and diff_k.flatten().norm() > 1e-10:
            dk = diff_k.flatten()
            proj = float(torch.dot(dk, d_unit))
            if proj < 0:
                bias = -2 * proj * d_unit * strength
                biases[target_layer] = bias
    
    elif intervention_type == "anti_align_flip":
        strength = kwargs.get("strength", 0.5)
        for li in range(num_layers):
            diff_k = increment_diffs.get(li)
            if diff_k is None or diff_k.flatten().norm() < 1e-10:
                continue
            dk = diff_k.flatten()
            proj = float(torch.dot(dk, d_unit))
            if proj < 0:
                bias = -proj * d_unit * strength * 2
                biases[li] = bias
    
    elif intervention_type == "amplify_cancel":
        strength = kwargs.get("strength", 0.5)
        for li in range(num_layers):
            diff_k = increment_diffs.get(li)
            if diff_k is None or diff_k.flatten().norm() < 1e-10:
                continue
            dk = diff_k.flatten()
            proj = float(torch.dot(dk, d_unit))
            if proj > 0:
                bias = -proj * d_unit * strength
                biases[li] = bias
    
    if not biases:
        return {
            "intervention": intervention_type,
            "margin_a_baseline": margin_a_baseline,
            "margin_b_baseline": margin_b_baseline,
            "margin_a_intervened": margin_a_baseline,
            "margin_b_intervened": margin_b_baseline,
            "delta_margin_a": 0,
            "delta_margin_b": 0,
            "n_layers_intervened": 0,
            "note": "no biases computed",
        }
    
    # Step 4: 应用偏置并测量
    # 使用hook方式注入偏置
    original_forwards = {}
    hooks = []
    
    def make_bias_hook(bias_tensor):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                bias_device = bias_tensor.to(hidden.device, hidden.dtype)
                # hidden shape: (batch, seq, dim), bias shape: (dim,)
                hidden = hidden + bias_device.unsqueeze(0).unsqueeze(0)
                return (hidden,) + output[1:]
            else:
                bias_device = bias_tensor.to(output.device, output.dtype)
                return output + bias_device.unsqueeze(0).unsqueeze(0)
        return hook_fn
    
    for li, bias in biases.items():
        if li < len(layers):
            hooks.append(layers[li].register_forward_hook(make_bias_hook(bias)))
    
    try:
        margin_a_intervened = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                              score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
        margin_b_intervened = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_a) - \
                              score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_a)
    finally:
        for h in hooks:
            h.remove()
    
    return {
        "intervention": intervention_type,
        "margin_a_baseline": margin_a_baseline,
        "margin_b_baseline": margin_b_baseline,
        "margin_a_intervened": margin_a_intervened,
        "margin_b_intervened": margin_b_intervened,
        "delta_margin_a": margin_a_intervened - margin_a_baseline,
        "delta_margin_b": margin_b_intervened - margin_b_baseline,
        "n_layers_intervened": len(biases),
        "n_anti_aligned_layers": len([li for li in biases]),
    }


def run_experiment(model_key: str):
    print(f"\n{'='*70}")
    print(f"Stage665: P18 抵消率因果干预 - {model_key}")
    print(f"{'='*70}")

    model, tokenizer = load_model_bundle(model_key)
    num_layers = len(discover_layers(model))
    device = next(model.parameters()).device
    
    all_results = {}

    for case in CASES:
        print(f"\n--- {case.name} ---")
        case_results = {}
        
        # Baseline
        margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
                   score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
        margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_a) - \
                   score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_a)
        print(f"  Baseline: margin_A={margin_a:.4f}, margin_B={margin_b:.4f}")
        
        # ========== 实验1: 全层反抵消 ==========
        print(f"\n  实验1: 全层反抵消...")
        for strength in [0.5, 1.0, 2.0]:
            r = run_causal_intervention(model, tokenizer, case, "full_anti_cancel", strength=strength)
            print(f"    strength={strength}: delta_A={r['delta_margin_a']:.4f}, "
                  f"delta_B={r['delta_margin_b']:.4f}, "
                  f"intervened_layers={r['n_layers_intervened']}")
            case_results[f"full_anti_cancel_s{strength}"] = r
        
        # ========== 实验2: 逐层反抵消 ==========
        print(f"\n  实验2: 逐层反抵消(每5层)...")
        step = max(1, num_layers // 5)
        for li in range(0, num_layers, step):
            r = run_causal_intervention(model, tokenizer, case, "layer_anti_cancel",
                                         target_layer=li, strength=1.0)
            print(f"    layer {li}: delta_A={r['delta_margin_a']:.4f}, delta_B={r['delta_margin_b']:.4f}")
            case_results[f"layer_anti_cancel_l{li}"] = r
        
        # ========== 实验3: 定向翻转 ==========
        print(f"\n  实验3: 定向翻转(反对齐分量)...")
        for strength in [0.3, 0.5, 1.0]:
            r = run_causal_intervention(model, tokenizer, case, "anti_align_flip", strength=strength)
            print(f"    strength={strength}: delta_A={r['delta_margin_a']:.4f}, "
                  f"delta_B={r['delta_margin_b']:.4f}")
            case_results[f"anti_align_flip_s{strength}"] = r
        
        # ========== 实验4: 放大抵消 ==========
        print(f"\n  实验4: 放大抵消...")
        for strength in [0.3, 0.5, 1.0]:
            r = run_causal_intervention(model, tokenizer, case, "amplify_cancel", strength=strength)
            print(f"    strength={strength}: delta_A={r['delta_margin_a']:.4f}, "
                  f"delta_B={r['delta_margin_b']:.4f}")
            case_results[f"amplify_cancel_s{strength}"] = r
        
        all_results[case.name] = case_results
    
    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print(f"汇总统计")
    print(f"{'='*70}")
    
    for case_name, cr in all_results.items():
        print(f"\n  {case_name}:")
        # 全层反抵消的效果
        for key in ["full_anti_cancel_s0.5", "full_anti_cancel_s1.0", "full_anti_cancel_s2.0"]:
            if key in cr:
                r = cr[key]
                print(f"    {key}: delta_A={r['delta_margin_a']:.4f}, delta_B={r['delta_margin_b']:.4f}")
    
    # INV-288判伪
    print(f"\n  INV-288判伪:")
    for case_name, cr in all_results.items():
        # 检查全层反抵消(s=1.0)是否降低了margin
        key = "full_anti_cancel_s1.0"
        if key in cr:
            r = cr[key]
            margin_decreased = r['delta_margin_a'] < 0
            print(f"    {case_name}: margin_A变化={r['delta_margin_a']:.4f} "
                  f"→ {'降低(抵消功能性)' if margin_decreased else '增加/不变(抵消非功能)'}")
    
    # 保存结果
    out_path = OUTPUT_DIR / f"stage665_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")
    
    free_model(model)
    return all_results


if __name__ == "__main__":
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_key)
