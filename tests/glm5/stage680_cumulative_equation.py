#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage680: P34 多步累积方程——从单步Logit到多步生成概率的数学桥梁

目标：建立单步margin→多步生成概率的累积方程，连接P27的精确Logit方程
     和P27多步生成过程分析中的动力学行为。

核心问题：
  P27发现单步logit方程margin=cos(h,u)×||h||×||u||精确到0.3%，
  但多步avg_logprob与单步margin差了100倍。原因是什么？

假说：
  1. 多步累积中，每一步的条件概率依赖前一步的输出（自回归效应）
  2. margin在每步的分布不同（信号放大/稀释效应）
  3. 概率空间的对数累积 vs 线性margin的尺度差异

实验方法：
  1. 对同一个prompt，测量生成每一步的margin和logit
  2. 建立 P(token_t | token_<t) = softmax(h_t · W) 的逐步追踪
  3. 验证累积方程: P(完整序列) = Π_t P(token_t | token_<t)
  4. 分析margin在多步中的衰减/放大模式

INV-322: 多步累积中，margin的log均值 ≈ avg_logprob + C（某个常数偏移）
INV-323: 信号放大模型的margin递增 → 总生成概率更高（模型更"自信"）
"""

from __future__ import annotations

import sys
import io
import json
import math

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import statistics
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


# 测试用例
TEST_PROMPTS = [
    {
        "prompt": "The capital of France is",
        "expected_next": "Paris",
        "label": "常识推理"
    },
    {
        "prompt": "If it rains tomorrow, then the ground will be",
        "expected_next": "wet",
        "label": "因果推理"
    },
    {
        "prompt": "The opposite of hot is",
        "expected_next": "cold",
        "label": "语义推理"
    },
    {
        "prompt": "In the equation 2x = 10, x equals",
        "expected_next": "5",
        "label": "数学推理"
    },
    {
        "prompt": "A cat is a type of",
        "expected_next": "animal",
        "label": "分类推理"
    },
]


def generate_and_trace(model, tokenizer, prompt, n_steps=8, temperature=1.0):
    """逐步生成并追踪每一步的margin/logit/probability"""
    model_device = next(model.parameters()).device

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model_device)
    input_ids = prompt_tokens.clone()
    generated_tokens = []

    trace = {
        "prompt": prompt,
        "steps": [],
        "total_log_prob": 0.0,
    }

    for step in range(n_steps):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # 最后一层的hidden state
        last_hs = outputs.hidden_states[-1][0, -1, :].float()  # (d,)

        # Logits
        logits = outputs.logits[0, -1, :]  # (V,)

        # 找top token
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        # 贪心解码
        top_token = logits.argmax().item()
        top_prob = probs[top_token].item()
        top_log_prob = log_probs[top_token].item()
        generated_text = tokenizer.decode([top_token], clean_up_tokenization_spaces=True)

        # margin分析：top-1 vs top-2
        sorted_logits, sorted_indices = logits.sort(descending=True)
        if sorted_logits.shape[0] > 1:
            margin = (sorted_logits[0] - sorted_logits[1]).item()
        else:
            margin = 0.0

        # Logit精确方程验证
        unembed = model.lm_head.weight.float()  # (V, d)
        if unembed.shape[0] == logits.shape[0]:
            # 直接计算：h · w_top1 vs h · w_top2
            h_dot_w1 = torch.dot(last_hs, unembed[top_token]).item()
            h_dot_w2 = torch.dot(last_hs, unembed[sorted_indices[1]]).item()
            logit_eq_margin = h_dot_w1 - h_dot_w2
            logit_eq_error = abs(logit_eq_margin - margin) / max(abs(margin), 1e-8) * 100
        else:
            logit_eq_margin = margin
            logit_eq_error = -1

        # ||h|| 和 ||w|| 分解
        h_norm = last_hs.norm().item()
        w_top1 = unembed[top_token] if unembed.shape[0] == logits.shape[0] else None
        w_norm = w_top1.norm().item() if w_top1 is not None else 0

        step_info = {
            "step": step,
            "token_id": top_token,
            "token_text": generated_text.strip(),
            "prob": top_prob,
            "log_prob": top_log_prob,
            "margin": margin,
            "h_norm": h_norm,
            "w_norm": w_norm,
            "logit_eq_error": logit_eq_error,
            "entropy": -(probs * log_probs).sum().item(),  # 信息熵
            "top5_probs": [probs[sorted_indices[i]].item() for i in range(min(5, sorted_logits.shape[0]))],
        }

        trace["steps"].append(step_info)
        trace["total_log_prob"] += top_log_prob

        # 拼接token继续生成
        input_ids = torch.cat([input_ids, torch.tensor([[top_token]], device=model_device)], dim=1)
        generated_tokens.append(top_token)

    # 计算平均log_prob
    trace["avg_log_prob"] = trace["total_log_prob"] / n_steps
    # 平均margin
    margins = [s["margin"] for s in trace["steps"]]
    trace["avg_margin"] = statistics.mean(margins)
    trace["margin_std"] = statistics.stdev(margins) if len(margins) > 1 else 0
    # margin趋势（最后3步 vs 前3步）
    if len(margins) >= 6:
        early_margins = margins[:3]
        late_margins = margins[-3:]
        trace["margin_early"] = statistics.mean(early_margins)
        trace["margin_late"] = statistics.mean(late_margins)
        trace["margin_amplification"] = trace["margin_late"] / max(trace["margin_early"], 1e-8)
    else:
        trace["margin_amplification"] = 1.0

    return trace


def run_cumulative_experiment(model_arg):
    """主函数：多步累积方程实验"""
    print("=" * 65)
    print(f"  P34 多步累积方程——从单步Logit到多步生成概率")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    all_traces = []

    for tc in TEST_PROMPTS:
        print(f"\n  --- {tc['label']}: \"{tc['prompt']}\" ---")
        trace = generate_and_trace(model, tokenizer, tc["prompt"], n_steps=8)
        all_traces.append(trace)

        # 逐步输出
        print(f"  生成: {''.join(s['token_text'] for s in trace['steps'][:5])}...")
        print(f"  avg_log_prob={trace['avg_log_prob']:.4f}, avg_margin={trace['avg_margin']:.3f}")
        print(f"  margin趋势: early={trace.get('margin_early', 0):.3f} → late={trace.get('margin_late', 0):.3f} ({trace['margin_amplification']:.2f}x)")

        # 逐步详情
        for s in trace["steps"][:4]:
            print(f"    step {s['step']}: '{s['token_text']}' prob={s['prob']:.4f} margin={s['margin']:.3f} h_norm={s['h_norm']:.1f} entropy={s['entropy']:.2f}")

    # 汇总分析
    print(f"\n{'='*50}")
    print(f"  P34 汇总——{model_name}")
    print(f"{'='*50}")

    avg_log_probs = [t["avg_log_prob"] for t in all_traces]
    avg_margins = [t["avg_margin"] for t in all_traces]
    amplifications = [t["margin_amplification"] for t in all_traces]

    print(f"\n  avg_log_prob范围: {min(avg_log_probs):.4f} ~ {max(avg_log_probs):.4f}")
    print(f"  avg_margin范围:  {min(avg_margins):.3f} ~ {max(avg_margins):.3f}")
    print(f"  放大系数范围:    {min(amplifications):.2f}x ~ {max(amplifications):.2f}x")
    print(f"  放大系数均值:    {statistics.mean(amplifications):.2f}x")

    # INV-322验证：margin的log均值 vs avg_logprob
    print(f"\n  --- INV-322: margin vs avg_logprob关系 ---")
    for t in all_traces:
        # 转换margin到log概率尺度
        log_margin = math.log(max(t["avg_margin"], 1e-8))
        gap = log_margin - t["avg_log_prob"]
        print(f"    {t['prompt'][:30]:30s}: log(margin)={log_margin:.2f}, avg_lp={t['avg_log_prob']:.4f}, gap={gap:.2f}")

    # INV-323验证：放大系数与总log_prob的关系
    print(f"\n  --- INV-323: 放大系数 vs 总log_prob ---")
    for t in all_traces:
        print(f"    {t['prompt'][:30]:30s}: amp={t['margin_amplification']:.2f}x, total_lp={t['total_log_prob']:.2f}")

    # 关键发现
    print(f"\n  --- 核心发现 ---")
    # 计算logit方程精度
    errors = []
    for t in all_traces:
        for s in t["steps"]:
            if s["logit_eq_error"] >= 0:
                errors.append(s["logit_eq_error"])
    if errors:
        print(f"  Logit精确方程精度: {statistics.mean(errors):.4f}% (误差)")

    # margin在多步中的变化模式
    print(f"\n  多步margin变化模式:")
    for t in all_traces:
        margins = [s["margin"] for s in t["steps"]]
        trend = "递增↑" if margins[-1] > margins[0] * 1.2 else ("递减↓" if margins[-1] < margins[0] * 0.8 else "平稳→")
        print(f"    {t['prompt'][:30]:30s}: {margins[0]:.2f} → {margins[-1]:.2f} ({trend})")

    free_model(model)

    return {"traces": all_traces, "model": model_name}


if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_cumulative_experiment(model_arg)
