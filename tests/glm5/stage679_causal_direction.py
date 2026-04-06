#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage679: P33 因果方向验证——纠缠度与推理能力的关系

目标：验证DS7B的高纠缠度是"原因"（RL训练导致融合，融合增强推理）
     还是"结果"（强推理能力导致编码需要更多维度共享）。

方法：
  1. 在四模型上测量"推理难度梯度"下的纠缠度变化
     - 简单推理（A→B直接推导）
     - 中等推理（需要2步逻辑链）
     - 困难推理（需要3+步逻辑链）
  2. 如果纠缠度随推理难度增加而增加 → 支持融合=结果假说
  3. 如果纠缠度在不同难度下保持稳定 → 支持融合=原因假说

  4. 额外测量：信息密度 vs 纠缠度的关系
     - 高信息密度句子（如数学/代码）vs 低信息密度（如日常对话）
     - 验证纠缠度是否与信息密度相关

INV-320: 推理难度↑ → 纠缠度↑（如果成立，说明纠缠是结果而非原因）
INV-321: 信息密度↑ → 纠缠度↑（如果成立，说明高维共享是信息处理需求）
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


# ============================================================
# 测试用例——按推理难度分级
# ============================================================
@dataclass(frozen=True)
class ReasoningPair:
    """一对推理句子"""
    difficulty: str  # easy/medium/hard
    domain: str  # math/logic/spatial/causal
    prompt_a: str
    prompt_b: str
    label: str


REASONING_PAIRS = [
    # === 简单推理（单步直接推导）===
    ReasoningPair("easy", "math", "3 + 4 = 7", "5 + 2 = 7", "简单加法"),
    ReasoningPair("easy", "math", "10 - 3 = 7", "9 - 2 = 7", "简单减法"),
    ReasoningPair("easy", "logic", "All cats are animals.", "All dogs are animals.", "简单分类"),
    ReasoningPair("easy", "spatial", "The book is on the table.", "The cup is on the desk.", "简单空间"),
    ReasoningPair("easy", "causal", "Rain makes the ground wet.", "Snow makes the ground cold.", "简单因果"),

    # === 中等推理（2步逻辑链）===
    ReasoningPair("medium", "math", "If x=3 then x squared = 9.", "If y=4 then y squared = 16.", "变量代换"),
    ReasoningPair("medium", "logic", "If A implies B and B implies C, then A implies C.", "If X implies Y and Y implies Z, then X implies Z.", "三段论"),
    ReasoningPair("medium", "spatial", "The key is in the box under the table.", "The letter is in the envelope on the shelf.", "嵌套空间"),
    ReasoningPair("medium", "causal", "Because it rained, the match was cancelled, so the team went home.", "Because it snowed, the flight was delayed, so the passengers waited.", "因果链"),
    ReasoningPair("medium", "math", "The average of 4 and 6 is 5.", "The average of 3 and 9 is 6.", "平均值计算"),

    # === 困难推理（3+步逻辑链）===
    ReasoningPair("hard", "math", "If f(x) = 2x + 1, then f(f(3)) = 15.", "If g(x) = x squared - 2, then g(g(3)) = 7.", "复合函数"),
    ReasoningPair("hard", "logic", "If all A are B, all B are C, and some C are D, then some A may be D.", "If all X are Y, all Y are Z, and some Z are W, then some X may be W.", "三段论扩展"),
    ReasoningPair("hard", "causal", "The policy increased taxes, reducing consumer spending, which lowered corporate profits, causing layoffs.", "The law reduced regulations, increasing production, which raised exports, creating jobs.", "多步因果"),
    ReasoningPair("hard", "math", "The probability of getting heads twice is one fourth.", "The probability of rolling six twice is one thirty-sixth.", "概率计算"),
    ReasoningPair("hard", "logic", "NOT(NOT(P AND Q)) is equivalent to P AND Q.", "NOT(NOT(A OR B)) is equivalent to A OR B.", "德摩根定律"),
]


# ============================================================
# 信息密度测试——高密度vs低密度
# ============================================================
@dataclass(frozen=True)
class DensityPair:
    density: str  # high/low
    domain: str
    prompt_a: str
    prompt_b: str
    label: str


DENSITY_PAIRS = [
    # 高信息密度（技术/数学/代码）
    DensityPair("high", "math", "The derivative of x cubed is 3x squared.", "The integral of 2x is x squared plus C.", "微积分"),
    DensityPair("high", "code", "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", "def factorial(n): return 1 if n == 0 else n * factorial(n-1)", "递归函数"),
    DensityPair("high", "science", "E equals m c squared relates energy to mass.", "F equals m a defines force in Newtonian mechanics.", "物理公式"),
    DensityPair("high", "logic", "The set of all prime numbers is infinite.", "The set of all real numbers between zero and one is uncountable.", "数学定理"),

    # 低信息密度（日常/社交/情感）
    DensityPair("low", "social", "How was your day at work today?", "Did you enjoy the movie last night?", "日常对话"),
    DensityPair("low", "emotion", "She felt happy and smiled warmly.", "He looked sad and sighed deeply.", "情感描述"),
    DensityPair("low", "narrative", "The old man walked slowly down the street.", "The young girl ran quickly across the field.", "简单叙事"),
    DensityPair("low", "routine", "I usually have breakfast at seven in the morning.", "She always goes to bed before ten at night.", "日常习惯"),
]


def get_direction(model, tokenizer, text, layer_idx=-1):
    """获取指定层的hidden state方向"""
    model_device = next(model.parameters()).device
    tokens = tokenizer.encode(text, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx]
    h = hs[0, -1, :].float().cpu()
    return h


def get_all_directions(model, tokenizer, pairs, layers_to_check=None):
    """获取所有pair在指定层的方向"""
    results = {}
    for pair in pairs:
        ha = get_direction(model, tokenizer, pair.prompt_a)
        hb = get_direction(model, tokenizer, pair.prompt_b)
        delta = ha - hb
        delta_norm = F.normalize(delta.unsqueeze(0)).squeeze(0)
        results[pair.label] = {
            "vector": delta_norm,
            "pair": pair,
            "norm": delta.norm().item(),
        }
    return results


def compute_entanglement_matrix(directions):
    """计算方向纠缠度矩阵"""
    labels = list(directions.keys())
    matrix = {}
    abs_values = []
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i < j:
                cos_v = F.cosine_similarity(
                    directions[l1]["vector"].unsqueeze(0),
                    directions[l2]["vector"].unsqueeze(0)
                ).item()
                matrix[(l1, l2)] = cos_v
                abs_values.append(abs(cos_v))
    mean_abs = statistics.mean(abs_values) if abs_values else 0
    std_abs = statistics.stdev(abs_values) if len(abs_values) > 1 else 0
    return matrix, mean_abs, std_abs


def run_causal_experiment(model_arg):
    """主函数：因果方向验证"""
    print("=" * 65)
    print(f"  P33 因果方向验证——纠缠度与推理能力的关系")
    print(f"  模型: {model_arg}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    model, tokenizer = load_model_bundle(model_arg)
    if model is None:
        print(f"无法加载模型: {model_arg}")
        return None

    model_name = MODEL_SPECS.get(model_arg, {}).get("label", model_arg)

    # 获取总层数
    try:
        total_layers = getattr(model.config, 'num_hidden_layers',
                                getattr(model.config, 'num_layers',
                                        getattr(model.config, 'n_layer', '?')))
        print(f"\n  总层数: {total_layers}")
    except Exception:
        print(f"\n  总层数: (无法获取)")

    # === 实验1：推理难度梯度 ===
    print(f"\n{'='*50}")
    print("  实验1: 推理难度梯度下的纠缠度")
    print(f"{'='*50}")

    all_dirs = get_all_directions(model, tokenizer, REASONING_PAIRS)

    # 按难度分组
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for label, info in all_dirs.items():
        by_difficulty[info["pair"].difficulty].append(label)

    diff_results = {}
    for diff in ["easy", "medium", "hard"]:
        sub_dirs = {l: all_dirs[l] for l in by_difficulty[diff]}
        if len(sub_dirs) < 2:
            diff_results[diff] = {"entangle": 0, "std": 0}
            continue
        _, mean_abs, std_abs = compute_entanglement_matrix(sub_dirs)
        diff_results[diff] = {"entangle": mean_abs, "std": std_abs}
        print(f"  {diff:>8}: 纠缠度={mean_abs:.4f} ± {std_abs:.4f} (n={len(sub_dirs)})")

    # 跨难度纠缠（easy↔medium, medium↔hard, easy↔hard）
    print(f"\n  跨难度纠缠:")
    for d1, d2 in [("easy", "medium"), ("medium", "hard"), ("easy", "hard")]:
        labels1 = by_difficulty[d1]
        labels2 = by_difficulty[d2]
        cross_pairs = []
        for l1 in labels1:
            for l2 in labels2:
                cos_v = F.cosine_similarity(
                    all_dirs[l1]["vector"].unsqueeze(0),
                    all_dirs[l2]["vector"].unsqueeze(0)
                ).item()
                cross_pairs.append(abs(cos_v))
        cross_mean = statistics.mean(cross_pairs) if cross_pairs else 0
        cross_std = statistics.stdev(cross_pairs) if len(cross_pairs) > 1 else 0
        print(f"    {d1}↔{d2}: {cross_mean:.4f} ± {cross_std:.4f}")

    # === 实验2：信息密度 ===
    print(f"\n{'='*50}")
    print("  实验2: 信息密度与纠缠度")
    print(f"{'='*50}")

    density_dirs = get_all_directions(model, tokenizer, DENSITY_PAIRS)

    high_labels = [l for l, info in density_dirs.items() if info["pair"].density == "high"]
    low_labels = [l for l, info in density_dirs.items() if info["pair"].density == "low"]

    high_dirs = {l: density_dirs[l] for l in high_labels}
    low_dirs = {l: density_dirs[l] for l in low_labels}

    _, high_ent, high_std = compute_entanglement_matrix(high_dirs)
    _, low_ent, low_std = compute_entanglement_matrix(low_dirs)
    print(f"  高信息密度: 纠缠度={high_ent:.4f} ± {high_std:.4f} (n={len(high_dirs)})")
    print(f"  低信息密度: 纠缠度={low_ent:.4f} ± {low_std:.4f} (n={len(low_dirs)})")

    # 高↔低
    cross_hl = []
    for l1 in high_labels:
        for l2 in low_labels:
            cos_v = F.cosine_similarity(
                density_dirs[l1]["vector"].unsqueeze(0),
                density_dirs[l2]["vector"].unsqueeze(0)
            ).item()
            cross_hl.append(abs(cos_v))
    cross_hl_mean = statistics.mean(cross_hl) if cross_hl else 0
    print(f"  高↔低交叉: {cross_hl_mean:.4f}")

    # === 实验3：方向范数分析 ===
    print(f"\n{'='*50}")
    print("  实验3: 方向范数（信号强度）分析")
    print(f"{'='*50}")

    all_norms = {l: info["norm"] for l, info in all_dirs.items()}
    diff_norms = {}
    for diff in ["easy", "medium", "hard"]:
        norms = [all_norms[l] for l in by_difficulty[diff]]
        diff_norms[diff] = statistics.mean(norms) if norms else 0
        print(f"  {diff:>8}: 平均范数={diff_norms[diff]:.2f}")

    # 密度范数
    high_norms = [density_dirs[l]["norm"] for l in high_labels]
    low_norms = [density_dirs[l]["norm"] for l in low_labels]
    print(f"  高密度: 平均范数={statistics.mean(high_norms):.2f}")
    print(f"  低密度: 平均范数={statistics.mean(low_norms):.2f}")

    # === 汇总 ===
    print(f"\n{'='*50}")
    print(f"  P33 汇总——{model_name}")
    print(f"{'='*50}")

    # INV-320验证
    ent_easy = diff_results["easy"]["entangle"]
    ent_hard = diff_results["hard"]["entangle"]
    inv320 = "✅确认" if ent_hard > ent_easy * 1.2 else "❌未确认"
    print(f"\n  INV-320 推理难度↑→纠缠度↑: {inv320}")
    print(f"    easy={ent_easy:.4f}, hard={ent_hard:.4f}, ratio={ent_hard/max(ent_easy,1e-8):.2f}x")

    # INV-321验证
    inv321 = "✅确认" if high_ent > low_ent * 1.2 else "❌未确认"
    print(f"\n  INV-321 信息密度↑→纠缠度↑: {inv321}")
    print(f"    high={high_ent:.4f}, low={low_ent:.4f}, ratio={high_ent/max(low_ent,1e-8):.2f}x")

    # 范数与纠缠度的关系
    print(f"\n  范数(信号强度) vs 纠缠度:")
    for diff in ["easy", "medium", "hard"]:
        print(f"    {diff}: norm={diff_norms[diff]:.2f}, entangle={diff_results[diff]['entangle']:.4f}")

    free_model(model)

    return {
        "model": model_name,
        "difficulty": diff_results,
        "density": {"high": high_ent, "low": low_ent, "cross": cross_hl_mean},
        "norms": diff_norms,
        "inv320": inv320,
        "inv321": inv321,
    }


if __name__ == "__main__":
    import sys
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_causal_experiment(model_arg)
