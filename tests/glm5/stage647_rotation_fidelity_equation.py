#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage647: 旋转-保真度定量方程

目标：建立"旋转角θ → 方向保留率R(θ)"的定量关系
方法：
1. 在多类能力（语法/推理/关系/共指）上提取差异方向
2. 对每种能力，追踪方向在10层内的cos衰减曲线
3. 拟合指数衰减模型 R(k) = exp(-α·k)，得到衰减速率α
4. 计算模型平均旋转速度θ，建立 α 与 θ 的关系
5. 跨模型验证：R(θ) = exp(-c·θ) 的拟合优度

预注册判伪条件：
INV-213: "衰减速率α与旋转速度θ正相关"
如果跨模型的α与θ的Pearson相关系数<0.3，则INV-213被推翻。

INV-214: "存在跨模型统一的衰减模型 R(k)=exp(-αk)"
如果四模型的平均R²<0.5（拟合优度差），则INV-214被推翻。
"""

from __future__ import annotations

import sys
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    discover_layers,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# 综合cases：语法+推理+关系+共指
CASES = {
    "syntax": [
        ("The key to the cabinet", " is", " are",
         "The keys to the cabinet", " are", " is"),
        ("The bouquet of roses", " smells", " smell",
         "The roses in the bouquet", " smell", " smells"),
    ],
    "syllogism": [
        ("All mammals are animals. All cats are mammals. Therefore, all cats are",
         " animals", " reptiles",
         "All birds are animals. All sparrows are birds. Therefore, all sparrows are",
         " animals", " insects"),
    ],
    "relation": [
        ("Paris is the capital of France. The capital of France is",
         " Paris", " Berlin",
         "Berlin is the capital of Germany. The capital of Germany is",
         " Berlin", " Paris"),
    ],
    "coref": [
        ("Alice thanked Mary because Alice had won the prize. The person who won was",
         " Alice", " Mary",
         "Alice thanked Mary because Mary had won the prize. The person who won was",
         " Mary", " Alice"),
    ],
    "arithmetic": [
        ("If x = 7 and y = 3, then x + y =",
         " 10", " 11",
         "If x = 15 and y = 8, then x + y =",
         " 23", " 22"),
    ],
}


def extract_layer_last_token(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    layers = discover_layers(model)
    device = get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captured = {}

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"]


def trace_cos_decay(model, tokenizer, prompt_a: str, prompt_b: str,
                    anchor_layer: int, num_layers: int, max_trace: int = 12) -> List[float]:
    """追踪方向余弦衰减，返回逐层cos值列表"""
    delta_anchor = extract_layer_last_token(model, tokenizer, prompt_a, anchor_layer) - \
                   extract_layer_last_token(model, tokenizer, prompt_b, anchor_layer)
    da_norm = delta_anchor / (torch.norm(delta_anchor) + 1e-10)

    cos_list = [1.0]  # offset 0 = 自身 = 1.0
    end_layer = min(anchor_layer + max_trace, num_layers)

    for target_layer in range(anchor_layer + 1, end_layer):
        delta_target = extract_layer_last_token(model, tokenizer, prompt_a, target_layer) - \
                       extract_layer_last_token(model, tokenizer, prompt_b, target_layer)
        dt_norm = delta_target / (torch.norm(delta_target) + 1e-10)
        cos_val = torch.dot(da_norm, dt_norm).item()
        cos_list.append(cos_val)

    return cos_list


def fit_exponential_decay(cos_list: List[float]) -> Dict:
    """拟合 R(k) = exp(-α·k)，返回α和R²"""
    if len(cos_list) < 3:
        return {"alpha": None, "r_squared": None, "fitted": []}

    offsets = np.arange(len(cos_list), dtype=float)
    values = np.array(cos_list)

    # 对|cos|取对数拟合
    abs_values = np.abs(values)
    abs_values = np.clip(abs_values, 1e-8, None)
    log_values = np.log(abs_values)

    # 线性回归: log(R) = -α·k + c
    try:
        coeffs = np.polyfit(offsets, log_values, 1)
        alpha = -coeffs[0]  # α（衰减速率）
        intercept = coeffs[1]

        # R²
        predicted = alpha * offsets + intercept
        ss_res = np.sum((log_values - predicted) ** 2)
        ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        # 拟合曲线
        fitted = list(np.exp(-alpha * offsets) * np.exp(intercept))

        return {
            "alpha": round(float(alpha), 4),
            "r_squared": round(float(max(r_squared, 0)), 4),
            "intercept": round(float(intercept), 4),
            "fitted": [round(v, 4) for v in fitted],
            "observed": [round(v, 4) for v in cos_list],
        }
    except Exception as e:
        return {"alpha": None, "r_squared": None, "error": str(e)}


def compute_avg_rotation(model, tokenizer, num_layers: int) -> float:
    """计算模型平均逐层旋转角度"""
    templates = [
        "The cat sat on the mat.", "The dog ran in the park.",
        "A bird flew over the house.", "She read a book yesterday.",
        "They went to the store.", "He plays piano every day.",
    ]
    rotations = []
    for prompt in templates:
        for l1, l2 in [(0, 1), (3, 4), (7, 8), (11, 12)]:
            if l2 >= num_layers:
                continue
            try:
                h1 = extract_layer_last_token(model, tokenizer, prompt, l1)
                h2 = extract_layer_last_token(model, tokenizer, prompt, l2)
                delta = h2 - h1
                if torch.norm(h1) > 1e-8 and torch.norm(delta) > 1e-8:
                    cos_v = torch.dot(h1 / torch.norm(h1), delta / torch.norm(delta)).item()
                    cos_v = max(-1.0, min(1.0, cos_v))
                    rotations.append(abs(np.degrees(np.arccos(cos_v))))
            except Exception:
                continue
    return float(np.mean(rotations)) if rotations else 0.0


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python stage647_rotation_fidelity_equation.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage647_rotation_fidelity_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage647] 加载模型 {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage647] 层数={num_layers}")

        # 平均旋转速度
        print(f"[Stage647] 计算平均旋转速度...")
        avg_rotation = compute_avg_rotation(model, tokenizer, num_layers)
        print(f"[Stage647] 平均旋转速度: {avg_rotation:.1f}°/层")

        records = []
        all_alphas = []

        for cap_name, case_list in CASES.items():
            print(f"\n[Stage647] 能力: {cap_name}")
            for j, (pa, ga, na, pb, gb, nb) in enumerate(case_list):
                print(f"  case {j+1}...")
                # 在L0开始追踪（所有模型delta都是rank-1，在L0最显著）
                anchor_layer = 0
                cos_decay = trace_cos_decay(model, tokenizer, pa, pb, anchor_layer, num_layers, max_trace=12)

                fit = fit_exponential_decay(cos_decay)
                alpha = fit.get("alpha")
                r2 = fit.get("r_squared")

                if alpha is not None:
                    all_alphas.append(alpha)
                    if len(cos_decay) > 5:
                        print(f"  alpha={alpha:.4f}, R2={r2:.4f}, cos5={cos_decay[5]:.4f}")
                    else:
                        print(f"  alpha={alpha:.4f}, R2={r2:.4f}")
                else:
                    print(f"  拟合失败: {fit.get('error', 'unknown')}")

                records.append({
                    "capability": cap_name,
                    "case_idx": j,
                    "anchor_layer": anchor_layer,
                    "cos_decay": [round(v, 4) for v in cos_decay],
                    "fit": fit,
                })

        # 汇总
        avg_alpha = float(np.mean(all_alphas)) if all_alphas else None
        avg_r2 = float(np.mean([r["fit"]["r_squared"] for r in records if r["fit"].get("r_squared") is not None])) if records else None

        print(f"\n{'='*60}")
        print(f"[Stage647] 汇总: {model_key}")
        print(f"  平均旋转速度: {avg_rotation:.1f}°/层")
        print(f"  平均衰减速率α: {avg_alpha:.4f}" if avg_alpha is not None else "  α: N/A")
        print(f"  平均拟合R²: {avg_r2:.4f}" if avg_r2 is not None else "  R²: N/A")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "avg_rotation_speed": round(avg_rotation, 1),
            "avg_decay_alpha": round(avg_alpha, 4) if avg_alpha is not None else None,
            "avg_fit_r_squared": round(avg_r2, 4) if avg_r2 is not None else None,
            "per_capability_alphas": {
                cap: round(float(np.mean([r["fit"]["alpha"] for r in records
                        if r["capability"] == cap and r["fit"].get("alpha") is not None])), 4)
                for cap in CASES.keys()
            },
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
