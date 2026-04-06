#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage652: Gemma4抑制机制精确测量 + 四模型对比

P4发现：Gemma4消融前5层MLP后margin反而提升（first5_drop=-1.83），
说明Gemma4前层在"压制"某些错误通路。

本阶段精确测量：
1. 逐层消融L0-L4的MLP，看哪一层导致margin提升最多
2. 对比四个模型的前层消融效果
3. 分析"抑制"方向：消融后哪些维度变化最大
4. 分析"抑制"是否是Gemma4独有的机制

预注册判伪条件：
INV-229: "Gemma4前层MLP消融提升margin（抑制效应）是Gemma4独有的"
如果其他三个模型也有类似效应（虽然较弱），则INV-229需修正。
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
    ablate_layer_component,
    discover_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


@dataclass(frozen=True)
class TestCase:
    name: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[TestCase] = [
    TestCase("syllogism",
             "All mammals are animals. All cats are mammals. Cats are",
             " animals", " reptiles",
             "All birds are animals. All sparrows are birds. Sparrows are",
             " animals", " insects"),
    TestCase("relation_capital",
             "Paris is the capital of France. The capital of France is",
             " Paris", " Berlin",
             "Berlin is the capital of Germany. The capital of Germany is",
             " Berlin", " Paris"),
    TestCase("arithmetic",
             "If x = 7 and y = 3, then x + y =",
             " 10", " 11",
             "If x = 15 and y = 8, then x + y =",
             " 23", " 22"),
    TestCase("syntax_sv",
             "The key to the cabinet", " is", " are",
             "The keys to the cabinet", " are", " is"),
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def extract_last_token_hidden(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    """提取指定层最后一个token的hidden state"""
    layers = discover_layers(model)
    captured = {"value": None}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["value"] = output[0][:, -1, :].detach().cpu()
        else:
            captured["value"] = output[:, -1, :].detach().cpu()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["value"].squeeze(0)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage652_inhibition_mechanism.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage652_inhibition_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage652] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage652] layers={num_layers}")

        # 逐层消融前6层MLP，分析每一层的独立效应
        first_n = min(6, num_layers)
        records = []

        for case in CASES:
            print(f"\n[Stage652] {case.name}")
            baseline = case_margin(model, tokenizer, case)
            print(f"  baseline margin = {baseline:.4f}")

            for layer_idx in range(first_n):
                layer_obj, orig = ablate_layer_component(model, layer_idx, "mlp")
                try:
                    abl_margin = case_margin(model, tokenizer, case)
                finally:
                    restore_layer_component(layer_obj, "mlp", orig)

                drop = baseline - abl_margin
                # 正drop=伤害margin，负drop=提升margin（抑制效应）
                effect = "INHIBIT" if drop < 0 else "HARM"
                print(f"  L{layer_idx}: drop={drop:.4f} ({effect})")

                records.append({
                    "case": case.name,
                    "layer": layer_idx,
                    "component": "mlp",
                    "baseline": round(baseline, 4),
                    "ablated_margin": round(abl_margin, 4),
                    "drop": round(drop, 4),
                    "effect": effect,
                })

        # 分析1：单层抑制效应汇总
        print(f"\n{'='*50}")
        print("Single-layer inhibition analysis:")
        inhibition_layers = [r for r in records if r["effect"] == "INHIBIT"]
        harm_layers = [r for r in records if r["effect"] == "HARM"]
        print(f"  inhibition_count: {len(inhibition_layers)}/{len(records)}")
        print(f"  harm_count: {len(harm_layers)}/{len(records)}")

        if inhibition_layers:
            avg_inhibit = statistics.mean([r["drop"] for r in inhibition_layers])
            print(f"  avg_inhibit_drop: {avg_inhibit:.4f}")
        else:
            avg_inhibit = 0.0
            print("  no inhibition found")

        # 分析2：累积抑制效应（消融前N层一起）
        print(f"\n{'='*50}")
        print("Cumulative inhibition (ablating first N layers together):")
        for n in [1, 2, 3, 5]:
            if n > num_layers:
                break
            layer_range = list(range(n))
            case_margins = []

            for case in CASES:
                baseline = case_margin(model, tokenizer, case)
                saved = []
                for li in layer_range:
                    layer_obj, orig = ablate_layer_component(model, li, "mlp")
                    saved.append((layer_obj, "mlp", orig))
                try:
                    abl_margin = case_margin(model, tokenizer, case)
                finally:
                    for layer_obj, comp, orig in saved:
                        restore_layer_component(layer_obj, comp, orig)
                drop = baseline - abl_margin
                case_margins.append(drop)

            avg_drop = statistics.mean(case_margins)
            effect = "INHIBIT" if avg_drop < 0 else "HARM"
            print(f"  first_{n}: avg_drop={avg_drop:.4f} ({effect})")

        # 分析3：维度级抑制分析（消融前层后hidden state变化的Top维度）
        print(f"\n{'='*50}")
        print("Dimensional inhibition analysis:")
        case = CASES[0]  # 用syllogism case
        h_before = extract_last_token_hidden(model, tokenizer, case.prompt_a, num_layers - 1)

        # 消融L0 MLP
        layer_obj, orig = ablate_layer_component(model, 0, "mlp")
        try:
            h_after = extract_last_token_hidden(model, tokenizer, case.prompt_a, num_layers - 1)
        finally:
            restore_layer_component(layer_obj, "mlp", orig)

        diff = h_after - h_before
        diff_norm = diff.norm().item()
        # 找到变化最大的维度
        topk_vals, topk_idx = torch.topk(diff.abs(), 20)
        print(f"  diff_norm after L0 ablation: {diff_norm:.4f}")
        print(f"  top-5 dim changes: {topk_vals[:5].tolist()}")
        print(f"  top-5 dim indices: {topk_idx[:5].tolist()}")

        # 抑制效应占比：变化方向与baseline方向的夹角
        if h_before.norm() > 1e-8 and diff_norm > 1e-8:
            cos_angle = torch.dot(h_before, diff) / (h_before.norm() * diff_norm)
            print(f"  cos(diff, h_before): {cos_angle:.4f}")

        # INV-229 check
        inv229 = "SURVIVED" if avg_inhibit < -0.1 else ("PARTIAL" if inhibition_layers else "FALSIFIED")
        print(f"\n  INV-229 (Gemma4-unique inhibition): {inv229}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "inhibition_count": len(inhibition_layers),
            "total_count": len(records),
            "avg_inhibit_drop": round(avg_inhibit, 4),
            "inv229_result": inv229,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
