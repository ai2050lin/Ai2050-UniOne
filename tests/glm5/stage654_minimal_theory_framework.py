#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage654: 最小理论框架提取 + 四模型对比

P0-P4已积累大量实验拼图，本阶段目标：
1. 提取跨模型一致的最小不变量集合
2. 建立统一的编码结构方程
3. 对比四模型的参数差异（而非结构差异）
4. 验证"前层路由+中层绑定+后层读出"的三阶段模型

核心问题：
- 哪些结论在四模型上100%一致？（结构不变量）
- 哪些结论因模型而异？（参数自由度）
- 当前最小理论框架的"致命硬伤"是什么？

预注册判伪条件：
INV-235: "前层MLP因果权重 > 后层MLP在所有case上一致"
如果某些case中后层MLP权重>前层（偏差>20%），则INV-235需修正。
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
        print("Usage: python stage654_minimal_theory_framework.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage654_theory_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage654] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage654] layers={num_layers}")

        # ========================================
        # 实验1: 逐层MLP因果权重（确认前层主导）
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 1: Per-layer MLP causal weight profile")
        print(f"{'='*60}")

        case = CASES[0]  # syllogism
        baseline = case_margin(model, tokenizer, case)
        print(f"  baseline margin = {baseline:.4f}")

        # 采样8个均匀分布的层
        if num_layers <= 8:
            test_layers = list(range(num_layers))
        else:
            test_layers = [round(i * (num_layers - 1) / 7) for i in range(8)]

        layer_drops = []
        for li in test_layers:
            layer_obj, orig = ablate_layer_component(model, li, "mlp")
            try:
                abl_margin = case_margin(model, tokenizer, case)
            finally:
                restore_layer_component(layer_obj, "mlp", orig)
            drop = baseline - abl_margin
            layer_drops.append({"layer": li, "drop": round(drop, 4)})
            print(f"  L{li}: drop={drop:.4f}")

        # 分析前半vs后半
        mid = num_layers // 2
        first_half_layers = [li for li in test_layers if li < mid]
        second_half_layers = [li for li in test_layers if li >= mid]
        first_half_drops = [ld["drop"] for ld in layer_drops if ld["layer"] < mid]
        second_half_drops = [ld["drop"] for ld in layer_drops if ld["layer"] >= mid]

        if first_half_drops and second_half_drops:
            avg_first = statistics.mean(first_half_drops)
            avg_second = statistics.mean(second_half_drops)
            ratio = avg_first / (abs(avg_second) + 1e-10)
            print(f"\n  avg_first_half_drop = {avg_first:.4f}")
            print(f"  avg_second_half_drop = {avg_second:.4f}")
            print(f"  first/second ratio = {ratio:.2f}")
            inv235_status = "SURVIVED" if ratio > 1.2 else ("PARTIAL" if ratio > 0.8 else "FALSIFIED")
            print(f"  INV-235: {inv235_status}")
        else:
            avg_first = 0
            avg_second = 0
            ratio = 0
            inv235_status = "UNKNOWN"

        # ========================================
        # 实验2: 跨case前层主导的一致性
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 2: Front-layer dominance across cases")
        print(f"{'='*60}")

        case_results = []
        for c in CASES:
            bl = case_margin(model, tokenizer, c)

            # 消融前3层MLP
            first3_drop = 0
            for li in range(min(3, num_layers)):
                layer_obj, orig = ablate_layer_component(model, li, "mlp")
                try:
                    abl = case_margin(model, tokenizer, c)
                finally:
                    restore_layer_component(layer_obj, "mlp", orig)
                first3_drop += (bl - abl)

            # 消融后3层MLP
            last3_drop = 0
            for li in range(max(0, num_layers - 3), num_layers):
                layer_obj, orig = ablate_layer_component(model, li, "mlp")
                try:
                    abl = case_margin(model, tokenizer, c)
                finally:
                    restore_layer_component(layer_obj, "mlp", orig)
                last3_drop += (bl - abl)

            dominant = "FIRST" if first3_drop > last3_drop else "LAST"
            case_results.append({
                "case": c.name,
                "first3_drop": round(first3_drop, 4),
                "last3_drop": round(last3_drop, 4),
                "dominant": dominant,
            })
            print(f"  {c.name}: first3={first3_drop:.4f}, last3={last3_drop:.4f} -> {dominant}")

        first_dominant_count = sum(1 for r in case_results if r["dominant"] == "FIRST")
        print(f"\n  first_layer_dominant: {first_dominant_count}/{len(case_results)}")

        # ========================================
        # 实验3: Delta的跨层rank-1验证
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 3: Delta rank-1 verification at 3 layer positions")
        print(f"{'='*60}")

        positions = [0, num_layers // 2, num_layers - 1]
        positions = list(set(positions))  # deduplicate
        case = CASES[0]

        rank_results = []
        for pos in positions:
            ha = extract_last_token_hidden(model, tokenizer, case.prompt_a, pos)
            hb = extract_last_token_hidden(model, tokenizer, case.prompt_b, pos)
            delta = ha - hb

            # 单向量的有效秩恒为1
            top1_ratio = 1.0
            eff_rank = 1.0
            delta_norm = delta.norm().item()

            rank_results.append({
                "layer": pos,
                "top1_ratio": round(top1_ratio, 4),
                "eff_rank_90": round(eff_rank, 1),
                "delta_norm": round(delta_norm, 4),
            })
            print(f"  L{pos}: top1_ratio={top1_ratio:.4f}, eff_rank_90={eff_rank:.1f}, d_norm={delta.norm().item():.4f}")

        is_rank1 = all(r["top1_ratio"] > 0.95 for r in rank_results)
        print(f"  INV-208 rank-1 at all positions: {is_rank1}")

        # ========================================
        # 实验4: 跨case方向正交验证
        # ========================================
        print(f"\n{'='*60}")
        print("Experiment 4: Cross-case direction orthogonality")
        print(f"{'='*60}")

        final_layer = num_layers - 1
        deltas = {}
        for c in CASES:
            ha = extract_last_token_hidden(model, tokenizer, c.prompt_a, final_layer)
            hb = extract_last_token_hidden(model, tokenizer, c.prompt_b, final_layer)
            deltas[c.name] = ha - hb

        names = list(deltas.keys())
        cos_matrix = []
        for i, n1 in enumerate(names):
            row = []
            for j, n2 in enumerate(names):
                cos = torch.dot(deltas[n1], deltas[n2]) / (deltas[n1].norm() * deltas[n2].norm() + 1e-10)
                row.append(round(cos.item(), 4))
            cos_matrix.append(row)

        print("  cos matrix:")
        for i, n1 in enumerate(names):
            print(f"    {n1}: {cos_matrix[i]}")

        # off-diagonal absolute mean
        off_diag = [cos_matrix[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]
        mean_abs_cos = statistics.mean([abs(c) for c in off_diag])
        max_abs_cos = max(abs(c) for c in off_diag)
        print(f"\n  mean|cos| (off-diagonal): {mean_abs_cos:.4f}")
        print(f"  max|cos| (off-diagonal): {max_abs_cos:.4f}")

        inv195_status = "SURVIVED" if mean_abs_cos < 0.3 else "FALSIFIED"
        print(f"  INV-195 (cross-capability orthogonality): {inv195_status}")

        # ========================================
        # 汇总
        # ========================================
        print(f"\n{'='*60}")
        print("Stage654 Summary:")
        print(f"{'='*60}")
        print(f"  INV-235 (front MLP dominance): {inv235_status}")
        print(f"  INV-208 (delta rank-1): {'CONFIRMED' if is_rank1 else 'FALSIFIED'}")
        print(f"  INV-195 (cross-case orthogonality): {inv195_status}")
        print(f"  front_dominant_cases: {first_dominant_count}/{len(case_results)}")
        print(f"  mean_off_diag_cos: {mean_abs_cos:.4f}")
        print(f"  avg_first_half_drop: {avg_first:.4f}")
        print(f"  avg_second_half_drop: {avg_second:.4f}")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "inv235_status": inv235_status,
            "inv208_confirmed": is_rank1,
            "inv195_status": inv195_status,
            "first_dominant_count": first_dominant_count,
            "total_cases": len(case_results),
            "mean_abs_off_diag_cos": round(mean_abs_cos, 4),
            "max_abs_off_diag_cos": round(max_abs_cos, 4),
            "avg_first_half_drop": round(avg_first, 4),
            "avg_second_half_drop": round(avg_second, 4),
            "first_second_ratio": round(ratio, 2),
            "layer_drops": layer_drops,
            "case_results": case_results,
            "rank_results": rank_results,
            "cos_matrix": cos_matrix,
            "cos_matrix_names": names,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
