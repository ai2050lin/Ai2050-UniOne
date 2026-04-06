#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage650: 解码层消融敏感度——反向验证解码层的关键性

目标：反向验证P3的"传播-解码二分"假说
方法：
1. 正常（不消融）状态下计算baseline margin
2. 消融最后1层/2层/5层/10层，看margin下降多少
3. 与消融前半层的下降量对比
4. 如果消融最后几层的伤害远大于消融前面几层，说明解码层确实承载了因果效力

预注册判伪条件：
INV-218: "消融最后5层的伤害 > 消融前5层的伤害"
如果四模型中消融最后5层的平均伤害不超过消融前5层，则INV-218被推翻。
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


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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
    TestCase("syntax_sv",
             "The key to the cabinet", " is", " are",
             "The keys to the cabinet", " are", " is"),
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
]


def case_margin(model, tokenizer, case: TestCase) -> float:
    ma = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.negative_a)
    mb = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - \
         score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.negative_b)
    return float((ma + mb) / 2.0)


def multi_layer_ablate_test(model, tokenizer, case: TestCase,
                             layer_ranges: Dict[str, List[int]],
                             component: str = "mlp") -> Dict[str, float]:
    """消融指定层范围，返回每个范围的margin下降量"""
    baseline = case_margin(model, tokenizer, case)
    results = {}

    for range_name, layer_list in layer_ranges.items():
        saved = []
        for li in layer_list:
            layer_obj, orig = ablate_layer_component(model, li, component)
            saved.append((layer_obj, component, orig))
        try:
            abl_margin = case_margin(model, tokenizer, case)
        finally:
            for layer_obj, comp, orig in saved:
                restore_layer_component(layer_obj, comp, orig)
        results[range_name] = {
            "margin": abl_margin,
            "drop": round(baseline - abl_margin, 4),
            "baseline": round(baseline, 4),
        }

    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stage650_decode_ablation_sensitivity.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)
    model_key = sys.argv[1]
    run_dir = OUTPUT_DIR / f"stage650_decode_ablation_{model_key}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        print(f"[Stage650] Loading {model_key}...")
        model, tokenizer = load_model_bundle(model_key, prefer_cuda=True)
        num_layers = len(discover_layers(model))
        print(f"[Stage650] layers={num_layers}")

        records = []
        for case in CASES:
            print(f"\n[Stage650] {case.name}")

            # 定义层范围
            last5 = list(range(num_layers - 5, num_layers))
            first5 = list(range(0, min(5, num_layers)))
            last1 = [num_layers - 1]
            first1 = [0]
            mid5_start = max(0, num_layers // 2 - 2)
            mid5 = list(range(mid5_start, mid5_start + 5))

            layer_ranges = {
                "last1": last1,
                "first1": first1,
                "last5": last5,
                "first5": first5,
                "mid5": mid5,
            }

            # 分别测试attn和mlp
            for comp in ("mlp", "attn"):
                try:
                    results = multi_layer_ablate_test(model, tokenizer, case, layer_ranges, component=comp)
                    # 计算比值
                    last5_drop = results["last5"]["drop"]
                    first5_drop = results["first5"]["drop"]
                    mid5_drop = results["mid5"]["drop"]
                    ratio_last_vs_first = last5_drop / max(first5_drop, 1e-8)

                    print(f"  {comp}: last5_drop={last5_drop:.4f}, first5_drop={first5_drop:.4f}, "
                          f"mid5_drop={mid5_drop:.4f}, ratio={ratio_last_vs_first:.2f}")

                    records.append({
                        "case_name": case.name,
                        "component": comp,
                        "ranges": results,
                        "last5_vs_first5_ratio": round(ratio_last_vs_first, 2),
                    })
                except Exception as e:
                    print(f"  {comp}: error={e}")

        # Summary per component
        for comp in ("mlp", "attn"):
            subset = [r for r in records if r["component"] == comp]
            avg_ratio = statistics.mean([r["last5_vs_first5_ratio"] for r in subset]) if subset else 0
            avg_last5 = statistics.mean([r["ranges"]["last5"]["drop"] for r in subset]) if subset else 0
            avg_first5 = statistics.mean([r["ranges"]["first5"]["drop"] for r in subset]) if subset else 0
            print(f"\n  {comp} avg: last5_drop={avg_last5:.4f}, first5_drop={avg_first5:.4f}, ratio={avg_ratio:.2f}")

        # INV-218 check
        mlp_records = [r for r in records if r["component"] == "mlp"]
        avg_mlp_ratio = statistics.mean([r["last5_vs_first5_ratio"] for r in mlp_records]) if mlp_records else 0
        inv218 = "SURVIVED" if avg_mlp_ratio > 1.0 else "FALSIFIED"
        print(f"\n  INV-218 (last5>first5): {inv218} (mlp ratio={avg_mlp_ratio:.2f})")

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "num_layers": num_layers,
            "avg_mlp_last5_vs_first5": round(avg_mlp_ratio, 2),
            "inv218_result": inv218,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {run_dir / 'summary.json'}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
