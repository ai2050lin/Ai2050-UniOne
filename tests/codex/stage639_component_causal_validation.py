#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage639: 组件级因果验证（syntax / relation / coref）

目标：
1. 在三类能力上验证当前变量体系是否具有组件级因果支持。
2. 对少量代表层分别零化 attention（注意力）和 MLP（多层感知机）。
3. 比较：
   - 哪类组件对 margin（正确候选与错误候选的平均对数概率差）影响更大
   - 哪些层是关键层
   - 不同模型是否复现相同模式

用法：
python tests/codex/stage639_component_causal_validation.py --model qwen3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from multimodel_language_shared import (
    ablate_layer_component,
    evenly_spaced_layers,
    free_model,
    load_model_bundle,
    restore_layer_component,
    score_candidate_avg_logprob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class CapabilityCase:
    capability: str
    pair_id: str
    prompt_a: str
    positive_a: str
    negative_a: str
    prompt_b: str
    positive_b: str
    negative_b: str


CASES: List[CapabilityCase] = [
    CapabilityCase(
        capability="syntax",
        pair_id="subject_verb_number",
        prompt_a="The key to the cabinet",
        positive_a=" is",
        negative_a=" are",
        prompt_b="The keys to the cabinet",
        positive_b=" are",
        negative_b=" is",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="distance_agreement",
        prompt_a="The bouquet of roses",
        positive_a=" smells",
        negative_a=" smell",
        prompt_b="The roses in the bouquet",
        positive_b=" smell",
        negative_b=" smells",
    ),
    CapabilityCase(
        capability="syntax",
        pair_id="collective_local_noun",
        prompt_a="The picture near the windows",
        positive_a=" was",
        negative_a=" were",
        prompt_b="The pictures near the window",
        positive_b=" were",
        negative_b=" was",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="capital_relation",
        prompt_a="Paris is the capital of France. The capital of France is",
        positive_a=" Paris",
        negative_a=" Berlin",
        prompt_b="Berlin is the capital of Germany. The capital of Germany is",
        positive_b=" Berlin",
        negative_b=" Paris",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="currency_relation",
        prompt_a="The currency used in Japan is",
        positive_a=" yen",
        negative_a=" euro",
        prompt_b="The currency used in Britain is",
        positive_b=" pound",
        negative_b=" yen",
    ),
    CapabilityCase(
        capability="relation",
        pair_id="inventor_relation",
        prompt_a="The telephone was invented by",
        positive_a=" Bell",
        negative_a=" Edison",
        prompt_b="The light bulb is associated with",
        positive_b=" Edison",
        negative_b=" Bell",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="winner_reference",
        prompt_a="Alice thanked Mary because Alice had won the prize. The person who won was",
        positive_a=" Alice",
        negative_a=" Mary",
        prompt_b="Alice thanked Mary because Mary had won the prize. The person who won was",
        positive_b=" Mary",
        negative_b=" Alice",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="help_reference",
        prompt_a="Emma called Sara because Emma needed advice. The one needing advice was",
        positive_a=" Emma",
        negative_a=" Sara",
        prompt_b="Emma called Sara because Sara needed advice. The one needing advice was",
        positive_b=" Sara",
        negative_b=" Emma",
    ),
    CapabilityCase(
        capability="coref",
        pair_id="blame_reference",
        prompt_a="Olivia blamed Mia because Olivia was careless. The careless person was",
        positive_a=" Olivia",
        negative_a=" Mia",
        prompt_b="Olivia blamed Mia because Mia was careless. The careless person was",
        positive_b=" Mia",
        negative_b=" Olivia",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage639 组件级因果验证")
    parser.add_argument("--model", required=True, choices=["qwen3", "deepseek7b", "gemma4", "glm4"])
    parser.add_argument("--layer-count", type=int, default=5, help="抽样层数")
    return parser.parse_args()


def case_margin(model, tokenizer, case: CapabilityCase) -> Dict[str, float | bool]:
    margin_a = score_candidate_avg_logprob(model, tokenizer, case.prompt_a, case.positive_a) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_a, case.negative_a
    )
    margin_b = score_candidate_avg_logprob(model, tokenizer, case.prompt_b, case.positive_b) - score_candidate_avg_logprob(
        model, tokenizer, case.prompt_b, case.negative_b
    )
    avg_margin = float((margin_a + margin_b) / 2.0)
    return {
        "margin_a": float(margin_a),
        "margin_b": float(margin_b),
        "avg_margin": avg_margin,
        "pair_correct": bool(margin_a > 0.0 and margin_b > 0.0),
    }


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def aggregate_capability(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_cap: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        by_cap.setdefault(record["capability"], []).append(record)

    summary: Dict[str, object] = {}
    for capability, items in by_cap.items():
        summary[capability] = {
            "count": len(items),
            "baseline_pair_accuracy": mean_or_zero([1.0 if item["baseline_pair_correct"] else 0.0 for item in items]),
            "baseline_margin": mean_or_zero([item["baseline_margin"] for item in items]),
            "attn_margin_drop": mean_or_zero([item["attn_margin_drop"] for item in items]),
            "mlp_margin_drop": mean_or_zero([item["mlp_margin_drop"] for item in items]),
            "attn_pair_accuracy_drop": mean_or_zero([item["attn_pair_accuracy_drop"] for item in items]),
            "mlp_pair_accuracy_drop": mean_or_zero([item["mlp_pair_accuracy_drop"] for item in items]),
        }
    return summary


def build_report(args: argparse.Namespace, layer_indices: Sequence[int], summary: Dict[str, object], records: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Stage639 组件级因果验证报告",
        "",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型: {args.model}",
        f"- 抽样层: {list(layer_indices)}",
        f"- 样本数: {len(records)}",
        "",
        "## 分能力摘要",
    ]
    for capability, item in summary.items():
        lines.extend(
            [
                f"### {capability}",
                f"- baseline_pair_accuracy: {item['baseline_pair_accuracy']:.4f}",
                f"- baseline_margin: {item['baseline_margin']:.4f}",
                f"- attn_margin_drop: {item['attn_margin_drop']:.4f}",
                f"- mlp_margin_drop: {item['mlp_margin_drop']:.4f}",
                f"- attn_pair_accuracy_drop: {item['attn_pair_accuracy_drop']:.4f}",
                f"- mlp_pair_accuracy_drop: {item['mlp_pair_accuracy_drop']:.4f}",
                "",
            ]
        )
    lines.append("## 单样本摘要")
    for record in records:
        lines.extend(
            [
                f"### {record['capability']} / {record['pair_id']}",
                f"- baseline_margin: {record['baseline_margin']:.4f}",
                f"- best_attn_layer: {record['best_attn_layer']}",
                f"- best_attn_margin_drop: {record['best_attn_margin_drop']:.4f}",
                f"- best_mlp_layer: {record['best_mlp_layer']}",
                f"- best_mlp_margin_drop: {record['best_mlp_margin_drop']:.4f}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT_ROOT / f"stage639_component_causal_validation_{args.model}_{TIMESTAMP}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = None
    try:
        model, tokenizer = load_model_bundle(args.model, prefer_cuda=True)
        layer_indices = evenly_spaced_layers(model, count=args.layer_count)
        records: List[Dict[str, object]] = []

        for case in CASES:
            baseline = case_margin(model, tokenizer, case)
            attn_results: List[Dict[str, object]] = []
            mlp_results: List[Dict[str, object]] = []

            for layer_idx in layer_indices:
                for component, store in (("attn", attn_results), ("mlp", mlp_results)):
                    layer, original = ablate_layer_component(model, layer_idx, component)
                    try:
                        ablated = case_margin(model, tokenizer, case)
                    finally:
                        restore_layer_component(layer, component, original)
                    store.append(
                        {
                            "layer": layer_idx,
                            "avg_margin": ablated["avg_margin"],
                            "pair_correct": ablated["pair_correct"],
                            "margin_drop": float(baseline["avg_margin"] - ablated["avg_margin"]),
                            "pair_accuracy_drop": float((1.0 if baseline["pair_correct"] else 0.0) - (1.0 if ablated["pair_correct"] else 0.0)),
                        }
                    )

            best_attn = max(attn_results, key=lambda item: item["margin_drop"])
            best_mlp = max(mlp_results, key=lambda item: item["margin_drop"])

            records.append(
                {
                    "capability": case.capability,
                    "pair_id": case.pair_id,
                    "baseline_margin": baseline["avg_margin"],
                    "baseline_pair_correct": baseline["pair_correct"],
                    "best_attn_layer": best_attn["layer"],
                    "best_attn_margin_drop": best_attn["margin_drop"],
                    "best_attn_pair_accuracy_drop": best_attn["pair_accuracy_drop"],
                    "best_mlp_layer": best_mlp["layer"],
                    "best_mlp_margin_drop": best_mlp["margin_drop"],
                    "best_mlp_pair_accuracy_drop": best_mlp["pair_accuracy_drop"],
                    "attn_margin_drop": mean_or_zero([item["margin_drop"] for item in attn_results]),
                    "mlp_margin_drop": mean_or_zero([item["margin_drop"] for item in mlp_results]),
                    "attn_pair_accuracy_drop": mean_or_zero([item["pair_accuracy_drop"] for item in attn_results]),
                    "mlp_pair_accuracy_drop": mean_or_zero([item["pair_accuracy_drop"] for item in mlp_results]),
                    "attn_results": attn_results,
                    "mlp_results": mlp_results,
                }
            )

        capability_summary = aggregate_capability(records)
        overall = {
            "model": args.model,
            "layer_indices": layer_indices,
            "case_count": len(records),
            "capability_summary": capability_summary,
        }
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall": overall,
            "records": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report = build_report(args, layer_indices, capability_summary, records)
        (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
        print(json.dumps(overall, ensure_ascii=False, indent=2))
        print(f"结果已写入: {run_dir}")
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    main()
