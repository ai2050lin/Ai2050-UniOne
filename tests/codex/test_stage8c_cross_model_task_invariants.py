#!/usr/bin/env python
"""
Score whether the current coding-law candidate preserves usable invariants across
models, tasks, and online risk stages.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8C cross-model task invariants")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8c_cross_model_task_invariants_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    structure = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json"
    )
    layer_band = load_json(
        ROOT / "tests" / "codex_temp" / "generator_network_real_layer_band_bridge_20260310.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    stage8ab = load_json(ROOT / "tests" / "codex_temp" / "stage8ab_adversarial_precision_master_20260311.json")

    model_stats: Dict[str, Dict[str, float]] = {}
    relation_means: Dict[str, Dict[str, float]] = {}
    for model_name, model_row in structure["models"].items():
        compat = []
        gains = []
        concept_scores = []
        grouped: Dict[str, List[float]] = defaultdict(list)
        for task_name, task_row in model_row["tasks"].items():
            compat.append(float(task_row["compatibility"]))
            gains.append(float(task_row["behavior_gain"]))
            concept_scores.append(float(task_row["concept_score"]))
            grouped[str(task_row["relation"])].append(float(task_row["behavior_gain"]))
        relation_means[model_name] = {
            relation: mean(rows) for relation, rows in grouped.items()
        }
        model_stats[model_name] = {
            "mean_behavior_gain": mean(gains),
            "compatibility_gain_corr": pearson(compat, gains),
            "concept_gain_corr": pearson(concept_scores, gains),
            "positive_gain_rate": mean(1.0 if g > 0.0 else 0.0 for g in gains),
        }

    qwen_relation_means = relation_means["qwen3_4b"]
    deepseek_relation_means = relation_means["deepseek_7b"]

    relation_order_invariance = {
        "both_models_tool_worst_stage": 1.0
        if layer_band["headline_metrics"]["qwen_worst_stage"] == "tool"
        and layer_band["headline_metrics"]["deepseek_worst_stage"] == "tool"
        else 0.0,
        "deepseek_searched_undercoverage_gt_qwen": normalize(
            float(layer_band["gains"]["deepseek_minus_qwen_searched_undercoverage"]),
            0.12,
            0.22,
        ),
        "deepseek_joint_success_lt_qwen": normalize(
            float(hard_interface["headline_metrics"]["qwen_joint_head_success"])
            - float(hard_interface["headline_metrics"]["deepseek_joint_head_success"]),
            0.35,
            0.50,
        ),
        "predicted_tool_first_matches_actual": 1.0
        if stage7b["recommended_policy"]["predicted_risk_order"]["qwen3_4b"][0] == "tool"
        and stage7b["recommended_policy"]["predicted_risk_order"]["deepseek_7b"][0] == "tool"
        else 0.0,
    }
    relation_order_invariance_score = mean(relation_order_invariance.values())

    compatibility_invariance = {
        "qwen_compatibility_gain_corr": normalize(model_stats["qwen3_4b"]["compatibility_gain_corr"], 0.15, 0.45),
        "deepseek_compatibility_gain_corr": normalize(
            model_stats["deepseek_7b"]["compatibility_gain_corr"],
            0.10,
            0.45,
        ),
        "qwen_positive_gain_rate": normalize(model_stats["qwen3_4b"]["positive_gain_rate"], 0.70, 0.90),
        "deepseek_positive_gain_rate": normalize(model_stats["deepseek_7b"]["positive_gain_rate"], 0.60, 0.85),
    }
    compatibility_invariance_score = mean(compatibility_invariance.values())

    relation_family_invariance = {
        "qwen_gender_beats_cause_effect": 1.0
        if qwen_relation_means.get("gender", 0.0) > qwen_relation_means.get("cause_effect", 0.0)
        else 0.0,
        "deepseek_gender_beats_cause_effect": 1.0
        if deepseek_relation_means.get("gender", 0.0) > deepseek_relation_means.get("cause_effect", 0.0)
        else 0.0,
        "qwen_hypernym_positive": 1.0 if qwen_relation_means.get("hypernym", 0.0) > 0.0 else 0.0,
        "deepseek_hypernym_positive": 1.0 if deepseek_relation_means.get("hypernym", 0.0) > 0.0 else 0.0,
    }
    relation_family_invariance_score = mean(relation_family_invariance.values())

    model_gap_structure = {
        "qwen_mean_gain": normalize(model_stats["qwen3_4b"]["mean_behavior_gain"], 0.03, 0.05),
        "deepseek_mean_gain": normalize(model_stats["deepseek_7b"]["mean_behavior_gain"], 0.015, 0.04),
        "deepseek_stays_harder_online": normalize(
            1.0 - float(hard_interface["headline_metrics"]["deepseek_joint_head_success"]),
            0.55,
            0.65,
        ),
        "stage8_counterexample_alignment": float(
            stage8ab["stage8_blocks"]["stage8a_adversarial_counterexample_search"]["score"]
        ),
    }
    model_gap_structure_score = mean(model_gap_structure.values())

    overall_score = mean(
        [
            relation_order_invariance_score,
            compatibility_invariance_score,
            relation_family_invariance_score,
            model_gap_structure_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8c_cross_model_task_invariants",
        },
        "model_stats": model_stats,
        "relation_means": relation_means,
        "pillars": {
            "relation_order_invariance": {
                "components": relation_order_invariance,
                "score": float(relation_order_invariance_score),
            },
            "compatibility_invariance": {
                "components": compatibility_invariance,
                "score": float(compatibility_invariance_score),
            },
            "relation_family_invariance": {
                "components": relation_family_invariance,
                "score": float(relation_family_invariance_score),
            },
            "model_gap_structure": {
                "components": model_gap_structure,
                "score": float(model_gap_structure_score),
            },
        },
        "headline_metrics": {
            "relation_order_invariance_score": float(relation_order_invariance_score),
            "compatibility_invariance_score": float(compatibility_invariance_score),
            "relation_family_invariance_score": float(relation_family_invariance_score),
            "model_gap_structure_score": float(model_gap_structure_score),
            "overall_stage8c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_tool_relation_risk_order_is_partly_invariant": bool(relation_order_invariance_score >= 0.76),
            "H2_compatibility_remains_a_cross_model_gain_driver": bool(compatibility_invariance_score >= 0.68),
            "H3_relation_family_structure_has_nontrivial_shared_order": bool(
                relation_family_invariance_score >= 0.75
            ),
            "H4_model_gap_itself_is_structured_not_random": bool(model_gap_structure_score >= 0.72),
            "H5_stage8c_cross_model_task_invariants_are_moderately_supported": bool(overall_score >= 0.73),
        },
        "project_readout": {
            "summary": (
                "Stage 8C is positive only if the current candidate law preserves some stable structure across models "
                "and tasks, even while absolute performance differs."
            ),
            "next_question": (
                "If this stage holds, the next step is to push the brain-side falsifiers harder rather than collecting "
                "more generic bridge metrics."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
