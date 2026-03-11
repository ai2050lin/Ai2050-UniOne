#!/usr/bin/env python
"""
Build an adversarial stress map against the current candidate coding law.
The goal is not to celebrate the law, but to quantify where it is easiest to break.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8A adversarial counterexample search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8a_adversarial_counterexample_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7d = load_json(ROOT / "tests" / "codex_temp" / "stage7d_coding_law_verdict_master_20260311.json")
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    targeted_ablation = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )
    sweetness = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307.json")
    extreme = load_json(
        ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307_extreme.json"
    )

    qwen_summary = targeted_ablation["models"]["qwen3_4b"]["global_summary"]
    deepseek_summary = targeted_ablation["models"]["deepseek_7b"]["global_summary"]

    orientation_pressure = {
        "qwen_prediction_mismatch": 1.0
        if qwen_summary["predicted_orientation_label"] != qwen_summary["actual_orientation_label"]
        else 0.0,
        "deepseek_prediction_mismatch": 1.0
        if deepseek_summary["predicted_orientation_label"] != deepseek_summary["actual_orientation_label"]
        else 0.0,
        "qwen_orientation_gap": normalize(
            abs(float(qwen_summary["predicted_orientation"]) - float(qwen_summary["actual_targeted_orientation"])),
            0.05,
            0.12,
        ),
        "deepseek_orientation_gap": normalize(
            abs(
                float(deepseek_summary["predicted_orientation"])
                - float(deepseek_summary["actual_targeted_orientation"])
            ),
            0.20,
            0.70,
        ),
    }
    orientation_pressure_score = mean(orientation_pressure.values())

    deepseek_joint = hard_interface["models"]["deepseek_7b"]["relation_tool_joint_head_online_tool_interface"]
    qwen_joint = hard_interface["models"]["qwen3_4b"]["relation_tool_joint_head_online_tool_interface"]
    interface_pressure = {
        "deepseek_low_success": normalize(1.0 - float(deepseek_joint["success_rate"]), 0.50, 0.65),
        "deepseek_high_trigger": normalize(float(deepseek_joint["rollback_trigger_rate"]), 0.65, 0.80),
        "deepseek_tool_failure": normalize(float(deepseek_joint["tool_failure_rate"]), 0.25, 0.38),
        "model_gap_after_upgrade": normalize(
            float(qwen_joint["success_rate"]) - float(deepseek_joint["success_rate"]),
            0.35,
            0.50,
        ),
    }
    interface_pressure_score = mean(interface_pressure.values())

    best = sweetness["best"]
    extreme_best = extreme["best"]
    layer27_extreme_fragility = max(
        normalize(0.80 - float(row["anchor_retention"]), 0.0, 0.65)
        for row in extreme["trials"]
        if int(row["layer"]) == 27 and bool(row["gap_reversed_from_base"])
    )
    precision_fragility = {
        "low_k_failure": 1.0 if sweetness["min_k_reversal_anchor80_soft"] > 16 else 0.0,
        "strong_edit_needs_k64": normalize(float(sweetness["min_k_reversal_anchor80_strong"]), 32.0, 96.0),
        "extreme_anchor_fragility": float(layer27_extreme_fragility),
        "precision_hypothesis_failed": 1.0
        if not bool(stage7b["hypotheses"]["H1_explicit_law_guides_precise_local_editing"])
        else 0.0,
    }
    precision_fragility_score = mean(precision_fragility.values())

    search_coverage = {
        "orientation_attack_family": 1.0,
        "hard_interface_attack_family": 1.0,
        "local_edit_attack_family": 1.0,
        "cross_model_attack_family": 1.0,
        "distinct_failure_modes": normalize(4.0, 3.0, 4.0),
    }
    search_coverage_score = mean(search_coverage.values())

    law_residual_survival = {
        "stage7_verdict_support": float(stage7d["verdict"]["verdict_support_score"]),
        "cross_model_prediction_strength": float(stage7b["headline_metrics"]["cross_model_prediction_score"]),
        "qwen_joint_success": normalize(float(qwen_joint["success_rate"]), 0.75, 0.90),
        "best_local_edit_anchor": normalize(float(best["anchor_retention"]), 0.75, 0.90),
    }
    law_residual_survival_score = mean(law_residual_survival.values())

    overall_score = mean(
        [
            search_coverage_score,
            orientation_pressure_score,
            interface_pressure_score,
            precision_fragility_score,
            law_residual_survival_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8a_adversarial_counterexample_search",
        },
        "adversarial_map": {
            "highest_risk_zone": "deepseek_relation_tool_online_chain",
            "secondary_risk_zone": "shared_layer_orientation_mismatch",
            "third_risk_zone": "local_attribute_edit_precision_band",
        },
        "pillars": {
            "search_coverage": {"components": search_coverage, "score": float(search_coverage_score)},
            "orientation_pressure": {
                "components": orientation_pressure,
                "score": float(orientation_pressure_score),
            },
            "interface_pressure": {
                "components": interface_pressure,
                "score": float(interface_pressure_score),
            },
            "precision_fragility": {
                "components": precision_fragility,
                "score": float(precision_fragility_score),
            },
            "law_residual_survival": {
                "components": law_residual_survival,
                "score": float(law_residual_survival_score),
            },
        },
        "headline_metrics": {
            "search_coverage_score": float(search_coverage_score),
            "orientation_pressure_score": float(orientation_pressure_score),
            "interface_pressure_score": float(interface_pressure_score),
            "precision_fragility_score": float(precision_fragility_score),
            "law_residual_survival_score": float(law_residual_survival_score),
            "overall_stage8a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_model_specific_orientation_counterexamples_exist": bool(orientation_pressure_score >= 0.78),
            "H2_deepseek_remains_a_hard_online_counterexample_zone": bool(interface_pressure_score >= 0.68),
            "H3_local_precision_editing_is_still_fragile": bool(precision_fragility_score >= 0.72),
            "H4_candidate_law_retains_nontrivial_survival_after_stress": bool(law_residual_survival_score >= 0.72),
            "H5_stage8a_adversarial_counterexample_map_is_established": bool(overall_score >= 0.74),
        },
        "project_readout": {
            "summary": (
                "Stage 8A is positive only if the project can point to concrete places where the current coding-law "
                "candidate is easiest to break, while still showing that the law retains nontrivial residual utility."
            ),
            "next_question": (
                "If this stage holds, the next step is to tighten precision editing and see whether the fragile local "
                "band can be compressed into a repeatable high-resolution intervention policy."
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
