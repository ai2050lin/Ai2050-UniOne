#!/usr/bin/env python
"""
Score whether current edit results can be compressed into a narrower and more
repeatable precision-tuning policy.
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


def inv_score(value: float, lo: float, hi: float) -> float:
    return float(1.0 - normalize(value, lo, hi))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8B high-resolution precision editing")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8b_high_resolution_precision_editing_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    sweetness = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307.json")
    quick = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307_quick.json")
    extreme = load_json(
        ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307_extreme.json"
    )
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    joint_upgrade = load_json(
        ROOT / "tests" / "codex_temp" / "relation_tool_joint_generator_network_upgrade_20260310.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )

    best = sweetness["best"]
    full_trials = sweetness["trials"]
    extreme_trials = extreme["trials"]
    viable_band = [
        row
        for row in full_trials
        if int(row["layer"]) == 27
        and bool(row["gap_reversed_from_base"])
        and float(row["anchor_retention"]) >= 0.80
    ]
    extreme_viable = [
        row
        for row in extreme_trials
        if int(row["layer"]) == 27
        and bool(row["gap_reversed_from_base"])
        and float(row["anchor_retention"]) >= 0.80
    ]

    localization = {
        "best_layer_is_top_candidate": 1.0
        if int(best["layer"]) == int(sweetness["layer_candidates"][1]["layer"])
        else 0.0,
        "reversal_band_count": inv_score(float(len(viable_band)), 1.0, 6.0),
        "extreme_viable_band_count": inv_score(float(len(extreme_viable)), 1.0, 4.0),
        "soft_min_k": inv_score(float(sweetness["min_k_reversal_anchor80_soft"]), 16.0, 64.0),
    }
    localization_score = mean(localization.values())

    repeatability = {
        "full_best_reverses": 1.0 if bool(best["gap_reversed_from_base"]) else 0.0,
        "full_anchor_retention": normalize(float(best["anchor_retention"]), 0.75, 0.90),
        "extreme_best_reverses": 1.0 if bool(extreme["best"]["gap_reversed_from_base"]) else 0.0,
        "extreme_best_anchor": normalize(float(extreme["best"]["anchor_retention"]), 0.75, 0.90),
        "best_scale_window": 1.0 if len(extreme_viable) >= 2 else 0.0,
    }
    repeatability_score = mean(repeatability.values())

    routing_precision = {
        "stage7b_routing_tuning": float(stage7b["headline_metrics"]["routing_tuning_score"]),
        "qwen_joint_success_gain": normalize(float(hard_interface["gains"]["qwen_joint_minus_tool_head_success"]), 0.05, 0.10),
        "deepseek_joint_success_gain": normalize(
            float(hard_interface["gains"]["deepseek_joint_minus_tool_head_success"]),
            0.03,
            0.07,
        ),
        "best_joint_trial_score": normalize(
            float(joint_upgrade["search_summary"]["top_trials"][0]["score"]),
            0.35,
            0.45,
        ),
    }
    routing_precision_score = mean(routing_precision.values())

    residual_risk = {
        "precision_hypothesis_gap": 1.0
        if not bool(stage7b["hypotheses"]["H1_explicit_law_guides_precise_local_editing"])
        else 0.0,
        "deepseek_joint_failure": normalize(
            1.0 - float(hard_interface["headline_metrics"]["deepseek_joint_head_success"]),
            0.55,
            0.65,
        ),
        "strong_min_k_penalty": normalize(float(sweetness["min_k_reversal_anchor80_strong"]), 32.0, 96.0),
        "extreme_k128_anchor_drop": normalize(
            0.80
            - max(
                float(row["anchor_retention"])
                for row in extreme_trials
                if int(row["layer"]) == 27 and int(row["k"]) == 128 and bool(row["gap_reversed_from_base"])
            ),
            0.0,
            0.70,
        ),
    }
    residual_risk_score = mean(residual_risk.values())

    overall_score = mean(
        [
            localization_score,
            repeatability_score,
            routing_precision_score,
            residual_risk_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8b_high_resolution_precision_editing",
        },
        "precision_policy": {
            "local_attribute_edit": {
                "target_layer": int(best["layer"]),
                "target_k": int(best["k"]),
                "target_scale": float(best["scale"]),
                "stable_alternative_scales": sorted(
                    {
                        float(row["scale"])
                        for row in extreme_viable
                    }
                ),
            },
            "routing_head_upgrade": joint_upgrade["generator_profiles"]["relation_tool_joint_head_generator_network"][
                "upgrade_spec"
            ],
        },
        "pillars": {
            "localization": {"components": localization, "score": float(localization_score)},
            "repeatability": {"components": repeatability, "score": float(repeatability_score)},
            "routing_precision": {"components": routing_precision, "score": float(routing_precision_score)},
            "residual_risk": {"components": residual_risk, "score": float(residual_risk_score)},
        },
        "headline_metrics": {
            "localization_score": float(localization_score),
            "repeatability_score": float(repeatability_score),
            "routing_precision_score": float(routing_precision_score),
            "residual_risk_score": float(residual_risk_score),
            "overall_stage8b_score": float(overall_score),
        },
        "hypotheses": {
            "H1_local_editing_now_localizes_to_a_narrow_band": bool(localization_score >= 0.72),
            "H2_local_reversal_is_repeatable_inside_that_band": bool(repeatability_score >= 0.78),
            "H3_joint_routing_upgrade_remains_the_best_precision_policy": bool(routing_precision_score >= 0.74),
            "H4_precision_editing_still_has_nontrivial_residual_risk": bool(residual_risk_score >= 0.70),
            "H5_stage8b_high_resolution_precision_editing_is_moderately_closed": bool(overall_score >= 0.74),
        },
        "project_readout": {
            "summary": (
                "Stage 8B is positive only if the current best edit policy stops looking like a coarse one-off and "
                "starts looking like a narrow but repeatable precision band, while residual risks remain explicit."
            ),
            "next_question": (
                "If this stage holds, the next step is to test whether the same policy remains invariant across more "
                "models and tasks, rather than only on the current attribute-edit and routing benchmarks."
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
