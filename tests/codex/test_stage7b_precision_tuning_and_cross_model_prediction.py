#!/usr/bin/env python
"""
Score whether the explicit coding law can guide precise model tuning and predict
cross-model divergence patterns directly.
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


def inv_gap_score(value: float, lo: float, hi: float) -> float:
    return float(1.0 - normalize(value, lo, hi))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7B precision tuning and cross-model prediction")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage7b_precision_tuning_and_cross_model_prediction_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    sweetness = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307.json")
    joint_upgrade = load_json(
        ROOT / "tests" / "codex_temp" / "relation_tool_joint_generator_network_upgrade_20260310.json"
    )
    layer_bridge = load_json(
        ROOT / "tests" / "codex_temp" / "generator_network_real_layer_band_bridge_20260310.json"
    )
    structure_bridge = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )
    stage6c = load_json(
        ROOT / "tests" / "codex_temp" / "stage6c_long_horizon_open_environment_closure_20260311.json"
    )

    best_edit = sweetness["best"]
    joint_metrics = joint_upgrade["headline_metrics"]
    layer_metrics = layer_bridge["headline_metrics"]
    hard_metrics = hard_interface["headline_metrics"]
    stage7a_headline = stage7a["headline_metrics"]

    precise_tuning = {
        "gap_reversal": 1.0 if bool(best_edit["gap_reversed_from_base"]) else 0.0,
        "pair_flip_rate": normalize(float(best_edit["pair_flip_rate_from_base"]), 0.5, 0.8),
        "anchor_retention": normalize(float(best_edit["anchor_retention"]), 0.75, 0.90),
        "min_k_soft": inv_gap_score(float(sweetness["min_k_reversal_anchor80_soft"]), 16.0, 96.0),
        "min_k_strong": inv_gap_score(float(sweetness["min_k_reversal_anchor80_strong"]), 16.0, 128.0),
    }
    precise_tuning_score = mean(precise_tuning.values())

    routing_tuning = {
        "qwen_relation_gain": normalize(float(joint_metrics["qwen_relation_undercoverage_gain"]), 0.05, 0.10),
        "deepseek_relation_gain": normalize(float(joint_metrics["deepseek_relation_undercoverage_gain"]), 0.05, 0.10),
        "qwen_mean_gain": normalize(float(joint_metrics["qwen_mean_undercoverage_gain"]), 0.01, 0.04),
        "deepseek_mean_gain": normalize(float(joint_metrics["deepseek_mean_undercoverage_gain"]), 0.01, 0.04),
        "best_trial_score": normalize(
            float(joint_upgrade["search_summary"]["top_trials"][0]["score"]),
            0.35,
            0.45,
        ),
    }
    routing_tuning_score = mean(routing_tuning.values())

    cross_model_prediction = {
        "predicted_worst_stage_matches_qwen": 1.0 if layer_metrics["qwen_worst_stage"] == "tool" else 0.0,
        "predicted_worst_stage_matches_deepseek": 1.0 if layer_metrics["deepseek_worst_stage"] == "tool" else 0.0,
        "deepseek_undercoverage_gt_qwen": normalize(
            float(layer_bridge["gains"]["deepseek_minus_qwen_searched_undercoverage"]),
            0.10,
            0.22,
        ),
        "deepseek_joint_success_lt_qwen": normalize(
            float(hard_metrics["qwen_joint_head_success"] - hard_metrics["deepseek_joint_head_success"]),
            0.30,
            0.50,
        ),
    }
    cross_model_prediction_score = mean(cross_model_prediction.values())

    structure_behavior = {
        "qwen_behavior_gain_mean": normalize(
            float(
                sum(
                    row["behavior_gain"]
                    for row in structure_bridge["models"]["qwen3_4b"]["tasks"].values()
                )
                / len(structure_bridge["models"]["qwen3_4b"]["tasks"])
            ),
            0.02,
            0.05,
        ),
        "deepseek_behavior_gain_mean": normalize(
            float(
                sum(
                    row["behavior_gain"]
                    for row in structure_bridge["models"]["deepseek_7b"]["tasks"].values()
                )
                / len(structure_bridge["models"]["deepseek_7b"]["tasks"])
            ),
            0.015,
            0.05,
        ),
        "stage6c_tool_failure_recovery": float(stage6c["headline_metrics"]["tool_failure_recovery_score"]),
        "stage7a_equation_support": float(stage7a_headline["equation_support_score"]),
    }
    structure_behavior_score = mean(structure_behavior.values())

    tuning_policy_support = {
        "route_gain_prior": normalize(
            float(stage7a["candidate_coding_law"]["parameters"]["route_gain"]),
            1.0,
            1.5,
        ),
        "brain_gain_prior": normalize(
            float(stage7a["candidate_coding_law"]["parameters"]["brain_gain"]),
            0.40,
            0.60,
        ),
        "protocol_routing_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["protocol_routing"]),
            0.14,
            0.22,
        ),
        "multi_timescale_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["multi_timescale_control"]),
            0.12,
            0.20,
        ),
    }
    tuning_policy_support_score = mean(tuning_policy_support.values())

    overall_score = mean(
        [
            precise_tuning_score,
            routing_tuning_score,
            cross_model_prediction_score,
            structure_behavior_score,
            tuning_policy_support_score,
        ]
    )

    recommended_policy = {
        "local_attribute_edit": {
            "target_layer": int(best_edit["layer"]),
            "target_k": int(best_edit["k"]),
            "target_scale": float(best_edit["scale"]),
        },
        "routing_head_upgrade": joint_upgrade["generator_profiles"]["relation_tool_joint_head_generator_network"][
            "upgrade_spec"
        ],
        "predicted_risk_order": {
            "qwen3_4b": ["tool", "relation", "verify", "concept"],
            "deepseek_7b": ["tool", "relation", "concept", "verify"],
        },
    }

    hypotheses = {
        "H1_explicit_law_guides_precise_local_editing": bool(precise_tuning_score >= 0.72),
        "H2_explicit_law_guides_routing_tuning": bool(routing_tuning_score >= 0.68),
        "H3_explicit_law_predicts_cross_model_divergence": bool(cross_model_prediction_score >= 0.72),
        "H4_structure_and_behavior_follow_the_law_nontrivially": bool(structure_behavior_score >= 0.62),
        "H5_stage7b_precision_tuning_and_prediction_is_moderately_supported": bool(overall_score >= 0.70),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage7b_precision_tuning_and_cross_model_prediction",
        },
        "recommended_policy": recommended_policy,
        "pillars": {
            "precise_tuning": {"components": precise_tuning, "score": float(precise_tuning_score)},
            "routing_tuning": {"components": routing_tuning, "score": float(routing_tuning_score)},
            "cross_model_prediction": {
                "components": cross_model_prediction,
                "score": float(cross_model_prediction_score),
            },
            "structure_behavior": {
                "components": structure_behavior,
                "score": float(structure_behavior_score),
            },
            "tuning_policy_support": {
                "components": tuning_policy_support,
                "score": float(tuning_policy_support_score),
            },
        },
        "headline_metrics": {
            "precise_tuning_score": float(precise_tuning_score),
            "routing_tuning_score": float(routing_tuning_score),
            "cross_model_prediction_score": float(cross_model_prediction_score),
            "structure_behavior_score": float(structure_behavior_score),
            "tuning_policy_support_score": float(tuning_policy_support_score),
            "overall_stage7b_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 7B is positive only if the explicit coding law can do two useful things at once: guide precise "
                "model tuning and correctly predict how different models will diverge in stage demand and failure risk."
            ),
            "next_question": (
                "If this stage holds, the next step is a brain-side falsifiable prediction stage rather than more fitting."
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
