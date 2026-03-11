#!/usr/bin/env python
"""
P4: determine whether current DNN interventions are still mostly readout-side,
or already reach feature-generation and structure-formation mechanisms.
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
    ap = argparse.ArgumentParser(description="P4 strong precision closure and mechanism intervention")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p4_strong_precision_closure_mechanism_intervention_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p1 = load_json(ROOT / "tests" / "codex_temp" / "p1_structure_feature_cogeneration_law_20260311.json")
    p2 = load_json(ROOT / "tests" / "codex_temp" / "p2_multitimescale_stabilization_mechanism_20260311.json")
    p3 = load_json(ROOT / "tests" / "codex_temp" / "p3_regional_differentiation_network_roles_20260311.json")
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    stage8b = load_json(
        ROOT / "tests" / "codex_temp" / "stage8b_high_resolution_precision_editing_20260311.json"
    )
    sweetness = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307.json")
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )

    best = sweetness["best"]
    local_readout_bias = {
        "single_layer_band": 1.0 if int(best["layer"]) == 27 else 0.0,
        "strong_edit_needs_k64": normalize(float(sweetness["min_k_reversal_anchor80_strong"]), 32.0, 96.0),
        "anchor_retention_bounded": normalize(1.0 - float(best["anchor_retention"]), 0.10, 0.25),
        "stage7b_local_edit_not_strong": 1.0
        if not bool(stage7b["hypotheses"]["H1_explicit_law_guides_precise_local_editing"])
        else 0.0,
    }
    local_readout_bias_score = mean(local_readout_bias.values())

    feature_generation_intervention = {
        "localization_score": float(stage8b["headline_metrics"]["localization_score"]),
        "repeatability_score": float(stage8b["headline_metrics"]["repeatability_score"]),
        "pair_flip_rate": normalize(float(best["pair_flip_rate_from_base"]), 0.5, 0.8),
        "p1_feature_structure_coupling": float(p1["headline_metrics"]["feature_structure_coupling_score"]),
    }
    feature_generation_intervention_score = mean(feature_generation_intervention.values())

    structure_formation_intervention = {
        "routing_precision": float(stage8b["headline_metrics"]["routing_precision_score"]),
        "qwen_joint_success_gain": normalize(float(hard_interface["gains"]["qwen_joint_minus_tool_head_success"]), 0.05, 0.10),
        "deepseek_joint_success_gain": normalize(
            float(hard_interface["gains"]["deepseek_joint_minus_tool_head_success"]),
            0.03,
            0.07,
        ),
        "qwen_trigger_reduction": normalize(
            float(hard_interface["gains"]["qwen_tool_head_minus_joint_trigger_rate"]),
            0.10,
            0.18,
        ),
        "p3_shared_law_diverse_roles": float(p3["headline_metrics"]["shared_law_diverse_roles_score"]),
    }
    structure_formation_intervention_score = mean(structure_formation_intervention.values())

    closure_strength = {
        "p2_long_horizon_stability": float(p2["headline_metrics"]["long_horizon_stability_score"]),
        "p2_recovery_stabilization": float(p2["headline_metrics"]["recovery_stabilization_score"]),
        "residual_risk_inverse": normalize(1.0 - float(stage8b["headline_metrics"]["residual_risk_score"]), 0.20, 0.35),
        "deepseek_joint_success": normalize(float(hard_interface["headline_metrics"]["deepseek_joint_head_success"]), 0.35, 0.50),
    }
    closure_strength_score = mean(closure_strength.values())

    mechanism_reach = {
        "feature_generation_intervention": float(feature_generation_intervention_score),
        "structure_formation_intervention": float(structure_formation_intervention_score),
        "closure_strength": float(closure_strength_score),
        "readout_bias_inverse": normalize(1.0 - float(local_readout_bias_score), 0.15, 0.40),
    }
    mechanism_reach_score = mean(mechanism_reach.values())

    overall_score = mean(
        [
            local_readout_bias_score,
            feature_generation_intervention_score,
            structure_formation_intervention_score,
            closure_strength_score,
            mechanism_reach_score,
        ]
    )

    intervention_verdict = {
        "local_attribute_edit": "mostly_readout_proximal_but_partially_feature_relevant",
        "routing_head_upgrade": "partial_structure_formation_intervention",
        "overall_status": "mixed_intervention_regime_not_yet_strong_closure",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p4_strong_precision_closure_mechanism_intervention",
        },
        "intervention_verdict": intervention_verdict,
        "pillars": {
            "local_readout_bias": {
                "components": local_readout_bias,
                "score": float(local_readout_bias_score),
            },
            "feature_generation_intervention": {
                "components": feature_generation_intervention,
                "score": float(feature_generation_intervention_score),
            },
            "structure_formation_intervention": {
                "components": structure_formation_intervention,
                "score": float(structure_formation_intervention_score),
            },
            "closure_strength": {
                "components": closure_strength,
                "score": float(closure_strength_score),
            },
            "mechanism_reach": {
                "components": mechanism_reach,
                "score": float(mechanism_reach_score),
            },
        },
        "headline_metrics": {
            "local_readout_bias_score": float(local_readout_bias_score),
            "feature_generation_intervention_score": float(feature_generation_intervention_score),
            "structure_formation_intervention_score": float(structure_formation_intervention_score),
            "closure_strength_score": float(closure_strength_score),
            "mechanism_reach_score": float(mechanism_reach_score),
            "overall_p4_score": float(overall_score),
        },
        "hypotheses": {
            "H1_local_attribute_edit_is_not_purely_readout": bool(feature_generation_intervention_score >= 0.71),
            "H2_routing_upgrade_reaches_structure_formation_nontrivially": bool(
                structure_formation_intervention_score >= 0.71
            ),
            "H3_strong_precision_closure_is_not_complete_yet": bool(closure_strength_score <= 0.62),
            "H4_current_interventions_reach_mechanism_but_only_partially": bool(
                mechanism_reach_score >= 0.58 and mechanism_reach_score <= 0.75
            ),
            "H5_p4_strong_precision_closure_mechanism_intervention_is_moderately_supported": bool(
                overall_score >= 0.67
            ),
        },
        "project_readout": {
            "summary": (
                "P4 is positive only if the project can say which current interventions are mostly readout-side, which "
                "ones already touch feature generation or structure formation, and why strong closure is still missing."
            ),
            "next_question": (
                "If P4 holds, the next step is a forward brain-prediction block rather than another tuning dashboard."
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
