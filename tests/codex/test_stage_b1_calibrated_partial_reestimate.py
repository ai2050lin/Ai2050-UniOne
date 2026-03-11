from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    stage_b = load_json("stage_b_bridge_role_kernel_master_20260311.json")
    g8b = load_json("g8b_high_margin_relation_bridge_discriminant_20260311.json")
    g9b = load_json("g9b_cross_model_intervention_stable_role_kernel_20260311.json")
    g10 = load_json("g10_surrogate_model_mismatch_calibration_20260311.json")
    g11 = load_json("g11_surrogate_sensitivity_decomposition_20260311.json")
    g12 = load_json("g12_cross_surrogate_family_calibration_20260311.json")
    g13 = load_json("g13_calibrated_critical_node_distance_reestimate_20260311.json")

    calibrated_bridge_rule_score = mean(
        [
            g10["adjusted_scores"]["g8b_calibrated"],
            g8b["headline_metrics"]["rule_separation_score"],
            g12["headline_metrics"]["invariant_anchor_score"],
        ]
    )

    calibrated_role_kernel_score = mean(
        [
            g10["adjusted_scores"]["g9b_calibrated"],
            g9b["headline_metrics"]["cross_model_kernel_consistency_score"],
            g12["headline_metrics"]["family_calibration_score"],
        ]
    )

    calibrated_support_score = mean(
        [
            g12["headline_metrics"]["overall_g12_score"],
            g13["headline_metrics"]["calibrated_readiness"],
            1.0 - g10["headline_metrics"]["surrogate_mismatch_pressure"],
        ]
    )

    transfer_risk_score = mean(
        [
            1.0 - g11["headline_metrics"]["block_sensitivity_score"],
            g11["headline_metrics"]["architecture_scale_carryover_score"],
            1.0 - min(1.0, g9b["supporting_readout"]["deepseek_orientation_gap_abs"]),
        ]
    )

    overall_stage_b1_score = mean(
        [
            calibrated_bridge_rule_score,
            calibrated_role_kernel_score,
            calibrated_support_score,
            transfer_risk_score,
        ]
    )

    hypotheses = {
        "H1_bridge_rule_becomes_nontrivial_after_calibration": calibrated_bridge_rule_score >= 0.72,
        "H2_role_kernel_strengthens_after_calibration": calibrated_role_kernel_score >= 0.64,
        "H3_calibrated_support_is_strong": calibrated_support_score >= 0.62,
        "H4_transfer_risk_remains_the_main_drag": transfer_risk_score < calibrated_support_score,
        "H5_stage_b1_remains_partial_not_moderate": 0.62 <= overall_stage_b1_score < 0.72,
    }

    if overall_stage_b1_score >= 0.72 and calibrated_bridge_rule_score >= 0.76 and calibrated_role_kernel_score >= 0.7:
        status = "stage_b_moderate_joint_closure_after_calibration"
    elif overall_stage_b1_score >= 0.62:
        status = "stage_b_partial_joint_closure_strengthened_by_calibration"
    else:
        status = "stage_b_still_not_ready_after_calibration"

    verdict = {
        "status": status,
        "core_answer": (
            "After calibrated reading, Stage B is stronger than the raw score suggests. Bridge-law is no longer weak enough to be read as near-failure, "
            "and role-kernel evidence becomes more robust. But transfer-risk and DeepSeek-side rotation still keep the block below moderate closure."
        ),
        "main_open_gap": "transfer_risk_and_deepseek_rotation_still_block_moderate_closure",
        "remaining_target": {
            "current_overall_stage_b1_score": overall_stage_b1_score,
            "moderate_target": 0.72,
            "lift_needed_for_moderate": max(0.0, 0.72 - overall_stage_b1_score),
        },
    }

    interpretation = {
        "bridge": "Calibration mainly rescues interpretation, not the underlying margin itself. The bridge side becomes clearly nontrivial but still not hard enough.",
        "role": "Role-kernel looks stronger after calibration, but family transfer and intervention-stability still do not behave like a stable kernel closure.",
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageB1_calibrated_partial_reestimate",
        },
        "headline_metrics": {
            "raw_stage_b_score": stage_b["headline_metrics"]["overall_stage_b_score"],
            "calibrated_bridge_rule_score": calibrated_bridge_rule_score,
            "calibrated_role_kernel_score": calibrated_role_kernel_score,
            "calibrated_support_score": calibrated_support_score,
            "transfer_risk_score": transfer_risk_score,
            "overall_stage_b1_score": overall_stage_b1_score,
        },
        "supporting_readout": {
            "g8b_raw": g8b["headline_metrics"]["overall_g8b_score"],
            "g8b_calibrated": g10["adjusted_scores"]["g8b_calibrated"],
            "g9b_raw": g9b["headline_metrics"]["overall_g9b_score"],
            "g9b_calibrated": g10["adjusted_scores"]["g9b_calibrated"],
            "g12_family_calibration_score": g12["headline_metrics"]["family_calibration_score"],
            "deepseek_orientation_gap_abs": g9b["supporting_readout"]["deepseek_orientation_gap_abs"],
        },
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_b1_calibrated_partial_reestimate_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
