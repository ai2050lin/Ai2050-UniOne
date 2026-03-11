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
    g8b = load_json("g8b_high_margin_relation_bridge_discriminant_20260311.json")
    g9b = load_json("g9b_cross_model_intervention_stable_role_kernel_20260311.json")
    g10 = load_json("g10_surrogate_model_mismatch_calibration_20260311.json")
    g11 = load_json("g11_surrogate_sensitivity_decomposition_20260311.json")
    g12 = load_json("g12_cross_surrogate_family_calibration_20260311.json")
    g13 = load_json("g13_calibrated_critical_node_distance_reestimate_20260311.json")

    bridge_rule_score = mean(
        [
            g8b["headline_metrics"]["discriminant_margin_score"],
            g8b["headline_metrics"]["behavior_alignment_score"],
            g8b["headline_metrics"]["rule_separation_score"],
        ]
    )

    role_kernel_score = mean(
        [
            g9b["headline_metrics"]["kernel_visibility_score"],
            g9b["headline_metrics"]["cross_model_kernel_consistency_score"],
            g9b["headline_metrics"]["intervention_stability_kernel_score"],
        ]
    )

    calibration_support_score = mean(
        [
            g10["headline_metrics"]["calibration_slack"] + 0.5,
            g12["headline_metrics"]["overall_g12_score"],
            g13["headline_metrics"]["calibrated_readiness"],
        ]
    )

    surrogate_clarity_score = mean(
        [
            1.0 - g10["headline_metrics"]["surrogate_mismatch_pressure"],
            g11["headline_metrics"]["architecture_scale_carryover_score"],
            g12["headline_metrics"]["family_calibration_score"],
        ]
    )

    overall_stage_b_score = mean(
        [
            bridge_rule_score,
            role_kernel_score,
            calibration_support_score,
            surrogate_clarity_score,
        ]
    )

    hypotheses = {
        "H1_bridge_rule_is_still_the_weaker_half": bridge_rule_score < role_kernel_score,
        "H2_role_kernel_is_nontrivial": role_kernel_score >= 0.62,
        "H3_calibration_support_is_strong": calibration_support_score >= 0.62,
        "H4_surrogate_clarity_is_only_moderate": 0.48 <= surrogate_clarity_score < 0.66,
        "H5_stage_b_is_not_fully_closed": overall_stage_b_score < 0.7,
    }

    if overall_stage_b_score >= 0.7 and bridge_rule_score >= 0.64 and role_kernel_score >= 0.68:
        status = "stage_b_joint_closure_ready"
    elif overall_stage_b_score >= 0.58 and role_kernel_score >= 0.62:
        status = "stage_b_partial_joint_closure"
    else:
        status = "stage_b_joint_closure_not_ready"

    verdict = {
        "status": status,
        "core_answer": (
            "Stage B is now centered on one shared middle-layer bottleneck: bridge-law margin and role-kernel stability "
            "must close together. Role-kernel evidence is already nontrivial, but bridge-law hardness is still weaker and "
            "keeps the joint block from strong closure."
        ),
        "main_open_gap": "bridge_margin_is_still_shallower_than_role_kernel_evidence",
        "next_step": (
            "Do not split bridge-law and role-kernel into separate tracks. Re-estimate them under calibrated reading and then "
            "search the smallest joint family that raises bridge margin without destroying role-kernel stability."
        ),
    }

    interpretation = {
        "bridge": "Bridge-law is visible but not yet high-margin. Rule separation is strong, but discriminant margin and behavior alignment are still uneven.",
        "role": "Role-kernel is already partial and real. The main weakness is intervention stability, especially on the DeepSeek side.",
        "calibration": "Calibration support is now strong enough that weak bridge/role readings cannot be interpreted naively as direct refutation.",
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageB_bridge_role_kernel_master",
        },
        "headline_metrics": {
            "bridge_rule_score": bridge_rule_score,
            "role_kernel_score": role_kernel_score,
            "calibration_support_score": calibration_support_score,
            "surrogate_clarity_score": surrogate_clarity_score,
            "overall_stage_b_score": overall_stage_b_score,
        },
        "supporting_readout": {
            "g8b_score": g8b["headline_metrics"]["overall_g8b_score"],
            "g9b_score": g9b["headline_metrics"]["overall_g9b_score"],
            "g10_g8b_calibrated": g10["adjusted_scores"]["g8b_calibrated"],
            "g10_g9b_calibrated": g10["adjusted_scores"]["g9b_calibrated"],
            "g12_score": g12["headline_metrics"]["overall_g12_score"],
            "g13_calibrated_readiness": g13["headline_metrics"]["calibrated_readiness"],
        },
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_b_bridge_role_kernel_master_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
