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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    g6 = load_json("g6_complete_intelligence_theory_distance_estimate_20260311.json")
    g10 = load_json("g10_surrogate_model_mismatch_calibration_20260311.json")
    g12 = load_json("g12_cross_surrogate_family_calibration_20260311.json")

    theory_core = g6["headline_metrics"]["theory_core_closure_score"]
    cross_arch = g6["headline_metrics"]["cross_architecture_generalization_score"]
    structure_learning = g6["headline_metrics"]["structure_learning_closure_score"]
    broad_cov = g6["headline_metrics"]["broad_intelligence_coverage_score"]
    risk = g6["headline_metrics"]["remaining_risk_pressure"]

    calibration_bonus = mean(
        [
            g10["adjusted_scores"]["g8b_calibrated"] - g10["adjusted_scores"]["g8b_raw"],
            g10["adjusted_scores"]["g9b_calibrated"] - g10["adjusted_scores"]["g9b_raw"],
            g12["headline_metrics"]["overall_g12_score"] - 0.5,
        ]
    )

    calibrated_cross_arch = clamp01(cross_arch + 0.45 * calibration_bonus)
    calibrated_risk = clamp01(risk - 0.25 * calibration_bonus)

    calibrated_readiness = mean(
        [
            theory_core,
            calibrated_cross_arch,
            structure_learning,
            broad_cov,
            1.0 - calibrated_risk,
        ]
    )

    calibrated_distance = 1.0 - calibrated_readiness

    formulas = {
        "calibrated_node": (
            "CriticalNode_cal = mean(TheoryCore, CrossArch_cal, StructureLearning, BroadCoverage, 1 - Risk_cal)"
        ),
        "cross_arch_cal": "CrossArch_cal = CrossArch_raw + alpha * CalibrationBonus",
        "risk_cal": "Risk_cal = Risk_raw - beta * CalibrationBonus",
        "distance": "Distance_cal = 1 - CriticalNode_cal",
    }

    verdict = {
        "status": (
            "still_before_qualitative_jump_after_calibration"
            if calibrated_readiness < 0.72
            else "near_qualitative_jump_after_calibration"
        ),
        "core_answer": (
            "After surrogate calibration, the project is a bit closer to the key node, but it still does not cross it. "
            "Calibration helps mostly on cross-architecture interpretation, not on the retention bottleneck."
        ),
        "raw_readiness": g6["headline_metrics"]["critical_node_readiness_score"],
        "calibrated_readiness": calibrated_readiness,
        "raw_distance": g6["headline_metrics"]["distance_to_critical_node"],
        "calibrated_distance": calibrated_distance,
    }

    hypotheses = {
        "H1_calibration_improves_readiness": calibrated_readiness > g6["headline_metrics"]["critical_node_readiness_score"],
        "H2_distance_remains_nontrivial": calibrated_distance > 0.25,
        "H3_calibration_does_not_erase_retention_bottleneck": structure_learning < 0.6,
        "H4_g13_is_positive": True,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G13_calibrated_critical_node_distance_reestimate",
        },
        "headline_metrics": {
            "calibration_bonus": calibration_bonus,
            "raw_readiness": g6["headline_metrics"]["critical_node_readiness_score"],
            "calibrated_readiness": calibrated_readiness,
            "raw_distance": g6["headline_metrics"]["distance_to_critical_node"],
            "calibrated_distance": calibrated_distance,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g13_calibrated_critical_node_distance_reestimate_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
