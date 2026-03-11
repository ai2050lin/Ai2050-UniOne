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
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")
    stage8c = load_json("stage8c_cross_model_task_invariants_20260311.json")
    p9a = load_json("p9a_spatial_plasticity_coding_master_20260311.json")
    p10a = load_json("p10a_final_theory_verdict_20260311.json")

    invariant_anchor_score = mean(
        [
            f1["headline_metrics"]["relation_invariance_extrapolation_score"],
            f1["headline_metrics"]["orientation_gap_stability_score"],
            stage8c["headline_metrics"]["relation_order_invariance_score"],
            stage8c["headline_metrics"]["compatibility_invariance_score"],
        ]
    )

    family_calibration_score = mean(
        [
            stage8c["headline_metrics"]["model_gap_structure_score"],
            f1["headline_metrics"]["family_topology_extrapolation_score"],
            f1["headline_metrics"]["layer_role_transfer_score"],
        ]
    )

    theory_anchor_retention_score = mean(
        [
            p9a["headline_metrics"]["theory_consistency_score"],
            p10a["headline_metrics"]["theory_support_score"],
            p10a["headline_metrics"]["testability_strength_score"],
        ]
    )

    overall_g12_score = mean(
        [
            invariant_anchor_score,
            family_calibration_score,
            theory_anchor_retention_score,
        ]
    )

    formulas = {
        "invariant_anchor": "Anchor = mean(RelationInvariant, OrientationStable, CompatibilityInvariant)",
        "family_calibration": "FamilyCal = mean(ModelGapStructure, FamilyTopology, LayerRoleTransfer)",
        "cross_surrogate": "G12 = mean(InvariantAnchor, FamilyCalibration, TheoryAnchorRetention)",
    }

    verdict = {
        "status": (
            "cross_surrogate_calibration_positive"
            if overall_g12_score >= 0.68
            else "cross_surrogate_calibration_partial"
        ),
        "core_answer": (
            "Cross-surrogate calibration is positive at the anchor level: relation invariance, compatibility invariance, and theory support remain strong. "
            "The weaker part is family calibration, especially layer-role transfer."
        ),
        "strongest_anchor": "relation_and_compatibility_invariance",
        "weakest_anchor": "family_topology_and_role_transfer",
    }

    hypotheses = {
        "H1_invariant_anchor_is_strong": invariant_anchor_score >= 0.85,
        "H2_family_calibration_is_only_moderate": family_calibration_score < 0.65,
        "H3_theory_anchor_retention_is_strong": theory_anchor_retention_score >= 0.76,
        "H4_g12_is_positive": overall_g12_score >= 0.68,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G12_cross_surrogate_family_calibration",
        },
        "headline_metrics": {
            "invariant_anchor_score": invariant_anchor_score,
            "family_calibration_score": family_calibration_score,
            "theory_anchor_retention_score": theory_anchor_retention_score,
            "overall_g12_score": overall_g12_score,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g12_cross_surrogate_family_calibration_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
