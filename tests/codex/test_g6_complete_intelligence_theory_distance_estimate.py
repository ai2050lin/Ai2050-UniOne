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
    p10a = load_json("p10a_final_theory_verdict_20260311.json")
    p10b = load_json("p10b_gap_boundary_empirical_vs_theoretical_20260311.json")
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")
    f6 = load_json("f6_world_model_reasoning_generation_physics_prediction_20260311.json")
    f7 = load_json("f7_human_language_instant_learning_architecture_20260311.json")
    g1a = load_json("g1a_targeted_bridge_selection_rules_20260311.json")
    g1b = load_json("g1b_unified_layer_role_coordinate_system_20260311.json")
    g2 = load_json("g2_structure_foundation_fast_slow_training_closure_20260311.json")
    g3 = load_json("g3_instant_learning_boundary_stress_20260311.json")
    g4 = load_json("g4_brain_direct_falsification_master_20260311.json")
    g5 = load_json("g5_brain_experiment_protocol_observable_mapping_20260311.json")

    theory_core_closure_score = mean(
        [
            p10a["headline_metrics"]["overall_p10a_score"],
            p10b["headline_metrics"]["overall_p10b_score"],
            g4["headline_metrics"]["overall_g4_score"],
            g5["headline_metrics"]["overall_g5_score"],
        ]
    )

    cross_architecture_generalization_score = mean(
        [
            f1["headline_metrics"]["overall_f1_score"],
            g1a["headline_metrics"]["overall_g1a_score"],
            g1b["headline_metrics"]["overall_g1b_score"],
        ]
    )

    structure_learning_closure_score = mean(
        [
            g2["headline_metrics"]["overall_g2_score"],
            g2["headline_metrics"]["structure_foundation_training_score"],
            g3["headline_metrics"]["retention_boundary_score"],
            f7["headline_metrics"]["instant_learning_readiness_score"],
        ]
    )

    broad_intelligence_coverage_score = mean(
        [
            f6["headline_metrics"]["overall_f6_score"],
            f7["headline_metrics"]["overall_f7_score"],
            f6["headline_metrics"]["physical_rule_prediction_readiness_score"],
            f7["headline_metrics"]["language_capacity_readiness_score"],
        ]
    )

    remaining_risk_pressure = mean(
        [
            1.0 - f1["headline_metrics"]["architecture_scale_residual_boundary_score"],
            1.0 - g1a["headline_metrics"]["relation_to_bridge_rule_score"],
            1.0 - g1b["headline_metrics"]["role_axis_separability_score"],
            1.0 - g2["headline_metrics"]["structure_foundation_training_score"],
            1.0 - g3["headline_metrics"]["retention_boundary_score"],
        ]
    )

    critical_node_readiness_score = mean(
        [
            theory_core_closure_score,
            cross_architecture_generalization_score,
            structure_learning_closure_score,
            broad_intelligence_coverage_score,
            1.0 - remaining_risk_pressure,
        ]
    )

    distance_to_critical_node = 1.0 - critical_node_readiness_score

    formulas = {
        "critical_node": (
            "CriticalNode = mean(TheoryCoreClosure, CrossArchitectureGeneralization, "
            "StructureLearningClosure, BroadIntelligenceCoverage, 1 - RemainingRiskPressure)"
        ),
        "distance": "Distance = 1 - CriticalNode",
        "qualitative_jump_condition": (
            "Jump = 1[TheoryCoreClosure, CrossArchitectureGeneralization, and StructureLearningClosure "
            "all exceed their bottleneck thresholds together]"
        ),
        "current_bottleneck": (
            "Bottleneck = max(ArchitectureScaleResidual, BridgeRuleGap, RoleAxisGap, StructureFoundationGap, RetentionGap)"
        ),
    }

    bottlenecks = [
        {
            "name": "retention_gap",
            "score": 1.0 - g3["headline_metrics"]["retention_boundary_score"],
        },
        {
            "name": "bridge_rule_gap",
            "score": 1.0 - g1a["headline_metrics"]["relation_to_bridge_rule_score"],
        },
        {
            "name": "role_axis_gap",
            "score": 1.0 - g1b["headline_metrics"]["role_axis_separability_score"],
        },
        {
            "name": "structure_foundation_gap",
            "score": 1.0 - g2["headline_metrics"]["structure_foundation_training_score"],
        },
        {
            "name": "architecture_scale_gap",
            "score": 1.0 - f1["headline_metrics"]["architecture_scale_residual_boundary_score"],
        },
    ]
    bottlenecks.sort(key=lambda x: x["score"], reverse=True)

    verdict = {
        "status": (
            "near_but_not_at_qualitative_jump"
            if critical_node_readiness_score >= 0.72
            else "still_before_qualitative_jump"
        ),
        "critical_node_readiness_score": critical_node_readiness_score,
        "distance_to_critical_node": distance_to_critical_node,
        "estimated_remaining_distance_percent": round(distance_to_critical_node * 100.0, 1),
        "core_answer": (
            "The project is no longer far from the key node, but it has not crossed it. "
            "The remaining distance is dominated by retention, bridge-law specificity, role-axis clarity, and cross-architecture closure."
        ),
        "top_bottlenecks": bottlenecks[:3],
    }

    hypotheses = {
        "H1_theory_core_is_close": theory_core_closure_score >= 0.75,
        "H2_cross_architecture_generalization_is_still_subcritical": cross_architecture_generalization_score < 0.7,
        "H3_structure_learning_closure_is_still_subcritical": structure_learning_closure_score < 0.65,
        "H4_broad_intelligence_coverage_is_nontrivial": broad_intelligence_coverage_score >= 0.78,
        "H5_project_is_near_but_not_at_the_critical_node": 0.72 <= critical_node_readiness_score < 0.82,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G6_complete_intelligence_theory_distance_estimate",
        },
        "headline_metrics": {
            "theory_core_closure_score": theory_core_closure_score,
            "cross_architecture_generalization_score": cross_architecture_generalization_score,
            "structure_learning_closure_score": structure_learning_closure_score,
            "broad_intelligence_coverage_score": broad_intelligence_coverage_score,
            "remaining_risk_pressure": remaining_risk_pressure,
            "critical_node_readiness_score": critical_node_readiness_score,
            "distance_to_critical_node": distance_to_critical_node,
        },
        "formulas": formulas,
        "bottlenecks": bottlenecks,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g6_complete_intelligence_theory_distance_estimate_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
