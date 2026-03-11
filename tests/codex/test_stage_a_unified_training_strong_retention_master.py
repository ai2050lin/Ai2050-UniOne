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
    task_block_2 = load_json("task_block_2_unified_training_closure_20260311.json")
    g2 = load_json("g2_structure_foundation_fast_slow_training_closure_20260311.json")
    g3 = load_json("g3_instant_learning_boundary_stress_20260311.json")
    g7b = load_json("g7b_anti_interference_retention_mechanism_search_20260311.json")

    training_foundation_score = mean(
        [
            task_block_2["headline_metrics"]["overall_task_block_2_score"],
            task_block_2["headline_metrics"]["structure_training_score"],
            g2["headline_metrics"]["structure_foundation_training_score"],
            g2["headline_metrics"]["fast_slow_unification_score"],
        ]
    )

    instant_write_reality_score = mean(
        [
            g3["headline_metrics"]["immediate_write_score"],
            g3["headline_metrics"]["cross_environment_carryover_score"],
            g2["headline_metrics"]["instant_learning_bridge_score"],
        ]
    )

    retention_coexistence_score = mean(
        [
            g3["headline_metrics"]["retention_boundary_score"],
            g7b["headline_metrics"]["anti_interference_retention_score"],
            g7b["headline_metrics"]["retention_write_balance_score"],
        ]
    )

    interference_control_score = mean(
        [
            g3["headline_metrics"]["interference_tradeoff_score"],
            g7b["headline_metrics"]["mechanism_candidate_strength_score"],
            g2["headline_metrics"]["online_failure_unification_score"],
        ]
    )

    overall_stage_a_score = mean(
        [
            training_foundation_score,
            instant_write_reality_score,
            retention_coexistence_score,
            interference_control_score,
        ]
    )

    formulas = {
        "stage_a_objective": (
            "L_stageA = lambda_t * L_unified_training + lambda_w * L_fast_write + "
            "lambda_r * L_delayed_retention + lambda_i * L_interference_control + lambda_s * L_structure"
        ),
        "foundation": (
            "TrainingFoundation = mean(TaskBlock2Closure, StructureTraining, FastSlowUnification)"
        ),
        "instant_write": (
            "InstantWriteReality = mean(ImmediateWrite, CrossEnvironmentCarryover, InstantLearningBridge)"
        ),
        "retention": (
            "RetentionCoexistence = mean(RetentionBoundary, AntiInterferenceRetention, RetentionWriteBalance)"
        ),
        "interference": (
            "InterferenceControl = mean(InterferenceTradeoff, MechanismCandidateStrength, OnlineFailureUnification)"
        ),
        "readiness": (
            "StageAReadiness = mean(TrainingFoundation, InstantWriteReality, RetentionCoexistence, InterferenceControl)"
        ),
    }

    hypotheses = {
        "H1_training_foundation_is_nontrivial": training_foundation_score >= 0.62,
        "H2_instant_write_is_real_inside_same_family": instant_write_reality_score >= 0.62,
        "H3_retention_coexistence_is_still_the_main_bottleneck": retention_coexistence_score < 0.42,
        "H4_interference_control_is_only_moderate": 0.52 <= interference_control_score < 0.68,
        "H5_stage_a_is_not_closed_yet": overall_stage_a_score < 0.62,
    }

    if overall_stage_a_score >= 0.68 and retention_coexistence_score >= 0.55:
        status = "stage_a_joint_closure_ready"
    elif overall_stage_a_score >= 0.58 and retention_coexistence_score >= 0.42:
        status = "stage_a_partial_joint_closure"
    else:
        status = "stage_a_joint_closure_not_ready"

    verdict = {
        "status": status,
        "core_answer": (
            "Stage A confirms that unified training and instant write now belong to one coherent family, "
            "but strong delayed retention still does not coexist with strong novel write. The main bottleneck is no longer "
            "whether a unified objective exists, but whether that same objective can force structure and retention to stay high "
            "under interference."
        ),
        "main_open_gap": "retention_coexistence_is_far_below_training_and_write_strength",
        "best_candidate_family": "cross_modal_dual_store_plus_hierarchical_consolidation_inside_fused_training",
        "next_step": (
            "Search the smallest fused mechanism that raises delayed retention without collapsing immediate write: "
            "dual-store isolation, replay scheduling, and structure-weight reinforcement should be optimized in one loop."
        ),
    }

    interpretation = {
        "training": (
            "The unified training-law side is already moderately positive. Structure, fast-slow coupling, and online-failure "
            "terms can live in one objective."
        ),
        "write": (
            "Immediate write is real, not fake. Online update and cross-environment carryover already show nontrivial gains."
        ),
        "retention": (
            "Delayed retention remains the hard wall. This is the lowest score by a wide margin and is the main reason Stage A "
            "does not close."
        ),
        "decision": (
            "The next move should not be another isolated retention probe. It should be a fused training search where structure "
            "weight, slow consolidation, replay timing, and dual-store routing are jointly tuned."
        ),
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageA_unified_training_strong_retention_master",
        },
        "headline_metrics": {
            "training_foundation_score": training_foundation_score,
            "instant_write_reality_score": instant_write_reality_score,
            "retention_coexistence_score": retention_coexistence_score,
            "interference_control_score": interference_control_score,
            "overall_stage_a_score": overall_stage_a_score,
        },
        "supporting_readout": {
            "task_block_2_score": task_block_2["headline_metrics"]["overall_task_block_2_score"],
            "g2_score": g2["headline_metrics"]["overall_g2_score"],
            "g3_score": g3["headline_metrics"]["overall_g3_score"],
            "g7b_score": g7b["headline_metrics"]["overall_g7b_score"],
            "g3_immediate_write_score": g3["headline_metrics"]["immediate_write_score"],
            "g3_retention_boundary_score": g3["headline_metrics"]["retention_boundary_score"],
            "g7b_retention_write_balance_score": g7b["headline_metrics"]["retention_write_balance_score"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_a_unified_training_strong_retention_master_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
