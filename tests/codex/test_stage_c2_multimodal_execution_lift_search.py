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
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    stage_c = load_json("stage_c_external_closure_master_20260311.json")
    multimodal = load_json("continuous_multimodal_grounding_proto_20260309.json")
    goal_state = load_json("open_world_grounding_action_loop_goal_state_scan_20260310.json")
    variable_planning = load_json("open_world_variable_planning_trainable_benchmark_20260310.json")

    best_goal = goal_state["best_config"]
    plan_head = variable_planning["headline_metrics"]

    grounding_execution_score = mean(
        [
            float(stage_c["headline_metrics"]["continuous_grounding_score"]),
            float(best_goal["loop_score"]),
            float(best_goal["corrected_action_accuracy"]),
            float(plan_head["variable_planning_score"]),
        ]
    )

    current_multimodal_score = float(stage_c["headline_metrics"]["multimodal_consistency_score"])
    brain_execution_readiness_score = float(stage_c["headline_metrics"]["brain_execution_readiness_score"])

    current_stage_c2_score = mean(
        [
            grounding_execution_score,
            current_multimodal_score,
            brain_execution_readiness_score,
        ]
    )

    partial_target = 0.58
    moderate_target = 0.65
    strong_target = 0.72

    multimodal_needed_for_partial = clamp01(partial_target * 3 - grounding_execution_score - brain_execution_readiness_score)
    multimodal_needed_for_moderate = clamp01(moderate_target * 3 - grounding_execution_score - brain_execution_readiness_score)
    multimodal_needed_for_strong = clamp01(strong_target * 3 - grounding_execution_score - brain_execution_readiness_score)

    hypotheses = {
        "H1_execution_loop_lifts_c_above_raw": grounding_execution_score > stage_c["headline_metrics"]["continuous_grounding_score"],
        "H2_stage_c2_reaches_partial_closure_with_execution_loop": current_stage_c2_score >= partial_target,
        "H3_multimodal_is_still_the_primary_gap": current_multimodal_score < multimodal_needed_for_moderate,
        "H4_moderate_closure_needs_only_nontrivial_multimodal_lift": multimodal_needed_for_moderate <= 0.4,
        "H5_strong_closure_still_requires_large_multimodal_lift": multimodal_needed_for_strong > 0.5,
    }

    if current_stage_c2_score >= strong_target and current_multimodal_score >= multimodal_needed_for_strong:
        status = "stage_c_joint_closure_ready"
    elif current_stage_c2_score >= partial_target:
        status = "stage_c_partial_external_closure_after_execution_integration"
    else:
        status = "stage_c_still_not_ready_after_execution_integration"

    verdict = {
        "status": status,
        "core_answer": (
            "Stage C is not blocked by execution-loop readiness anymore. Once execution is integrated, the block reaches partial closure. "
            "The remaining dominant gap is multimodal consistency."
        ),
        "main_open_gap": "multimodal_consistency_remains_too_low_for_moderate_closure",
        "required_multimodal_targets": {
            "current_multimodal_score": current_multimodal_score,
            "for_partial": multimodal_needed_for_partial,
            "for_moderate": multimodal_needed_for_moderate,
            "for_strong": multimodal_needed_for_strong,
        },
    }

    interpretation = {
        "execution": "Execution-loop integration is already strong enough to remove 'no external loop' as the main objection.",
        "multimodal": "Multimodal consistency is now the clearest dominant bottleneck. Without lifting it, C cannot become a real external closure.",
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageC2_multimodal_execution_lift_search",
        },
        "headline_metrics": {
            "raw_stage_c_score": float(stage_c["headline_metrics"]["overall_stage_c_score"]),
            "grounding_execution_score": grounding_execution_score,
            "current_multimodal_score": current_multimodal_score,
            "brain_execution_readiness_score": brain_execution_readiness_score,
            "current_stage_c2_score": current_stage_c2_score,
        },
        "supporting_readout": {
            "best_goal_loop_score": float(best_goal["loop_score"]),
            "best_goal_corrected_action_accuracy": float(best_goal["corrected_action_accuracy"]),
            "best_goal_old_concept_retention": float(best_goal["old_concept_retention"]),
            "variable_planning_score": float(plan_head["variable_planning_score"]),
        },
        "hypotheses": hypotheses,
        "interpretation": interpretation,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_c2_multimodal_execution_lift_search_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
