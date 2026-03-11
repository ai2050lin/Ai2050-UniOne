from __future__ import annotations

import json
import math
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
    precision = load_json("continuous_input_grounding_precision_scan_20260309.json")
    hierarchical = load_json("real_multistep_memory_hierarchical_state_scan_20260309.json")
    task_block_2 = load_json("task_block_2_unified_training_closure_20260311.json")
    g2 = load_json("g2_structure_foundation_fast_slow_training_closure_20260311.json")
    g7a = load_json("g7a_slow_consolidation_replay_closure_20260311.json")
    g7b = load_json("g7b_anti_interference_retention_mechanism_search_20260311.json")

    training_support_score = mean(
        [
            task_block_2["headline_metrics"]["overall_task_block_2_score"],
            g2["headline_metrics"]["overall_g2_score"],
            g2["headline_metrics"]["fast_slow_unification_score"],
        ]
    )

    replay_support_score = mean(
        [
            g7a["headline_metrics"]["consolidation_balance_score"],
            g7a["headline_metrics"]["replay_controller_gain_score"],
            g7b["headline_metrics"]["mechanism_candidate_strength_score"],
        ]
    )

    precision_candidates: list[dict] = []
    for system_name, row in precision["systems"].items():
        novel = float(row["novel_concept_accuracy"])
        retention = float(row["retention_concept_accuracy"])
        coexistence = mean([retention, math.sqrt(max(0.0, novel * retention)), min(novel, retention)])
        precision_candidates.append(
            {
                "write_system": system_name,
                "novel_concept_accuracy": novel,
                "retention_concept_accuracy": retention,
                "grounding_score": float(row["grounding_score"]),
                "write_coexistence_score": coexistence,
            }
        )

    state_candidates: list[dict] = []
    for row in hierarchical["ranking"]:
        state_candidates.append(
            {
                "state_system": row["system"],
                "policy": row["policy"],
                "state_mode": row["state_mode"],
                "mean_closure_score": float(row["mean_closure_score"]),
                "mean_retention_score": float(row["mean_retention_score"]),
                "max_length_score": float(row["max_length_score"]),
                "state_support_score": mean(
                    [
                        float(row["mean_closure_score"]),
                        float(row["mean_retention_score"]),
                        float(row["max_length_score"]),
                    ]
                ),
            }
        )

    fused_candidates: list[dict] = []
    for write_row in precision_candidates:
        for state_row in state_candidates:
            overall = mean(
                [
                    training_support_score,
                    replay_support_score,
                    write_row["novel_concept_accuracy"],
                    write_row["write_coexistence_score"],
                    state_row["state_support_score"],
                ]
            )
            fused_candidates.append(
                {
                    "write_system": write_row["write_system"],
                    "state_system": state_row["state_system"],
                    "policy": state_row["policy"],
                    "state_mode": state_row["state_mode"],
                    "training_support_score": training_support_score,
                    "replay_support_score": replay_support_score,
                    "write_strength_score": write_row["novel_concept_accuracy"],
                    "write_coexistence_score": write_row["write_coexistence_score"],
                    "state_support_score": state_row["state_support_score"],
                    "overall_fused_candidate_score": overall,
                }
            )

    fused_candidates.sort(key=lambda row: row["overall_fused_candidate_score"], reverse=True)
    top5 = fused_candidates[:5]
    best = top5[0]

    verdict = {
        "status": (
            "fused_candidate_found_but_not_closed"
            if best["write_coexistence_score"] < 0.32
            else "fused_candidate_crosses_first_balance_gate"
        ),
        "core_answer": (
            "The best Stage-A fused direction is no longer ambiguous: keep the strong unified training support, "
            "preserve replay-aware consolidation, and prioritize high-write precision routes plus hierarchical state. "
            "But the best synthetic candidate still fails the delayed-retention coexistence gate."
        ),
        "best_candidate_family": {
            "write_system": best["write_system"],
            "state_system": best["state_system"],
            "policy": best["policy"],
            "state_mode": best["state_mode"],
        },
        "main_open_gap": "best_fused_candidate_still_has_low_write_retention_coexistence",
        "recommended_real_implementation": (
            "Implement the top candidate family inside one real fused loop first, then explicitly penalize write-retention "
            "collapse instead of optimizing write and consolidation separately."
        ),
    }

    hypotheses = {
        "H1_training_support_is_already_strong_enough_to_try_fused_search": training_support_score >= 0.66,
        "H2_replay_support_is_nontrivial": replay_support_score >= 0.52,
        "H3_best_candidate_keeps_high_write": best["write_strength_score"] >= 0.95,
        "H4_best_candidate_still_fails_retention_coexistence_gate": best["write_coexistence_score"] < 0.32,
        "H5_no_candidate_is_stage_a_closed": max(row["write_coexistence_score"] for row in fused_candidates) < 0.42,
    }

    formulas = {
        "candidate_score": (
            "FusedCandidate = mean(TrainingSupport, ReplaySupport, WriteStrength, WriteCoexistence, StateSupport)"
        ),
        "write_coexistence": (
            "WriteCoexistence = mean(RetentionConcept, sqrt(NovelConcept * RetentionConcept), min(NovelConcept, RetentionConcept))"
        ),
        "state_support": (
            "StateSupport = mean(MeanClosure, MeanRetention, MaxLengthScore)"
        ),
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "StageA1_fused_write_retention_search",
        },
        "headline_metrics": {
            "training_support_score": training_support_score,
            "replay_support_score": replay_support_score,
            "best_overall_fused_candidate_score": best["overall_fused_candidate_score"],
            "best_write_strength_score": best["write_strength_score"],
            "best_write_coexistence_score": best["write_coexistence_score"],
            "best_state_support_score": best["state_support_score"],
        },
        "top_candidates": top5,
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "stage_a1_fused_write_retention_search_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
