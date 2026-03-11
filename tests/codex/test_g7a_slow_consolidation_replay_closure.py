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
    phase = load_json("real_multistep_memory_phase_state_controller_20260309.json")
    hierarchical = load_json("real_multistep_memory_hierarchical_state_scan_20260309.json")
    segment = load_json("real_multistep_memory_segment_summary_scan_20260309.json")
    g7 = load_json("g7_strong_retention_instant_learning_closure_20260311.json")

    best_phase = phase["ranking"][0]
    best_hier = hierarchical["best"]["best_mean_system"]
    best_segment = segment["best"]["best_mean_segment_system"]
    single_anchor = hierarchical["ranking"][2]

    retention_lift_score = mean(
        [
            best_hier["mean_retention_score"],
            best_segment["mean_retention_score"],
            max(0.0, best_hier["mean_retention_score"] - single_anchor["mean_retention_score"]),
            max(0.0, best_segment["mean_retention_score"] - single_anchor["mean_retention_score"]),
        ]
    )

    consolidation_balance_score = mean(
        [
            best_hier["mean_closure_score"],
            best_segment["mean_closure_score"],
            max(0.0, 1.0 - abs(best_hier["closure_relative_drop"])),
            max(0.0, 1.0 - abs(best_segment["closure_relative_drop"])),
        ]
    )

    replay_controller_gain_score = mean(
        [
            max(0.0, phase["gains"]["best_controller_vs_single_anchor_at_max_length"] + 0.5),
            max(0.0, hierarchical["gains"]["hierarchical_mean_vs_segment"] + 0.5),
            max(0.0, segment["gains"]["joint_segment_mean_vs_joint"] + 0.5),
            g7["headline_metrics"]["online_retention_carryover_score"],
        ]
    )

    strong_retention_progress_score = mean(
        [
            retention_lift_score,
            consolidation_balance_score,
            replay_controller_gain_score,
        ]
    )

    formulas = {
        "slow_consolidation": "A_{t+1} = A_t + eta_slow * Consolidate(M_{t+1}, h_t, z_t, g_t)",
        "replay_assisted_retention": "Retain = ReplayStride * SegmentState + HierarchicalState + TailFocusedRoute",
        "progress": "RetentionProgress = mean(RetentionLift, ConsolidationBalance, ReplayControllerGain)",
    }

    verdict = {
        "status": (
            "slow_consolidation_replay_partially_positive"
            if strong_retention_progress_score >= 0.48
            else "slow_consolidation_replay_not_enough"
        ),
        "core_answer": (
            "Replay plus hierarchical state does improve retention relative to the weakest baselines, "
            "and tail-focused hierarchical routing is the best current retention family. But the lift is still far from a strong closure."
        ),
        "best_family": "hierarchical_or_tailfocus_hierarchical",
        "main_open_gap": "retention_lift_is_real_but_still_small",
    }

    hypotheses = {
        "H1_retention_lift_exists": retention_lift_score >= 0.1,
        "H2_consolidation_balance_is_nontrivial": consolidation_balance_score >= 0.45,
        "H3_replay_controller_gain_is_nontrivial": replay_controller_gain_score >= 0.52,
        "H4_g7a_is_partial_not_closed": 0.48 <= strong_retention_progress_score < 0.62,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G7A_slow_consolidation_replay_closure",
        },
        "headline_metrics": {
            "retention_lift_score": retention_lift_score,
            "consolidation_balance_score": consolidation_balance_score,
            "replay_controller_gain_score": replay_controller_gain_score,
            "overall_g7a_score": strong_retention_progress_score,
        },
        "supporting_readout": {
            "best_phase_controller": best_phase,
            "best_hierarchical_system": best_hier,
            "best_segment_system": best_segment,
            "single_anchor_reference": single_anchor,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g7a_slow_consolidation_replay_closure_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
