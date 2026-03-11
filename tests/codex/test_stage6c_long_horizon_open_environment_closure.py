#!/usr/bin/env python
"""
Score whether the compressed training loop survives a long-horizon open
environment with planning, tool failure, drift, and recovery pressure.
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
    ap = argparse.ArgumentParser(description="Stage 6C long-horizon open environment closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage6c_long_horizon_open_environment_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    long_horizon = load_json(ROOT / "tests" / "codex_temp" / "open_world_long_horizon_goal_state_benchmark_20260310.json")
    subgoal = load_json(ROOT / "tests" / "codex_temp" / "open_world_subgoal_planning_benchmark_20260310.json")
    variable = load_json(ROOT / "tests" / "codex_temp" / "open_world_variable_planning_trainable_benchmark_20260310.json")
    hard_interface = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json")
    recovery = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    long_best = long_horizon["best_config"]
    subgoal_best = subgoal["best_config"]
    trainable = variable["systems"]["trainable_planner"]
    hard_gain = hard_interface["gains"]
    recovery_headline = recovery["headline_metrics"]
    stage6b_headline = stage6b["headline_metrics"]

    long_horizon_goal = {
        "loop_score": normalize(float(long_best["long_horizon_loop_score"]), 0.84, 0.88),
        "phase_switch_gain": normalize(float(long_best["phase_switch_gain_vs_direct"]), 0.0, 0.02),
        "retention_gain": normalize(float(long_best["retention_gain_vs_direct"]), 0.0, 0.03),
        "loop_gain_vs_stateful": normalize(float(long_best["loop_score_gain_vs_stateful"]), 0.0, 0.03),
        "mean_family_trust": normalize(float(long_best["mean_family_trust"]), 0.97, 1.0),
    }
    long_horizon_goal_score = mean(long_horizon_goal.values())

    subgoal_chain = {
        "planning_loop_score": normalize(float(subgoal_best["planning_loop_score"]), 0.82, 0.87),
        "episode_success_rate": normalize(float(subgoal_best["episode_success_rate"]), 0.60, 0.80),
        "planning_gain_vs_direct": normalize(float(subgoal_best["planning_gain_vs_direct"]), 0.0, 0.03),
        "planning_gain_vs_stateful": normalize(float(subgoal_best["planning_gain_vs_stateful"]), 0.02, 0.06),
        "transition_gain_vs_direct": normalize(float(subgoal_best["transition_gain_vs_direct"]), 0.0, 0.01),
    }
    subgoal_chain_score = mean(subgoal_chain.values())

    variable_planner = {
        "episode_success_rate": normalize(float(trainable["episode_success_rate"]), 0.85, 1.0),
        "rollback_recovery_rate": normalize(float(trainable["rollback_recovery_rate"]), 0.80, 1.0),
        "open_environment_stability": normalize(float(trainable["open_environment_stability"]), 0.90, 0.97),
        "variable_planning_score": normalize(float(trainable["variable_planning_score"]), 0.85, 0.98),
        "target_capture_accuracy": normalize(float(trainable["target_capture_accuracy"]), 0.95, 0.99),
    }
    variable_planner_score = mean(variable_planner.values())

    tool_failure_recovery = {
        "qwen_success_gain": normalize(float(hard_gain["qwen_joint_minus_tool_head_success"]), 0.04, 0.10),
        "deepseek_success_gain": normalize(float(hard_gain["deepseek_joint_minus_tool_head_success"]), 0.04, 0.08),
        "qwen_trigger_reduction": normalize(float(hard_gain["qwen_tool_head_minus_joint_trigger_rate"]), 0.03, 0.18),
        "deepseek_trigger_reduction": normalize(float(hard_gain["deepseek_tool_head_minus_joint_trigger_rate"]), 0.03, 0.08),
        "qwen_recovery_gain": normalize(float(recovery["gains"]["qwen_online_recovery_gain"]), 0.10, 0.22),
        "deepseek_recovery_gain": normalize(float(recovery["gains"]["deepseek_online_recovery_gain"]), 0.10, 0.22),
        "qwen_recovery_rate": normalize(float(recovery_headline["qwen_recovery_rate"]), 0.35, 0.65),
        "deepseek_recovery_rate": normalize(float(recovery_headline["deepseek_recovery_rate"]), 0.35, 0.65),
    }
    tool_failure_recovery_score = mean(tool_failure_recovery.values())

    training_anchor = {
        "stage6b_overall": float(stage6b_headline["overall_stage6b_score"]),
        "compressed_core": float(stage6b_headline["compressed_core_score"]),
        "online_carryover": float(stage6b_headline["online_carryover_score"]),
        "cocalibration": float(stage6b_headline["cocalibration_score"]),
    }
    training_anchor_score = mean(training_anchor.values())

    overall_score = mean(
        [
            long_horizon_goal_score,
            subgoal_chain_score,
            variable_planner_score,
            tool_failure_recovery_score,
            training_anchor_score,
        ]
    )

    hypotheses = {
        "H1_long_horizon_goal_maintenance_is_nontrivial": bool(long_horizon_goal_score >= 0.66),
        "H2_subgoal_chain_planning_is_nontrivial": bool(subgoal_chain_score >= 0.66),
        "H3_variable_planner_stays_strong": bool(variable_planner_score >= 0.78),
        "H4_tool_failure_and_recovery_are_integrated": bool(tool_failure_recovery_score >= 0.56),
        "H5_stage6c_long_horizon_open_environment_is_moderately_closed": bool(overall_score >= 0.69),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage6c_long_horizon_open_environment_closure",
        },
        "pillars": {
            "long_horizon_goal": {"components": long_horizon_goal, "score": float(long_horizon_goal_score)},
            "subgoal_chain": {"components": subgoal_chain, "score": float(subgoal_chain_score)},
            "variable_planner": {"components": variable_planner, "score": float(variable_planner_score)},
            "tool_failure_recovery": {
                "components": tool_failure_recovery,
                "score": float(tool_failure_recovery_score),
            },
            "training_anchor": {"components": training_anchor, "score": float(training_anchor_score)},
        },
        "headline_metrics": {
            "long_horizon_goal_score": float(long_horizon_goal_score),
            "subgoal_chain_score": float(subgoal_chain_score),
            "variable_planner_score": float(variable_planner_score),
            "tool_failure_recovery_score": float(tool_failure_recovery_score),
            "training_anchor_score": float(training_anchor_score),
            "overall_stage6c_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 6C is positive only if the compressed training loop survives long-horizon goal maintenance, "
                "subgoal chaining, variable planning, tool failure, and recovery within one open-environment view."
            ),
            "next_question": (
                "If this stage is positive, the next step is stage 6D: turn brain-side constraints into a direct "
                "freedom-reduction mechanism for the compressed core instead of keeping them as external penalties."
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
