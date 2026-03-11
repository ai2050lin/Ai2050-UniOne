#!/usr/bin/env python
"""
Score whether hard online failure types can be treated as training-integrated
penalties instead of a separate post-hoc benchmark.
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
    ap = argparse.ArgumentParser(description="Stage 5C online failure integrated training closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5c_online_failure_integrated_training_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage5 = load_json(ROOT / "tests" / "codex_temp" / "stage5_fused_unified_law_objective_20260311.json")
    hard_interface = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json")
    learnable = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_learnable_stage_heads_smoke_20260311.json"
    )
    recovery = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
    control = load_json(ROOT / "tests" / "codex_temp" / "real_multistep_unified_control_manifold_benchmark_20260310.json")

    hard_gain = hard_interface["gains"]
    learnable_gain = learnable["gains"]
    recovery_metrics = recovery["headline_metrics"]
    control_metrics = control["headline_metrics"]
    stage5_best = stage5["best_config"]

    hard_interface_score_parts = {
        "qwen_success_gain": normalize(float(hard_gain["qwen_joint_minus_tool_head_success"]), 0.04, 0.10),
        "deepseek_success_gain": normalize(float(hard_gain["deepseek_joint_minus_tool_head_success"]), 0.04, 0.10),
        "qwen_trigger_reduction": normalize(
            float(hard_gain["qwen_tool_head_minus_joint_trigger_rate"]),
            0.03,
            0.18,
        ),
        "deepseek_trigger_reduction": normalize(
            float(hard_gain["deepseek_tool_head_minus_joint_trigger_rate"]),
            0.03,
            0.18,
        ),
    }
    hard_interface_score = mean(hard_interface_score_parts.values())

    learnable_heads_score_parts = {
        "qwen_success_gain": normalize(float(learnable_gain["qwen_learned_minus_fixed_success"]), 0.05, 0.20),
        "deepseek_success_gain": normalize(
            float(learnable_gain["deepseek_learned_minus_fixed_success"]),
            0.05,
            0.20,
        ),
        "qwen_trigger_reduction": normalize(
            float(learnable_gain["qwen_fixed_minus_learned_trigger_rate"]),
            0.02,
            0.20,
        ),
        "deepseek_trigger_reduction": normalize(
            float(learnable_gain["deepseek_fixed_minus_learned_trigger_rate"]),
            0.02,
            0.20,
        ),
        "qwen_tool_failure_reduction": normalize(
            float(learnable_gain["qwen_fixed_minus_learned_tool_failure"]),
            0.01,
            0.05,
        ),
        "deepseek_tool_failure_reduction": normalize(
            float(learnable_gain["deepseek_fixed_minus_learned_tool_failure"]),
            0.01,
            0.05,
        ),
    }
    learnable_heads_score = mean(learnable_heads_score_parts.values())

    recovery_score_parts = {
        "qwen_recovery_gain": normalize(float(recovery["gains"]["qwen_online_recovery_gain"]), 0.10, 0.22),
        "deepseek_recovery_gain": normalize(
            float(recovery["gains"]["deepseek_online_recovery_gain"]),
            0.10,
            0.22,
        ),
        "qwen_recovery_rate": normalize(float(recovery_metrics["qwen_recovery_rate"]), 0.35, 0.65),
        "deepseek_recovery_rate": normalize(float(recovery_metrics["deepseek_recovery_rate"]), 0.35, 0.65),
    }
    recovery_score = mean(recovery_score_parts.values())

    control_score_parts = {
        "max_length_unified_score": normalize(float(control_metrics["max_length_unified_score"]), 0.34, 0.40),
        "max_length_episode_success": normalize(
            float(control_metrics["max_length_episode_success"]),
            0.18,
            0.28,
        ),
        "max_length_recovery_rate": normalize(
            float(control_metrics["max_length_recovery_rate"]),
            0.08,
            0.15,
        ),
        "max_length_retention": normalize(float(control_metrics["max_length_retention"]), 0.30, 0.40),
    }
    control_score = mean(control_score_parts.values())

    fused_online_guard = {
        "stage5_online_score": normalize(float(stage5_best["online_score"]), 0.54, 0.65),
        "stage5_imbalance_guard": normalize(
            float(0.12 - min(0.12, stage5_best["imbalance"])),
            0.0,
            0.09,
        ),
        "stage5_upgrade_share": normalize(float(stage5_best["online_weights"]["upgrade"]), 0.40, 0.65),
    }
    fused_online_guard_score = mean(fused_online_guard.values())

    overall_score = mean(
        [
            hard_interface_score,
            learnable_heads_score,
            recovery_score,
            control_score,
            fused_online_guard_score,
        ]
    )

    hypotheses = {
        "H1_hard_interface_penalties_are_nontrivial": bool(hard_interface_score >= 0.45),
        "H2_online_learnable_heads_are_nontrivial": bool(learnable_heads_score >= 0.50),
        "H3_recovery_chain_is_stably_positive": bool(recovery_score >= 0.56),
        "H4_unified_control_manifold_preserves_long_horizon_signal": bool(control_score >= 0.60),
        "H5_stage5c_online_failure_integration_is_moderately_closed": bool(overall_score >= 0.58),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5c_online_failure_integrated_training_closure",
        },
        "pillars": {
            "hard_interface": {"components": hard_interface_score_parts, "score": float(hard_interface_score)},
            "learnable_heads": {"components": learnable_heads_score_parts, "score": float(learnable_heads_score)},
            "recovery_chain": {"components": recovery_score_parts, "score": float(recovery_score)},
            "control_manifold": {"components": control_score_parts, "score": float(control_score)},
            "fused_online_guard": {"components": fused_online_guard, "score": float(fused_online_guard_score)},
        },
        "headline_metrics": {
            "hard_interface_score": float(hard_interface_score),
            "learnable_heads_score": float(learnable_heads_score),
            "recovery_chain_score": float(recovery_score),
            "control_manifold_score": float(control_score),
            "fused_online_guard_score": float(fused_online_guard_score),
            "overall_stage5c_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 5C is positive only if schema mismatch, timeout pressure, state drift, and recovery "
                "signals can all be treated as one integrated online-training pressure rather than disconnected tests."
            ),
            "next_question": (
                "If this stage is stable, the next hard step is to force the same fused objective to stay "
                "balanced across Qwen and DeepSeek rather than relying on one stronger route."
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
