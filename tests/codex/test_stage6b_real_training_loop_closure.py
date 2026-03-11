#!/usr/bin/env python
"""
Score whether the compressed causal core is ready to function as a real training
loop rather than remaining a scorecard-only abstraction.
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


def inv_gap_score(value: float, lo: float, hi: float) -> float:
    return float(1.0 - normalize(value, lo, hi))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6B real training loop closure scorecard")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage6b_real_training_loop_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage6a = load_json(ROOT / "tests" / "codex_temp" / "stage6a_causal_core_compression_20260311.json")
    stage5c = load_json(
        ROOT / "tests" / "codex_temp" / "stage5c_online_failure_integrated_training_closure_20260311.json"
    )
    learnable = load_json(ROOT / "tests" / "codex_temp" / "learnable_two_layer_unified_law_20260309.json")
    learnable_rank = load_json(
        ROOT / "tests" / "codex_temp" / "learnable_ranking_two_layer_unified_law_20260310.json"
    )
    real_task = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")
    d_real = load_json(ROOT / "tests" / "codex_temp" / "d_real_task_cocalibrated_two_layer_unified_law_20260310.json")
    brain = load_json(ROOT / "tests" / "codex_temp" / "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json")
    control = load_json(ROOT / "tests" / "codex_temp" / "real_multistep_unified_control_manifold_benchmark_20260310.json")

    compressed_core = {
        "stage6a_overall": float(stage6a["headline_metrics"]["overall_stage6a_score"]),
        "two_param_core": float(stage6a["headline_metrics"]["two_param_core_score"]),
        "minimal_interface": float(stage6a["headline_metrics"]["minimal_interface_score"]),
        "causal_anchor": float(stage6a["headline_metrics"]["causal_anchor_score"]),
    }
    compressed_core_score = mean(compressed_core.values())

    trainable_core = {
        "learnable_gap": inv_gap_score(
            float(learnable["learnable_two_layer_law"]["mean_absolute_gap"]),
            0.01,
            0.05,
        ),
        "learnable_corr": normalize(
            float(learnable["learnable_two_layer_law"]["score_correlation"]),
            0.95,
            0.995,
        ),
        "ranking_gap": inv_gap_score(
            float(learnable_rank["learnable_ranking_two_layer_law"]["mean_absolute_gap"]),
            0.001,
            0.02,
        ),
        "ranking_corr": normalize(
            float(learnable_rank["learnable_ranking_two_layer_law"]["score_correlation"]),
            0.98,
            1.0,
        ),
        "ranking_held_out_gap": inv_gap_score(
            float(learnable_rank["learnable_ranking_two_layer_law"]["held_out_mean_gap"]),
            0.005,
            0.02,
        ),
    }
    trainable_core_score = mean(trainable_core.values())

    real_task_loop = {
        "real_task_gap": inv_gap_score(
            float(real_task["real_task_two_layer_law"]["mean_absolute_gap"]),
            0.008,
            0.02,
        ),
        "real_task_corr": normalize(
            float(real_task["real_task_two_layer_law"]["score_correlation"]),
            0.78,
            0.86,
        ),
        "real_task_held_out_gap": inv_gap_score(
            float(real_task["real_task_two_layer_law"]["held_out_mean_gap"]),
            0.010,
            0.02,
        ),
        "real_task_held_out_corr": normalize(
            float(real_task["real_task_two_layer_law"]["held_out_score_correlation"]),
            0.75,
            0.82,
        ),
        "correlation_improvement_vs_baseline": normalize(
            float(real_task["real_task_two_layer_law"]["correlation_improvement_vs_baseline"]),
            0.03,
            0.10,
        ),
    }
    real_task_loop_score = mean(real_task_loop.values())

    cocalibration = {
        "d_real_gap": inv_gap_score(
            float(d_real["cocalibrated_two_layer_law"]["mean_absolute_gap"]),
            0.001,
            0.01,
        ),
        "d_real_held_out_gap": inv_gap_score(
            float(d_real["cocalibrated_two_layer_law"]["held_out_mean_gap"]),
            0.001,
            0.01,
        ),
        "brain_gap": inv_gap_score(
            float(brain["brain_d_real_cocalibrated_two_layer_law"]["mean_absolute_gap"]),
            0.001,
            0.01,
        ),
        "brain_held_out_gap": inv_gap_score(
            float(brain["brain_d_real_cocalibrated_two_layer_law"]["held_out_mean_gap"]),
            0.001,
            0.01,
        ),
        "brain_corr": normalize(
            float(brain["brain_d_real_cocalibrated_two_layer_law"]["held_out_score_correlation"]),
            0.98,
            1.0,
        ),
    }
    cocalibration_score = mean(cocalibration.values())

    online_carryover = {
        "stage5c_score": normalize(float(stage5c["headline_metrics"]["overall_stage5c_score"]), 0.58, 0.66),
        "control_unified_score": normalize(
            float(control["headline_metrics"]["max_length_unified_score"]),
            0.34,
            0.40,
        ),
        "control_recovery_rate": normalize(
            float(control["headline_metrics"]["max_length_recovery_rate"]),
            0.08,
            0.15,
        ),
        "control_retention": normalize(float(control["headline_metrics"]["max_length_retention"]), 0.30, 0.40),
    }
    online_carryover_score = mean(online_carryover.values())

    overall_score = mean(
        [
            compressed_core_score,
            trainable_core_score,
            real_task_loop_score,
            cocalibration_score,
            online_carryover_score,
        ]
    )

    hypotheses = {
        "H1_compressed_core_is_ready_for_training_loop": bool(compressed_core_score >= 0.72),
        "H2_trainable_core_is_strong": bool(trainable_core_score >= 0.80),
        "H3_real_task_loop_is_nontrivial": bool(real_task_loop_score >= 0.58),
        "H4_cocalibration_stays_strong_inside_training_view": bool(cocalibration_score >= 0.84),
        "H5_online_carryover_is_nontrivial": bool(online_carryover_score >= 0.58),
        "H6_stage6b_real_training_loop_is_moderately_closed": bool(overall_score >= 0.70),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage6b_real_training_loop_closure",
        },
        "pillars": {
            "compressed_core": {"components": compressed_core, "score": float(compressed_core_score)},
            "trainable_core": {"components": trainable_core, "score": float(trainable_core_score)},
            "real_task_loop": {"components": real_task_loop, "score": float(real_task_loop_score)},
            "cocalibration": {"components": cocalibration, "score": float(cocalibration_score)},
            "online_carryover": {"components": online_carryover, "score": float(online_carryover_score)},
        },
        "headline_metrics": {
            "compressed_core_score": float(compressed_core_score),
            "trainable_core_score": float(trainable_core_score),
            "real_task_loop_score": float(real_task_loop_score),
            "cocalibration_score": float(cocalibration_score),
            "online_carryover_score": float(online_carryover_score),
            "overall_stage6b_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 6B is positive only if the compressed causal core stays trainable, transfers to real-task "
                "rows, remains cocalibrated with D and brain constraints, and does not obviously drop online pressure."
            ),
            "next_question": (
                "If this stage holds, the next step is a larger stage-6C environment where long-horizon task flow, "
                "tool failure, and recovery are all driven by the same compressed training loop."
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
