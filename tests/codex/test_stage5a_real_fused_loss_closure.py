#!/usr/bin/env python
"""
Score whether the stage-5 fused objective can be interpreted as a real fused loss
once D, real-task, and brain-side calibration layers are brought back together.
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
    ap = argparse.ArgumentParser(description="Stage 5A real fused loss closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage5a_real_fused_loss_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage5 = load_json(ROOT / "tests" / "codex_temp" / "stage5_fused_unified_law_objective_20260311.json")
    task2 = load_json(ROOT / "tests" / "codex_temp" / "task_block_2_unified_training_closure_20260311.json")
    d_real = load_json(ROOT / "tests" / "codex_temp" / "d_real_task_cocalibrated_two_layer_unified_law_20260310.json")
    brain_d_real = load_json(ROOT / "tests" / "codex_temp" / "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json")

    d_law = d_real["cocalibrated_two_layer_law"]
    brain_law = brain_d_real["brain_d_real_cocalibrated_two_layer_law"]
    stage5_best = stage5["best_config"]

    d_alignment = {
        "mean_gap": inv_gap_score(float(d_law["mean_absolute_gap"]), 0.0010, 0.0100),
        "held_out_gap": inv_gap_score(float(d_law["held_out_mean_gap"]), 0.0010, 0.0120),
        "score_corr": normalize(float(d_law["score_correlation"]), 0.9700, 1.0),
        "held_out_corr": normalize(float(d_law["held_out_score_correlation"]), 0.9700, 1.0),
        "real_task_gap": inv_gap_score(float(d_law["real_task_mean_gap"]), 0.0004, 0.0030),
    }
    d_alignment_score = mean(d_alignment.values())

    brain_alignment = {
        "mean_gap": inv_gap_score(float(brain_law["mean_absolute_gap"]), 0.0010, 0.0120),
        "held_out_gap": inv_gap_score(float(brain_law["held_out_mean_gap"]), 0.0010, 0.0180),
        "score_corr": normalize(float(brain_law["score_correlation"]), 0.9800, 1.0),
        "held_out_corr": normalize(float(brain_law["held_out_score_correlation"]), 0.9800, 1.0),
        "brain_held_out_gap": inv_gap_score(float(brain_law["brain_held_out_gap"]), 0.0050, 0.0400),
    }
    brain_alignment_score = mean(brain_alignment.values())

    fused_guard = {
        "stage5_fused_score": normalize(float(stage5_best["fused_score"]), 0.60, 0.76),
        "stage5_training_score": normalize(float(stage5_best["training_score"]), 0.60, 0.67),
        "stage5_online_score": normalize(float(stage5_best["online_score"]), 0.54, 0.65),
        "stage5_brain_score": normalize(float(stage5_best["brain_score"]), 0.68, 0.76),
        "imbalance_guard": inv_gap_score(float(stage5_best["imbalance"]), 0.03, 0.12),
    }
    fused_guard_score = mean(fused_guard.values())

    structure_guard = {
        "task2_structure_score": normalize(
            float(task2["headline_metrics"]["structure_training_score"]),
            0.54,
            0.64,
        ),
        "task2_recovery_score": normalize(
            float(task2["headline_metrics"]["recovery_training_score"]),
            0.55,
            0.72,
        ),
        "task2_generator_score": normalize(
            float(task2["headline_metrics"]["generator_training_score"]),
            0.58,
            0.70,
        ),
    }
    structure_guard_score = mean(structure_guard.values())

    overall_score = mean(
        [d_alignment_score, brain_alignment_score, fused_guard_score, structure_guard_score]
    )

    hypotheses = {
        "H1_d_real_two_layer_alignment_is_strong": bool(d_alignment_score >= 0.82),
        "H2_brain_d_real_two_layer_alignment_is_strong": bool(brain_alignment_score >= 0.84),
        "H3_stage5_proxy_weights_transfer_to_real_alignment_view": bool(fused_guard_score >= 0.68),
        "H4_real_fused_loss_keeps_nontrivial_structure_guard": bool(structure_guard_score >= 0.42),
        "H5_stage5a_real_fused_loss_is_moderately_closed": bool(overall_score >= 0.66),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage5a_real_fused_loss_closure",
        },
        "pillars": {
            "d_real_alignment": {"components": d_alignment, "score": float(d_alignment_score)},
            "brain_d_real_alignment": {"components": brain_alignment, "score": float(brain_alignment_score)},
            "stage5_fused_guard": {"components": fused_guard, "score": float(fused_guard_score)},
            "structure_guard": {"components": structure_guard, "score": float(structure_guard_score)},
        },
        "headline_metrics": {
            "d_real_alignment_score": float(d_alignment_score),
            "brain_d_real_alignment_score": float(brain_alignment_score),
            "fused_guard_score": float(fused_guard_score),
            "structure_guard_score": float(structure_guard_score),
            "overall_stage5a_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 5A treats the proxy fused objective as acceptable only if D-side, real-task, "
                "and brain-side calibration stay jointly aligned when pulled back into one scorecard."
            ),
            "next_question": (
                "If this stage is positive, the next hard step is not more proxy search but explicit "
                "structure reinforcement inside the same fused training view."
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
