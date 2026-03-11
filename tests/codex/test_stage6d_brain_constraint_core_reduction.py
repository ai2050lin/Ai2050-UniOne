#!/usr/bin/env python
"""
Score whether brain-side candidates directly reduce the freedom of the compressed
core rather than remaining external penalties.
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
    ap = argparse.ArgumentParser(description="Stage 6D brain constraint core reduction")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage6d_brain_constraint_core_reduction_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    task4 = load_json(ROOT / "tests" / "codex_temp" / "task_block_4_brain_constraint_closure_20260311.json")
    brain_rank = load_json(
        ROOT / "tests" / "codex_temp" / "brain_learnable_ranking_two_layer_unified_law_20260310.json"
    )
    brain_d_real = load_json(ROOT / "tests" / "codex_temp" / "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json")
    stage6a = load_json(ROOT / "tests" / "codex_temp" / "stage6a_causal_core_compression_20260311.json")
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    stage6c = load_json(ROOT / "tests" / "codex_temp" / "stage6c_long_horizon_open_environment_closure_20260311.json")

    task4_pillars = task4["pillars"]
    brain_rank_law = brain_rank["brain_learnable_ranking_two_layer_law"]
    brain_coeffs = brain_rank["ranking_layer"]["brain_component_coefficients"]
    coeff_values = [float(v) for v in brain_coeffs.values()]
    coeff_sum = sum(coeff_values)
    top3_ratio = float(sum(sorted(coeff_values, reverse=True)[:3]) / max(1e-9, coeff_sum))

    brain_constraint_quality = {
        "brain_positive": float(task4["headline_metrics"]["brain_positive_score"]),
        "brain_robustness": float(task4["headline_metrics"]["brain_robustness_score"]),
        "candidate_coverage": float(task4_pillars["candidate_coverage"]["score"]),
        "augmentation_stability": float(task4_pillars["augmentation_stability"]["score"]),
        "brain_learnable_ranking": float(task4_pillars["brain_learnable_ranking"]["score"]),
    }
    brain_constraint_quality_score = mean(brain_constraint_quality.values())

    learnable_brain_reduction = {
        "brain_mean_gap": inv_gap_score(float(brain_rank_law["brain_mean_gap"]), 0.0, 0.01),
        "brain_held_out_gap": inv_gap_score(float(brain_rank_law["brain_held_out_gap"]), 0.01, 0.03),
        "held_out_corr": normalize(float(brain_rank_law["held_out_score_correlation"]), 0.99, 1.0),
        "brain_gap_improvement": normalize(float(brain_rank_law["brain_gap_improvement"]), 0.0, 0.01),
        "top3_component_ratio": normalize(top3_ratio, 0.40, 0.60),
    }
    learnable_brain_reduction_score = mean(learnable_brain_reduction.values())

    brain_d_real_alignment = {
        "mean_gap": inv_gap_score(
            float(brain_d_real["brain_d_real_cocalibrated_two_layer_law"]["mean_absolute_gap"]),
            0.001,
            0.01,
        ),
        "held_out_gap": inv_gap_score(
            float(brain_d_real["brain_d_real_cocalibrated_two_layer_law"]["held_out_mean_gap"]),
            0.001,
            0.01,
        ),
        "brain_held_out_gap": inv_gap_score(
            float(brain_d_real["brain_d_real_cocalibrated_two_layer_law"]["brain_held_out_gap"]),
            0.01,
            0.03,
        ),
        "held_out_corr": normalize(
            float(brain_d_real["brain_d_real_cocalibrated_two_layer_law"]["held_out_score_correlation"]),
            0.99,
            1.0,
        ),
        "brain_alignment_score": normalize(
            float(brain_d_real["brain_breakdown"][0]["brain_alignment_score"]),
            0.85,
            0.95,
        ),
    }
    brain_d_real_alignment_score = mean(brain_d_real_alignment.values())

    core_freedom_reduction = {
        "two_param_core": float(stage6a["headline_metrics"]["two_param_core_score"]),
        "compressed_core": float(stage6b["headline_metrics"]["compressed_core_score"]),
        "training_anchor": float(stage6c["headline_metrics"]["training_anchor_score"]),
        "compression_ratio_to_core": float(
            stage6a["pillars"]["two_param_core_candidate"]["components"]["compression_ratio_to_core"]
        ),
        "component_concentration": normalize(top3_ratio, 0.40, 0.60),
    }
    core_freedom_reduction_score = mean(core_freedom_reduction.values())

    modality_robustness = {
        "parameterized_modality": float(task4_pillars["parameterized_modality"]["score"]),
        "negative_control": float(task4_pillars["constraint_expansion_negative_control"]["score"]),
        "stage6c_tool_failure_recovery": float(stage6c["headline_metrics"]["tool_failure_recovery_score"]),
        "brain_robustness": float(task4["headline_metrics"]["brain_robustness_score"]),
    }
    modality_robustness_score = mean(modality_robustness.values())

    overall_score = mean(
        [
            brain_constraint_quality_score,
            learnable_brain_reduction_score,
            brain_d_real_alignment_score,
            core_freedom_reduction_score,
            modality_robustness_score,
        ]
    )

    hypotheses = {
        "H1_brain_constraint_quality_is_nontrivial": bool(brain_constraint_quality_score >= 0.72),
        "H2_learnable_brain_reduction_is_nontrivial": bool(learnable_brain_reduction_score >= 0.74),
        "H3_brain_d_real_alignment_stays_strong": bool(brain_d_real_alignment_score >= 0.84),
        "H4_brain_constraints_reduce_core_freedom_nontrivially": bool(core_freedom_reduction_score >= 0.74),
        "H5_stage6d_brain_constraint_core_reduction_is_moderately_closed": bool(overall_score >= 0.72),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage6d_brain_constraint_core_reduction",
        },
        "pillars": {
            "brain_constraint_quality": {
                "components": brain_constraint_quality,
                "score": float(brain_constraint_quality_score),
            },
            "learnable_brain_reduction": {
                "components": learnable_brain_reduction,
                "score": float(learnable_brain_reduction_score),
            },
            "brain_d_real_alignment": {
                "components": brain_d_real_alignment,
                "score": float(brain_d_real_alignment_score),
            },
            "core_freedom_reduction": {
                "components": core_freedom_reduction,
                "score": float(core_freedom_reduction_score),
            },
            "modality_robustness": {
                "components": modality_robustness,
                "score": float(modality_robustness_score),
            },
        },
        "headline_metrics": {
            "brain_constraint_quality_score": float(brain_constraint_quality_score),
            "learnable_brain_reduction_score": float(learnable_brain_reduction_score),
            "brain_d_real_alignment_score": float(brain_d_real_alignment_score),
            "core_freedom_reduction_score": float(core_freedom_reduction_score),
            "modality_robustness_score": float(modality_robustness_score),
            "overall_stage6d_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 6D is positive only if brain-side candidates stop acting like external bonuses and instead "
                "reduce the freedom of the compressed core in a learnable, cross-domain way."
            ),
            "next_question": (
                "If this stage is positive, the next move is not another bridge dashboard but a stage-6 master "
                "closure that marks the full phase complete."
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
