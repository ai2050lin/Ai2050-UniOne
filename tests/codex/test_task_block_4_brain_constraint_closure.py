#!/usr/bin/env python
"""
Score whether task block 4 (brain-side constraint front-loading) is sufficiently closed.
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
    ap = argparse.ArgumentParser(description="Task block 4 brain constraint closure scorecard")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/task_block_4_brain_constraint_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    coverage = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_brain_candidate_coverage_expansion_20260310.json")
    augmentation = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_brain_augmentation_stability_20260310.json")
    expansion = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_brain_constraint_expansion_20260310.json")
    learnable = load_json(ROOT / "tests" / "codex_temp" / "brain_learnable_ranking_two_layer_unified_law_20260310.json")
    cocalibrated = load_json(ROOT / "tests" / "codex_temp" / "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json")
    modality = load_json(ROOT / "tests" / "codex_temp" / "parameterized_shared_modality_law_20260310.json")

    pillars = {
        "candidate_coverage": {
            "components": {
                "brain_gap_improvement": normalize(float(coverage["best_config"]["brain_gap_improvement_vs_baseline"]), 0.01, 0.04),
                "held_out_gap_gain": normalize(float(coverage["best_config"]["mean_gap_improvement_vs_baseline"]), 0.0002, 0.0010),
                "held_out_corr": normalize(float(coverage["best_config"]["held_out_score_correlation"]), 0.998, 1.0),
                "constraint_density": normalize(float(coverage["best_config"]["mean_focus_count"]), 2.0, 4.2),
            }
        },
        "augmentation_stability": {
            "components": {
                "brain_gap_improvement": normalize(float(augmentation["improvement"]["brain_gap_improvement"]), 0.02, 0.055),
                "mean_gap_improvement": normalize(float(augmentation["improvement"]["mean_gap_improvement"]), 0.0004, 0.0014),
                "brain_held_out_gap": normalize(0.06 - float(augmentation["brain_augmented_leave_one_out"]["brain_held_out_gap"]), 0.0, 0.06),
                "held_out_corr": normalize(float(augmentation["brain_augmented_leave_one_out"]["held_out_score_correlation"]), 0.998, 1.0),
            }
        },
        "brain_learnable_ranking": {
            "components": {
                "brain_gap_improvement": normalize(float(learnable["brain_learnable_ranking_two_layer_law"]["brain_gap_improvement"]), 0.001, 0.008),
                "brain_held_out_improvement": normalize(float(learnable["brain_learnable_ranking_two_layer_law"]["brain_held_out_improvement"]), 0.001, 0.003),
                "held_out_corr": normalize(float(learnable["brain_learnable_ranking_two_layer_law"]["held_out_score_correlation"]), 0.9985, 1.0),
                "brain_mean_gap": normalize(0.01 - float(learnable["brain_learnable_ranking_two_layer_law"]["brain_mean_gap"]), 0.0, 0.01),
            }
        },
        "brain_d_real_cocalibration": {
            "components": {
                "held_out_corr": normalize(float(cocalibrated["brain_d_real_cocalibrated_two_layer_law"]["held_out_score_correlation"]), 0.9985, 1.0),
                "brain_held_out_gap": normalize(0.02 - float(cocalibrated["brain_d_real_cocalibrated_two_layer_law"]["brain_held_out_gap"]), 0.0, 0.02),
                "brain_mean_gap": normalize(0.01 - float(cocalibrated["brain_d_real_cocalibrated_two_layer_law"]["brain_mean_gap"]), 0.0, 0.01),
                "overall_mean_gap": normalize(0.003 - float(cocalibrated["brain_d_real_cocalibrated_two_layer_law"]["held_out_mean_gap"]), 0.0, 0.003),
            }
        },
        "parameterized_modality": {
            "components": {
                "mean_gap": normalize(0.0045 - float(modality["parameterized_shared_law"]["mean_held_out_gap"]), 0.0, 0.002),
                "held_out_corr": normalize(float(modality["parameterized_shared_law"]["held_out_score_correlation"]), 0.988, 0.992),
                "tactile_gap": normalize(0.0045 - float(modality["parameterized_shared_law"]["modality_held_out_gap"]["tactile"]), 0.0, 0.003),
                "oracle_closeness": normalize(
                    float(modality["modality_separate_oracle"]["mean_held_out_gap"]) - float(modality["parameterized_shared_law"]["mean_held_out_gap"]),
                    0.0,
                    0.001,
                ),
            }
        },
        "constraint_expansion_negative_control": {
            "components": {
                "noncollapse": normalize(0.001 - abs(float(expansion["improvement"]["mean_gap_improvement"])), 0.0, 0.001),
                "brain_penalty_control": normalize(0.0 - float(expansion["improvement"]["brain_gap_improvement"]), -0.02, 0.01),
                "held_out_corr": normalize(float(expansion["brain_constraint_expansion_leave_one_out"]["held_out_score_correlation"]), 0.995, 1.0),
                "constraint_density": normalize(float(expansion["brain_constraint_expansion_leave_one_out"]["mean_constraint_count"]), 10.0, 21.0),
            }
        },
    }

    for row in pillars.values():
        row["score"] = mean(row["components"].values())

    positive_score = mean(
        [
            pillars["candidate_coverage"]["score"],
            pillars["augmentation_stability"]["score"],
            pillars["brain_learnable_ranking"]["score"],
            pillars["brain_d_real_cocalibration"]["score"],
            pillars["parameterized_modality"]["score"],
        ]
    )
    robustness_score = mean(
        [
            pillars["constraint_expansion_negative_control"]["score"],
            pillars["augmentation_stability"]["score"],
        ]
    )
    overall_score = mean([positive_score, robustness_score])

    hypotheses = {
        "H1_brain_candidate_coverage_improves_brain_gap": bool(pillars["candidate_coverage"]["score"] >= 0.55),
        "H2_brain_augmentation_is_stable": bool(pillars["augmentation_stability"]["score"] >= 0.70),
        "H3_brain_learnable_ranking_is_nontrivial": bool(pillars["brain_learnable_ranking"]["score"] >= 0.60),
        "H4_same_mechanism_different_modality_params_is_viable": bool(pillars["parameterized_modality"]["score"] >= 0.45),
        "H5_task_block_4_is_moderately_closed": bool(overall_score >= 0.52),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "task_block_4_brain_constraint_closure",
        },
        "pillars": pillars,
        "headline_metrics": {
            "brain_positive_score": float(positive_score),
            "brain_robustness_score": float(robustness_score),
            "overall_task_block_4_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Task block 4 is treated as closed only if brain-side candidates improve held-out brain fit without "
                "collapsing D-side or real-task terms, and if parameterized modality variants remain viable."
            ),
            "next_question": (
                "If this scorecard is positive, the next step is to turn these brain-side terms from post-hoc scores "
                "into actual training penalties or controller updates in the unified law."
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
