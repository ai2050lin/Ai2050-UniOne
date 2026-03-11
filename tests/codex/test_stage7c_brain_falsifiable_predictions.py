#!/usr/bin/env python
"""
Turn the explicit stage-7 coding-law candidate into brain-side falsifiable
predictions and score whether existing brain-bridge results support them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    ap = argparse.ArgumentParser(description="Stage 7C brain falsifiable predictions")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage7c_brain_falsifiable_predictions_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    brain_bridge = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    coverage = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_candidate_coverage_expansion_20260310.json"
    )
    augmentation = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_augmentation_stability_20260310.json"
    )
    expansion = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_constraint_expansion_20260310.json"
    )

    top3_components: List[str] = [
        row["component"] for row in stage7a["candidate_coding_law"]["strongest_brain_components"]
    ]
    baseline_gap = float(coverage["baseline_semantic_4d_vector"]["brain_held_out_gap"])
    light_gap = float(coverage["light_mix_best"]["brain_held_out_gap"])
    focused_gap = float(coverage["best_config"]["brain_held_out_gap"])
    broad_gap = float(expansion["brain_constraint_expansion_leave_one_out"]["brain_held_out_gap"])
    augmentation_gap = float(augmentation["brain_augmented_leave_one_out"]["brain_held_out_gap"])
    baseline_mean_gap = float(coverage["baseline_semantic_4d_vector"]["held_out_mean_gap"])
    light_mean_gap = float(coverage["light_mix_best"]["mean_held_out_gap"])
    broad_mean_gap = float(expansion["brain_constraint_expansion_leave_one_out"]["mean_held_out_gap"])

    selective_constraint_prediction = {
        "light_focus_beats_baseline": normalize(baseline_gap - light_gap, 0.02, 0.04),
        "focused_coverage_beats_baseline": normalize(baseline_gap - focused_gap, 0.025, 0.04),
        "broad_expansion_hurts_vs_baseline": normalize(broad_gap - baseline_gap, 0.003, 0.012),
        "light_focus_beats_broad_expansion": normalize(broad_gap - light_gap, 0.03, 0.05),
        "light_mean_gap_beats_broad": normalize(broad_mean_gap - light_mean_gap, 0.001, 0.002),
    }
    selective_constraint_score = mean(selective_constraint_prediction.values())

    augmentation_prediction = {
        "brain_gap_improvement": normalize(float(augmentation["improvement"]["brain_gap_improvement"]), 0.03, 0.06),
        "mean_gap_improvement": normalize(float(augmentation["improvement"]["mean_gap_improvement"]), 0.0008, 0.0014),
        "corr_improvement": normalize(float(augmentation["improvement"]["corr_improvement"]), 0.0015, 0.0030),
        "augmentation_pass": 1.0 if bool(augmentation["brain_augmented_leave_one_out"]["pass"]) else 0.0,
        "augmentation_beats_focused_gap": normalize(focused_gap - augmentation_gap, 0.01, 0.02),
    }
    augmentation_prediction_score = mean(augmentation_prediction.values())

    top3_model_means = {}
    rest_model_means = {}
    for model_name, model_row in brain_bridge["models"].items():
        component_scores = {
            component_name: float(component_row["score"])
            for component_name, component_row in model_row["components"].items()
        }
        top_values = [component_scores[name] for name in top3_components]
        rest_values = [score for name, score in component_scores.items() if name not in top3_components]
        top3_model_means[model_name] = mean(top_values)
        rest_model_means[model_name] = mean(rest_values)

    bridge_alignment_prediction = {
        "gpt2_top3_advantage": normalize(top3_model_means["gpt2"] - rest_model_means["gpt2"], 0.08, 0.20),
        "qwen_top3_advantage": normalize(
            top3_model_means["qwen3_4b"] - rest_model_means["qwen3_4b"],
            0.02,
            0.12,
        ),
        "top3_weight_sum": normalize(
            sum(stage7a["candidate_coding_law"]["normalized_brain_weights"][name] for name in top3_components),
            0.34,
            0.42,
        ),
        "bridge_score_mean": normalize(
            mean(
                [
                    float(brain_bridge["models"]["gpt2"]["overall_bridge_score"]),
                    float(brain_bridge["models"]["qwen3_4b"]["overall_bridge_score"]),
                ]
            ),
            0.78,
            0.83,
        ),
    }
    bridge_alignment_score = mean(bridge_alignment_prediction.values())

    falsifiability_quality = {
        "prediction_1_selective_constraints_help": 1.0 if light_gap < baseline_gap else 0.0,
        "prediction_2_broad_constraints_can_hurt": 1.0 if broad_gap > baseline_gap else 0.0,
        "prediction_3_augmentation_should_help": 1.0 if augmentation_gap < focused_gap else 0.0,
        "prediction_4_top3_components_dominate": 1.0
        if min(
            top3_model_means["gpt2"] - rest_model_means["gpt2"],
            top3_model_means["qwen3_4b"] - rest_model_means["qwen3_4b"],
        )
        > 0.0
        else 0.0,
        "prediction_5_has_directional_sign_change": 1.0
        if (baseline_gap - light_gap) > 0.0 and (baseline_gap - broad_gap) < 0.0
        else 0.0,
    }
    falsifiability_quality_score = mean(falsifiability_quality.values())

    overall_score = mean(
        [
            selective_constraint_score,
            augmentation_prediction_score,
            bridge_alignment_score,
            falsifiability_quality_score,
        ]
    )

    observed_predictions = {
        "law_prediction": (
            "Brain-side support should be selective rather than monotonic: focused constraints and augmentation "
            "should lower brain held-out gap, while broad unfocused expansion can raise it."
        ),
        "observed_brain_held_out_gaps": {
            "baseline": baseline_gap,
            "light_focus": light_gap,
            "focused_coverage": focused_gap,
            "augmentation": augmentation_gap,
            "broad_expansion": broad_gap,
        },
        "top3_brain_components": top3_components,
        "top3_component_bridge_means": top3_model_means,
        "rest_component_bridge_means": rest_model_means,
    }

    hypotheses = {
        "H1_selective_brain_constraints_beat_broad_expansion": bool(selective_constraint_score >= 0.56),
        "H2_brain_augmentation_supports_sample_thin_hypothesis": bool(augmentation_prediction_score >= 0.78),
        "H3_stage7a_top3_brain_prior_aligns_with_bridge_structure": bool(bridge_alignment_score >= 0.72),
        "H4_stage7_candidate_generates_genuinely_falsifiable_predictions": bool(
            falsifiability_quality_score >= 0.95
        ),
        "H5_stage7c_brain_falsifiable_predictions_are_moderately_supported": bool(overall_score >= 0.76),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage7c_brain_falsifiable_predictions",
        },
        "observed_predictions": observed_predictions,
        "pillars": {
            "selective_constraint_prediction": {
                "components": selective_constraint_prediction,
                "score": float(selective_constraint_score),
            },
            "augmentation_prediction": {
                "components": augmentation_prediction,
                "score": float(augmentation_prediction_score),
            },
            "bridge_alignment_prediction": {
                "components": bridge_alignment_prediction,
                "score": float(bridge_alignment_score),
            },
            "falsifiability_quality": {
                "components": falsifiability_quality,
                "score": float(falsifiability_quality_score),
            },
        },
        "headline_metrics": {
            "selective_constraint_score": float(selective_constraint_score),
            "augmentation_prediction_score": float(augmentation_prediction_score),
            "bridge_alignment_score": float(bridge_alignment_score),
            "falsifiability_quality_score": float(falsifiability_quality_score),
            "overall_stage7c_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 7C is positive only if the explicit coding law generates brain-side predictions that could have "
                "failed in a directional way: selective support should help, broad expansion can hurt, and the top "
                "brain-prior components should align with the bridge structure."
            ),
            "next_question": (
                "If this stage holds, the next step is a truth-status master view: is the coding law merely plausible, "
                "or has it become the best current mechanism guess?"
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
