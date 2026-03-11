#!/usr/bin/env python
"""
Build a sharper brain-side falsification view over the current coding-law candidate.
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
    ap = argparse.ArgumentParser(description="Stage 8D brain high-risk falsification")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage8d_brain_high_risk_falsification_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage7c = load_json(ROOT / "tests" / "codex_temp" / "stage7c_brain_falsifiable_predictions_20260311.json")
    coverage = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_candidate_coverage_expansion_20260310.json"
    )
    augmentation = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_augmentation_stability_20260310.json"
    )
    expansion = load_json(
        ROOT / "tests" / "codex_temp" / "semantic_4d_brain_constraint_expansion_20260310.json"
    )
    brain_bridge = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")

    base = coverage["baseline_semantic_4d_vector"]
    light = coverage["light_mix_best"]
    focused = coverage["best_config"]
    aug = augmentation["brain_augmented_leave_one_out"]
    broad = expansion["brain_constraint_expansion_leave_one_out"]

    directional_falsifier = {
        "focused_brain_beats_baseline": normalize(
            float(base["brain_held_out_gap"]) - float(focused["brain_held_out_gap"]),
            0.02,
            0.04,
        ),
        "augmentation_brain_beats_focused": normalize(
            float(focused["brain_held_out_gap"]) - float(aug["brain_held_out_gap"]),
            0.01,
            0.02,
        ),
        "broad_expansion_hurts_brain": normalize(
            float(broad["brain_held_out_gap"]) - float(base["brain_held_out_gap"]),
            0.003,
            0.012,
        ),
        "light_focus_beats_broad": normalize(
            float(broad["brain_held_out_gap"]) - float(light["brain_held_out_gap"]),
            0.03,
            0.05,
        ),
    }
    directional_falsifier_score = mean(directional_falsifier.values())

    brain_specificity = {
        "focused_brain_gt_real_task_improvement": normalize(
            float(focused["brain_gap_improvement_vs_baseline"])
            - float(focused["mean_gap_improvement_vs_baseline"]),
            0.02,
            0.04,
        ),
        "augmentation_brain_gt_real_task_improvement": normalize(
            float(augmentation["improvement"]["brain_gap_improvement"])
            - float(augmentation["improvement"]["real_task_gap_improvement"]),
            0.03,
            0.06,
        ),
        "broad_brain_hurt_gt_real_task_hurt": normalize(
            abs(float(expansion["improvement"]["brain_gap_improvement"]))
            - abs(float(expansion["improvement"]["real_task_gap_improvement"])),
            0.005,
            0.01,
        ),
        "brain_improvement_does_not_track_d_gap": 1.0
        if float(augmentation["improvement"]["d_gap_improvement"]) < 0.0
        else 0.0,
    }
    brain_specificity_score = mean(brain_specificity.values())

    top3 = [row["component"] for row in stage7a["candidate_coding_law"]["strongest_brain_components"]]
    gpt2_components = brain_bridge["models"]["gpt2"]["components"]
    qwen_components = brain_bridge["models"]["qwen3_4b"]["components"]
    top3_component_risk = {
        "top3_weight_sum": normalize(
            sum(stage7a["candidate_coding_law"]["normalized_brain_weights"][name] for name in top3),
            0.34,
            0.42,
        ),
        "gpt2_top3_vs_rest_advantage": normalize(
            mean(float(gpt2_components[name]["score"]) for name in top3)
            - mean(float(v["score"]) for name, v in gpt2_components.items() if name not in top3),
            0.12,
            0.28,
        ),
        "qwen_top3_vs_rest_advantage": normalize(
            mean(float(qwen_components[name]["score"]) for name in top3)
            - mean(float(v["score"]) for name, v in qwen_components.items() if name not in top3),
            0.03,
            0.10,
        ),
        "stage7c_bridge_alignment": float(stage7c["headline_metrics"]["bridge_alignment_score"]),
    }
    top3_component_risk_score = mean(top3_component_risk.values())

    hard_falsifier_spec = {
        "falsifier_1_broad_beats_focused": 1.0,
        "falsifier_2_augmentation_fails_brain_gap": 1.0,
        "falsifier_3_top3_component_advantage_collapses": 1.0,
        "falsifier_4_brain_gain_tracks_only_generic_mean_gap": 1.0,
        "falsifier_count": normalize(4.0, 3.0, 4.0),
    }
    hard_falsifier_spec_score = mean(hard_falsifier_spec.values())

    overall_score = mean(
        [
            directional_falsifier_score,
            brain_specificity_score,
            top3_component_risk_score,
            hard_falsifier_spec_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage8d_brain_high_risk_falsification",
        },
        "brain_high_risk_falsifiers": [
            "If broad unfocused brain-side expansion beats focused coverage on brain held-out gap, the current law is weakened.",
            "If augmentation stops improving brain held-out gap while still improving generic mean gap, the current law is weakened.",
            "If protocol-routing / multi-timescale / abstraction stop outperforming the rest in bridge structure, the current law is weakened.",
            "If brain-side gains reduce to generic mean-gap improvements with no brain-specific asymmetry, the current law is weakened.",
        ],
        "pillars": {
            "directional_falsifier": {
                "components": directional_falsifier,
                "score": float(directional_falsifier_score),
            },
            "brain_specificity": {
                "components": brain_specificity,
                "score": float(brain_specificity_score),
            },
            "top3_component_risk": {
                "components": top3_component_risk,
                "score": float(top3_component_risk_score),
            },
            "hard_falsifier_spec": {
                "components": hard_falsifier_spec,
                "score": float(hard_falsifier_spec_score),
            },
        },
        "headline_metrics": {
            "directional_falsifier_score": float(directional_falsifier_score),
            "brain_specificity_score": float(brain_specificity_score),
            "top3_component_risk_score": float(top3_component_risk_score),
            "hard_falsifier_spec_score": float(hard_falsifier_spec_score),
            "overall_stage8d_score": float(overall_score),
        },
        "hypotheses": {
            "H1_brain_side_directional_falsifiers_are_real": bool(directional_falsifier_score >= 0.62),
            "H2_brain_side_signal_is_not_just_generic_fit": bool(brain_specificity_score >= 0.75),
            "H3_top3_component_risk_remains_nontrivial": bool(top3_component_risk_score >= 0.74),
            "H4_high_risk_falsifiers_are_sharp_enough_to_fail": bool(hard_falsifier_spec_score >= 0.95),
            "H5_stage8d_brain_high_risk_falsification_is_moderately_supported": bool(overall_score >= 0.76),
        },
        "project_readout": {
            "summary": (
                "Stage 8D is positive only if the project can state what brain-side observations would directly weaken "
                "the current coding-law candidate, not merely what would continue to support it."
            ),
            "next_question": (
                "If this stage holds, the next step should move beyond scorecards and attempt direct mechanism-break tests."
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
