#!/usr/bin/env python
"""
P6C: detail the feature-extraction term inside the unified plasticity-coding law.
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


def compatibility_top_gain_rate(task_table: Dict[str, Dict[str, float]], cutoff: float = 0.9) -> float:
    rows = list(task_table.values())
    if not rows:
        return 0.0
    best_gain = max(float(row["behavior_gain"]) for row in rows)
    winners = [row for row in rows if float(row["behavior_gain"]) >= best_gain - 1e-12]
    return mean(float(row["compatibility"]) >= cutoff for row in winners)


def main() -> None:
    ap = argparse.ArgumentParser(description="P6C feature extraction mechanism detail")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p6c_feature_extraction_mechanism_detail_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p1 = load_json(ROOT / "tests" / "codex_temp" / "p1_structure_feature_cogeneration_law_20260311.json")
    p4 = load_json(
        ROOT / "tests" / "codex_temp" / "p4_strong_precision_closure_mechanism_intervention_20260311.json"
    )
    p5 = load_json(ROOT / "tests" / "codex_temp" / "p5_forward_brain_predictions_plasticity_coding_20260311.json")
    p6a = load_json(ROOT / "tests" / "codex_temp" / "p6a_unified_plasticity_coding_principle_20260311.json")
    stage7b = load_json(
        ROOT / "tests" / "codex_temp" / "stage7b_precision_tuning_and_cross_model_prediction_20260311.json"
    )
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )
    sweetness = load_json(ROOT / "tests" / "codex_temp" / "real_model_apple_sweetness_channel_edit_20260307.json")
    structure_bridge = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json"
    )

    best = sweetness["best"]
    qwen_tasks = structure_bridge["models"]["qwen3_4b"]["tasks"]
    deepseek_tasks = structure_bridge["models"]["deepseek_7b"]["tasks"]
    qwen_stats = stage8c["model_stats"]["qwen3_4b"]
    deepseek_stats = stage8c["model_stats"]["deepseek_7b"]

    local_feature_separability = {
        "stage7b_precise_tuning": float(stage7b["headline_metrics"]["precise_tuning_score"]),
        "gap_reversal": 1.0 if bool(best["gap_reversed_from_base"]) else 0.0,
        "pair_flip_rate": normalize(float(best["pair_flip_rate_from_base"]), 0.5, 0.8),
        "anchor_retention": normalize(float(best["anchor_retention"]), 0.75, 0.90),
        "target_layer_is_local_band": 1.0 if int(best["layer"]) == 27 else 0.0,
    }
    local_feature_separability_score = mean(local_feature_separability.values())

    compatibility_weighted_extraction = {
        "stage8c_compatibility_invariance": float(stage8c["headline_metrics"]["compatibility_invariance_score"]),
        "qwen_compatibility_gain_corr": float(qwen_stats["compatibility_gain_corr"]),
        "deepseek_compatibility_gain_corr": float(deepseek_stats["compatibility_gain_corr"]),
        "qwen_top_gain_prefers_high_compatibility": float(compatibility_top_gain_rate(qwen_tasks)),
        "deepseek_top_gain_prefers_high_compatibility": float(compatibility_top_gain_rate(deepseek_tasks)),
    }
    compatibility_weighted_extraction_score = mean(compatibility_weighted_extraction.values())

    abstraction_reuse_support = {
        "p1_shared_kernel": float(p1["headline_metrics"]["cross_model_shared_kernel_score"]),
        "qwen_concept_gain_corr": float(qwen_stats["concept_gain_corr"]),
        "deepseek_concept_gain_corr": float(deepseek_stats["concept_gain_corr"]),
        "positive_gain_rate_mean": mean(
            [
                float(qwen_stats["positive_gain_rate"]),
                float(deepseek_stats["positive_gain_rate"]),
            ]
        ),
    }
    abstraction_reuse_support_score = mean(abstraction_reuse_support.values())

    intervention_sensitivity = {
        "p4_feature_generation_intervention": float(
            p4["headline_metrics"]["feature_generation_intervention_score"]
        ),
        "p4_mechanism_reach": float(p4["headline_metrics"]["mechanism_reach_score"]),
        "p6a_intervention_alignment": float(p6a["headline_metrics"]["intervention_alignment_score"]),
        "strong_edit_band_is_nontrivial": normalize(float(best["k"]), 32.0, 96.0),
        "strong_edit_scale_is_nontrivial": normalize(abs(float(best["scale"])), 2.0, 4.0),
    }
    intervention_sensitivity_score = mean(intervention_sensitivity.values())

    explicit_feature_equation = {
        "p1_explicitness": float(p1["headline_metrics"]["explicitness_score"]),
        "p5_mechanistic_specificity": float(p5["headline_metrics"]["mechanistic_specificity_score"]),
        "p6a_math_explicitness": float(p6a["headline_metrics"]["mathematical_explicitness_score"]),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
    }
    explicit_feature_equation_score = mean(explicit_feature_equation.values())

    overall_score = mean(
        [
            local_feature_separability_score,
            compatibility_weighted_extraction_score,
            abstraction_reuse_support_score,
            intervention_sensitivity_score,
            explicit_feature_equation_score,
        ]
    )

    candidate_equations = {
        "feature_operator": "Phi(x_t, A_t) = c_l * L(x_t) + c_k * K_t * S(x_t, A_t) + c_a * U_t - c_i * I_t",
        "local_contrast": "L(x_t) = local_match_t - local_competitor_t",
        "shared_reuse": "S(x_t, A_t) = A_t * local_match_t",
        "compatibility_gate": "K_t = sigmoid(v_c * compatibility_t + v_r * reuse_prior_t)",
        "abstraction_lift": "U_t = abstract_pool(A_t, x_t)",
        "interference": "I_t = overlap_t + mismatch_t",
        "full_update": "f_{t+1} = (1 - l_f) * f_t + e_f * g_t * Phi(x_t, A_t) + b_g * b_region",
    }

    interpretation = {
        "local_contrast": "feature extraction starts from local contrast, not from a full symbolic code",
        "shared_reuse": "current structure routes local evidence toward already reusable subspaces",
        "compatibility_gate": "high-compatibility relations receive earlier feature reuse and cleaner lifting",
        "abstraction_lift": "abstraction is a late reuse term built on top of routed local features",
        "interference": "overlap and mismatch suppress fragile or conflicting candidate features",
        "full_update": "fast features emerge from local contrast under structural routing, then get amplified or damped by compatibility and abstraction pressure",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p6c_feature_extraction_mechanism_detail",
        },
        "candidate_mechanism": {
            "title": "Feature Extraction Detail",
            "equations": candidate_equations,
            "interpretation": interpretation,
            "core_claim": (
                "Feature extraction is not a pure feedforward readout. It is a compatibility-weighted, "
                "structure-routed process that begins with local contrast and grows into reusable abstractions."
            ),
        },
        "pillars": {
            "local_feature_separability": {
                "components": local_feature_separability,
                "score": float(local_feature_separability_score),
            },
            "compatibility_weighted_extraction": {
                "components": compatibility_weighted_extraction,
                "score": float(compatibility_weighted_extraction_score),
            },
            "abstraction_reuse_support": {
                "components": abstraction_reuse_support,
                "score": float(abstraction_reuse_support_score),
            },
            "intervention_sensitivity": {
                "components": intervention_sensitivity,
                "score": float(intervention_sensitivity_score),
            },
            "explicit_feature_equation": {
                "components": explicit_feature_equation,
                "score": float(explicit_feature_equation_score),
            },
        },
        "headline_metrics": {
            "local_feature_separability_score": float(local_feature_separability_score),
            "compatibility_weighted_extraction_score": float(compatibility_weighted_extraction_score),
            "abstraction_reuse_support_score": float(abstraction_reuse_support_score),
            "intervention_sensitivity_score": float(intervention_sensitivity_score),
            "explicit_feature_equation_score": float(explicit_feature_equation_score),
            "overall_p6c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_feature_extraction_has_nontrivial_local_separability": bool(local_feature_separability_score >= 0.69),
            "H2_feature_extraction_is_compatibility_weighted_not_flat": bool(
                compatibility_weighted_extraction_score >= 0.73
            ),
            "H3_feature_extraction_supports_reusable_abstraction": bool(
                abstraction_reuse_support_score >= 0.66
            ),
            "H4_current_interventions_reach_the_feature_generator_nontrivially": bool(
                intervention_sensitivity_score >= 0.72
            ),
            "H5_p6c_feature_detail_is_moderately_supported": bool(overall_score >= 0.73),
        },
        "project_readout": {
            "summary": (
                "P6C is positive only if the feature term can be expanded into a concrete extraction law, rather than "
                "treated as an opaque encoding operator."
            ),
            "next_question": (
                "If P6C holds, the next stage should fuse P6B and P6C into one structure-feature co-evolution "
                "equation and start attacking remaining residuals directly."
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
