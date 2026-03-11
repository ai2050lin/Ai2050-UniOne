#!/usr/bin/env python
"""
Build an explicit candidate coding law from the strongest stage-6 signals.
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
    ap = argparse.ArgumentParser(description="Stage 7A explicit coding law candidate")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage7a_explicit_coding_law_candidate_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    update_law = load_json(ROOT / "tests" / "codex_temp" / "unified_update_law_candidate_20260309.json")
    phase_law = load_json(ROOT / "tests" / "codex_temp" / "phase_gated_unified_update_law_20260309.json")
    state_law = load_json(ROOT / "tests" / "codex_temp" / "state_variable_calibrated_unified_law_20260309.json")
    brain_rank = load_json(
        ROOT / "tests" / "codex_temp" / "brain_learnable_ranking_two_layer_unified_law_20260310.json"
    )
    stage6 = load_json(ROOT / "tests" / "codex_temp" / "stage6_master_closure_20260311.json")
    stage6a = load_json(ROOT / "tests" / "codex_temp" / "stage6a_causal_core_compression_20260311.json")
    stage6d = load_json(
        ROOT / "tests" / "codex_temp" / "stage6d_brain_constraint_core_reduction_20260311.json"
    )

    best_update = update_law["best_law"]
    phase_candidate = phase_law["phase_gated_law"]
    state_candidate = state_law["state_variable_law"]
    brain_coeffs = {
        k: float(v) for k, v in brain_rank["ranking_layer"]["brain_component_coefficients"].items()
    }

    coeff_sum = sum(brain_coeffs.values())
    normalized_brain_weights = {
        k: float(v / coeff_sum) for k, v in sorted(brain_coeffs.items(), key=lambda item: item[1], reverse=True)
    }
    top3_keys = list(normalized_brain_weights.keys())[:3]
    top3_weight_sum = float(sum(normalized_brain_weights[k] for k in top3_keys))

    explicit_law = {
        "state_variable": "z = tanh(alpha * (stabilization - routing) + beta * (adaptive_offset - 0.5) + bias)",
        "phase_gate": "g = sigmoid(gate_temp * z)",
        "brain_prior": "b = sum_i w_i * component_i",
        "effective_offset": (
            "adaptive_offset + route_gain * (1 - g) * routing * (1 - adaptive_offset) "
            "- stabilize_drag * g * (1 - stabilization) * adaptive_offset + brain_gain * b"
        ),
        "readout": "coding_score = mean(base, effective_offset, routing, stabilization)",
    }

    explicit_params = {
        "route_gain": float(best_update["route_gain"]),
        "stabilize_drag": float(best_update["stabilize_drag"]),
        "gate_temp": float(phase_candidate["gate_temp"]),
        "gate_bias": float(phase_candidate["gate_bias"]),
        "alpha": float(state_candidate["alpha"]),
        "beta": float(state_candidate["beta"]),
        "bias": float(state_candidate["bias"]),
        "brain_gain": float(top3_weight_sum),
    }

    strongest_brain_components = [
        {"component": key, "weight": float(normalized_brain_weights[key])}
        for key in top3_keys
    ]

    equation_support = {
        "stage6_overall": float(stage6["stage6_headline_metrics"]["overall_stage6_score"]),
        "two_param_core": float(stage6a["headline_metrics"]["two_param_core_score"]),
        "brain_core_reduction": float(stage6d["headline_metrics"]["core_freedom_reduction_score"]),
        "brain_learnable_reduction": float(stage6d["headline_metrics"]["learnable_brain_reduction_score"]),
    }
    equation_support_score = mean(equation_support.values())

    law_fit = {
        "two_param_gap": inv_gap_score(float(best_update["mean_absolute_gap"]), 0.02, 0.06),
        "two_param_held_out_gap": inv_gap_score(
            float(update_law["leave_one_out"]["mean_held_out_gap"]),
            0.02,
            0.06,
        ),
        "two_param_corr": normalize(float(best_update["score_correlation"]), 0.78, 0.84),
        "two_param_pass": 1.0 if bool(best_update["pass"]) else 0.0,
    }
    law_fit_score = mean(law_fit.values())

    gating_support = {
        "phase_corr": normalize(float(phase_candidate["score_correlation"]), 0.90, 0.97),
        "phase_corr_improvement": normalize(
            float(phase_candidate["correlation_improvement_vs_fixed"]),
            0.15,
            0.26,
        ),
        "state_gap_improvement": normalize(
            float(state_candidate["gap_improvement_vs_phase_gated"]),
            0.03,
            0.06,
        ),
        "state_d_gap": inv_gap_score(float(state_candidate["d_mean_gap"]), 0.005, 0.02),
    }
    gating_support_score = mean(gating_support.values())

    brain_prior_support = {
        "top3_weight_sum": normalize(top3_weight_sum, 0.45, 0.65),
        "brain_mean_gap": inv_gap_score(
            float(brain_rank["brain_learnable_ranking_two_layer_law"]["brain_mean_gap"]),
            0.0,
            0.01,
        ),
        "brain_held_out_gap": inv_gap_score(
            float(brain_rank["brain_learnable_ranking_two_layer_law"]["brain_held_out_gap"]),
            0.01,
            0.03,
        ),
        "brain_corr": normalize(
            float(brain_rank["brain_learnable_ranking_two_layer_law"]["held_out_score_correlation"]),
            0.99,
            1.0,
        ),
    }
    brain_prior_support_score = mean(brain_prior_support.values())

    explicitness = {
        "equation_count": float(max(0.0, 1.0 - (5.0 - 4.0) / 5.0)),
        "param_count": float(max(0.0, 1.0 - (len(explicit_params) - 8.0) / 12.0)),
        "top3_component_concentration": normalize(top3_weight_sum, 0.45, 0.65),
        "compression_ratio_to_core": float(
            stage6a["pillars"]["two_param_core_candidate"]["components"]["compression_ratio_to_core"]
        ),
    }
    explicitness_score = mean(explicitness.values())

    overall_score = mean(
        [
            equation_support_score,
            law_fit_score,
            gating_support_score,
            brain_prior_support_score,
            explicitness_score,
        ]
    )

    hypotheses = {
        "H1_candidate_equation_is_supported_by_stage6": bool(equation_support_score >= 0.76),
        "H2_two_param_law_fit_is_nontrivial": bool(law_fit_score >= 0.76),
        "H3_phase_gating_support_remains_nontrivial": bool(gating_support_score >= 0.60),
        "H4_brain_prior_support_is_nontrivial": bool(brain_prior_support_score >= 0.74),
        "H5_stage7a_explicit_coding_law_is_moderately_supported": bool(overall_score >= 0.74),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage7a_explicit_coding_law_candidate",
        },
        "candidate_coding_law": {
            "equations": explicit_law,
            "parameters": explicit_params,
            "strongest_brain_components": strongest_brain_components,
            "normalized_brain_weights": normalized_brain_weights,
            "verbal_guess": (
                "The current best guess is a phase-gated local update law: routing pressure increases an adaptive "
                "offset when the system is in a routing-dominant phase, stabilization pressure suppresses that offset "
                "when the system is in a grounding-dominant phase, and a small weighted brain prior biases the update "
                "toward protocol-routing, multi-timescale control, and abstraction-like components."
            ),
        },
        "pillars": {
            "equation_support": {"components": equation_support, "score": float(equation_support_score)},
            "law_fit": {"components": law_fit, "score": float(law_fit_score)},
            "gating_support": {"components": gating_support, "score": float(gating_support_score)},
            "brain_prior_support": {
                "components": brain_prior_support,
                "score": float(brain_prior_support_score),
            },
            "explicitness": {"components": explicitness, "score": float(explicitness_score)},
        },
        "headline_metrics": {
            "equation_support_score": float(equation_support_score),
            "law_fit_score": float(law_fit_score),
            "gating_support_score": float(gating_support_score),
            "brain_prior_support_score": float(brain_prior_support_score),
            "explicitness_score": float(explicitness_score),
            "overall_stage7a_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 7A is positive only if the project can stop speaking in dashboards and write down an explicit "
                "candidate coding law with concrete equations, parameters, and brain-side prior terms."
            ),
            "next_question": (
                "If this stage holds, the next step is a direct cross-model test: does this explicit law predict "
                "which models fail where, rather than merely fitting after the fact?"
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
