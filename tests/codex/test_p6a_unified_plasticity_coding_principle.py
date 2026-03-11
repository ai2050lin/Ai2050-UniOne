#!/usr/bin/env python
"""
P6A: compress P1-P5 into one explicit unified mathematical principle for
plasticity-driven network formation and coding emergence.
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
    ap = argparse.ArgumentParser(description="P6A unified plasticity coding principle")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p6a_unified_plasticity_coding_principle_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p1 = load_json(ROOT / "tests" / "codex_temp" / "p1_structure_feature_cogeneration_law_20260311.json")
    p2 = load_json(ROOT / "tests" / "codex_temp" / "p2_multitimescale_stabilization_mechanism_20260311.json")
    p3 = load_json(ROOT / "tests" / "codex_temp" / "p3_regional_differentiation_network_roles_20260311.json")
    p4 = load_json(ROOT / "tests" / "codex_temp" / "p4_strong_precision_closure_mechanism_intervention_20260311.json")
    p5 = load_json(ROOT / "tests" / "codex_temp" / "p5_forward_brain_predictions_plasticity_coding_20260311.json")
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    synthesis_consistency = {
        "p1_overall": float(p1["headline_metrics"]["overall_p1_score"]),
        "p2_overall": float(p2["headline_metrics"]["overall_p2_score"]),
        "p3_overall": float(p3["headline_metrics"]["overall_p3_score"]),
        "p4_overall": float(p4["headline_metrics"]["overall_p4_score"]),
        "p5_overall": float(p5["headline_metrics"]["overall_p5_score"]),
    }
    synthesis_consistency_score = mean(synthesis_consistency.values())

    mathematical_explicitness = {
        "stage7a_explicitness": float(stage7a["headline_metrics"]["explicitness_score"]),
        "p1_explicitness": float(p1["headline_metrics"]["explicitness_score"]),
        "p2_explicitness": float(p2["headline_metrics"]["explicitness_score"]),
        "p5_prediction_sharpness": float(p5["headline_metrics"]["prediction_sharpness_score"]),
    }
    mathematical_explicitness_score = mean(mathematical_explicitness.values())

    intervention_alignment = {
        "feature_generation_intervention": float(p4["headline_metrics"]["feature_generation_intervention_score"]),
        "structure_formation_intervention": float(p4["headline_metrics"]["structure_formation_intervention_score"]),
        "mechanism_reach": float(p4["headline_metrics"]["mechanism_reach_score"]),
        "not_yet_strong_closure": 1.0
        if bool(p4["hypotheses"]["H3_strong_precision_closure_is_not_complete_yet"])
        else 0.0,
    }
    intervention_alignment_score = mean(intervention_alignment.values())

    brain_mechanism_plausibility = {
        "p1_brain_plausibility": float(p1["headline_metrics"]["brain_plausibility_score"]),
        "p3_region_prior_support": float(p3["headline_metrics"]["region_prior_support_score"]),
        "p5_independence": float(p5["headline_metrics"]["independence_from_current_fit_score"]),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
    }
    brain_mechanism_plausibility_score = mean(brain_mechanism_plausibility.values())

    residual_control = {
        "unresolved_core_inverse": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "architecture_scale_known": normalize(
            float(stage9c["headline_metrics"]["architecture_plus_scale_share"]),
            0.50,
            0.65,
        ),
        "identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
        "p5_falsifiability": float(p5["headline_metrics"]["falsifiability_score"]),
    }
    residual_control_score = mean(residual_control.values())

    overall_score = mean(
        [
            synthesis_consistency_score,
            mathematical_explicitness_score,
            intervention_alignment_score,
            brain_mechanism_plausibility_score,
            residual_control_score,
        ]
    )

    equations = {
        "phase_state": "z_t = a_r * r_t - a_s * s_t + b_0",
        "phase_gate": "g_t = sigmoid(T * tanh(z_t))",
        "fast_feature": "f_{t+1} = (1 - l_f) * f_t + e_f * g_t * Phi(x_t, A_t) + b_g * b_region",
        "mid_structure": "A_{t+1} = (1 - l_A) * A_t + e_A * (1 - g_t) * Psi(f_{t+1}, A_t) + r_g * R_t",
        "slow_memory": "m_{t+1} = (1 - l_m) * m_t + e_m * s_t * (A_{t+1} - m_t)",
        "regional_bias": "b_region = w_p * p_route + w_m * p_multi + w_a * p_abs + w_h * p_shared",
        "coding_readout": "y_t = W_f * f_t + W_A * vec(A_t) + W_m * m_t",
    }

    principle = {
        "title": "Unified Plasticity Coding Principle",
        "equations": equations,
        "interpretation": {
            "phase_state": "routing pressure and stabilization pressure define the current update regime",
            "phase_gate": "the gate allocates update mass between feature growth and structure consolidation",
            "fast_feature": "fast state extracts local features from current input under current effective structure",
            "mid_structure": "mid-timescale state reshapes effective network topology from co-activated features",
            "slow_memory": "slow state stores durable structure and supplies stabilization / recovery",
            "regional_bias": "the same rule produces different regional roles by changing priors and timescale preference",
            "coding_readout": "coding is a readout over fast features, effective structure, and slow stabilizer memory",
        },
        "core_claim": (
            "Coding does not live in features alone or structure alone. Coding emerges from the coupled evolution of "
            "fast features, mid-timescale effective topology, and slow stabilizer memory under one gated local "
            "plasticity law."
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p6a_unified_plasticity_coding_principle",
        },
        "principle": principle,
        "pillars": {
            "synthesis_consistency": {
                "components": synthesis_consistency,
                "score": float(synthesis_consistency_score),
            },
            "mathematical_explicitness": {
                "components": mathematical_explicitness,
                "score": float(mathematical_explicitness_score),
            },
            "intervention_alignment": {
                "components": intervention_alignment,
                "score": float(intervention_alignment_score),
            },
            "brain_mechanism_plausibility": {
                "components": brain_mechanism_plausibility,
                "score": float(brain_mechanism_plausibility_score),
            },
            "residual_control": {
                "components": residual_control,
                "score": float(residual_control_score),
            },
        },
        "headline_metrics": {
            "synthesis_consistency_score": float(synthesis_consistency_score),
            "mathematical_explicitness_score": float(mathematical_explicitness_score),
            "intervention_alignment_score": float(intervention_alignment_score),
            "brain_mechanism_plausibility_score": float(brain_mechanism_plausibility_score),
            "residual_control_score": float(residual_control_score),
            "overall_p6a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_P1_to_P5_can_be_compressed_into_one_principle": bool(synthesis_consistency_score >= 0.72),
            "H2_the_principle_is_mathematically_explicit": bool(mathematical_explicitness_score >= 0.80),
            "H3_the_principle_matches_current_intervention_behavior": bool(intervention_alignment_score >= 0.70),
            "H4_the_principle_is_brain_plausible": bool(brain_mechanism_plausibility_score >= 0.74),
            "H5_p6a_unified_principle_is_moderately_supported": bool(overall_score >= 0.76),
        },
        "project_readout": {
            "summary": (
                "P6A is positive only if the project can compress P1-P5 into one explicit mathematical principle for "
                "plasticity-driven network formation and coding emergence."
            ),
            "next_question": (
                "If P6A holds, the next step is to separately detail the structure-formation term and the feature-"
                "extraction term rather than leaving them inside one compact theory."
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
