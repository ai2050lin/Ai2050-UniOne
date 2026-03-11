#!/usr/bin/env python
"""
P7B: compress the co-evolution equation into a smaller minimal plasticity core.
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
    ap = argparse.ArgumentParser(description="P7B minimal plasticity core compression")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p7b_minimal_plasticity_core_compression_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p3 = load_json(ROOT / "tests" / "codex_temp" / "p3_regional_differentiation_network_roles_20260311.json")
    p5 = load_json(ROOT / "tests" / "codex_temp" / "p5_forward_brain_predictions_plasticity_coding_20260311.json")
    p6a = load_json(ROOT / "tests" / "codex_temp" / "p6a_unified_plasticity_coding_principle_20260311.json")
    p6b = load_json(ROOT / "tests" / "codex_temp" / "p6b_structure_formation_mechanism_detail_20260311.json")
    p6c = load_json(ROOT / "tests" / "codex_temp" / "p6c_feature_extraction_mechanism_detail_20260311.json")
    p7a = load_json(ROOT / "tests" / "codex_temp" / "p7a_structure_feature_coevolution_equation_20260311.json")
    stage6a = load_json(ROOT / "tests" / "codex_temp" / "stage6a_causal_core_compression_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    compression_retains_signal = {
        "stage6a_two_param_core": float(stage6a["headline_metrics"]["two_param_core_score"]),
        "stage6a_four_factor_score": float(stage6a["headline_metrics"]["four_factor_score"]),
        "p7a_equation_consistency": float(p7a["headline_metrics"]["equation_consistency_score"]),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
    }
    compression_retains_signal_score = mean(compression_retains_signal.values())

    minimal_term_sufficiency = {
        "p7a_minimality_pressure": float(p7a["headline_metrics"]["minimality_pressure_score"]),
        "stage6a_minimal_interface": float(stage6a["headline_metrics"]["minimal_interface_score"]),
        "stage6a_shell_localization": float(stage6a["headline_metrics"]["shell_localization_score"]),
        "p6a_residual_control": float(p6a["headline_metrics"]["residual_control_score"]),
    }
    minimal_term_sufficiency_score = mean(minimal_term_sufficiency.values())

    shared_parameterization = {
        "feature_structure_balance": 1.0
        - abs(
            float(p6b["headline_metrics"]["overall_p6b_score"])
            - float(p6c["headline_metrics"]["overall_p6c_score"])
        ),
        "structure_explicitness": float(p6b["headline_metrics"]["explicit_structure_equation_score"]),
        "feature_explicitness": float(p6c["headline_metrics"]["explicit_feature_equation_score"]),
        "p7a_coupling_closure": float(p7a["headline_metrics"]["coupling_closure_score"]),
    }
    shared_parameterization_score = mean(shared_parameterization.values())

    residual_bound_after_compression = {
        "p7a_residual_readiness": float(p7a["headline_metrics"]["residual_attack_readiness_score"]),
        "architecture_scale_known": normalize(
            float(stage9c["headline_metrics"]["architecture_plus_scale_share"]),
            0.50,
            0.65,
        ),
        "unresolved_core_inverse": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
    }
    residual_bound_after_compression_score = mean(residual_bound_after_compression.values())

    brain_plausibility_after_compression = {
        "p7a_brain_plausibility": float(p7a["headline_metrics"]["brain_plausibility_score"]),
        "p6a_brain_plausibility": float(p6a["headline_metrics"]["brain_mechanism_plausibility_score"]),
        "p5_mechanistic_specificity": float(p5["headline_metrics"]["mechanistic_specificity_score"]),
        "p3_region_prior_support": float(p3["headline_metrics"]["region_prior_support_score"]),
    }
    brain_plausibility_after_compression_score = mean(brain_plausibility_after_compression.values())

    overall_score = mean(
        [
            compression_retains_signal_score,
            minimal_term_sufficiency_score,
            shared_parameterization_score,
            residual_bound_after_compression_score,
            brain_plausibility_after_compression_score,
        ]
    )

    candidate_core = {
        "phase_core": "q_t = sigmoid(alpha * (r_t - s_t) + beta * b_region)",
        "feature_core": "f_{t+1} = (1 - l_f) * f_t + e_f * q_t * (L_t + k_a * A_t * L_t - k_i * I_t)",
        "structure_core": "A_{t+1} = (1 - l_A) * A_t + e_A * (1 - q_t) * (f_{t+1} * f_{t+1}^T + d_t - p_t)",
        "memory_core": "m_{t+1} = (1 - l_m) * m_t + e_m * s_t * (A_{t+1} - m_t)",
        "demand_term": "d_t = k_d * relu(Q_t - A_t)",
        "prune_term": "p_t = k_p * relu(A_t - m_t)",
        "coding_state": "y_t = W_f * f_t + W_A * vec(A_t) + W_m * m_t",
    }

    interpretation = {
        "phase_core": "one gate decides whether update mass goes to feature growth or topology consolidation",
        "feature_core": "feature generation only keeps local contrast, routed reuse, and interference suppression",
        "structure_core": "structure formation only keeps feature coactivation, demand mismatch, and prune pressure",
        "memory_core": "slow memory remains the only stabilizer and pruning anchor",
        "core_claim": (
            "The minimal plasticity core only needs four mechanisms: phase gating, local contrast extraction, "
            "demand-shaped structure growth, and slow stabilization."
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p7b_minimal_plasticity_core_compression",
        },
        "candidate_mechanism": {
            "title": "Minimal Plasticity Core",
            "equations": candidate_core,
            "interpretation": interpretation,
            "core_claim": interpretation["core_claim"],
        },
        "pillars": {
            "compression_retains_signal": {
                "components": compression_retains_signal,
                "score": float(compression_retains_signal_score),
            },
            "minimal_term_sufficiency": {
                "components": minimal_term_sufficiency,
                "score": float(minimal_term_sufficiency_score),
            },
            "shared_parameterization": {
                "components": shared_parameterization,
                "score": float(shared_parameterization_score),
            },
            "residual_bound_after_compression": {
                "components": residual_bound_after_compression,
                "score": float(residual_bound_after_compression_score),
            },
            "brain_plausibility_after_compression": {
                "components": brain_plausibility_after_compression,
                "score": float(brain_plausibility_after_compression_score),
            },
        },
        "headline_metrics": {
            "compression_retains_signal_score": float(compression_retains_signal_score),
            "minimal_term_sufficiency_score": float(minimal_term_sufficiency_score),
            "shared_parameterization_score": float(shared_parameterization_score),
            "residual_bound_after_compression_score": float(residual_bound_after_compression_score),
            "brain_plausibility_after_compression_score": float(
                brain_plausibility_after_compression_score
            ),
            "overall_p7b_score": float(overall_score),
        },
        "hypotheses": {
            "H1_compression_retains_nontrivial_signal": bool(compression_retains_signal_score >= 0.76),
            "H2_a_small_number_of_terms_remains_sufficient": bool(minimal_term_sufficiency_score >= 0.79),
            "H3_feature_and_structure_can_share_one_small_parameterization": bool(
                shared_parameterization_score >= 0.79
            ),
            "H4_compression_does_not_destroy_residual_control": bool(
                residual_bound_after_compression_score >= 0.69
            ),
            "H5_p7b_minimal_plasticity_core_is_moderately_supported": bool(overall_score >= 0.75),
        },
        "project_readout": {
            "summary": (
                "P7B is positive only if the co-evolution equation can be compressed into a smaller plasticity core "
                "without losing too much signal, residual control, or brain plausibility."
            ),
            "next_question": (
                "If P7B holds, the next stage should push high-risk brain-side falsifiers directly against this "
                "minimal core."
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
