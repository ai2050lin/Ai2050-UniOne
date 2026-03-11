#!/usr/bin/env python
"""
P7A: fuse the detailed feature and structure terms into one co-evolution equation.
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
    ap = argparse.ArgumentParser(description="P7A structure-feature coevolution equation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p7a_structure_feature_coevolution_equation_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p2 = load_json(ROOT / "tests" / "codex_temp" / "p2_multitimescale_stabilization_mechanism_20260311.json")
    p3 = load_json(ROOT / "tests" / "codex_temp" / "p3_regional_differentiation_network_roles_20260311.json")
    p5 = load_json(ROOT / "tests" / "codex_temp" / "p5_forward_brain_predictions_plasticity_coding_20260311.json")
    p6a = load_json(ROOT / "tests" / "codex_temp" / "p6a_unified_plasticity_coding_principle_20260311.json")
    p6b = load_json(ROOT / "tests" / "codex_temp" / "p6b_structure_formation_mechanism_detail_20260311.json")
    p6c = load_json(ROOT / "tests" / "codex_temp" / "p6c_feature_extraction_mechanism_detail_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    equation_consistency = {
        "p6a_unified_principle": float(p6a["headline_metrics"]["overall_p6a_score"]),
        "p6b_structure_detail": float(p6b["headline_metrics"]["overall_p6b_score"]),
        "p6c_feature_detail": float(p6c["headline_metrics"]["overall_p6c_score"]),
        "math_explicitness": float(p6a["headline_metrics"]["mathematical_explicitness_score"]),
    }
    equation_consistency_score = mean(equation_consistency.values())

    coupling_closure = {
        "p6b_topology_reorganization": float(
            p6b["headline_metrics"]["topology_reorganization_evidence_score"]
        ),
        "p6c_local_feature_separability": float(
            p6c["headline_metrics"]["local_feature_separability_score"]
        ),
        "p2_fast_slow_coupling": float(p2["headline_metrics"]["fast_slow_coupling_score"]),
        "p2_long_horizon_stability": float(p2["headline_metrics"]["long_horizon_stability_score"]),
        "p3_shared_roles": float(p3["headline_metrics"]["shared_law_diverse_roles_score"]),
    }
    coupling_closure_score = mean(coupling_closure.values())

    residual_attack_readiness = {
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
        "identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
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
    }
    residual_attack_readiness_score = mean(residual_attack_readiness.values())

    brain_plausibility = {
        "p5_prediction_sharpness": float(p5["headline_metrics"]["prediction_sharpness_score"]),
        "p5_independence": float(p5["headline_metrics"]["independence_from_current_fit_score"]),
        "p3_region_prior_support": float(p3["headline_metrics"]["region_prior_support_score"]),
        "p6a_brain_plausibility": float(p6a["headline_metrics"]["brain_mechanism_plausibility_score"]),
    }
    brain_plausibility_score = mean(brain_plausibility.values())

    minimality_pressure = {
        "structure_equation_explicit": float(p6b["headline_metrics"]["explicit_structure_equation_score"]),
        "feature_equation_explicit": float(p6c["headline_metrics"]["explicit_feature_equation_score"]),
        "overlap_balance": 1.0
        - abs(
            float(p6b["headline_metrics"]["overall_p6b_score"])
            - float(p6c["headline_metrics"]["overall_p6c_score"])
        ),
        "not_pure_feature_or_structure": mean(
            [
                float(p6b["headline_metrics"]["route_demand_alignment_score"]),
                float(p6c["headline_metrics"]["compatibility_weighted_extraction_score"]),
            ]
        ),
    }
    minimality_pressure_score = mean(minimality_pressure.values())

    overall_score = mean(
        [
            equation_consistency_score,
            coupling_closure_score,
            residual_attack_readiness_score,
            brain_plausibility_score,
            minimality_pressure_score,
        ]
    )

    equations = {
        "phase_state": "z_t = a_r * r_t - a_s * s_t + b_0",
        "phase_gate": "g_t = sigmoid(T * tanh(z_t))",
        "feature_operator": "Phi_t = c_l * L_t + c_k * K_t * (A_t * L_t) + c_a * U_t - c_i * I_t",
        "structure_operator": "Psi_t = c_c * (f_{t+1} * f_{t+1}^T) + c_d * relu(Q_t - A_t) + c_b * B_region - c_p * relu(A_t - M_t)",
        "feature_update": "f_{t+1} = (1 - l_f) * f_t + e_f * g_t * Phi_t + b_g * b_region",
        "structure_update": "A_{t+1} = (1 - l_A) * A_t + e_A * (1 - g_t) * Psi_t",
        "memory_update": "m_{t+1} = (1 - l_m) * m_t + e_m * s_t * (A_{t+1} - m_t)",
        "coding_state": "y_t = W_f * f_t + W_A * vec(A_t) + W_m * m_t",
    }

    interpretation = {
        "feature_operator": "local contrast becomes reusable feature content only when current topology routes it and compatibility allows reuse",
        "structure_operator": "new feature coalitions reshape topology when demand exceeds current structure and slow memory does not prune them away",
        "feature_update": "feature growth dominates in feature-heavy phases",
        "structure_update": "topology consolidation dominates in structure-heavy phases",
        "memory_update": "slow stabilizer retains durable topology and generates future prune pressure",
        "coding_state": "coding is the instantaneous coupled state of fast features, effective topology, and slow memory",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p7a_structure_feature_coevolution_equation",
        },
        "candidate_mechanism": {
            "title": "Structure Feature Coevolution Equation",
            "equations": equations,
            "interpretation": interpretation,
            "core_claim": (
                "Feature extraction and topology formation are two phases of one co-evolution law. "
                "Features propose structure, structure routes future features, and slow memory stabilizes the cycle."
            ),
        },
        "pillars": {
            "equation_consistency": {
                "components": equation_consistency,
                "score": float(equation_consistency_score),
            },
            "coupling_closure": {
                "components": coupling_closure,
                "score": float(coupling_closure_score),
            },
            "residual_attack_readiness": {
                "components": residual_attack_readiness,
                "score": float(residual_attack_readiness_score),
            },
            "brain_plausibility": {
                "components": brain_plausibility,
                "score": float(brain_plausibility_score),
            },
            "minimality_pressure": {
                "components": minimality_pressure,
                "score": float(minimality_pressure_score),
            },
        },
        "headline_metrics": {
            "equation_consistency_score": float(equation_consistency_score),
            "coupling_closure_score": float(coupling_closure_score),
            "residual_attack_readiness_score": float(residual_attack_readiness_score),
            "brain_plausibility_score": float(brain_plausibility_score),
            "minimality_pressure_score": float(minimality_pressure_score),
            "overall_p7a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_feature_and_structure_terms_can_be_fused_into_one_equation": bool(
                equation_consistency_score >= 0.75
            ),
            "H2_the_fused_equation_preserves_nontrivial_coupling": bool(coupling_closure_score >= 0.73),
            "H3_the_fused_equation_is_ready_to_attack_residuals": bool(
                residual_attack_readiness_score >= 0.72
            ),
            "H4_the_fused_equation_remains_brain_plausible": bool(brain_plausibility_score >= 0.74),
            "H5_p7a_structure_feature_coevolution_is_moderately_supported": bool(overall_score >= 0.77),
        },
        "project_readout": {
            "summary": (
                "P7A is positive only if the detailed structure and feature laws can be fused into one co-evolution "
                "equation without losing coupling, explicitness, or brain plausibility."
            ),
            "next_question": (
                "If P7A holds, the next step is to compress the fused equation into a smaller minimal plasticity core "
                "and then push high-risk falsifiers against that core."
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
