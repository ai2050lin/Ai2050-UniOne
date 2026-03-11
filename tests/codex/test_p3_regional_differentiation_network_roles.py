#!/usr/bin/env python
"""
P3: score a candidate mechanism for regional differentiation and network-role
emergence under one shared plasticity law.
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
    ap = argparse.ArgumentParser(description="P3 regional differentiation and network roles")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p3_regional_differentiation_network_roles_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p1 = load_json(ROOT / "tests" / "codex_temp" / "p1_structure_feature_cogeneration_law_20260311.json")
    p2 = load_json(ROOT / "tests" / "codex_temp" / "p2_multitimescale_stabilization_mechanism_20260311.json")
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    orientation = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json"
    )
    targeted = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json"
    )
    stage6d = load_json(ROOT / "tests" / "codex_temp" / "stage6d_brain_constraint_core_reduction_20260311.json")

    qwen_o = orientation["models"]["qwen3_4b"]["global_summary"]
    deepseek_o = orientation["models"]["deepseek_7b"]["global_summary"]
    qwen_t = targeted["models"]["qwen3_4b"]["global_summary"]
    deepseek_t = targeted["models"]["deepseek_7b"]["global_summary"]

    regional_role_emergence = {
        "qwen_orientation_strength": normalize(abs(float(qwen_o["shared_layer_orientation"])), 0.04, 0.16),
        "deepseek_orientation_strength": normalize(abs(float(deepseek_o["shared_layer_orientation"])), 0.15, 0.30),
        "cross_model_orientation_gap": normalize(
            abs(float(deepseek_o["shared_layer_orientation"]) - float(qwen_o["shared_layer_orientation"])),
            0.18,
            0.35,
        ),
        "deepseek_targeted_orientation_strength": normalize(
            abs(float(deepseek_t["actual_targeted_orientation"])),
            0.20,
            0.45,
        ),
    }
    regional_role_emergence_score = mean(regional_role_emergence.values())

    shared_law_diverse_roles = {
        "shared_kernel_survives": float(p1["headline_metrics"]["cross_model_shared_kernel_score"]),
        "fast_slow_shared_core": float(p2["headline_metrics"]["fast_slow_coupling_score"]),
        "compatibility_invariance": float(stage8c["headline_metrics"]["compatibility_invariance_score"]),
        "model_gap_structure": float(stage8c["headline_metrics"]["model_gap_structure_score"]),
    }
    shared_law_diverse_roles_score = mean(shared_law_diverse_roles.values())

    region_prior_support = {
        "protocol_routing_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["protocol_routing"]),
            0.14,
            0.22,
        ),
        "multi_timescale_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["multi_timescale_control"]),
            0.12,
            0.20,
        ),
        "abstraction_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["abstraction_operator"]),
            0.12,
            0.18,
        ),
        "brain_constraint_concentration": normalize(
            float(stage6d["pillars"]["core_freedom_reduction"]["components"]["component_concentration"]),
            0.55,
            0.70,
        ),
    }
    region_prior_support_score = mean(region_prior_support.values())

    role_specific_failure_patterns = {
        "qwen_predicted_actual_mismatch": 1.0
        if qwen_t["predicted_orientation_label"] != qwen_t["actual_orientation_label"]
        else 0.0,
        "deepseek_predicted_actual_mismatch": 1.0
        if deepseek_t["predicted_orientation_label"] != deepseek_t["actual_orientation_label"]
        else 0.0,
        "qwen_role_is_more_balanced": normalize(abs(float(qwen_t["actual_targeted_orientation"])), 0.0, 0.10),
        "deepseek_role_is_more_specialized": normalize(
            abs(float(deepseek_t["actual_targeted_orientation"])),
            0.20,
            0.45,
        ),
    }
    role_specific_failure_patterns_score = mean(role_specific_failure_patterns.values())

    overall_score = mean(
        [
            regional_role_emergence_score,
            shared_law_diverse_roles_score,
            region_prior_support_score,
            role_specific_failure_patterns_score,
        ]
    )

    candidate_mechanism = {
        "equations": {
            "shared_plasticity": "Δh_i = η_i * gate_t * local_signal_i + region_bias_i - decay_i * h_i",
            "role_bias": "region_bias_i = rho_region * prior_region + lambda_tau * timescale_preference_i",
            "structure_specialization": "ΔA_{ij} = coactivate(h_i, h_j) * gate_t - prune_t * A_{ij}",
        },
        "verbal_guess": (
            "The current best P3 guess is that brain areas do not need different fundamental learning rules. "
            "They can inherit different network roles because the same plasticity law is instantiated under different "
            "regional priors, timescale preferences, and input statistics."
        ),
        "candidate_roles": [
            "routing_dominant_region",
            "stabilization_dominant_region",
            "abstraction_dominant_region",
            "mixed_transition_region",
        ],
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p3_regional_differentiation_network_roles",
        },
        "candidate_mechanism": candidate_mechanism,
        "pillars": {
            "regional_role_emergence": {
                "components": regional_role_emergence,
                "score": float(regional_role_emergence_score),
            },
            "shared_law_diverse_roles": {
                "components": shared_law_diverse_roles,
                "score": float(shared_law_diverse_roles_score),
            },
            "region_prior_support": {
                "components": region_prior_support,
                "score": float(region_prior_support_score),
            },
            "role_specific_failure_patterns": {
                "components": role_specific_failure_patterns,
                "score": float(role_specific_failure_patterns_score),
            },
        },
        "headline_metrics": {
            "regional_role_emergence_score": float(regional_role_emergence_score),
            "shared_law_diverse_roles_score": float(shared_law_diverse_roles_score),
            "region_prior_support_score": float(region_prior_support_score),
            "role_specific_failure_patterns_score": float(role_specific_failure_patterns_score),
            "overall_p3_score": float(overall_score),
        },
        "hypotheses": {
            "H1_same_law_can_generate_different_regional_roles": bool(regional_role_emergence_score >= 0.52),
            "H2_shared_plasticity_core_survives_despite_role_diversity": bool(
                shared_law_diverse_roles_score >= 0.74
            ),
            "H3_regional_priors_enter_the_mechanism_nontrivially": bool(region_prior_support_score >= 0.62),
            "H4_role_specific_failure_patterns_are_real": bool(role_specific_failure_patterns_score >= 0.72),
            "H5_p3_regional_differentiation_is_moderately_supported": bool(overall_score >= 0.68),
        },
        "project_readout": {
            "summary": (
                "P3 is positive only if the project can explain why one shared plasticity law produces different "
                "network roles across regions rather than forcing every region into the same function."
            ),
            "next_question": (
                "If P3 holds, the next step is to reconnect this theory to stronger mechanism interventions on the "
                "DNN side rather than staying at the descriptive role level."
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
