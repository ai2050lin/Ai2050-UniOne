#!/usr/bin/env python
"""
P1: propose and score a unified plasticity law where feature extraction and
network-structure formation co-evolve under the same local update mechanism.
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
    ap = argparse.ArgumentParser(description="P1 structure-feature co-generation law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p1_structure_feature_cogeneration_law_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")
    stage9a = load_json(ROOT / "tests" / "codex_temp" / "stage9a_mechanism_adversarial_break_test_20260311.json")
    stage9c = load_json(ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json")
    brain_bridge = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")

    top3 = [row["component"] for row in stage7a["candidate_coding_law"]["strongest_brain_components"]]
    qwen_components = brain_bridge["models"]["qwen3_4b"]["components"]
    gpt2_components = brain_bridge["models"]["gpt2"]["components"]

    feature_structure_coupling = {
        "gating_support": float(stage7a["headline_metrics"]["gating_support_score"]),
        "protocol_routing_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["protocol_routing"]),
            0.14,
            0.22,
        ),
        "topology_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["topology_basis"]),
            0.09,
            0.14,
        ),
        "multi_timescale_weight": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["multi_timescale_control"]),
            0.12,
            0.20,
        ),
    }
    feature_structure_coupling_score = mean(feature_structure_coupling.values())

    cross_model_shared_kernel = {
        "compatibility_invariance": float(stage8c["headline_metrics"]["compatibility_invariance_score"]),
        "relation_family_invariance": float(stage8c["headline_metrics"]["relation_family_invariance_score"]),
        "top3_bridge_advantage_gpt2": normalize(
            mean(float(gpt2_components[name]["score"]) for name in top3)
            - mean(float(v["score"]) for name, v in gpt2_components.items() if name not in top3),
            0.12,
            0.28,
        ),
        "top3_bridge_advantage_qwen": normalize(
            mean(float(qwen_components[name]["score"]) for name in top3)
            - mean(float(v["score"]) for name, v in qwen_components.items() if name not in top3),
            0.03,
            0.10,
        ),
    }
    cross_model_shared_kernel_score = mean(cross_model_shared_kernel.values())

    residual_boundedness = {
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
        "unresolved_core_is_bounded": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "architecture_scale_dominate": normalize(
            float(stage9c["headline_metrics"]["architecture_plus_scale_share"]),
            0.50,
            0.65,
        ),
        "data_pressure_not_dominant": normalize(
            1.0 - float(stage9c["residual_shares"]["data_share"]),
            0.70,
            0.85,
        ),
    }
    residual_boundedness_score = mean(residual_boundedness.values())

    brain_plausibility = {
        "brain_specificity": float(stage8d["headline_metrics"]["brain_specificity_score"]),
        "top3_component_risk": float(stage8d["headline_metrics"]["top3_component_risk_score"]),
        "gpt2_protocol_score": normalize(float(gpt2_components["protocol_routing"]["score"]), 0.75, 0.95),
        "qwen_multi_timescale_score": normalize(
            float(qwen_components["multi_timescale_control"]["score"]),
            0.72,
            0.90,
        ),
    }
    brain_plausibility_score = mean(brain_plausibility.values())

    explicitness = {
        "equation_support": float(stage7a["headline_metrics"]["equation_support_score"]),
        "explicitness_score": float(stage7a["headline_metrics"]["explicitness_score"]),
        "identifiability_score": float(stage9c["headline_metrics"]["identifiability_score"]),
        "break_test_support": float(stage9a["headline_metrics"]["overall_stage9a_score"]),
    }
    explicitness_score = mean(explicitness.values())

    overall_score = mean(
        [
            feature_structure_coupling_score,
            cross_model_shared_kernel_score,
            residual_boundedness_score,
            brain_plausibility_score,
            explicitness_score,
        ]
    )

    route_gain = float(stage7a["candidate_coding_law"]["parameters"]["route_gain"])
    gate_temp = float(stage7a["candidate_coding_law"]["parameters"]["gate_temp"])
    brain_gain = float(stage7a["candidate_coding_law"]["parameters"]["brain_gain"])

    candidate_law = {
        "equations": {
            "phase_gate": "g_t = sigmoid(gate_temp * tanh(alpha * (routing_t - stabilization_t) + bias))",
            "feature_update": (
                "f_{t+1} = f_t + eta_f * g_t * (input_match_t * A_t - decay_f * f_t) + brain_gain * b_r"
            ),
            "structure_update": (
                "A_{t+1} = A_t + eta_s * (1 - g_t) * (coactivate(f_{t+1}) - decay_s * A_t) + route_gain * route_t"
            ),
            "readout": "coding_t = mean(shared_basis_t, sparse_offset_t, routing_t, abstraction_t)",
        },
        "verbal_guess": (
            "The current best P1 guess is a unified local plasticity law: data-driven feature activity updates local "
            "effective structure, and the resulting structure changes the next round of feature extraction. Phase "
            "gating controls when the system prioritizes feature growth versus structural consolidation."
        ),
        "parameter_seeds": {
            "route_gain": route_gain,
            "gate_temp": gate_temp,
            "brain_gain": brain_gain,
            "dominant_brain_components": top3,
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p1_structure_feature_cogeneration_law",
        },
        "candidate_law": candidate_law,
        "pillars": {
            "feature_structure_coupling": {
                "components": feature_structure_coupling,
                "score": float(feature_structure_coupling_score),
            },
            "cross_model_shared_kernel": {
                "components": cross_model_shared_kernel,
                "score": float(cross_model_shared_kernel_score),
            },
            "residual_boundedness": {
                "components": residual_boundedness,
                "score": float(residual_boundedness_score),
            },
            "brain_plausibility": {
                "components": brain_plausibility,
                "score": float(brain_plausibility_score),
            },
            "explicitness": {
                "components": explicitness,
                "score": float(explicitness_score),
            },
        },
        "headline_metrics": {
            "feature_structure_coupling_score": float(feature_structure_coupling_score),
            "cross_model_shared_kernel_score": float(cross_model_shared_kernel_score),
            "residual_boundedness_score": float(residual_boundedness_score),
            "brain_plausibility_score": float(brain_plausibility_score),
            "explicitness_score": float(explicitness_score),
            "overall_p1_score": float(overall_score),
        },
        "hypotheses": {
            "H1_feature_and_structure_can_be_described_by_one_local_plasticity_law": bool(
                feature_structure_coupling_score >= 0.66
            ),
            "H2_shared_kernel_survives_across_models_and_tasks": bool(cross_model_shared_kernel_score >= 0.78),
            "H3_remaining_residual_does_not_destroy_the_core_guess": bool(residual_boundedness_score >= 0.60),
            "H4_brain_side_plausibility_is_nontrivial": bool(brain_plausibility_score >= 0.76),
            "H5_p1_structure_feature_cogeneration_is_moderately_supported": bool(overall_score >= 0.74),
        },
        "project_readout": {
            "summary": (
                "P1 is positive only if the project can stop treating feature extraction and structure formation as two "
                "separate stories and write them as one unified plasticity mechanism."
            ),
            "next_question": (
                "If P1 holds, the next step is a multi-timescale stabilization block: why does this co-generation law "
                "not collapse under continued data wash?"
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
