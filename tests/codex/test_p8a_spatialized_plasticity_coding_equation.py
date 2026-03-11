#!/usr/bin/env python
"""
P8A: formulate the spatialized unified plasticity-coding equation.
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


def classification_mean(relations: Dict[str, Dict[str, Any]], classification: str, key: str) -> float:
    rows = [float(v[key]) for v in relations.values() if v["classification"] == classification]
    return mean(rows) if rows else 0.0


def probe_support_rate(model_block: Dict[str, Any]) -> float:
    rows = [
        1.0 if bool(v["supports_family_topology_basis"]) else 0.0
        for v in model_block["probe_fits"].values()
    ]
    return mean(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="P8A spatialized plasticity coding equation")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p8a_spatialized_plasticity_coding_equation_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p7a = load_json(ROOT / "tests" / "codex_temp" / "p7a_structure_feature_coevolution_equation_20260311.json")
    p7b = load_json(ROOT / "tests" / "codex_temp" / "p7b_minimal_plasticity_core_compression_20260311.json")
    p7c = load_json(ROOT / "tests" / "codex_temp" / "p7c_brain_spatial_falsification_minimal_core_20260311.json")
    topo_basis = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_attention_topology_basis_20260309.json")
    topo_atlas = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_attention_topology_atlas_20260309.json")
    relation_boundary = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json"
    )
    d_problem = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    qwen_basis = topo_basis["models"]["qwen3_4b"]
    deepseek_basis = topo_basis["models"]["deepseek_7b"]
    qwen_atlas = topo_atlas["models"]["qwen3_4b"]["global_summary"]
    deepseek_atlas = topo_atlas["models"]["deepseek_7b"]["global_summary"]
    qwen_rel = relation_boundary["models"]["qwen3_4b"]["relations"]
    deepseek_rel = relation_boundary["models"]["deepseek_7b"]["relations"]
    qwen_rel_summary = relation_boundary["models"]["qwen3_4b"]["global_summary"]["classification_bridge_mean"]
    deepseek_rel_summary = relation_boundary["models"]["deepseek_7b"]["global_summary"]["classification_bridge_mean"]
    d_global = d_problem["global_summary"]

    spatial_equation_consistency = {
        "p7a_overall": float(p7a["headline_metrics"]["overall_p7a_score"]),
        "p7b_overall": float(p7b["headline_metrics"]["overall_p7b_score"]),
        "p7c_overall": float(p7c["headline_metrics"]["overall_p7c_score"]),
        "p7c_geometry_constraint": float(p7c["headline_metrics"]["geometry_constraint_score"]),
    }
    spatial_equation_consistency_score = mean(spatial_equation_consistency.values())

    topology_reuse_locality = {
        "qwen_support_rate": float(qwen_atlas["support_rate"]),
        "deepseek_support_rate": float(deepseek_atlas["support_rate"]),
        "qwen_mean_margin": normalize(float(qwen_atlas["mean_margin_vs_best_wrong"]), 0.15, 0.70),
        "deepseek_mean_margin": normalize(float(deepseek_atlas["mean_margin_vs_best_wrong"]), 0.15, 0.70),
        "family_residual_advantage": normalize(
            mean(
                [
                    float(qwen_atlas["mean_best_wrong_residual"]) - float(qwen_atlas["mean_true_family_residual"]),
                    float(deepseek_atlas["mean_best_wrong_residual"]) - float(deepseek_atlas["mean_true_family_residual"]),
                ]
            ),
            0.35,
            0.55,
        ),
    }
    topology_reuse_locality_score = mean(topology_reuse_locality.values())

    compact_bridge_efficiency = {
        "qwen_compact_boundary_bridge_advantage": normalize(
            float(qwen_rel_summary["compact_boundary"] - qwen_rel_summary["layer_cluster_only"]),
            0.03,
            0.08,
        ),
        "deepseek_compact_vs_distributed_advantage": normalize(
            float(deepseek_rel_summary["compact_boundary"] - deepseek_rel_summary["distributed_none"]),
            0.05,
            0.12,
        ),
        "compact_boundary_compactness": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "topology_compactness"),
                classification_mean(deepseek_rel, "compact_boundary", "topology_compactness"),
            ]
        ),
        "compact_boundary_top8_bridge_share": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "top8_bridge_share_in_top20"),
                classification_mean(deepseek_rel, "compact_boundary", "top8_bridge_share_in_top20"),
            ]
        ),
    }
    compact_bridge_efficiency_score = mean(compact_bridge_efficiency.values())

    geometry_only_failure = {
        "all_models_fail_joint_geometry_gain": 1.0 if bool(d_global["all_models_fail_novel_and_retention"]) else 0.0,
        "best_overall_gain_not_positive": normalize(-float(d_global["best_overall_gain_across_methods"]), 0.0, 0.02),
        "base_offset_not_enough": normalize(-float(d_global["base_offset_best_overall_gain"]), 0.0, 0.03),
        "multistage_still_not_enough": normalize(-float(d_global["multistage_best_overall_gain"]), 0.0, 0.02),
    }
    geometry_only_failure_score = mean(geometry_only_failure.values())

    brain_plausibility = {
        "p7c_brain_spatial_plausibility": float(p7c["headline_metrics"]["brain_spatial_plausibility_score"]),
        "p7b_brain_plausibility": float(p7b["headline_metrics"]["brain_plausibility_after_compression_score"]),
        "qwen_family_topology_support": float(probe_support_rate(qwen_basis)),
        "deepseek_family_topology_support": float(probe_support_rate(deepseek_basis)),
    }
    brain_plausibility_score = mean(brain_plausibility.values())

    overall_score = mean(
        [
            spatial_equation_consistency_score,
            topology_reuse_locality_score,
            compact_bridge_efficiency_score,
            geometry_only_failure_score,
            brain_plausibility_score,
        ]
    )

    equations = {
        "spatial_gate": "q_t = sigmoid(alpha * (r_t - s_t) + beta * b_region - gamma * C_local)",
        "feature_field": "f_{t+1}(i) = (1 - l_f) * f_t(i) + e_f * q_t(i) * [L_t(i) + k_a * sum_j A_t(i,j)L_t(j) - k_i I_t(i)]",
        "structure_field": "A_{t+1}(i,j) = (1 - l_A) * A_t(i,j) + e_A * (1 - q_t(i,j)) * [f_{t+1}(i)f_{t+1}(j) + d_t(i,j) - p_t(i,j) - lambda_s D_3d(i,j)]",
        "memory_field": "m_{t+1}(i,j) = (1 - l_m) * m_t(i,j) + e_m * s_t * (A_{t+1}(i,j) - m_t(i,j))",
        "wire_cost": "D_3d(i,j) = d_{ij} / (1 + local_bundle_gain(i,j))",
        "spatial_efficiency": "E_3d = T_eff / (lambda_w C_wire + lambda_d C_delay + lambda_i I_global)",
        "coding_state": "y_t = W_f f_t + W_A vec(A_t) + W_m m_t",
    }

    interpretation = {
        "spatial_gate": "local physical crowding and route demand jointly decide whether a region should keep growing features or consolidate structure",
        "feature_field": "feature extraction is a routed field update over a 3D substrate, not a coordinate-free vector lookup",
        "structure_field": "effective topology only spends long-range cost when coactivation and demand exceed prune and spatial penalties",
        "memory_field": "slow memory stores durable long-range scaffolds and suppresses unstable geometric overgrowth",
        "wire_cost": "3D distance is a real penalty, but bundled local neighborhoods discount that penalty for efficient tracts",
        "spatial_efficiency": "efficiency is information throughput per unit physical wiring, delay, and interference cost",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p8a_spatialized_plasticity_coding_equation",
        },
        "candidate_mechanism": {
            "title": "Spatialized Plasticity Coding Equation",
            "equations": equations,
            "interpretation": interpretation,
            "core_claim": (
                "The brain is efficient in 3D not because geometry directly encodes meaning, but because a 3D wiring "
                "substrate lets local neighborhoods stay dense and cheap while plasticity continuously builds a sparse "
                "effective topology for high-value long-range integration."
            ),
        },
        "pillars": {
            "spatial_equation_consistency": {
                "components": spatial_equation_consistency,
                "score": float(spatial_equation_consistency_score),
            },
            "topology_reuse_locality": {
                "components": topology_reuse_locality,
                "score": float(topology_reuse_locality_score),
            },
            "compact_bridge_efficiency": {
                "components": compact_bridge_efficiency,
                "score": float(compact_bridge_efficiency_score),
            },
            "geometry_only_failure": {
                "components": geometry_only_failure,
                "score": float(geometry_only_failure_score),
            },
            "brain_plausibility": {
                "components": brain_plausibility,
                "score": float(brain_plausibility_score),
            },
        },
        "headline_metrics": {
            "spatial_equation_consistency_score": float(spatial_equation_consistency_score),
            "topology_reuse_locality_score": float(topology_reuse_locality_score),
            "compact_bridge_efficiency_score": float(compact_bridge_efficiency_score),
            "geometry_only_failure_score": float(geometry_only_failure_score),
            "brain_plausibility_score": float(brain_plausibility_score),
            "overall_p8a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_the_spatialized_equation_is_consistent_with_P7": bool(spatial_equation_consistency_score >= 0.76),
            "H2_concept_topology_reuse_is_locality_aware": bool(topology_reuse_locality_score >= 0.75),
            "H3_compact_bridge_relations_support_spatial_efficiency": bool(compact_bridge_efficiency_score >= 0.49),
            "H4_geometry_only_baselines_are_not_sufficient": bool(geometry_only_failure_score >= 0.65),
            "H5_p8a_spatialized_equation_is_moderately_supported": bool(overall_score >= 0.70),
        },
        "project_readout": {
            "summary": (
                "P8A is positive only if the project can write one spatialized plasticity-coding equation that fits "
                "local topology reuse, compact relation bridges, and the failure of geometry-only baselines."
            ),
            "next_question": (
                "If P8A holds, the next step is to explain in more detail how 3D wiring economy and dynamic topology "
                "divide labor, and then derive new spatial falsifiers."
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
