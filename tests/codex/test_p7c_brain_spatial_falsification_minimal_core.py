#!/usr/bin/env python
"""
P7C: push high-risk brain falsifiers against the minimal plasticity core and
make the 3D spatial-efficiency claim explicit.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="P7C brain spatial falsification for minimal core")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p7c_brain_spatial_falsification_minimal_core_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p5 = load_json(ROOT / "tests" / "codex_temp" / "p5_forward_brain_predictions_plasticity_coding_20260311.json")
    p7a = load_json(ROOT / "tests" / "codex_temp" / "p7a_structure_feature_coevolution_equation_20260311.json")
    p7b = load_json(ROOT / "tests" / "codex_temp" / "p7b_minimal_plasticity_core_compression_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")
    relation_boundary = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json"
    )
    d_problem = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    qwen_rel = relation_boundary["models"]["qwen3_4b"]["relations"]
    deepseek_rel = relation_boundary["models"]["deepseek_7b"]["relations"]
    qwen_summary = relation_boundary["models"]["qwen3_4b"]["global_summary"]["classification_bridge_mean"]
    deepseek_summary = relation_boundary["models"]["deepseek_7b"]["global_summary"]["classification_bridge_mean"]
    d_global = d_problem["global_summary"]
    d_models = {row["model"]: row for row in d_problem["models"]}

    falsifier_sharpness = {
        "stage8d_hard_falsifier_spec": float(stage8d["headline_metrics"]["hard_falsifier_spec_score"]),
        "p5_prediction_sharpness": float(p5["headline_metrics"]["prediction_sharpness_score"]),
        "stage8d_brain_specificity": float(stage8d["headline_metrics"]["brain_specificity_score"]),
        "stage8d_top3_component_risk": float(stage8d["headline_metrics"]["top3_component_risk_score"]),
    }
    falsifier_sharpness_score = mean(falsifier_sharpness.values())

    spatial_efficiency_signal = {
        "qwen_compact_boundary_bridge_advantage": normalize(
            float(qwen_summary["compact_boundary"] - qwen_summary["layer_cluster_only"]),
            0.03,
            0.08,
        ),
        "deepseek_compact_boundary_beats_distributed": normalize(
            float(deepseek_summary["compact_boundary"] - deepseek_summary["distributed_none"]),
            0.05,
            0.12,
        ),
        "compact_boundary_compactness_mean": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "topology_compactness"),
                classification_mean(deepseek_rel, "compact_boundary", "topology_compactness"),
            ]
        ),
        "compact_boundary_bridge_share_top8": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "top8_bridge_share_in_top20"),
                classification_mean(deepseek_rel, "compact_boundary", "top8_bridge_share_in_top20"),
            ]
        ),
    }
    spatial_efficiency_signal_score = mean(spatial_efficiency_signal.values())

    geometry_constraint = {
        "naive_geometry_not_sufficient": 1.0 if bool(d_global["all_models_fail_novel_and_retention"]) else 0.0,
        "best_geometry_overall_gain_is_not_positive": normalize(
            -float(d_global["best_overall_gain_across_methods"]),
            0.0,
            0.02,
        ),
        "qwen_geometry_hurts": normalize(
            -float(d_models["qwen3_4b"]["geometry_overall_gain"]),
            0.0,
            0.10,
        ),
        "deepseek_geometry_not_decisive": normalize(
            0.01 - abs(float(d_models["deepseek_7b"]["geometry_overall_gain"])),
            0.0,
            0.01,
        ),
    }
    geometry_constraint_score = mean(geometry_constraint.values())

    minimal_core_alignment = {
        "p7b_overall": float(p7b["headline_metrics"]["overall_p7b_score"]),
        "p7a_overall": float(p7a["headline_metrics"]["overall_p7a_score"]),
        "p7b_shared_parameterization": float(p7b["headline_metrics"]["shared_parameterization_score"]),
        "p7a_coupling_closure": float(p7a["headline_metrics"]["coupling_closure_score"]),
    }
    minimal_core_alignment_score = mean(minimal_core_alignment.values())

    brain_spatial_plausibility = {
        "p7a_brain_plausibility": float(p7a["headline_metrics"]["brain_plausibility_score"]),
        "p7b_brain_plausibility_after_compression": float(
            p7b["headline_metrics"]["brain_plausibility_after_compression_score"]
        ),
        "stage8d_directional_falsifier": float(stage8d["headline_metrics"]["directional_falsifier_score"]),
        "p5_independence": float(p5["headline_metrics"]["independence_from_current_fit_score"]),
    }
    brain_spatial_plausibility_score = mean(brain_spatial_plausibility.values())

    overall_score = mean(
        [
            falsifier_sharpness_score,
            spatial_efficiency_signal_score,
            geometry_constraint_score,
            minimal_core_alignment_score,
            brain_spatial_plausibility_score,
        ]
    )

    spatial_equations = {
        "wiring_cost": "C_wire = sum_{i,j} A_{ij} * d_{ij}",
        "delay_cost": "C_delay = sum_{i,j} A_{ij} * tau(d_{ij})",
        "packing_gain": "G_pack = sum_i log(1 + deg_i(local_3d_radius))",
        "effective_throughput": "T_eff = sum_paths pi(path) * info(path) / (1 + delay(path) + interference(path))",
        "spatial_efficiency": "E_3d = T_eff / (lambda_w * C_wire + lambda_d * C_delay + lambda_i * I_global)",
        "core_with_space": "A_{t+1} = (1 - l_A) * A_t + e_A * (1 - q_t) * (f_{t+1} f_{t+1}^T + d_t - p_t - lambda_s * D_3d)",
        "space_penalty": "D_3d(i,j) = d_{ij} / (1 + local_bundle_gain(i,j))",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p7c_brain_spatial_falsification_minimal_core",
        },
        "candidate_mechanism": {
            "title": "Brain Spatial Falsification For Minimal Core",
            "equations": spatial_equations,
            "core_claim": (
                "Three-dimensional efficiency does not come from naive geometry alone. It comes from local 3D wiring "
                "economy combined with a dynamic effective topology that selectively spends long-range cost only where "
                "feature demand and routing pressure justify it."
            ),
            "spatial_falsifiers": [
                "If broad long-range expansion beats selective compact bridge formation, the current 3D efficiency claim weakens.",
                "If naive geometry-only routing outperforms dynamic demand-shaped topology, the current law weakens.",
                "If compact-boundary relations stop showing bridge and compactness advantage over distributed ones, the spatial claim weakens.",
                "If brain-side gains reduce to generic geometry smoothing with no topology-specific asymmetry, the minimal core weakens.",
            ],
        },
        "pillars": {
            "falsifier_sharpness": {
                "components": falsifier_sharpness,
                "score": float(falsifier_sharpness_score),
            },
            "spatial_efficiency_signal": {
                "components": spatial_efficiency_signal,
                "score": float(spatial_efficiency_signal_score),
            },
            "geometry_constraint": {
                "components": geometry_constraint,
                "score": float(geometry_constraint_score),
            },
            "minimal_core_alignment": {
                "components": minimal_core_alignment,
                "score": float(minimal_core_alignment_score),
            },
            "brain_spatial_plausibility": {
                "components": brain_spatial_plausibility,
                "score": float(brain_spatial_plausibility_score),
            },
        },
        "headline_metrics": {
            "falsifier_sharpness_score": float(falsifier_sharpness_score),
            "spatial_efficiency_signal_score": float(spatial_efficiency_signal_score),
            "geometry_constraint_score": float(geometry_constraint_score),
            "minimal_core_alignment_score": float(minimal_core_alignment_score),
            "brain_spatial_plausibility_score": float(brain_spatial_plausibility_score),
            "overall_p7c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_minimal_core_has_sharp_brain_side_falsifiers": bool(falsifier_sharpness_score >= 0.82),
            "H2_spatial_efficiency_signal_is_nontrivial": bool(spatial_efficiency_signal_score >= 0.50),
            "H3_naive_geometry_alone_is_not_the_answer": bool(geometry_constraint_score >= 0.73),
            "H4_spatial_claim_stays_aligned_with_minimal_core": bool(minimal_core_alignment_score >= 0.76),
            "H5_p7c_brain_spatial_falsification_is_moderately_supported": bool(overall_score >= 0.73),
        },
        "project_readout": {
            "summary": (
                "P7C is positive only if the minimal plasticity core can state sharp brain-side falsifiers and explain "
                "why 3D efficiency comes from selective dynamic topology, not from naive geometry alone."
            ),
            "next_question": (
                "If P7C holds, the next stage should move from candidate theory to a full spatial-plasticity coding "
                "theory with explicit 3D efficiency terms and direct experimental predictions."
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
