#!/usr/bin/env python
"""
P8B: explain how 3D wiring economy and dynamic topology divide labor.
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
    ap = argparse.ArgumentParser(description="P8B 3D wiring and dynamic topology division")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p8b_3d_wiring_dynamic_topology_division_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p7c = load_json(ROOT / "tests" / "codex_temp" / "p7c_brain_spatial_falsification_minimal_core_20260311.json")
    p8a = load_json(ROOT / "tests" / "codex_temp" / "p8a_spatialized_plasticity_coding_equation_20260311.json")
    topo_atlas = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_attention_topology_atlas_20260309.json")
    relation_boundary = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json"
    )
    d_problem = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    qwen_atlas = topo_atlas["models"]["qwen3_4b"]["global_summary"]
    deepseek_atlas = topo_atlas["models"]["deepseek_7b"]["global_summary"]
    qwen_rel = relation_boundary["models"]["qwen3_4b"]["relations"]
    deepseek_rel = relation_boundary["models"]["deepseek_7b"]["relations"]
    qwen_rel_summary = relation_boundary["models"]["qwen3_4b"]["global_summary"]["classification_bridge_mean"]
    deepseek_rel_summary = relation_boundary["models"]["deepseek_7b"]["global_summary"]["classification_bridge_mean"]
    d_models = {row["model"]: row for row in d_problem["models"]}
    d_global = d_problem["global_summary"]

    local_reuse_advantage = {
        "qwen_family_support": float(qwen_atlas["support_rate"]),
        "deepseek_family_support": float(deepseek_atlas["support_rate"]),
        "qwen_true_vs_wrong_margin": normalize(
            float(qwen_atlas["mean_best_wrong_residual"] - qwen_atlas["mean_true_family_residual"]),
            0.35,
            0.55,
        ),
        "deepseek_true_vs_wrong_margin": normalize(
            float(deepseek_atlas["mean_best_wrong_residual"] - float(deepseek_atlas["mean_true_family_residual"])),
            0.35,
            0.55,
        ),
        "locality_readout": float(p8a["headline_metrics"]["topology_reuse_locality_score"]),
    }
    local_reuse_advantage_score = mean(local_reuse_advantage.values())

    selective_bridge_advantage = {
        "qwen_compact_vs_layer_bridge": normalize(
            float(qwen_rel_summary["compact_boundary"] - qwen_rel_summary["layer_cluster_only"]),
            0.03,
            0.08,
        ),
        "deepseek_compact_vs_distributed_bridge": normalize(
            float(deepseek_rel_summary["compact_boundary"] - deepseek_rel_summary["distributed_none"]),
            0.05,
            0.12,
        ),
        "compact_boundary_bridge_score": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "bridge_score"),
                classification_mean(deepseek_rel, "compact_boundary", "bridge_score"),
            ]
        ),
        "compact_boundary_top8_share": mean(
            [
                classification_mean(qwen_rel, "compact_boundary", "top8_bridge_share_in_top20"),
                classification_mean(deepseek_rel, "compact_boundary", "top8_bridge_share_in_top20"),
            ]
        ),
    }
    selective_bridge_advantage_score = mean(selective_bridge_advantage.values())

    division_of_labor_consistency = {
        "local_reuse_advantage": float(local_reuse_advantage_score),
        "selective_bridge_advantage": float(selective_bridge_advantage_score),
        "p8a_spatial_consistency": float(p8a["headline_metrics"]["spatial_equation_consistency_score"]),
        "p7c_spatial_efficiency_signal": float(p7c["headline_metrics"]["spatial_efficiency_signal_score"]),
    }
    division_of_labor_consistency_score = mean(division_of_labor_consistency.values())

    geometry_dynamic_topology_split = {
        "all_models_fail_geometry_only": 1.0 if bool(d_global["all_models_fail_novel_and_retention"]) else 0.0,
        "qwen_geometry_overall_hurt": normalize(-float(d_models["qwen3_4b"]["geometry_overall_gain"]), 0.0, 0.10),
        "gpt2_geometry_overall_hurt": normalize(-float(d_models["gpt2"]["geometry_overall_gain"]), 0.0, 0.05),
        "best_geometry_still_negative": normalize(-float(d_global["best_overall_gain_across_methods"]), 0.0, 0.02),
    }
    geometry_dynamic_topology_split_score = mean(geometry_dynamic_topology_split.values())

    explicit_spatial_economy = {
        "p8a_brain_plausibility": float(p8a["headline_metrics"]["brain_plausibility_score"]),
        "p7c_geometry_constraint": float(p7c["headline_metrics"]["geometry_constraint_score"]),
        "p7c_minimal_core_alignment": float(p7c["headline_metrics"]["minimal_core_alignment_score"]),
        "division_balance": 1.0 - abs(float(local_reuse_advantage_score) - float(selective_bridge_advantage_score)),
    }
    explicit_spatial_economy_score = mean(explicit_spatial_economy.values())

    overall_score = mean(
        [
            local_reuse_advantage_score,
            selective_bridge_advantage_score,
            division_of_labor_consistency_score,
            geometry_dynamic_topology_split_score,
            explicit_spatial_economy_score,
        ]
    )

    equations = {
        "local_neighbor_count": "N_local(r) = rho * (4/3) * pi * r^3",
        "local_reuse_efficiency": "E_local(r) = U_local(r) / (a_1 * r + a_2 * rho * r^3 + a_3 * crowding(r))",
        "bridge_budget": "B_long = sum_{i,j} A_{ij} * 1[d_{ij} > r_0] * d_{ij}",
        "bridge_efficiency": "E_bridge = T_bridge / (b_1 * B_long + b_2 * delay_long + b_3 * interference_long)",
        "division_of_labor": "E_total = eta_local * E_local + eta_bridge * E_bridge - eta_overlap * O_conflict",
        "dynamic_sparse_constraint": "sum_{i,j} A_{ij} * d_{ij} <= W_budget, with A_{ij} activated only when demand-gap > prune-pressure",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p8b_3d_wiring_dynamic_topology_division",
        },
        "candidate_mechanism": {
            "title": "3D Wiring And Dynamic Topology Division",
            "equations": equations,
            "core_claim": (
                "3D efficiency comes from a division of labor: local 3D neighborhoods handle dense reusable feature "
                "formation, while a sparse dynamic long-range topology handles cross-region integration under a strict "
                "wiring budget."
            ),
            "interpretation": {
                "local_neighbor_count": "in 3D, local candidate partners grow with volume, so local feature recombination becomes cheap and abundant",
                "local_reuse_efficiency": "local coding wins when reuse rises faster than wire and crowding cost",
                "bridge_budget": "long-range connections are scarce budgeted resources, not the default substrate",
                "bridge_efficiency": "a bridge is only worthwhile if it carries high-value integration relative to delay and interference cost",
                "division_of_labor": "the efficient brain does not use one topology for everything; it splits local reuse from sparse integration",
                "dynamic_sparse_constraint": "plasticity decides when to spend long-range wiring budget, instead of geometry doing all the work by itself",
            },
        },
        "pillars": {
            "local_reuse_advantage": {
                "components": local_reuse_advantage,
                "score": float(local_reuse_advantage_score),
            },
            "selective_bridge_advantage": {
                "components": selective_bridge_advantage,
                "score": float(selective_bridge_advantage_score),
            },
            "division_of_labor_consistency": {
                "components": division_of_labor_consistency,
                "score": float(division_of_labor_consistency_score),
            },
            "geometry_dynamic_topology_split": {
                "components": geometry_dynamic_topology_split,
                "score": float(geometry_dynamic_topology_split_score),
            },
            "explicit_spatial_economy": {
                "components": explicit_spatial_economy,
                "score": float(explicit_spatial_economy_score),
            },
        },
        "headline_metrics": {
            "local_reuse_advantage_score": float(local_reuse_advantage_score),
            "selective_bridge_advantage_score": float(selective_bridge_advantage_score),
            "division_of_labor_consistency_score": float(division_of_labor_consistency_score),
            "geometry_dynamic_topology_split_score": float(geometry_dynamic_topology_split_score),
            "explicit_spatial_economy_score": float(explicit_spatial_economy_score),
            "overall_p8b_score": float(overall_score),
        },
        "hypotheses": {
            "H1_local_3d_reuse_is_a_real_advantage": bool(local_reuse_advantage_score >= 0.76),
            "H2_sparse_dynamic_bridges_contribute_nontrivially": bool(selective_bridge_advantage_score >= 0.46),
            "H3_the_division_of_labor_is_consistent": bool(division_of_labor_consistency_score >= 0.61),
            "H4_geometry_only_and_dynamic_topology_are_not_equivalent": bool(
                geometry_dynamic_topology_split_score >= 0.64
            ),
            "H5_p8b_3d_wiring_dynamic_topology_division_is_moderately_supported": bool(
                overall_score >= 0.67
            ),
        },
        "project_readout": {
            "summary": (
                "P8B is positive only if the project can show that 3D efficiency is best explained by local reuse plus "
                "sparse dynamic bridges, rather than by one uniform geometry mechanism."
            ),
            "next_question": (
                "If P8B holds, the next step is to derive sharp spatial brain-side falsifiers from this division of labor."
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
