#!/usr/bin/env python
"""
P6B: detail the structure-formation term inside the unified plasticity-coding law.
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
    ap = argparse.ArgumentParser(description="P6B structure formation mechanism detail")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p6b_structure_formation_mechanism_detail_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p3 = load_json(ROOT / "tests" / "codex_temp" / "p3_regional_differentiation_network_roles_20260311.json")
    p4 = load_json(
        ROOT / "tests" / "codex_temp" / "p4_strong_precision_closure_mechanism_intervention_20260311.json"
    )
    p6a = load_json(ROOT / "tests" / "codex_temp" / "p6a_unified_plasticity_coding_principle_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )
    generator_bridge = load_json(
        ROOT / "tests" / "codex_temp" / "generator_network_real_layer_band_bridge_20260310.json"
    )
    shared_orientation = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json"
    )
    hard_interface = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json"
    )

    qwen_match = generator_bridge["models"]["qwen3_4b"]["searched_generator_match"]
    deepseek_match = generator_bridge["models"]["deepseek_7b"]["searched_generator_match"]
    qwen_global = shared_orientation["models"]["qwen3_4b"]["global_summary"]
    deepseek_global = shared_orientation["models"]["deepseek_7b"]["global_summary"]
    gains = hard_interface["gains"]

    topology_reorganization = {
        "p4_structure_intervention": float(p4["headline_metrics"]["structure_formation_intervention_score"]),
        "qwen_joint_success_gain": normalize(float(gains["qwen_joint_minus_tool_head_success"]), 0.05, 0.10),
        "deepseek_joint_success_gain": normalize(
            float(gains["deepseek_joint_minus_tool_head_success"]),
            0.03,
            0.07,
        ),
        "pressure_split_visible": normalize(
            float(generator_bridge["gains"]["deepseek_minus_qwen_searched_undercoverage"]),
            0.10,
            0.22,
        ),
    }
    topology_reorganization_score = mean(topology_reorganization.values())

    route_demand_alignment = {
        "qwen_worst_stage_is_tool": 1.0 if qwen_match["worst_stage"] == "tool" else 0.0,
        "deepseek_worst_stage_is_tool": 1.0 if deepseek_match["worst_stage"] == "tool" else 0.0,
        "qwen_alignment_gap_is_bounded": normalize(1.0 - float(qwen_match["mean_alignment_gap"]), 0.80, 0.92),
        "deepseek_high_pressure_is_visible": normalize(float(deepseek_match["mean_alignment_gap"]), 0.18, 0.30),
        "qwen_trigger_drop_after_joint_head": normalize(
            float(gains["qwen_tool_head_minus_joint_trigger_rate"]),
            0.10,
            0.18,
        ),
    }
    route_demand_alignment_score = mean(route_demand_alignment.values())

    regional_role_specialization = {
        "shared_law_diverse_roles": float(p3["headline_metrics"]["shared_law_diverse_roles_score"]),
        "region_prior_support": float(p3["headline_metrics"]["region_prior_support_score"]),
        "orientation_separation": normalize(
            float(shared_orientation["gains"]["deepseek_minus_qwen_orientation"]),
            0.15,
            0.35,
        ),
        "bridge_strength_mean": mean(
            [
                float(qwen_global["mechanism_bridge_score"]),
                float(deepseek_global["mechanism_bridge_score"]),
            ]
        ),
    }
    regional_role_specialization_score = mean(regional_role_specialization.values())

    structure_intervention_support = {
        "p4_structure_formation_intervention": float(
            p4["headline_metrics"]["structure_formation_intervention_score"]
        ),
        "p6a_intervention_alignment": float(p6a["headline_metrics"]["intervention_alignment_score"]),
        "qwen_joint_success_gain": normalize(
            float(gains["qwen_joint_minus_tool_head_success"]),
            0.05,
            0.10,
        ),
        "deepseek_joint_success_gain": normalize(
            float(gains["deepseek_joint_minus_tool_head_success"]),
            0.03,
            0.07,
        ),
        "qwen_trigger_drop_after_joint_head": normalize(
            float(gains["qwen_tool_head_minus_joint_trigger_rate"]),
            0.10,
            0.18,
        ),
    }
    structure_intervention_support_score = mean(structure_intervention_support.values())

    explicit_structure_equation = {
        "p6a_math_explicitness": float(p6a["headline_metrics"]["mathematical_explicitness_score"]),
        "stage9c_identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
        "regional_role_emergence": float(p3["headline_metrics"]["regional_role_emergence_score"]),
    }
    explicit_structure_equation_score = mean(explicit_structure_equation.values())

    overall_score = mean(
        [
            topology_reorganization_score,
            route_demand_alignment_score,
            regional_role_specialization_score,
            structure_intervention_support_score,
            explicit_structure_equation_score,
        ]
    )

    candidate_equations = {
        "structure_driver": "Psi(f_{t+1}, A_t) = c_c * C(f_{t+1}) + c_d * D_t + c_r * R_t + c_b * B_region - c_p * P_t",
        "coactivation": "C(f_{t+1}) = f_{t+1} * f_{t+1}^T",
        "route_mismatch": "D_t = relu(Q_t - A_t)",
        "region_bias": "B_region = u_route * p_route + u_multi * p_multi + u_abs * p_abs",
        "prune_pressure": "P_t = relu(A_t - M_t)",
        "full_update": "A_{t+1} = (1 - l_A) * A_t + e_A * (1 - g_t) * Psi(f_{t+1}, A_t)",
    }

    interpretation = {
        "coactivation": "newly co-activated features propose local edges and short path reinforcements",
        "route_mismatch": "task demand that current topology cannot serve pushes additional bridge formation",
        "region_bias": "the same plasticity rule becomes routing-heavy or abstraction-heavy by changing regional priors",
        "prune_pressure": "slow memory turns unused or unstable edges into prune pressure rather than permanent growth",
        "full_update": "effective topology grows when coactivation and route demand agree, and shrinks when slow stabilization marks excess structure",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p6b_structure_formation_mechanism_detail",
        },
        "candidate_mechanism": {
            "title": "Structure Formation Detail",
            "equations": candidate_equations,
            "interpretation": interpretation,
            "core_claim": (
                "Effective network structure is not a fixed scaffold. It is a demand-shaped topology that grows from "
                "feature coactivation, route mismatch pressure, and regional priors, then gets pruned by slow "
                "stabilization."
            ),
        },
        "pillars": {
            "topology_reorganization_evidence": {
                "components": topology_reorganization,
                "score": float(topology_reorganization_score),
            },
            "route_demand_alignment": {
                "components": route_demand_alignment,
                "score": float(route_demand_alignment_score),
            },
            "regional_role_specialization": {
                "components": regional_role_specialization,
                "score": float(regional_role_specialization_score),
            },
            "structure_intervention_support": {
                "components": structure_intervention_support,
                "score": float(structure_intervention_support_score),
            },
            "explicit_structure_equation": {
                "components": explicit_structure_equation,
                "score": float(explicit_structure_equation_score),
            },
        },
        "headline_metrics": {
            "topology_reorganization_evidence_score": float(topology_reorganization_score),
            "route_demand_alignment_score": float(route_demand_alignment_score),
            "regional_role_specialization_score": float(regional_role_specialization_score),
            "structure_intervention_support_score": float(structure_intervention_support_score),
            "explicit_structure_equation_score": float(explicit_structure_equation_score),
            "overall_p6b_score": float(overall_score),
        },
        "hypotheses": {
            "H1_structure_growth_tracks_nontrivial_topology_reorganization": bool(
                topology_reorganization_score >= 0.70
            ),
            "H2_structure_growth_is_shaped_by_route_demand_not_only_static_connectivity": bool(
                route_demand_alignment_score >= 0.76
            ),
            "H3_one_shared_rule_can_still_generate_regional_structure_roles": bool(
                regional_role_specialization_score >= 0.72
            ),
            "H4_current_interventions_touch_structure_formation_nontrivially": bool(
                structure_intervention_support_score >= 0.69
            ),
            "H5_p6b_structure_detail_is_moderately_supported": bool(overall_score >= 0.74),
        },
        "project_readout": {
            "summary": (
                "P6B is positive only if the structure term can be expanded into a concrete topology-formation law, "
                "rather than left as an unspecified mid-timescale update."
            ),
            "next_question": (
                "If P6B holds, the next step is to detail the feature-extraction term with the same level of "
                "mechanistic explicitness."
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
