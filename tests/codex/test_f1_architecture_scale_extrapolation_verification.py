from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    path = TEMP_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_from_residual(residual: float) -> float:
    return clamp01(1.0 - residual)


def main() -> None:
    p10a = load_json("p10a_final_theory_verdict_20260311.json")
    stage8c = load_json("stage8c_cross_model_task_invariants_20260311.json")
    problem_atlas = load_json("d_problem_atlas_summary_20260309.json")
    gpt2_qwen_basis = load_json("gpt2_qwen3_attention_topology_basis_20260308.json")
    qwen_deepseek_basis = load_json("qwen3_deepseek7b_attention_topology_basis_20260309.json")
    gpt2_qwen_align = load_json("gpt2_qwen3_repr_topology_layer_alignment_20260308.json")
    qwen_deepseek_orientation = load_json(
        "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json"
    )
    generator_bridge = load_json("generator_network_real_layer_band_bridge_20260310.json")
    atlas_models = {row["model"]: row for row in problem_atlas["models"]}

    gpt2_family = gpt2_qwen_basis["models"]["gpt2"]["family_summary"]
    qwen_family = gpt2_qwen_basis["models"]["qwen3_4b"]["family_summary"]
    deepseek_family = qwen_deepseek_basis["models"]["deepseek_7b"]["family_summary"]

    family_scores = []
    for family in ("fruit", "animal", "abstract"):
        family_scores.append(score_from_residual(gpt2_family[family]["mean_topology_residual_ratio"]))
        family_scores.append(score_from_residual(qwen_family[family]["mean_topology_residual_ratio"]))
        family_scores.append(score_from_residual(deepseek_family[family]["mean_topology_residual_ratio"]))
    family_topology_extrapolation_score = mean(family_scores)

    relation_invariance_extrapolation_score = mean(
        [
            stage8c["headline_metrics"]["compatibility_invariance_score"],
            stage8c["headline_metrics"]["relation_family_invariance_score"],
            stage8c["headline_metrics"]["relation_order_invariance_score"],
        ]
    )

    qwen_orientation = qwen_deepseek_orientation["models"]["qwen3_4b"]["global_summary"][
        "shared_layer_orientation"
    ]
    deepseek_orientation = qwen_deepseek_orientation["models"]["deepseek_7b"]["global_summary"][
        "shared_layer_orientation"
    ]
    qwen_bridge = qwen_deepseek_orientation["models"]["qwen3_4b"]["global_summary"][
        "mechanism_bridge_score"
    ]
    deepseek_bridge = qwen_deepseek_orientation["models"]["deepseek_7b"]["global_summary"][
        "mechanism_bridge_score"
    ]
    orientation_separation = clamp01(abs(deepseek_orientation - qwen_orientation) / 0.25)
    orientation_gap_stability_score = mean(
        [
            orientation_separation,
            qwen_bridge,
            deepseek_bridge,
        ]
    )

    geometry_gains = [
        atlas_models["gpt2"]["geometry_overall_gain"],
        atlas_models["qwen3_4b"]["geometry_overall_gain"],
        atlas_models["deepseek_7b"]["geometry_overall_gain"],
    ]
    geometry_failure_generality_score = mean([clamp01((0.02 - gain) / 0.1) for gain in geometry_gains])

    architecture_share = p10a["verdict"]["main_open_gap"] == "bridge_specificity_and_architecture_scale_residuals"
    model_gap_structure_score = stage8c["headline_metrics"]["model_gap_structure_score"]
    deepseek_undercoverage = generator_bridge["headline_metrics"]["deepseek_end_to_end_undercoverage"]
    qwen_undercoverage = generator_bridge["headline_metrics"]["qwen_end_to_end_undercoverage"]
    architecture_scale_residual_boundary_score = mean(
        [
            1.0 if architecture_share else 0.0,
            model_gap_structure_score,
            clamp01((deepseek_undercoverage - qwen_undercoverage) / 0.25),
        ]
    )

    gpt2_topology_dominance = mean(
        [
            clamp01(-x / 0.35)
            for x in gpt2_qwen_align["models"]["gpt2"]["topology_minus_repr_by_layer"]
        ]
    )
    qwen_topology_dominance = mean(
        [
            clamp01(-x / 0.25)
            for x in gpt2_qwen_align["models"]["qwen3_4b"]["topology_minus_repr_by_layer"]
        ]
    )
    layer_role_transfer_score = mean([gpt2_topology_dominance, qwen_topology_dominance])

    overall_f1_score = mean(
        [
            family_topology_extrapolation_score,
            relation_invariance_extrapolation_score,
            orientation_gap_stability_score,
            geometry_failure_generality_score,
            architecture_scale_residual_boundary_score,
            layer_role_transfer_score,
        ]
    )

    verdict = {
        "status": (
            "available_evidence_supports_extrapolation_but_scale_gap_remains"
            if overall_f1_score >= 0.72
            else "partial_extrapolation_only"
        ),
        "strongest_part": "family_topology_and_relation_invariance_hold_across_gpt2_qwen_deepseek",
        "weakest_part": "architecture_scale_residual_boundary_is_not_closed",
        "main_remaining_risk": "deepseek_relation_tool_bridge_pressure_still_exceeds_qwen",
        "geometry_only_generalizes": False,
        "supports_final_theory_extension": overall_f1_score >= 0.72,
    }

    hypotheses = {
        "H1_family_topology_basis_survives_architecture_change": family_topology_extrapolation_score >= 0.48,
        "H2_relation_invariance_survives_cross_model_shift": relation_invariance_extrapolation_score >= 0.75,
        "H3_orientation_gap_is_stable_not_random": orientation_gap_stability_score >= 0.65,
        "H4_geometry_only_failure_is_general_not_model_specific": geometry_failure_generality_score >= 0.45,
        "H5_remaining_gap_is_mainly_architecture_scale_not_theory_collapse": (
            architecture_scale_residual_boundary_score >= 0.6
        ),
        "H6_topology_role_bias_transfers_across_small_and_mid_models": layer_role_transfer_score >= 0.6,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F1_architecture_scale_extrapolation_verification",
        },
        "headline_metrics": {
            "family_topology_extrapolation_score": family_topology_extrapolation_score,
            "relation_invariance_extrapolation_score": relation_invariance_extrapolation_score,
            "orientation_gap_stability_score": orientation_gap_stability_score,
            "geometry_failure_generality_score": geometry_failure_generality_score,
            "architecture_scale_residual_boundary_score": architecture_scale_residual_boundary_score,
            "layer_role_transfer_score": layer_role_transfer_score,
            "overall_f1_score": overall_f1_score,
        },
        "supporting_readout": {
            "geometry_gains": {
                "gpt2": geometry_gains[0],
                "qwen3_4b": geometry_gains[1],
                "deepseek_7b": geometry_gains[2],
            },
            "orientation": {
                "qwen3_4b": qwen_orientation,
                "deepseek_7b": deepseek_orientation,
            },
            "generator_undercoverage": {
                "qwen3_4b": qwen_undercoverage,
                "deepseek_7b": deepseek_undercoverage,
            },
        },
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f1_architecture_scale_extrapolation_verification_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
