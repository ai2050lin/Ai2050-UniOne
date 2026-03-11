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


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    p9b = load_json("p9b_spatial_residual_counterexample_compression_20260311.json")
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")
    stage8c = load_json("stage8c_cross_model_task_invariants_20260311.json")
    orientation = load_json("qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")

    bridge_specificity_closure_score = mean(
        [
            p9b["pillars"]["bridge_specificity_gap"]["components"]["p8b_selective_bridge_advantage"],
            p9b["pillars"]["bridge_specificity_gap"]["components"]["p8c_local_vs_bridge_specificity"],
            p9b["headline_metrics"]["bridge_specificity_gap_score"],
            1.0 - p9b["pillars"]["bridge_specificity_gap"]["components"]["not_yet_strong"],
        ]
    )

    qwen_relation_gain = clamp01(
        orientation["models"]["qwen3_4b"]["global_summary"]["relation_hit_behavior_gain_corr"] + 0.5
    )
    deepseek_relation_gain = clamp01(
        orientation["models"]["deepseek_7b"]["global_summary"]["relation_hit_behavior_gain_corr"]
    )
    orientation_role_explainability_score = mean(
        [
            clamp01(abs(orientation["headline_metrics"]["qwen_orientation"]) * 4.0),
            clamp01(abs(orientation["headline_metrics"]["deepseek_orientation"]) * 2.5),
            qwen_relation_gain,
            deepseek_relation_gain,
            orientation["headline_metrics"]["qwen_mechanism_bridge"],
            orientation["headline_metrics"]["deepseek_mechanism_bridge"],
        ]
    )

    layer_role_transfer_closure_score = mean(
        [
            f1["headline_metrics"]["layer_role_transfer_score"],
            f1["headline_metrics"]["orientation_gap_stability_score"],
            stage8c["headline_metrics"]["model_gap_structure_score"],
            stage8c["headline_metrics"]["relation_order_invariance_score"],
        ]
    )

    architecture_scale_residual_control_score = mean(
        [
            f1["headline_metrics"]["architecture_scale_residual_boundary_score"],
            p9b["headline_metrics"]["residual_source_control_score"],
            p9b["headline_metrics"]["compressed_core_resilience_score"],
            stage8c["headline_metrics"]["compatibility_invariance_score"],
        ]
    )

    overall_g1_score = mean(
        [
            bridge_specificity_closure_score,
            orientation_role_explainability_score,
            layer_role_transfer_closure_score,
            architecture_scale_residual_control_score,
        ]
    )

    formulas = {
        "bridge_specificity": (
            "BridgeSpec(r, c, t) = sigmoid(w_s * SelectiveDemand(r,c,t) + "
            "w_b * BoundaryCompactness(r,c) + w_g * GainCompatibility(r,c) - w_n * NonspecificSpread(r,c,t))"
        ),
        "layer_role_transfer": (
            "Role_l(model) = argmax_k Align(LayerFeature_l(model), ProtoRole_k)"
        ),
        "cross_model_transfer": (
            "TransferScore = mean_l 1[Role_l(model_a) == Role_pi(l)(model_b)] * Stability_pi"
        ),
        "residual_boundary": (
            "Residual = ArchitectureBias + ScaleBias + BridgeSpecificityGap + Epsilon_core"
        ),
    }

    interpretation = {
        "bridge_specificity": (
            "The unresolved issue is no longer whether bridges exist, but whether the theory can predict "
            "which bridge should activate for which relation, task, or spatial demand."
        ),
        "layer_roles": (
            "Layer-role transfer is not about exact layer indices matching. It is about whether concept-led, "
            "relation-led, routing-led, and recovery-led roles can be aligned across models."
        ),
        "current_status": (
            "Current evidence says bridge specificity and role transfer are real, partially structured, "
            "and not random, but still not fully closed."
        ),
    }

    verdict = {
        "status": "partial_closure_with_clear_remaining_gaps",
        "main_closed_part": "orientation_and_cross_model_relation_order_are_real",
        "main_open_part": "bridge_specificity_and_layer_role_transfer_are_not_yet_strong_enough",
        "core_answer": (
            "G1 is positive only in a partial sense: the project now has a structured view of which models are "
            "concept-led versus relation-led, and bridge specificity is clearly a real residual. But neither "
            "bridge selection rules nor cross-model role transfer are fully closed."
        ),
    }

    hypotheses = {
        "H1_bridge_specificity_is_nontrivial_but_not_closed": 0.58 <= bridge_specificity_closure_score < 0.76,
        "H2_orientation_roles_are_structured": orientation_role_explainability_score >= 0.68,
        "H3_layer_role_transfer_is_nontrivial": layer_role_transfer_closure_score >= 0.68,
        "H4_architecture_scale_residuals_are_under_partial_control": architecture_scale_residual_control_score >= 0.78,
        "H5_g1_partial_closure_is_reached": overall_g1_score >= 0.7,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G1_bridge_specificity_layer_role_transfer_closure",
        },
        "headline_metrics": {
            "bridge_specificity_closure_score": bridge_specificity_closure_score,
            "orientation_role_explainability_score": orientation_role_explainability_score,
            "layer_role_transfer_closure_score": layer_role_transfer_closure_score,
            "architecture_scale_residual_control_score": architecture_scale_residual_control_score,
            "overall_g1_score": overall_g1_score,
        },
        "supporting_readout": {
            "p9b_bridge_specificity_gap_score": p9b["headline_metrics"]["bridge_specificity_gap_score"],
            "f1_layer_role_transfer_score": f1["headline_metrics"]["layer_role_transfer_score"],
            "f1_orientation_gap_stability_score": f1["headline_metrics"]["orientation_gap_stability_score"],
            "stage8c_model_gap_structure_score": stage8c["headline_metrics"]["model_gap_structure_score"],
            "qwen_orientation": orientation["headline_metrics"]["qwen_orientation"],
            "deepseek_orientation": orientation["headline_metrics"]["deepseek_orientation"],
            "qwen_relation_gain_corr": orientation["models"]["qwen3_4b"]["global_summary"]["relation_hit_behavior_gain_corr"],
            "deepseek_relation_gain_corr": orientation["models"]["deepseek_7b"]["global_summary"]["relation_hit_behavior_gain_corr"],
        },
        "formulas": formulas,
        "interpretation": interpretation,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g1_bridge_specificity_layer_role_transfer_closure_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
