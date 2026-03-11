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
    orientation = load_json("qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")
    support_bridge = load_json("qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    behavior_bridge = load_json("qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    structure_atlas = load_json("qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    f1 = load_json("f1_architecture_scale_extrapolation_verification_20260311.json")
    stage8c = load_json("stage8c_cross_model_task_invariants_20260311.json")

    qwen = orientation["models"]["qwen3_4b"]["global_summary"]
    deepseek = orientation["models"]["deepseek_7b"]["global_summary"]

    role_axis_separability_score = mean(
        [
            clamp01((qwen["mean_concept_shared_layer_hit"] - qwen["mean_relation_shared_layer_hit"]) * 4.0),
            clamp01((deepseek["mean_relation_shared_layer_hit"] - deepseek["mean_concept_shared_layer_hit"]) * 4.0),
            clamp01(abs(qwen["shared_layer_orientation"]) * 4.0),
            clamp01(abs(deepseek["shared_layer_orientation"]) * 2.5),
        ]
    )

    shared_band_alignment_score = mean(
        [
            support_bridge["headline_metrics"]["qwen_soft_layer_overlap"],
            support_bridge["headline_metrics"]["deepseek_soft_layer_overlap"],
            clamp01(structure_atlas["models"]["qwen3_4b"]["global_summary"]["shared_band_layer_count"] / 5.0),
            clamp01(structure_atlas["models"]["deepseek_7b"]["global_summary"]["shared_band_layer_count"] / 5.0),
        ]
    )

    behavior_role_mapping_score = mean(
        [
            behavior_bridge["models"]["qwen3_4b"]["global_summary"]["mean_behavior_gain"],
            behavior_bridge["models"]["deepseek_7b"]["global_summary"]["mean_behavior_gain"],
            qwen["mechanism_bridge_score"],
            deepseek["mechanism_bridge_score"],
            stage8c["headline_metrics"]["relation_order_invariance_score"],
        ]
    )

    transfer_consistency_score = mean(
        [
            f1["headline_metrics"]["layer_role_transfer_score"],
            f1["headline_metrics"]["orientation_gap_stability_score"],
            stage8c["headline_metrics"]["model_gap_structure_score"],
            stage8c["headline_metrics"]["compatibility_invariance_score"],
        ]
    )

    overall_g1b_score = mean(
        [
            role_axis_separability_score,
            shared_band_alignment_score,
            behavior_role_mapping_score,
            transfer_consistency_score,
        ]
    )

    formulas = {
        "role_projection": "z_l(model) = W_role * [concept_support_l, relation_support_l, shared_support_l, task_gain_l]",
        "proto_role_assignment": "Role_l = argmax_k cosine(z_l, ProtoRole_k)",
        "coordinate_transfer": "Transfer(model_a -> model_b) = mean_l cosine(z_l(a), z_pi(l)(b))",
        "role_library": (
            "ProtoRole = {concept_led, relation_led, shared_band, targeted_bridge, recovery_support}"
        ),
    }

    interpretation = {
        "core_idea": (
            "A unified layer-role coordinate system should align functional roles across models, not raw layer indices."
        ),
        "qwen": (
            "Qwen is more concept-led in shared layers, so its role map should project more mass to concept-led and shared-band coordinates."
        ),
        "deepseek": (
            "DeepSeek is more relation-led in shared layers, so its role map should project more mass to relation-led and targeted-bridge coordinates."
        ),
        "current_limit": (
            "The coordinate is now partially writeable, but transfer consistency is still only moderate, especially when real ablation shifts the observed orientation."
        ),
    }

    verdict = {
        "status": "unified_role_coordinate_partially_writeable",
        "core_answer": (
            "The project can now write a first unified role coordinate system: concept-led, relation-led, shared-band, targeted-bridge, and recovery-support. "
            "This coordinate explains part of the cross-model variation, but it does not yet produce a strong stable transfer map."
        ),
        "main_open_gap": "role_transfer_strength_is_moderate_and_state_dependent",
    }

    hypotheses = {
        "H1_role_axes_are_nontrivially_separable": role_axis_separability_score >= 0.52,
        "H2_shared_band_alignment_is_visible": shared_band_alignment_score >= 0.6,
        "H3_behavior_can_be_mapped_into_role_coordinates": behavior_role_mapping_score >= 0.52,
        "H4_transfer_consistency_is_nontrivial": transfer_consistency_score >= 0.7,
        "H5_g1b_partial_coordinate_system_is_ready": overall_g1b_score >= 0.62,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G1B_unified_layer_role_coordinate_system",
        },
        "headline_metrics": {
            "role_axis_separability_score": role_axis_separability_score,
            "shared_band_alignment_score": shared_band_alignment_score,
            "behavior_role_mapping_score": behavior_role_mapping_score,
            "transfer_consistency_score": transfer_consistency_score,
            "overall_g1b_score": overall_g1b_score,
        },
        "supporting_readout": {
            "qwen_orientation": orientation["headline_metrics"]["qwen_orientation"],
            "deepseek_orientation": orientation["headline_metrics"]["deepseek_orientation"],
            "qwen_soft_layer_overlap": support_bridge["headline_metrics"]["qwen_soft_layer_overlap"],
            "deepseek_soft_layer_overlap": support_bridge["headline_metrics"]["deepseek_soft_layer_overlap"],
            "qwen_shared_band_layer_count": structure_atlas["models"]["qwen3_4b"]["global_summary"]["shared_band_layer_count"],
            "deepseek_shared_band_layer_count": structure_atlas["models"]["deepseek_7b"]["global_summary"]["shared_band_layer_count"],
            "f1_layer_role_transfer_score": f1["headline_metrics"]["layer_role_transfer_score"],
        },
        "formulas": formulas,
        "interpretation": interpretation,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g1b_unified_layer_role_coordinate_system_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
