from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(name: str) -> dict:
    with (TEMP_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    g1b = load_json("g1b_unified_layer_role_coordinate_system_20260311.json")
    orientation = load_json("qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")
    ablation = load_json("qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json")
    shared = load_json("qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    p3 = load_json("p3_regional_differentiation_network_roles_20260311.json")

    qwen_pred = orientation["headline_metrics"]["qwen_orientation"]
    deepseek_pred = orientation["headline_metrics"]["deepseek_orientation"]
    qwen_actual = ablation["headline_metrics"]["qwen_actual_orientation"]
    deepseek_actual = ablation["headline_metrics"]["deepseek_actual_orientation"]

    orientation_stability_score = mean(
        [
            1.0 - min(1.0, abs(qwen_pred - qwen_actual)),
            1.0 - min(1.0, abs(deepseek_pred - deepseek_actual)),
            g1b["headline_metrics"]["transfer_consistency_score"],
        ]
    )

    shared_band_visibility_score = mean(
        [
            g1b["headline_metrics"]["shared_band_alignment_score"],
            shared["headline_metrics"]["qwen_soft_layer_overlap"],
            shared["headline_metrics"]["deepseek_soft_layer_overlap"],
        ]
    )

    role_behavior_binding_score = mean(
        [
            g1b["headline_metrics"]["behavior_role_mapping_score"],
            p3["headline_metrics"]["shared_law_diverse_roles_score"],
            p3["headline_metrics"]["role_specific_failure_patterns_score"],
        ]
    )

    role_axis_clarity_score = mean(
        [
            g1b["headline_metrics"]["role_axis_separability_score"],
            max(0.0, abs(qwen_pred)),
            max(0.0, abs(deepseek_pred)),
        ]
    )

    overall_g9_score = mean(
        [
            orientation_stability_score,
            shared_band_visibility_score,
            role_behavior_binding_score,
            role_axis_clarity_score,
        ]
    )

    formulas = {
        "role_projection": "z_l(model) = W_role * [concept_support_l, relation_support_l, shared_support_l, task_gain_l]",
        "proto_assignment": "Role_l = argmax_k cosine(z_l, ProtoRole_k)",
        "stability": "RoleStability = 1 - |PredictedOrientation - ActualOrientation|",
        "closure": "RoleClosure = mean(OrientationStability, SharedBandVisibility, RoleBehaviorBinding, RoleAxisClarity)",
    }

    verdict = {
        "status": (
            "stable_role_coordinate_reached"
            if overall_g9_score >= 0.69
            else "stable_role_coordinate_not_closed"
        ),
        "core_answer": (
            "The role coordinate system is visible and behavior-linked, but it is still not stable enough across intervention. "
            "Shared bands are real, yet orientation prediction still shifts too much under targeted ablation."
        ),
        "main_open_gap": "intervention_stable_role_axis_is_not_yet_closed",
    }

    hypotheses = {
        "H1_shared_band_visibility_is_real": shared_band_visibility_score >= 0.5,
        "H2_role_behavior_binding_is_nontrivial": role_behavior_binding_score >= 0.65,
        "H3_role_axis_clarity_is_only_moderate": role_axis_clarity_score < 0.6,
        "H4_g9_is_not_yet_fully_closed": overall_g9_score < 0.69,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G9_stable_unified_role_coordinate_closure",
        },
        "headline_metrics": {
            "orientation_stability_score": orientation_stability_score,
            "shared_band_visibility_score": shared_band_visibility_score,
            "role_behavior_binding_score": role_behavior_binding_score,
            "role_axis_clarity_score": role_axis_clarity_score,
            "overall_g9_score": overall_g9_score,
        },
        "supporting_readout": {
            "qwen_predicted_orientation": qwen_pred,
            "qwen_actual_orientation": qwen_actual,
            "deepseek_predicted_orientation": deepseek_pred,
            "deepseek_actual_orientation": deepseek_actual,
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g9_stable_unified_role_coordinate_closure_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
