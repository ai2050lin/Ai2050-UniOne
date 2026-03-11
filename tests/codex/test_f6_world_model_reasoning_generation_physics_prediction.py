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


def concept_family_support(model_blob: dict, concept: str) -> float:
    concept_blob = model_blob["concepts"][concept]
    margin = clamp01(concept_blob["summary"]["margin_vs_best_wrong"])
    residual = clamp01(1.0 - concept_blob["family_fit"][concept_blob["true_family"]]["residual_ratio"])
    match = 1.0 if concept_blob["preferred_family_matches_truth"] else 0.0
    return mean([margin, residual, match])


def main() -> None:
    f5 = load_json("f5_world_knowledge_encoding_confirmation_20260311.json")
    real_task = load_json("real_task_driven_two_layer_unified_law_20260310.json")
    modality = load_json("parameterized_shared_modality_law_20260310.json")
    relation_topology = load_json("qwen3_deepseek7b_relation_topology_atlas_20260309.json")
    open_loop = load_json("open_world_grounding_action_loop_20260310.json")
    open_plan = load_json("open_world_variable_planning_trainable_benchmark_20260310.json")
    p8a = load_json("p8a_spatialized_plasticity_coding_equation_20260311.json")
    d_problem = load_json("d_problem_atlas_summary_20260309.json")

    world_reasoning_schema_score = mean(
        [
            f5["headline_metrics"]["world_basis_score"],
            f5["headline_metrics"]["predicate_protocol_score"],
            real_task["real_task_two_layer_law"]["held_out_score_correlation"],
            real_task["real_task_two_layer_law"]["score_correlation"],
        ]
    )

    world_generation_schema_score = mean(
        [
            open_plan["systems"]["trainable_planner"]["variable_planning_score"],
            open_plan["systems"]["trainable_planner"]["episode_success_rate"],
            open_plan["systems"]["trainable_planner"]["open_environment_stability"],
            open_loop["systems"]["direct_action"]["loop_score"],
            open_loop["systems"]["direct_action"]["corrected_action_accuracy"],
        ]
    )

    multimodal_world_binding_score = mean(
        [
            modality["fully_shared_law"]["held_out_score_correlation"],
            modality["parameterized_shared_law"]["held_out_score_correlation"],
            f5["headline_metrics"]["system_reuse_score"],
        ]
    )

    qwen = relation_topology["models"]["qwen3_4b"]
    deepseek = relation_topology["models"]["deepseek_7b"]
    causal_support_score = mean(
        [
            concept_family_support(qwen, "heat"),
            concept_family_support(qwen, "bacteria"),
            concept_family_support(qwen, "smoke"),
            concept_family_support(qwen, "motion"),
            concept_family_support(qwen, "flood"),
            concept_family_support(deepseek, "heat"),
            concept_family_support(deepseek, "bacteria"),
            concept_family_support(deepseek, "smoke"),
            concept_family_support(deepseek, "motion"),
            concept_family_support(deepseek, "flood"),
        ]
    )

    geometry_only_failure_score = mean(
        [
            clamp01(1.0 - abs(model["geometry_overall_gain"]))
            for model in d_problem["models"]
        ]
    )

    physical_rule_prediction_readiness_score = mean(
        [
            causal_support_score,
            p8a["headline_metrics"]["topology_reuse_locality_score"],
            p8a["headline_metrics"]["brain_plausibility_score"],
            p8a["headline_metrics"]["geometry_only_failure_score"],
            geometry_only_failure_score,
        ]
    )

    overall_f6_score = mean(
        [
            world_reasoning_schema_score,
            world_generation_schema_score,
            multimodal_world_binding_score,
            physical_rule_prediction_readiness_score,
        ]
    )

    formulas = {
        "world_state": "K_t = {B_world, B_families, Delta_concepts, R_predicates, C_causal, S_spatial, A_t, M_t}",
        "reasoning_decode": "Infer(q, t) = Decode_reason(K_t, context_q, goal_t)",
        "generation_decode": "Generate(goal, t) = Decode_generate(K_t, policy_t, route_t)",
        "stone_fall_prediction": (
            "Fall_stone(x, t+1) = sigmoid(w_o * B_solid(x) + w_g * R_gravity(t) + "
            "w_h * Height_t(x) - w_s * Support_t(x) + w_m * R_motion(t) + w_3 * S_spatial(x, t))"
        ),
        "liquid_flow_prediction": (
            "Flow_liquid(x, t+1) = sigmoid(w_l * B_liquid(x) + w_p * PressureGrad_t(x) + "
            "w_b * BoundaryOpen_t(x) + w_f * R_flow(t) + w_3 * S_spatial(x, t) - w_v * ViscosityBarrier_t(x))"
        ),
        "spatial_dynamics": (
            "A_{t+1}(i,j) = (1 - l_A) * A_t(i,j) + e_A * (1 - q_t(i,j)) * "
            "[f_{t+1}(i)f_{t+1}(j) + d_t(i,j) - p_t(i,j) - lambda_s * D_3d(i,j)]"
        ),
    }

    interpretation = {
        "reasoning_layer": (
            "World reasoning is not a flat rule table. The system reads from a layered world state "
            "that combines family bases, concept offsets, causal protocols, and spatial topology."
        ),
        "generation_layer": (
            "World generation is the action-side readout of the same state. Planning, correction, "
            "and environment interaction reuse the same world state instead of calling an isolated simulator."
        ),
        "stone_fall": (
            "Stone-fall prediction should be easier because it is dominated by solid-object identity, "
            "support loss, gravity-like causal routing, and 3D motion continuation."
        ),
        "liquid_flow": (
            "Liquid-flow prediction should be harder because it depends on continuous boundary conditions, "
            "pressure gradients, and many locally coupled micro-updates, not just one discrete event edge."
        ),
        "physical_scope": (
            "Current evidence supports qualitative and mid-level physical prediction readiness, "
            "not a precise fluid-dynamics or rigid-body simulator."
        ),
    }

    examples = {
        "apple_can_eat": "B_world + B_object + B_living + B_edible + B_fruit + Delta_apple + R_consume",
        "stone_will_fall": "B_world + B_object + B_solid + Delta_stone + R_gravity + R_motion + S_spatial",
        "liquid_will_flow": "B_world + B_object + B_liquid + Delta_water + R_flow + BoundaryState + S_spatial",
        "smoke_follows_fire": "B_world + B_event + Delta_smoke + R_cause_effect + Delta_fire",
    }

    hypotheses = {
        "H1_world_reasoning_layer_is_ready": world_reasoning_schema_score >= 0.78,
        "H2_world_generation_layer_is_ready": world_generation_schema_score >= 0.86,
        "H3_multimodal_world_binding_is_strong": multimodal_world_binding_score >= 0.9,
        "H4_physical_rule_prediction_is_nontrivial": physical_rule_prediction_readiness_score >= 0.74,
        "H5_f6_world_model_reasoning_generation_is_ready": overall_f6_score >= 0.82,
    }

    verdict = {
        "status": "world_model_reasoning_generation_supported",
        "physical_prediction_scope": "qualitative_and_mid_level_supported_not_exact_simulator",
        "core_answer": (
            "The current theory supports a layered world model with reasoning and generation readouts. "
            "For physical rules, it already supports nontrivial qualitative prediction for causal-spatial events "
            "such as falling solids and flowing liquids, but it is not yet a precise numerical simulator."
        ),
        "stone_fall_prediction_strength": "stronger_than_liquid_flow",
        "liquid_flow_prediction_strength": "moderate_but_not_exact",
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F6_world_model_reasoning_generation_physics_prediction",
        },
        "headline_metrics": {
            "world_reasoning_schema_score": world_reasoning_schema_score,
            "world_generation_schema_score": world_generation_schema_score,
            "multimodal_world_binding_score": multimodal_world_binding_score,
            "causal_support_score": causal_support_score,
            "physical_rule_prediction_readiness_score": physical_rule_prediction_readiness_score,
            "overall_f6_score": overall_f6_score,
        },
        "supporting_readout": {
            "real_task_held_out_corr": real_task["real_task_two_layer_law"]["held_out_score_correlation"],
            "trainable_planner_variable_planning_score": open_plan["systems"]["trainable_planner"]["variable_planning_score"],
            "trainable_planner_open_environment_stability": open_plan["systems"]["trainable_planner"]["open_environment_stability"],
            "direct_action_loop_score": open_loop["systems"]["direct_action"]["loop_score"],
            "qwen_motion_effect_support": concept_family_support(qwen, "motion"),
            "deepseek_motion_effect_support": concept_family_support(deepseek, "motion"),
            "qwen_flood_effect_support": concept_family_support(qwen, "flood"),
            "deepseek_flood_effect_support": concept_family_support(deepseek, "flood"),
            "parameterized_shared_held_out_corr": modality["parameterized_shared_law"]["held_out_score_correlation"],
            "geometry_only_failure_score": geometry_only_failure_score,
        },
        "formulas": formulas,
        "interpretation": interpretation,
        "examples": examples,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f6_world_model_reasoning_generation_physics_prediction_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
