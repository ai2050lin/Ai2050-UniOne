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
    f6 = load_json("f6_world_model_reasoning_generation_physics_prediction_20260311.json")
    stage6b = load_json("stage6b_real_training_loop_closure_20260311.json")
    stage6c = load_json("stage6c_long_horizon_open_environment_closure_20260311.json")
    stage7a = load_json("stage7a_explicit_coding_law_candidate_20260311.json")
    online_heads = load_json("qwen3_deepseek7b_online_learnable_stage_heads_20260310.json")
    trainable_generator = load_json("local_pulse_trainable_region_family_generator_20260310.json")
    end_to_end_generator = load_json("local_pulse_end_to_end_region_family_generator_network_20260310.json")
    long_memory = load_json("real_multistep_memory_learnable_state_machine_long_validation_20260309.json")

    language_capacity_readiness_score = mean(
        [
            f6["headline_metrics"]["world_reasoning_schema_score"],
            f6["headline_metrics"]["multimodal_world_binding_score"],
            stage6b["headline_metrics"]["trainable_core_score"],
            stage7a["headline_metrics"]["overall_stage7a_score"],
        ]
    )

    qwen_gain = online_heads["gains"]["qwen_learned_minus_fixed_success"]
    deepseek_gain = online_heads["gains"]["deepseek_learned_minus_fixed_success"]
    qwen_success = online_heads["models"]["qwen3_4b"]["online_learnable_stage_heads"]["success_rate"]
    deepseek_success = online_heads["models"]["deepseek_7b"]["online_learnable_stage_heads"]["success_rate"]
    memory_closure = mean(
        [
            long_memory["systems"]["single_anchor_beta_086"]["per_length"]["32"]["real_closure_score"],
            long_memory["systems"]["single_anchor_beta_086"]["per_length"]["40"]["real_closure_score"],
            long_memory["systems"]["single_anchor_beta_086"]["per_length"]["48"]["real_closure_score"],
        ]
    )

    instant_learning_readiness_score = mean(
        [
            clamp01(qwen_success),
            clamp01(deepseek_success),
            clamp01(0.5 + qwen_gain),
            clamp01(0.5 + deepseek_gain),
            stage6b["headline_metrics"]["online_carryover_score"],
            stage6c["headline_metrics"]["tool_failure_recovery_score"],
            memory_closure,
        ]
    )

    architecture_constructibility_score = mean(
        [
            clamp01(0.5 + trainable_generator["headline_metrics"]["trainable_three_stage_gain_vs_fixed"]),
            clamp01(0.5 + trainable_generator["headline_metrics"]["trainable_balance_gain_vs_fixed"]),
            clamp01(trainable_generator["systems"]["trainable_generated_family"]["aggregate_objective"]),
            clamp01(end_to_end_generator["systems"]["end_to_end_generator_eval_family"]["aggregate_objective"]),
            clamp01(1.0 - end_to_end_generator["headline_metrics"]["end_to_end_generalization_gap"]),
        ]
    )

    overall_f7_score = mean(
        [
            language_capacity_readiness_score,
            instant_learning_readiness_score,
            architecture_constructibility_score,
        ]
    )

    formulas = {
        "language_backbone": (
            "h_t = Readout(B_world, B_language, Delta_token, R_syntax, R_semantics, A_t, M_t)"
        ),
        "fast_learning_write": (
            "M_{t+1} = (1 - lambda_m) * M_t + eta_fast * gate_fast * Novelty_t * LocalBind(x_t, h_t)"
        ),
        "slow_consolidation": (
            "A_{t+1} = (1 - lambda_A) * A_t + eta_slow * (1 - gate_fast) * Consolidate(M_{t+1}, h_t)"
        ),
        "routing_gate": "gate_fast = sigmoid(alpha * novelty_t + beta * uncertainty_t - gamma * interference_t)",
        "language_decode": "y_t = Decode_text(h_t, M_t, A_t)",
        "teacherless_update": "Delta theta_local = -lr * dL_local/dtheta_local + rho * Hebb(h_t, x_t, context_t)",
    }

    architecture = {
        "module_1_world_language_basis": "shared basis for world structure, syntax, semantics, and concept families",
        "module_2_dynamic_protocol_router": "routes current context into syntax, semantics, tool, memory, and recovery protocols",
        "module_3_fast_local_memory": "writes new events and bindings immediately with bounded local plasticity",
        "module_4_slow_scaffold": "consolidates stable patterns into reusable long-term structure",
        "module_5_generation_head": "reads the same latent state for next-token generation and action/prediction generation",
        "module_6_online_failure_controller": "uses schema mismatch, drift, timeout, and verification failures to retune local heads online",
    }

    verdict = {
        "status": "candidate_architecture_writeable_but_not_human_complete",
        "can_write_new_network": True,
        "human_language_now": "not_proven",
        "instant_learning_now": "partial_local_supported_not_full_human_grade",
        "core_answer": (
            "A new network can be designed from the current theory, and it should support strong language plus bounded online local learning. "
            "But current evidence does not justify claiming human-level language or fully human-grade instant learning."
        ),
    }

    hypotheses = {
        "H1_language_capacity_is_strong_enough_for_candidate_architecture": language_capacity_readiness_score >= 0.82,
        "H2_instant_learning_is_partially_supported": instant_learning_readiness_score >= 0.62,
        "H3_architecture_constructibility_is_nontrivial": architecture_constructibility_score >= 0.7,
        "H4_candidate_network_is_writeable_now": overall_f7_score >= 0.72,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "F7_human_language_instant_learning_architecture",
        },
        "headline_metrics": {
            "language_capacity_readiness_score": language_capacity_readiness_score,
            "instant_learning_readiness_score": instant_learning_readiness_score,
            "architecture_constructibility_score": architecture_constructibility_score,
            "overall_f7_score": overall_f7_score,
        },
        "supporting_readout": {
            "qwen_online_success": qwen_success,
            "deepseek_online_success": deepseek_success,
            "qwen_online_gain": qwen_gain,
            "deepseek_online_gain": deepseek_gain,
            "memory_long_validation_mean_closure": memory_closure,
            "trainable_generator_aggregate_objective": trainable_generator["systems"]["trainable_generated_family"]["aggregate_objective"],
            "end_to_end_generator_aggregate_objective": end_to_end_generator["systems"]["end_to_end_generator_eval_family"]["aggregate_objective"],
            "end_to_end_generalization_gap": end_to_end_generator["headline_metrics"]["end_to_end_generalization_gap"],
        },
        "formulas": formulas,
        "architecture": architecture,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "f7_human_language_instant_learning_architecture_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
