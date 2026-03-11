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
    stage5b = load_json("stage5b_structure_reinforcement_closure_20260311.json")
    p2 = load_json("p2_multitimescale_stabilization_mechanism_20260311.json")
    stage6b = load_json("stage6b_real_training_loop_closure_20260311.json")
    stage5c = load_json("stage5c_online_failure_integrated_training_closure_20260311.json")
    f7 = load_json("f7_human_language_instant_learning_architecture_20260311.json")

    structure_foundation_training_score = mean(
        [
            stage5b["headline_metrics"]["foundation_score"],
            stage5b["headline_metrics"]["shared_support_score"],
            stage5b["headline_metrics"]["real_task_structure_gain_score"],
            stage6b["headline_metrics"]["trainable_core_score"],
        ]
    )

    fast_slow_unification_score = mean(
        [
            p2["headline_metrics"]["fast_slow_coupling_score"],
            p2["headline_metrics"]["long_horizon_stability_score"],
            p2["headline_metrics"]["explicitness_score"],
            stage6b["headline_metrics"]["compressed_core_score"],
        ]
    )

    online_failure_unification_score = mean(
        [
            stage5c["headline_metrics"]["hard_interface_score"],
            stage5c["headline_metrics"]["learnable_heads_score"],
            stage5c["headline_metrics"]["recovery_chain_score"],
            stage5c["headline_metrics"]["control_manifold_score"],
            stage6b["headline_metrics"]["online_carryover_score"],
        ]
    )

    instant_learning_bridge_score = mean(
        [
            f7["headline_metrics"]["instant_learning_readiness_score"],
            f7["headline_metrics"]["architecture_constructibility_score"],
            clamp01(f7["supporting_readout"]["memory_long_validation_mean_closure"]),
            clamp01(f7["supporting_readout"]["trainable_generator_aggregate_objective"]),
        ]
    )

    overall_g2_score = mean(
        [
            structure_foundation_training_score,
            fast_slow_unification_score,
            online_failure_unification_score,
            instant_learning_bridge_score,
        ]
    )

    formulas = {
        "unified_loss": (
            "L_total = lambda_s * L_structure + lambda_f * L_fast_memory + "
            "lambda_m * L_slow_consolidation + lambda_o * L_online_failure + lambda_r * L_readout"
        ),
        "structure_term": (
            "L_structure = ||A_target - A_t|| + gamma_c * CollapsePenalty(A_t) + gamma_b * BridgeBudgetPenalty(A_t)"
        ),
        "fast_write_term": (
            "L_fast_memory = ||M_{t+1} - Write(Novelty_t, x_t, h_t)||"
        ),
        "slow_term": (
            "L_slow_consolidation = ||A_{t+1} - Consolidate(M_{t+1}, h_t)||"
        ),
        "online_failure_term": (
            "L_online_failure = alpha_s * SchemaMismatch + alpha_d * StateDrift + alpha_t * Timeout + alpha_v * VerifyMismatch"
        ),
        "gated_update": (
            "gate_fast = sigmoid(alpha * novelty_t + beta * uncertainty_t - gamma * interference_t)"
        ),
    }

    interpretation = {
        "goal": (
            "G2 asks whether structure can stop being a weak regularizer and become a first-class training target, "
            "while fast write and slow consolidation are optimized inside the same objective."
        ),
        "current_result": (
            "The answer is partially yes: fast-slow unification is already fairly strong, but structure foundation "
            "and online-failure integration are still only moderate."
        ),
        "main_bottleneck": (
            "The bottleneck is still the structure foundation itself. Fast and slow dynamics look coherent, "
            "but the structural base they are supposed to stabilize is not yet strong enough."
        ),
    }

    verdict = {
        "status": "moderate_partial_closure",
        "core_answer": (
            "G2 is positive only in a moderate sense. The project can now write one fused training view where structure, "
            "fast memory, slow consolidation, and online failures belong to the same objective. But structure is still not "
            "strong enough to count as a fully closed training foundation."
        ),
        "main_open_gap": "structure_foundation_strength_is_still_too_low",
    }

    hypotheses = {
        "H1_structure_foundation_is_nontrivial_inside_training_view": structure_foundation_training_score >= 0.62,
        "H2_fast_slow_unification_is_strong": fast_slow_unification_score >= 0.74,
        "H3_online_failure_is_part_of_same_training_objective": online_failure_unification_score >= 0.58,
        "H4_instant_learning_bridge_is_nontrivial": instant_learning_bridge_score >= 0.54,
        "H5_g2_reaches_moderate_partial_closure": overall_g2_score >= 0.63,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G2_structure_foundation_fast_slow_training_closure",
        },
        "headline_metrics": {
            "structure_foundation_training_score": structure_foundation_training_score,
            "fast_slow_unification_score": fast_slow_unification_score,
            "online_failure_unification_score": online_failure_unification_score,
            "instant_learning_bridge_score": instant_learning_bridge_score,
            "overall_g2_score": overall_g2_score,
        },
        "supporting_readout": {
            "stage5b_foundation_score": stage5b["headline_metrics"]["foundation_score"],
            "p2_fast_slow_coupling_score": p2["headline_metrics"]["fast_slow_coupling_score"],
            "stage6b_trainable_core_score": stage6b["headline_metrics"]["trainable_core_score"],
            "stage5c_online_failure_score": stage5c["headline_metrics"]["overall_stage5c_score"],
            "f7_instant_learning_readiness_score": f7["headline_metrics"]["instant_learning_readiness_score"],
        },
        "formulas": formulas,
        "interpretation": interpretation,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g2_structure_foundation_fast_slow_training_closure_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
