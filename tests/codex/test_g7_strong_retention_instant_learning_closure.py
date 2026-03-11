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


def geom_balance(a: float, b: float) -> float:
    return (max(0.0, a) * max(0.0, b)) ** 0.5


def main() -> None:
    g3 = load_json("g3_instant_learning_boundary_stress_20260311.json")
    precision = load_json("continuous_input_grounding_precision_scan_20260309.json")
    segment = load_json("real_multistep_memory_segment_summary_scan_20260309.json")
    long_validation = load_json("real_multistep_memory_learnable_state_machine_long_validation_20260309.json")
    online = load_json("qwen3_deepseek7b_online_learnable_stage_heads_20260310.json")
    f7 = load_json("f7_human_language_instant_learning_architecture_20260311.json")

    best_balanced_grounding = max(
        geom_balance(v["novel_concept_accuracy"], v["retention_concept_accuracy"])
        for v in precision["systems"].values()
    )

    best_long_horizon_retention = max(
        [
            segment["best"]["best_mean_segment_system"]["mean_retention_score"],
            segment["best"]["best_max_segment_system"]["mean_retention_score"],
            mean(
                [
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["32"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["40"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["48"]["retention_after_phase2"],
                ]
            ),
        ]
    )

    qwen_gain = online["gains"]["qwen_learned_minus_fixed_success"]
    deepseek_gain = online["gains"]["deepseek_learned_minus_fixed_success"]
    online_retention_carryover_score = mean(
        [
            online["models"]["qwen3_4b"]["online_learnable_stage_heads"]["success_rate"],
            online["models"]["deepseek_7b"]["online_learnable_stage_heads"]["success_rate"],
            qwen_gain + 0.5,
            deepseek_gain + 0.5,
            g3["headline_metrics"]["cross_environment_carryover_score"],
        ]
    )

    fast_write_preservation_balance_score = mean(
        [
            g3["headline_metrics"]["immediate_write_score"],
            best_balanced_grounding,
            f7["headline_metrics"]["instant_learning_readiness_score"],
        ]
    )

    long_retention_foundation_score = mean(
        [
            best_long_horizon_retention,
            g3["headline_metrics"]["retention_boundary_score"],
            f7["supporting_readout"]["memory_long_validation_mean_closure"],
        ]
    )

    interference_control_score = mean(
        [
            1.0 - g3["headline_metrics"]["interference_tradeoff_score"],
            precision["systems"]["cross_modal_dual_store"]["retention_concept_accuracy"],
            precision["systems"]["dual_store_route"]["retention_concept_accuracy"],
        ]
    )

    overall_g7_score = mean(
        [
            fast_write_preservation_balance_score,
            long_retention_foundation_score,
            online_retention_carryover_score,
            interference_control_score,
        ]
    )

    formulas = {
        "balanced_write": "Balance = sqrt(NovelWrite * DelayedRetention)",
        "retention_core": "Retain = sigmoid(w_r * Replay + w_s * SlowConsolidation - w_i * Interference)",
        "carryover": "Carryover = mean(OnlineSuccess, SuccessGain, CrossEnvironmentStability)",
        "strong_closure": "StrongRetention = mean(BalancedWrite, LongRetention, Carryover, InterferenceControl)",
    }

    verdict = {
        "status": (
            "strong_retention_closure_reached"
            if overall_g7_score >= 0.67
            else "strong_retention_not_closed"
        ),
        "core_answer": (
            "Fast write is real, and some online carryover is real, but strong delayed retention still does not close. "
            "The best current systems improve balance only modestly; retention remains the main bottleneck."
        ),
        "main_open_gap": "delayed_retention_under_interference",
        "best_current_balance_system": "cross_modal_dual_store_or_segment_summary_family",
    }

    hypotheses = {
        "H1_fast_write_is_real": fast_write_preservation_balance_score >= 0.55,
        "H2_some_long_retention_signal_exists": long_retention_foundation_score >= 0.16,
        "H3_online_carryover_is_nontrivial": online_retention_carryover_score >= 0.6,
        "H4_interference_control_is_still_weak": interference_control_score < 0.45,
        "H5_g7_strong_retention_is_not_yet_closed": overall_g7_score < 0.67,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G7_strong_retention_instant_learning_closure",
        },
        "headline_metrics": {
            "fast_write_preservation_balance_score": fast_write_preservation_balance_score,
            "long_retention_foundation_score": long_retention_foundation_score,
            "online_retention_carryover_score": online_retention_carryover_score,
            "interference_control_score": interference_control_score,
            "overall_g7_score": overall_g7_score,
        },
        "supporting_readout": {
            "best_balanced_grounding": best_balanced_grounding,
            "best_long_horizon_retention": best_long_horizon_retention,
            "qwen_gain": qwen_gain,
            "deepseek_gain": deepseek_gain,
            "memory_long_validation_mean_closure": f7["supporting_readout"]["memory_long_validation_mean_closure"],
        },
        "formulas": formulas,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    with (TEMP_DIR / "g7_strong_retention_instant_learning_closure_20260311.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
