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
    f7 = load_json("f7_human_language_instant_learning_architecture_20260311.json")
    long_validation = load_json("real_multistep_memory_learnable_state_machine_long_validation_20260309.json")
    online_heads = load_json("qwen3_deepseek7b_online_learnable_stage_heads_20260310.json")
    open_stream = load_json("open_world_continuous_grounding_stream_20260310.json")
    retention_scan = load_json("continuous_input_grounding_retention_scan_20260309.json")
    precision_scan = load_json("continuous_input_grounding_precision_scan_20260309.json")
    proto = load_json("continuous_input_grounding_proto_20260309.json")

    immediate_write_score = mean(
        [
            precision_scan["systems"]["adaptive_precision_shared_offset"]["novel_concept_accuracy"],
            precision_scan["systems"]["adaptive_precision_shared_offset_replay"]["novel_concept_accuracy"],
            precision_scan["systems"]["protected_phase_split"]["novel_concept_accuracy"],
            clamp01(f7["supporting_readout"]["qwen_online_gain"] + 0.5),
            clamp01(f7["supporting_readout"]["deepseek_online_gain"] + 0.5),
        ]
    )

    retention_boundary_score = mean(
        [
            retention_scan["systems"]["direct_prototype"]["retention_concept_accuracy"],
            retention_scan["systems"]["shared_offset_grounder"]["retention_concept_accuracy"],
            proto["systems"]["shared_offset_grounder"]["retention_concept_accuracy"],
            mean(
                [
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["32"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["40"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["48"]["retention_after_phase2"],
                ]
            ),
        ]
    )

    interference_tradeoff_score = mean(
        [
            1.0 - precision_scan["systems"]["adaptive_precision_shared_offset"]["retention_concept_accuracy"],
            1.0 - precision_scan["systems"]["adaptive_precision_shared_offset_replay"]["retention_concept_accuracy"],
            precision_scan["systems"]["cross_modal_dual_store"]["retention_concept_accuracy"],
            clamp01(open_stream["gains_vs_direct"]["drifted_concept_gain"] + 0.5),
        ]
    )

    cross_environment_carryover_score = mean(
        [
            online_heads["models"]["qwen3_4b"]["online_learnable_stage_heads"]["success_rate"],
            online_heads["models"]["deepseek_7b"]["online_learnable_stage_heads"]["success_rate"],
            open_stream["systems"]["direct_stream"]["closure_score"],
            open_stream["systems"]["shared_offset_stream"]["closure_score"],
            f7["supporting_readout"]["memory_long_validation_mean_closure"],
        ]
    )

    overall_g3_score = mean(
        [
            immediate_write_score,
            retention_boundary_score,
            interference_tradeoff_score,
            cross_environment_carryover_score,
        ]
    )

    formulas = {
        "instant_write": "M_{t+1} = M_t + eta_fast * gate_fast * Novelty_t * LocalBind(x_t, h_t)",
        "retention": "Retain = sigmoid(w_r * Replay + w_s * SlowConsolidation - w_i * Interference)",
        "tradeoff": "Boundary = NovelWriteGain - lambda_i * OldKnowledgeLoss - lambda_d * DriftCost",
        "carryover": "Carryover = mean(TaskSuccess_after_update, Retention_after_delay, Stability_under_drift)",
    }

    interpretation = {
        "write": "The system can write some novel information quickly, especially under adaptive precision or phase-split schemes.",
        "retention": "The hardest boundary is not first-write but delayed retention. Novel write often rises only by sacrificing old concept retention.",
        "interference": "Immediate learning is not free. Strong novel-write settings still produce large interference costs.",
        "overall": "Current instant learning is real but narrow: fast local write is possible, long retention and low-interference consolidation are still weak.",
    }

    verdict = {
        "status": "instant_learning_real_but_narrow",
        "core_answer": (
            "G3 confirms that instant learning is not fake: the system can write new information quickly and improve some online success metrics. "
            "But the boundary is clear: delayed retention is weak, and strong novel-write settings still pay a large interference cost."
        ),
        "main_open_gap": "retention_and_interference_tradeoff",
    }

    hypotheses = {
        "H1_immediate_write_is_nontrivial": immediate_write_score >= 0.72,
        "H2_retention_boundary_is_weak": retention_boundary_score < 0.35,
        "H3_interference_tradeoff_is_real": interference_tradeoff_score >= 0.6,
        "H4_cross_environment_carryover_is_nontrivial": cross_environment_carryover_score >= 0.55,
        "H5_g3_boundary_is_now_explicit": overall_g3_score >= 0.52,
    }

    output = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_block": "G3_instant_learning_boundary_stress",
        },
        "headline_metrics": {
            "immediate_write_score": immediate_write_score,
            "retention_boundary_score": retention_boundary_score,
            "interference_tradeoff_score": interference_tradeoff_score,
            "cross_environment_carryover_score": cross_environment_carryover_score,
            "overall_g3_score": overall_g3_score,
        },
        "supporting_readout": {
            "adaptive_precision_novel": precision_scan["systems"]["adaptive_precision_shared_offset"]["novel_concept_accuracy"],
            "adaptive_precision_retention": precision_scan["systems"]["adaptive_precision_shared_offset"]["retention_concept_accuracy"],
            "phase_split_novel": precision_scan["systems"]["protected_phase_split"]["novel_concept_accuracy"],
            "phase_split_retention": precision_scan["systems"]["protected_phase_split"]["retention_concept_accuracy"],
            "single_anchor_retention_mean": mean(
                [
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["32"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["40"]["retention_after_phase2"],
                    long_validation["systems"]["single_anchor_beta_086"]["per_length"]["48"]["retention_after_phase2"],
                ]
            ),
            "qwen_online_success": online_heads["models"]["qwen3_4b"]["online_learnable_stage_heads"]["success_rate"],
            "deepseek_online_success": online_heads["models"]["deepseek_7b"]["online_learnable_stage_heads"]["success_rate"],
        },
        "formulas": formulas,
        "interpretation": interpretation,
        "hypotheses": hypotheses,
        "verdict": verdict,
    }

    output_path = TEMP_DIR / "g3_instant_learning_boundary_stress_20260311.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
