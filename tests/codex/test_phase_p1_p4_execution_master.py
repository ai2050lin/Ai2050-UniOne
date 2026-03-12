from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    ap = argparse.ArgumentParser(description="P1-P4 execution master rollup")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/phase_p1_p4_execution_master_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_a4 = load("stage_a4_partial_closure_reestimate_20260311.json")
    stage_b1 = load("stage_b1_calibrated_partial_reestimate_20260311.json")
    stage_c42 = load("stage_c42_strong_cross_law_manifold_lift_search_20260312.json")
    stage_c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    g4 = load("g4_brain_direct_falsification_master_20260311.json")
    g5 = load("g5_brain_experiment_protocol_observable_mapping_20260311.json")

    p1_bridge_role = float(stage_b1["headline_metrics"]["overall_stage_b1_score"])
    p1_crossmodal_default = float(stage_c42["headline_metrics"]["best_compatible_consistency"])
    p1_crossmodal_moderate = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    p1_retention_moderate = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["retention_concept_accuracy"])
    p1_score = (
        0.40 * p1_bridge_role
        + 0.30 * clamp01(p1_crossmodal_default / 0.30)
        + 0.15 * clamp01(p1_crossmodal_moderate / 0.30)
        + 0.15 * clamp01(p1_retention_moderate / 0.40)
    )

    p2_update_score = float(stage_a4["headline_metrics"]["overall_stage_a4_score"])
    p2_retention_score = float(stage_a4["headline_metrics"]["retention_coexistence_score"])
    p2_interference_score = float(stage_a4["headline_metrics"]["interference_control_score"])
    p2_easy_retention = float(stage_c49["controls"]["easy_regime_c42"]["summary"]["retention_concept_accuracy"])
    p2_score = (
        0.40 * p2_update_score
        + 0.25 * p2_retention_score
        + 0.20 * p2_interference_score
        + 0.15 * clamp01(p2_easy_retention / 0.50)
    )

    p3_default_closure_ratio = clamp01(p1_crossmodal_default / max(1e-6, p1_crossmodal_moderate))
    p3_stagnation_penalty = 0.25
    p3_readout_capacity = clamp01(float(stage_c49["controls"]["oracle_exact_prototype"]["summary"]["crossmodal_consistency"]) / 0.20)
    p3_score = (
        0.45 * p3_default_closure_ratio
        + 0.30 * p3_readout_capacity
        + 0.25 * p3_stagnation_penalty
    )

    p4_g4 = float(g4["headline_metrics"]["overall_g4_score"])
    p4_g5 = float(g5["headline_metrics"]["overall_g5_score"])
    p4_score = 0.5 * p4_g4 + 0.5 * p4_g5

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "P1_P4_execution_master",
        },
        "phases": {
            "P1_object_manifold_modeling": {
                "bridge_role_score": p1_bridge_role,
                "crossmodal_default_consistency": p1_crossmodal_default,
                "crossmodal_moderate_consistency": p1_crossmodal_moderate,
                "retention_moderate_score": p1_retention_moderate,
                "overall_score": p1_score,
                "status": "active_with_strong_partial_support",
            },
            "P2_controlled_update_law_modeling": {
                "stage_a4_score": p2_update_score,
                "retention_coexistence_score": p2_retention_score,
                "interference_control_score": p2_interference_score,
                "easy_regime_retention": p2_easy_retention,
                "overall_score": p2_score,
                "status": "partial_closure_with_retention_gap",
            },
            "P3_compatibility_geometry_modeling": {
                "default_closure_ratio_vs_moderate": p3_default_closure_ratio,
                "oracle_readout_capacity": p3_readout_capacity,
                "stagnation_penalty": p3_stagnation_penalty,
                "overall_score": p3_score,
                "status": "main_open_bottleneck",
            },
            "P4_brain_mapping_and_falsification": {
                "g4_score": p4_g4,
                "g5_score": p4_g5,
                "overall_score": p4_score,
                "status": "protocol_ready_not_executed",
            },
        },
        "verdict": {
            "core_answer": "P1 and P2 are already in strong partial support, P4 protocol readiness is high, and P3 compatibility geometry is the current bottleneck.",
            "main_open_gap": "shared_object_manifold_to_discriminative_geometry_compatibility",
            "next_priority": "P3_then_P4",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
