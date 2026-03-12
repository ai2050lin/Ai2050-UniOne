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
    ap = argparse.ArgumentParser(description="P1-P4 staged push plan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/phase_p1_p4_push_plan_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    phase_master = load("phase_p1_p4_execution_master_20260312.json")
    stage_c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    stage_c58 = load("stage_c58_higher_order_feasible_manifold_search_20260312.json")
    g4 = load("g4_brain_direct_falsification_master_20260311.json")
    g5 = load("g5_brain_experiment_protocol_observable_mapping_20260311.json")

    p1_score = float(phase_master["phases"]["P1_object_manifold_modeling"]["overall_score"])
    p2_score = float(phase_master["phases"]["P2_controlled_update_law_modeling"]["overall_score"])
    p3_score = float(phase_master["phases"]["P3_compatibility_geometry_modeling"]["overall_score"])
    p4_score = float(phase_master["phases"]["P4_brain_mapping_and_falsification"]["overall_score"])

    c58_consistency = float(stage_c58["headline_metrics"]["best_compatible_consistency"])
    moderate_control = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    p3_gap = max(0.0, moderate_control - c58_consistency)
    p3_pressure = clamp01(p3_gap / 0.08)

    g4_score = float(g4["headline_metrics"]["overall_g4_score"])
    g5_score = float(g5["headline_metrics"]["overall_g5_score"])
    p4_readiness = 0.5 * g4_score + 0.5 * g5_score

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "P1_P4_push_plan",
        },
        "phases": {
            "P1": {
                "name": "object_manifold_bridge_role_atlas",
                "current_score": p1_score,
                "status": "supportive_base_ready",
                "goal": "freeze object manifold / bridge / role atlas as a stable substrate",
                "next_block": "P1A atlas consolidation and invariant extraction",
                "priority": 3,
            },
            "P2": {
                "name": "controlled_update_law",
                "current_score": p2_score,
                "status": "partial_closure_ready_for_freeze",
                "goal": "freeze local update law, write-read separation, and admissible update geometry",
                "next_block": "P2A update-law freeze and anti-interference calibration",
                "priority": 4,
            },
            "P3": {
                "name": "compatibility_geometry",
                "current_score": p3_score,
                "higher_order_feasible_consistency": c58_consistency,
                "moderate_control_consistency": moderate_control,
                "remaining_gap_vs_control": p3_gap,
                "pressure": p3_pressure,
                "status": "primary_bottleneck",
                "goal": "close shared object manifold -> discriminative geometry compatibility",
                "next_block": "P3A transport/phase geometry formalism and execution block",
                "priority": 1,
            },
            "P4": {
                "name": "brain_mapping_and_falsification",
                "current_score": p4_score,
                "readiness": p4_readiness,
                "status": "protocol_ready_waiting_execution",
                "goal": "map abstract mechanism to brain experiments, regions, topology, and falsification",
                "next_block": "P4A execute protocol-ready brain-side mapping bundle",
                "priority": 2,
            },
        },
        "execution_order": [
            "P3A transport/phase geometry formalism and execution block",
            "P4A execute protocol-ready brain-side mapping bundle",
            "P1A atlas consolidation and invariant extraction",
            "P2A update-law freeze and anti-interference calibration",
        ],
        "verdict": {
            "core_answer": "The practical execution route is no longer seven parallel questions, but a four-phase program where P3 is the main bottleneck and P4 is the next leverage point.",
            "main_open_gap": "phase-conditioned compatibility between shared object manifold and discriminative geometry",
            "next_priority": "P3_then_P4_then_P1_then_P2",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
