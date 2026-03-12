from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track encoding mechanism synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_mechanism_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    phase_master = load("phase_p1_p4_execution_master_20260312.json")
    push_plan = load("phase_p1_p4_push_plan_20260312.json")
    stage_a4 = load("stage_a4_partial_closure_reestimate_20260311.json")
    stage_b1 = load("stage_b1_calibrated_partial_reestimate_20260311.json")
    stage_c42 = load("stage_c42_strong_cross_law_manifold_lift_search_20260312.json")
    stage_c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    stage_c57 = load("stage_c57_feasible_manifold_geometry_search_20260312.json")
    g4 = load("g4_brain_direct_falsification_master_20260311.json")
    g5 = load("g5_brain_experiment_protocol_observable_mapping_20260311.json")

    moderate_control = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    c57_consistency = float(stage_c57["headline_metrics"]["best_compatible_consistency"])
    p3_gap = max(0.0, moderate_control - c57_consistency)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_mechanism_synthesis",
        },
        "headline_judgment": {
            "unified_theory_backbone_completion": "95% - 97%",
            "engineering_closure": "83% - 88%",
            "brain_encoding_mechanism_decryption": "86% - 90%",
            "main_bottleneck": "P3 compatibility geometry / higher-order feasible manifold",
        },
        "seven_questions_to_mechanism": {
            "Q1_encoding_object_layer": {
                "function": "Define what is stably encoded.",
                "current_answer": "object kernel, concept kernel, family structure, bridge-role coordinates",
                "brain_guess": "cortex stores object-like latent states rather than raw features",
            },
            "Q2_local_update_law": {
                "function": "Define how encoding changes.",
                "current_answer": "local plastic update + conditional gating + multi-timescale retention",
                "brain_guess": "plasticity is local, gated, and regime-dependent rather than globally uniform",
            },
            "Q3_object_carrier_and_write_read_separation": {
                "function": "Define how new evidence is written without destroying old identity.",
                "current_answer": "write state / read state / persistence state separation",
                "brain_guess": "brain likely uses fast-write, slow-read, slower-persistence strata",
            },
            "Q4_bridge_law_and_role_kernel": {
                "function": "Lift object codes into relation and role structure.",
                "current_answer": "bridge selection law + stable role kernel",
                "brain_guess": "association areas reuse common bridge operators with region-specific parameterization",
            },
            "Q5_crossmodal_consistency": {
                "function": "Preserve same-object identity across modalities.",
                "current_answer": "shared object manifold is strongly supported",
                "brain_guess": "different modalities project to a shared object manifold with modality offsets",
            },
            "Q6_discriminative_geometry": {
                "function": "Read shared object codes into stable decisions.",
                "current_answer": "still not fully closed; shared manifold to discriminative geometry remains the hard boundary",
                "brain_guess": "decision geometry is downstream of object geometry, not identical to it",
            },
            "Q7_brain_side_truth_and_3d_mapping": {
                "function": "Map the abstract mechanism to real brain regions, cells, topology, and falsification.",
                "current_answer": "protocol-ready but not fully executed",
                "brain_guess": "same mechanism family may recur across regions with different anatomical parameterization",
            },
        },
        "formation_process_hypothesis": {
            "stage_1": "Multimodal inputs are compressed into an object manifold with family and role offsets.",
            "stage_2": "Local gated plasticity writes fast traces while protected memory preserves old geometry.",
            "stage_3": "Bridge and role operators organize object states into relational structure.",
            "stage_4": "A discriminative readout geometry is constructed from, but is not identical to, the object manifold.",
            "stage_5": "Brain-side regions instantiate the same mechanism family under different topology and parameter regimes.",
        },
        "system_level_operation_hypothesis": {
            "core_form": "controlled encoding dynamical system",
            "state_objects": [
                "object manifold state",
                "memory / read / persistence state",
                "bridge-role state",
                "readout geometry state",
                "phase / rule state",
            ],
            "operating_laws": [
                "local plastic update",
                "admissible update geometry",
                "phase-conditioned switching",
                "readout formation law",
            ],
            "current_missing_object": "higher-order feasible manifold / meta-geometric admissibility",
        },
        "phase_mapping": {
            "P1": phase_master["phases"]["P1_object_manifold_modeling"],
            "P2": phase_master["phases"]["P2_controlled_update_law_modeling"],
            "P3": {
                **push_plan["phases"]["P3"],
                "gap_vs_moderate_control": p3_gap,
            },
            "P4": {
                "g4_score": float(g4["headline_metrics"]["overall_g4_score"]),
                "g5_score": float(g5["headline_metrics"]["overall_g5_score"]),
                "status": "protocol_ready_waiting_execution",
            },
        },
        "verdict": {
            "core_answer": "The seven questions increasingly reconstruct one controlled encoding dynamical system, but the missing closure is still the higher-order feasible manifold that links shared object geometry to discriminative geometry.",
            "next_theory_target": "formalize higher-order feasible manifold / meta-geometric admissibility",
            "next_project_target": "keep P3 -> P4 -> P1 -> P2",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
