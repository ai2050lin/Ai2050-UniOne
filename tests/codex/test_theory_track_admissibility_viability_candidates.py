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
    ap = argparse.ArgumentParser(description="Theory-track admissibility and viability candidates")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_admissibility_viability_candidates_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_a4 = load("stage_a4_partial_closure_reestimate_20260311.json")
    stage_c42 = load("stage_c42_strong_cross_law_manifold_lift_search_20260312.json")
    stage_c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    stage_c58 = load("stage_c58_higher_order_feasible_manifold_search_20260312.json")
    formal = load("theory_track_global_feasibility_law_formalization_20260312.json")

    moderate_consistency = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    moderate_retention = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["retention_concept_accuracy"])
    easy_retention = float(stage_c49["controls"]["easy_regime_c42"]["summary"]["retention_concept_accuracy"])
    current_consistency = float(stage_c58["best_retention_compatible_candidate"]["summary"]["crossmodal_consistency"])
    current_retention = float(stage_c58["best_retention_compatible_candidate"]["summary"]["retention_concept_accuracy"])
    current_overall = float(stage_c58["best_retention_compatible_candidate"]["summary"]["overall_concept_accuracy"])
    stage_a4_retention = float(stage_a4["headline_metrics"]["retention_coexistence_score"])

    consistency_gap = max(0.0, moderate_consistency - current_consistency)
    retention_gap = max(0.0, moderate_retention - current_retention)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_admissibility_viability_candidates",
        },
        "candidate_admissible_set_A": {
            "informal_definition": "A is the set of updates that preserve old geometry, maintain object identity continuity, and do not over-compress discriminative readout.",
            "constraint_families": {
                "A1_retention_safe": "||Delta_mem|| <= tau_mem and post-update retention >= tau_ret",
                "A2_identity_safe": "identity contraction must not exceed the rate at which readout geometry can absorb it",
                "A3_readout_safe": "discriminative compression cannot move faster than object-manifold stabilization",
                "A4_phase_safe": "phase-switch updates are allowed only if switching cost stays below tau_switch",
                "A5_bridge_safe": "relation/role lift must remain compatible with object manifold anchors",
            },
            "candidate_thresholds": {
                "tau_ret_lower_bound": current_retention,
                "tau_ret_target": moderate_retention,
                "tau_consistency_current": current_consistency,
                "tau_consistency_target": moderate_consistency,
                "tau_overall_lower_bound": current_overall,
                "tau_stage_a4_retention_proxy": stage_a4_retention,
            },
        },
        "candidate_viability_manifold_M_feas": {
            "informal_definition": "M_feas is the region where object, memory, relation, and discriminative states can coexist without causing catastrophic overwrite or readout collapse.",
            "state_constraints": {
                "M1_object_memory_coupling": "object manifold and memory manifold must stay within a bounded compatibility band",
                "M2_object_readout_coupling": "discriminative geometry must remain downstream of object geometry rather than collapsing it",
                "M3_relation_consistency": "bridge-role states must remain anchored to object-family structure",
                "M4_phase_viability": "phase/rule state must not force transitions that leave the compatibility band",
                "M5_temporal_viability": "trajectory continuity must remain sufficient for same-object identity",
            },
            "empirical_pressure": {
                "consistency_gap_vs_moderate_control": consistency_gap,
                "retention_gap_vs_moderate_control": retention_gap,
                "easy_regime_retention_headroom": easy_retention,
            },
        },
        "constructive_hypothesis": {
            "A_constructive_form": [
                "A may be represented as an intersection of retention-safe, identity-safe, readout-safe, and phase-safe cones.",
                "Each cone can be parameterized by a different subsystem: memory, object, readout, and controller.",
                "The effective update is admissible only if it lies in their intersection.",
            ],
            "M_feas_constructive_form": [
                "M_feas may be represented as a stratified manifold with object, relation, memory, and readout leaves.",
                "Phase/rule state selects which local chart is active.",
                "Catastrophic failure occurs when dynamics leave the overlap region between object and readout charts.",
            ],
        },
        "encoding_process_reconstruction": {
            "step_1": "Input is projected into a shared object chart.",
            "step_2": "Admissible local plasticity updates the memory chart under A.",
            "step_3": "Bridge-role operators lift the object chart into a relational chart.",
            "step_4": "Readout geometry is generated on a separate discriminative chart.",
            "step_5": "Phase/rule control keeps the trajectory inside M_feas while switching between charts.",
        },
        "verdict": {
            "core_answer": "The next formal step is not another heuristic block, but an explicit parameterization of A as an admissible-update intersection and M_feas as a stratified viability manifold.",
            "next_theory_target": "write explicit algebraic or geometric candidate forms for the cones/charts that define A and M_feas",
            "next_project_target": "use P3 experiments to falsify candidate forms of A and M_feas",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
