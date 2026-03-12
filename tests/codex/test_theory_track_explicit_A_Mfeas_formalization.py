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
    ap = argparse.ArgumentParser(description="Theory-track explicit A and M_feas formalization")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_explicit_A_Mfeas_formalization_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    c58 = load("stage_c58_higher_order_feasible_manifold_search_20260312.json")
    stage_a4 = load("stage_a4_partial_closure_reestimate_20260311.json")
    formal = load("theory_track_global_feasibility_law_formalization_20260312.json")
    admiss = load("theory_track_admissibility_viability_candidates_20260312.json")

    moderate_consistency = float(c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    moderate_retention = float(c49["controls"]["moderate_regime_c42"]["summary"]["retention_concept_accuracy"])
    current_consistency = float(c58["best_retention_compatible_candidate"]["summary"]["crossmodal_consistency"])
    current_retention = float(c58["best_retention_compatible_candidate"]["summary"]["retention_concept_accuracy"])
    current_overall = float(c58["best_retention_compatible_candidate"]["summary"]["overall_concept_accuracy"])
    easy_retention = float(c49["controls"]["easy_regime_c42"]["summary"]["retention_concept_accuracy"])
    stage_a4_retention = float(stage_a4["headline_metrics"]["retention_coexistence_score"])

    tau_ret_floor = current_retention
    tau_ret_target = moderate_retention
    tau_consistency_floor = current_consistency
    tau_consistency_target = moderate_consistency
    tau_overall_floor = current_overall
    tau_switch = 0.15
    tau_bridge = 0.12
    beta_read = 0.8
    beta_phase = 0.7

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_explicit_A_Mfeas_formalization",
        },
        "explicit_A": {
            "name": "admissible_update_set_A",
            "high_level_form": "A = K_ret INTERSECT K_id INTERSECT K_read INTERSECT K_phase INTERSECT K_bridge",
            "cone_families": {
                "K_ret": {
                    "meaning": "retention-safe cone",
                    "definition": [
                        "||P_mem Delta|| <= tau_mem",
                        "||P_ret Delta|| <= tau_ret_budget",
                        "retention(z + Delta) >= tau_ret_floor",
                    ],
                    "current_parameters": {
                        "tau_ret_floor": tau_ret_floor,
                        "tau_ret_target": tau_ret_target,
                        "tau_stage_a4_retention_proxy": stage_a4_retention,
                    },
                },
                "K_id": {
                    "meaning": "identity-safe cone",
                    "definition": [
                        "||P_id Delta|| <= alpha_id ||P_obj Delta|| + epsilon_id",
                        "same_object_contraction(Delta) <= absorption_capacity(z_disc)",
                    ],
                    "current_parameters": {
                        "tau_consistency_floor": tau_consistency_floor,
                        "tau_consistency_target": tau_consistency_target,
                    },
                },
                "K_read": {
                    "meaning": "readout-safe cone",
                    "definition": [
                        "||P_disc Delta|| <= beta_read ||P_obj Delta|| + gamma_read",
                        "overall(z + Delta) >= tau_overall_floor",
                    ],
                    "current_parameters": {
                        "beta_read": beta_read,
                        "tau_overall_floor": tau_overall_floor,
                    },
                },
                "K_phase": {
                    "meaning": "phase-safe cone",
                    "definition": [
                        "c_switch(r, r', Delta) <= tau_switch",
                        "||P_phase Delta|| <= beta_phase ||P_obj Delta|| + gamma_phase",
                    ],
                    "current_parameters": {
                        "tau_switch": tau_switch,
                        "beta_phase": beta_phase,
                    },
                },
                "K_bridge": {
                    "meaning": "bridge-safe cone",
                    "definition": [
                        "||B(Delta_obj) - Delta_rel|| <= tau_bridge",
                        "bridge_anchor_drift(z + Delta) <= tau_bridge",
                    ],
                    "current_parameters": {
                        "tau_bridge": tau_bridge,
                    },
                },
            },
            "interpretation": [
                "A is not one threshold but the intersection of five update cones.",
                "An update is admissible only if it is simultaneously retention-safe, identity-safe, readout-safe, phase-safe, and bridge-safe.",
                "This explains why local patches keep failing: they improve one cone while leaving the intersection unchanged.",
            ],
        },
        "explicit_M_feas": {
            "name": "stratified_viability_manifold_M_feas",
            "high_level_form": "M_feas = UNION_m U_m with transition maps phi_(m->n) defined only on overlaps U_m INTERSECT U_n",
            "mode_strata": {
                "U_object": {
                    "meaning": "shared object chart",
                    "coordinates": ["z_obj", "family offset", "role offset"],
                    "local_condition": "object identity continuity remains above tau_consistency_floor",
                },
                "U_memory": {
                    "meaning": "memory retention chart",
                    "coordinates": ["write state", "read state", "persistence state"],
                    "local_condition": "retention remains above tau_ret_floor",
                },
                "U_relation": {
                    "meaning": "bridge-role relational chart",
                    "coordinates": ["bridge coordinate", "role kernel", "family relation state"],
                    "local_condition": "relation lift stays anchored to object-family structure",
                },
                "U_disc": {
                    "meaning": "discriminative readout chart",
                    "coordinates": ["margin state", "query state", "decision geometry"],
                    "local_condition": "decision geometry remains downstream of object geometry",
                },
                "U_phase": {
                    "meaning": "phase/rule switching chart",
                    "coordinates": ["memory mode", "identity mode", "stabilize mode"],
                    "local_condition": "switch cost remains below tau_switch",
                },
            },
            "chart_overlaps": {
                "U_object INTERSECT U_memory": "object-memory compatibility band",
                "U_object INTERSECT U_disc": "shared-object to readout overlap, current main bottleneck",
                "U_object INTERSECT U_relation": "bridge-anchor overlap",
                "U_memory INTERSECT U_phase": "safe switching overlap for protected updates",
                "U_disc INTERSECT U_phase": "safe switching overlap for readout updates",
            },
            "transition_maps": {
                "phi_object_to_memory": "stabilized write projection",
                "phi_memory_to_object": "memory-protected object reconstruction",
                "phi_object_to_relation": "bridge-role lift",
                "phi_object_to_disc": "readout formation map",
                "phi_phase_to_*": "mode-conditioned chart selector",
            },
            "failure_modes": [
                "Leaving U_object INTERSECT U_disc causes discriminative collapse.",
                "Leaving U_object INTERSECT U_memory causes catastrophic overwrite.",
                "Leaving U_memory INTERSECT U_phase causes unsafe switching.",
            ],
            "interpretation": [
                "M_feas is not a single smooth manifold everywhere; it is better modeled as a stratified manifold.",
                "Different functional regimes live in different local charts.",
                "Switching is only valid when the system is inside the overlap between charts.",
            ],
        },
        "detailed_principles": {
            "why_cones_for_A": [
                "Update safety is directional, so cone-like constraints are natural.",
                "An update can be small in norm but still point in a dangerous direction; cones capture this better than scalar thresholds.",
                "Each subsystem defines its own safe directional family, and admissibility is their intersection.",
            ],
            "why_stratified_manifold_for_M_feas": [
                "The system behaves differently in object, memory, relation, readout, and phase regimes.",
                "These regimes are not best represented by one global coordinate chart.",
                "A stratified manifold with chart overlaps explains why switching and readout formation are hard boundaries.",
            ],
            "why_this_matches_current_failures": [
                "C52-C58 repeatedly failed because they added new local coordinates without changing the intersection structure of A or the overlap structure of M_feas.",
                "The repeated plateau near 0.20 consistency indicates an unchanged global feasible region, not just a weak local mechanism.",
            ],
        },
        "compressed_theory_answer": {
            "core_statement": "Brain encoding may be governed by an admissible-update intersection A and a stratified viability manifold M_feas; coding forms when local plasticity stays inside A and system trajectories remain inside M_feas.",
            "next_formal_step": "write explicit algebraic forms for the projection operators P_mem, P_obj, P_disc, P_phase and the overlap maps phi_(m->n)",
        },
        "verdict": {
            "core_answer": "A can now be formalized as an intersection of subsystem-safe update cones, and M_feas can now be formalized as a stratified viability manifold with chart overlaps and transition maps.",
            "next_theory_target": "derive explicit candidate operators and overlap conditions that can be falsified by P3 experiments",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
