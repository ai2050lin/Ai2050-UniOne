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
    ap = argparse.ArgumentParser(description="Theory-track global feasibility law formalization")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_global_feasibility_law_formalization_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage_a4 = load("stage_a4_partial_closure_reestimate_20260311.json")
    stage_b1 = load("stage_b1_calibrated_partial_reestimate_20260311.json")
    stage_c42 = load("stage_c42_strong_cross_law_manifold_lift_search_20260312.json")
    stage_c49 = load("stage_c49_model_sanity_diagnostics_20260312.json")
    stage_c58 = load("stage_c58_higher_order_feasible_manifold_search_20260312.json")
    phase_push = load("phase_p1_p4_push_plan_20260312.json")
    theory_synth = load("theory_track_encoding_mechanism_synthesis_20260312.json")

    moderate_control = float(stage_c49["controls"]["moderate_regime_c42"]["summary"]["crossmodal_consistency"])
    current_best = float(stage_c58["headline_metrics"]["best_compatible_consistency"])
    retention_level = float(stage_c58["best_retention_compatible_candidate"]["summary"]["retention_concept_accuracy"])
    overall_level = float(stage_c58["best_retention_compatible_candidate"]["summary"]["overall_concept_accuracy"])
    p3_gap = max(0.0, moderate_control - current_best)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_global_feasibility_law_formalization",
        },
        "formal_object": {
            "name": "global_feasibility_law",
            "type": "controlled_encoding_dynamical_system_with_viability_constraint",
            "core_equations": [
                "z_obj(t+1) = F_obj(z_obj(t), x(t), r(t), z_mem(t))",
                "z_mem(t+1) = F_mem(z_mem(t), x(t), r(t), z_obj(t))",
                "z_rel(t+1) = F_rel(z_obj(t), z_rel(t), r(t))",
                "z_disc(t+1) = F_disc(z_obj(t), z_rel(t), q(t), r(t))",
                "r(t+1) = S(r(t), z_obj(t), z_mem(t), z_disc(t), h(t))",
                "q(t) = Q(x(t), z_obj(t), z_mem(t), z_disc(t), r(t), h(t))",
                "Delta(t) in A(z_obj(t), z_mem(t), z_disc(t), r(t), x(t))",
                "(z_obj(t), z_mem(t), z_rel(t), z_disc(t), r(t)) in M_feas",
            ],
            "state_spaces": {
                "Z_obj": "shared object manifold state",
                "Z_mem": "write/read/persistence memory state",
                "Z_rel": "bridge-role relational state",
                "Z_disc": "discriminative readout geometry state",
                "R": "phase/rule/meta-switching state",
                "Q": "query/readout access state",
            },
            "operators": {
                "F_obj": "object formation and object-state update operator",
                "F_mem": "memory retention and admissible update operator",
                "F_rel": "bridge-role lifting operator",
                "F_disc": "readout geometry formation operator",
                "S": "phase/rule selection operator",
                "Q": "query formation and selective access operator",
                "A": "admissible update set",
                "M_feas": "global feasible manifold / viability region",
            },
        },
        "meaning": {
            "plain_language": [
                "The brain first forms object-like latent states rather than storing raw stimuli directly.",
                "Updates are local and gated, but only some updates are admissible because old geometry must remain viable.",
                "Relation and role structure are downstream lifts from object states, not separate primary codes.",
                "Decision geometry is downstream of object geometry and cannot be arbitrarily compressed back into it.",
                "The whole system is constrained by a global feasibility law that decides which object, memory, relation, and readout states can coexist.",
            ],
            "current_missing_piece": "The explicit mathematical form of M_feas and the admissible-update/phase-switching constraint that keeps object geometry and discriminative geometry jointly viable.",
        },
        "empirical_constraints": {
            "stage_a4_partial_closure": float(stage_a4["headline_metrics"]["overall_stage_a4_score"]),
            "stage_b1_partial_closure": float(stage_b1["headline_metrics"]["overall_stage_b1_score"]),
            "stage_c42_retention_compatible_consistency": float(stage_c42["headline_metrics"]["best_compatible_consistency"]),
            "moderate_control_consistency": moderate_control,
            "current_best_consistency_c58": current_best,
            "current_retention_c58": retention_level,
            "current_overall_c58": overall_level,
            "remaining_gap_vs_moderate_control": p3_gap,
            "interpretation": "The object manifold clearly exists, but the shared-object-to-discriminative mapping is stuck below the moderate-control ceiling, indicating a missing feasibility constraint rather than a missing local component.",
        },
        "seven_question_mapping": {
            "Q1": "Constrains the ontology of Z_obj.",
            "Q2": "Constrains F_obj, F_mem, and A.",
            "Q3": "Constrains the internal decomposition of Z_mem into write/read/persistence strata.",
            "Q4": "Constrains F_rel and the stable relational basis.",
            "Q5": "Constrains manifold sharing and object identity continuity inside Z_obj.",
            "Q6": "Constrains F_disc and Q, and their compatibility with Z_obj.",
            "Q7": "Constrains how the abstract tuple (Z_obj, Z_mem, Z_rel, Z_disc, R) projects to real brain regions and 3D topology.",
        },
        "phase_mapping": {
            "P1": {
                "purpose": "Freeze and refine Z_obj and the bridge-role atlas.",
                "linked_questions": ["Q1", "Q4", "Q5"],
                "current_state": phase_push["phases"]["P1"],
            },
            "P2": {
                "purpose": "Freeze F_obj, F_mem, A, and write/read separation.",
                "linked_questions": ["Q2", "Q3"],
                "current_state": phase_push["phases"]["P2"],
            },
            "P3": {
                "purpose": "Close compatibility between Z_obj and Z_disc under M_feas.",
                "linked_questions": ["Q5", "Q6"],
                "current_state": phase_push["phases"]["P3"],
            },
            "P4": {
                "purpose": "Map the formal object to real brain-side falsification and topology.",
                "linked_questions": ["Q7"],
                "current_state": phase_push["phases"]["P4"],
            },
        },
        "formation_and_operation_hypothesis": {
            "formation_process": theory_synth["formation_process_hypothesis"],
            "system_operation": theory_synth["system_level_operation_hypothesis"],
            "compressed_hypothesis": "Brain encoding is formed by controlled local plasticity over a shared object manifold, then lifted by bridge-role operators and read out through a separate discriminative geometry, all under a global viability constraint.",
        },
        "verdict": {
            "core_answer": "Global feasibility law can now be stated as a formal object: a controlled encoding dynamical system whose state tuple must remain inside a viability manifold M_feas while all updates stay inside an admissible set A.",
            "next_theory_target": "identify an explicit constructive form for M_feas and the admissibility relation A",
            "next_project_target": "keep P3 -> P4 -> P1 -> P2 and use P3 to narrow the structure of M_feas",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
