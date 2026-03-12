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
    ap = argparse.ArgumentParser(description="Theory-track path-conditioned encoding law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_path_conditioned_encoding_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    switching = load("theory_track_switching_aware_readout_law_20260312.json")
    inv_sys = load("theory_track_inventory_unified_system_formalization_20260312.json")
    op_family = load("theory_track_inventory_operator_family_closure_20260312.json")
    concept_transport = load("theory_track_inventory_stress_to_readout_transport_coupling_20260312.json")
    phase_transport = load("theory_track_phase_level_transport_operator_20260312.json")

    family_blocks = op_family["family_operator_blocks"]
    concept_profiles = concept_transport["concept_transport_profiles"]
    phase_ops = phase_transport["phase_transport_operator"]["operators"]

    concept_path_entries: dict[str, dict] = {}
    stabilize_open = 0
    novelty_narrow = 0
    relation_conditional = 0

    for concept, row in concept_profiles.items():
        family = str(row["family"])
        family_block = family_blocks[family]
        stabilize_is_open = str(phase_ops["stabilize_to_read"]["status"]) == "candidate_open" and float(row["transport_budget"]) > 0.0
        novelty_is_narrow = str(phase_ops["novelty_to_read"]["status"]) == "candidate_narrow"
        relation_is_conditional = str(phase_ops["relation_to_read"]["status"]) == "candidate_conditional"

        if stabilize_is_open:
            stabilize_open += 1
        if novelty_is_narrow:
            novelty_narrow += 1
        if relation_is_conditional:
            relation_conditional += 1

        concept_path_entries[concept] = {
            "family": family,
            "path_signature": {
                "object_operator": family_block["update_block"]["object_operator"],
                "memory_operator": family_block["update_block"]["memory_operator"],
                "identity_operator": family_block["update_block"]["identity_operator"],
                "disc_operator": family_block["readout_block"]["disc_operator"],
            },
            "transport_budget": float(row["transport_budget"]),
            "phase_status": {
                "stabilize_to_read": "open" if stabilize_is_open else "closed",
                "novelty_to_read": "narrow" if novelty_is_narrow else "open",
                "relation_to_read": "conditional" if relation_is_conditional else "open",
            },
            "path_open_condition": "Delta in A(I) AND trajectory subset of M_feas(I) AND Tau_readout(c, mode_1 -> mode_2) > 0",
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_path_conditioned_encoding_law",
        },
        "path_conditioned_encoding_law": {
            "core_statement": "Encoding is not only a concept state. It is a concept state plus the admissible path that allows safe transport into memory, relation, and readout.",
            "high_level_form": "Enc_path(c, mode_1 -> mode_2) = (E_c, Omega^(f_c)_upd, Omega^(f_c)_read, Tau_readout(c, mode_1 -> mode_2), chi_A, chi_M)",
            "path_open_indicator": "Pi_path(c, mode_1 -> mode_2) = 1[Delta in A(I)] * 1[trajectory subset of M_feas(I)] * 1[Tau_readout(c, mode_1 -> mode_2) > 0]",
            "switching_aware_readout_form": switching["switching_aware_readout_law"]["formal_form"],
            "system_link": inv_sys["unified_object"]["high_level_form"],
        },
        "layer_interpretation": {
            "state_layer": "E_c gives the concept entry on the family-patched object atlas",
            "operator_layer": "Omega^(f_c)_upd and Omega^(f_c)_read determine which local directions can be safely traversed",
            "transport_layer": "Tau_readout(c, mode_1 -> mode_2) determines whether object state can reach readout under switching",
            "constraint_layer": "chi_A and chi_M enforce admissibility and viability along the path",
        },
        "concept_path_entries": concept_path_entries,
        "headline_metrics": {
            "concept_count": int(len(concept_path_entries)),
            "stabilize_open_count": int(stabilize_open),
            "novelty_narrow_count": int(novelty_narrow),
            "relation_conditional_count": int(relation_conditional),
        },
        "mathematical_meaning": {
            "core_answer": "A usable encoding is no longer a static address. It is an atlas entry plus an admissible, phase-conditioned transport path.",
            "why_it_matters": [
                "explains why object manifold existence alone does not guarantee successful readout",
                "ties concept identity to local operator blocks and switching-aware transport",
                "lets engineering prune P3 by path openness instead of broader geometric heuristics",
            ],
        },
        "verdict": {
            "core_answer": "The theory track can now rewrite 'encoding-as-path' as a formal path-conditioned encoding law rather than only a qualitative intuition.",
            "next_theory_target": "promote this law into a full path-conditioned readout and bridge-lift framework",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
