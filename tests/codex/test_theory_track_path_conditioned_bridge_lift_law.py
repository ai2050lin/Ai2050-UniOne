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
    ap = argparse.ArgumentParser(description="Theory-track path-conditioned bridge-lift law")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_path_conditioned_bridge_lift_law_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    bridge = load("theory_track_inventory_bridge_role_coupling_20260312.json")
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    op_family = load("theory_track_inventory_operator_family_closure_20260312.json")

    family_blocks = op_family["family_operator_blocks"]
    concept_path_entries = path_law["concept_path_entries"]

    bridge_ready = 0
    bridge_entries: dict[str, dict] = {}
    for concept, row in concept_path_entries.items():
        family = row["family"]
        relation_overlap = float(family_blocks[family]["bridge_block"]["object_relation_overlap"])
        conditional = row["phase_status"]["relation_to_read"] == "conditional"
        if conditional and relation_overlap > 0.5:
            bridge_ready += 1

        bridge_entries[concept] = {
            "family": family,
            "relation_overlap": relation_overlap,
            "path_condition": "conditional" if conditional else "open",
            "bridge_open_indicator": "1[relation_overlap > tau_rel] * 1[sigma_rel(c) > tau_sigma] * 1[Delta in A(I)]",
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_path_conditioned_bridge_lift_law",
        },
        "path_conditioned_bridge_lift_law": {
            "core_statement": "Relation and role do not lift from object entries unconditionally. They open only when the object path remains admissible and the object-relation overlap is sufficiently wide.",
            "formal_form": "BridgeLift(c) = G_rel(E_c) only if Pi_bridge(c) = 1",
            "expanded_form": "Pi_bridge(c) = 1[relation_overlap(f_c) > tau_rel] * 1[sigma_rel(c) > tau_sigma] * 1[Delta in A(I)]",
            "bridge_support": bridge["candidate_equations"]["bridge_lift"],
        },
        "concept_bridge_entries": bridge_entries,
        "headline_metrics": {
            "concept_count": int(len(bridge_entries)),
            "bridge_ready_count": int(bridge_ready),
            "mean_relation_overlap": bridge["support_metrics"]["mean_object_relation_overlap"],
            "mean_relation_lift_capacity": bridge["support_metrics"]["mean_relation_lift_capacity"],
        },
        "mathematical_meaning": {
            "core_answer": "Bridge-role structure is best modeled as a path-conditioned lift from object entries, not as a free symbolic layer.",
            "why_it_helps": [
                "ties Q4 directly back to the object atlas",
                "explains why role structure should remain anchored to object-relation overlaps",
                "gives a direct route from inventory to B-line engineering tests",
            ],
        },
        "verdict": {
            "core_answer": "The bridge-role side of the theory track can now be expressed in the same path-conditioned language as readout.",
            "next_theory_target": "attach this bridge-lift law to B-line and brain-side relational probes",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
