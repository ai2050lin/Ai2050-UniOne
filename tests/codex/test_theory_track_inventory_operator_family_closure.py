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
    ap = argparse.ArgumentParser(description="Theory-track inventory operator family closure")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_operator_family_closure_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_A = load("theory_track_inventory_to_A_coupling_20260312.json")
    inv_M = load("theory_track_inventory_to_Mfeas_coupling_20260312.json")
    operators = load("theory_track_family_conditioned_projection_operators_20260312.json")
    limitations = load("theory_track_inventory_limitations_analysis_20260312.json")

    family_operator_blocks = {}
    for family, row in operators["core_operators"].items():
        family_operator_blocks[family] = {
            "update_block": {
                "object_operator": row["P_obj_family"]["support_dims"],
                "memory_operator": row["P_mem_family"]["support_dims"],
                "identity_operator": row["P_id_family"]["support_dims"],
            },
            "readout_block": {
                "disc_operator": row["P_disc_family"]["support_dims"],
                "object_disc_overlap": inv_M["family_charts"][family]["U_disc_family"]["overlap_width"],
            },
            "bridge_block": {
                "bridge_operator": row["P_obj_family"]["support_dims"],
                "object_relation_overlap": inv_M["family_charts"][family]["U_relation_family"]["overlap_width"],
            },
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_operator_family_closure",
        },
        "operator_family_closure": {
            "core_statement": "Inventory now induces a family of local operator blocks rather than a single global operator.",
            "formal_form": "Omega(I) = {Omega^(f)_upd, Omega^(f)_read, Omega^(f)_bridge}_f",
            "meaning": "each family patch carries its own update, readout, and bridge operators constrained by local overlaps",
        },
        "family_operator_blocks": family_operator_blocks,
        "closure_status": {
            "object_operator_family": "strong",
            "readout_operator_family": "partial",
            "bridge_operator_family": "partial",
            "phase_operator_family": limitations["core_limitations"]["phase_switch_limitation"]["severity"],
        },
        "why_not_closed_yet": [
            "family-conditioned operators are now explicit, but cross-family transition operators are still weakly specified",
            "stress gating is known, but readout transport law is not yet fully closed",
            "phase switching remains constrained by overlaps without a full transition operator",
        ],
        "verdict": {
            "core_answer": "The inventory now supports operator-family closure at the patch level, but not yet full transition closure across patches and phases.",
            "next_theory_target": "close stress-to-readout transport and phase-transition operators on top of the family operator blocks",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
