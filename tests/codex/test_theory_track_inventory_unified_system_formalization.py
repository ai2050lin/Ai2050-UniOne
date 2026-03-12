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
    ap = argparse.ArgumentParser(description="Theory-track inventory unified system formalization")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_unified_system_formalization_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    inv_A = load("theory_track_inventory_to_A_coupling_20260312.json")
    inv_M = load("theory_track_inventory_to_Mfeas_coupling_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_unified_system_formalization",
        },
        "unified_object": {
            "name": "inventory_conditioned_controlled_encoding_system",
            "high_level_form": "Sys(I) = (I, A(I), M_feas(I), F, Q, R)",
            "meaning": "inventory becomes the conditioning object that shapes admissible updates and feasible trajectories",
        },
        "candidate_equations": {
            "inventory_entry": inv_math["inventory_object_definition"]["entry_form"],
            "admissibility": inv_A["inventory_conditioned_form"],
            "viability": inv_M["inventory_conditioned_form"],
            "controlled_dynamics": "z_(t+1) = F(z_t, x_t, r_t, I)",
            "query_readout": "q_t = Q(x_t, z_t, r_t, I)",
            "rule_update": "r_(t+1) = R(r_t, z_t, x_t, I)",
            "joint_constraint": "Delta_t in A(I) and trajectory(z_t) subset of M_feas(I)",
        },
        "system_interpretation": {
            "core_statement": "The inventory is no longer only a record of encoded concepts. It becomes the object that parameterizes local geometry, safe updates, and viable transitions.",
            "formation_answer": "Encoding forms by entering a family patch, selecting concept-local offsets and attribute directions, and then updating only along inventory-conditioned admissible directions.",
            "runtime_answer": "System operation becomes a trajectory over family-patched charts, with readout and rule switching constrained by inventory-conditioned overlap bands.",
        },
        "verdict": {
            "core_answer": "The encoding inventory can now be treated as the central mathematical object of the theory track: it conditions both A and M_feas and therefore enters the core system equations.",
            "next_theory_target": "use this unified inventory-conditioned system to guide the next P3 engineering block and to refine brain-side probe design",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
