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
    ap = argparse.ArgumentParser(description="Theory-track inventory to bridge-role coupling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_bridge_role_coupling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    relation_atlas = load("theory_track_concept_relation_attribute_atlas_synthesis_20260312.json")
    stress = load("theory_track_inventory_stress_profiling_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")

    relation_capacity = float(stress["headline_metrics"]["mean_relation_lift_capacity"])
    mean_relation_overlap = sum(
        row["object_relation_overlap"] for row in overlap["restricted_overlap_maps"].values()
    ) / max(1, len(overlap["restricted_overlap_maps"]))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_bridge_role_coupling",
        },
        "bridge_role_upgrade": {
            "starting_point": "R_c as relation template attachment set",
            "upgraded_form": "B_c = (T_rel(c), K_role(c), sigma_rel(c))",
            "field_meaning": {
                "T_rel(c)": "concept-conditioned relation template family",
                "K_role(c)": "role kernel coordinates available to concept c",
                "sigma_rel(c)": "relation-lift capacity from the stress field",
            },
        },
        "candidate_equations": {
            "bridge_lift": "rho_c = G_rel(E_c) = G_rel(f_c, z_c, delta_c, A_c, S_c)",
            "role_kernel": "kappa_c = H_role(rho_c, N_same(c), N_cross(c))",
            "bridge_admissibility": "bridge lift is valid only if sigma_rel(c) stays above the minimum relation threshold",
            "family_anchor_condition": "relation lift must remain inside the family-conditioned object-relation overlap band",
        },
        "support_metrics": {
            "mean_relation_lift_capacity": relation_capacity,
            "mean_object_relation_overlap": float(mean_relation_overlap),
        },
        "mathematical_meaning": {
            "core_statement": "Bridge-role structure should be generated from inventory entries rather than built as a separate symbolic module.",
            "formation_answer": "A concept first exists as an object entry, then relation templates and role kernels are lifted from that entry through restricted object-relation overlaps.",
            "runtime_answer": "Bridge-role dynamics are inventory-conditioned and stress-gated, not free-floating.",
        },
        "verdict": {
            "core_answer": "The inventory can now be extended to parameterize bridge-role lift and role kernel generation.",
            "next_theory_target": "connect this bridge-role lift to engineering B-line and to brain-side relational probe design",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
