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
    ap = argparse.ArgumentParser(description="Theory-track inventory to M_feas coupling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_to_Mfeas_coupling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    stress = load("theory_track_inventory_stress_profiling_20260312.json")
    explicit = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")

    mean_stable = float(stress["headline_metrics"]["stable_under_stress_ratio"])
    coupled_charts = {}
    for family, patch in atlas["family_atlas"].items():
        coupled_charts[family] = {
            "U_object_family": {
                "radius": patch["family_radius"],
                "concepts": patch["concepts"],
            },
            "U_memory_family": {
                "overlap_width": overlap["restricted_overlap_maps"][family]["object_memory_overlap"],
                "meaning": "family-local protected retention band",
            },
            "U_disc_family": {
                "overlap_width": overlap["restricted_overlap_maps"][family]["object_disc_overlap"],
                "meaning": "family-local readout band",
            },
            "U_relation_family": {
                "overlap_width": overlap["restricted_overlap_maps"][family]["object_relation_overlap"],
                "meaning": "family-local bridge-role band",
            },
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_to_Mfeas_coupling",
        },
        "starting_form": explicit["explicit_M_feas"]["high_level_form"],
        "inventory_conditioned_form": (
            "M_feas(I) = UNION_f [U_object^(f)(I) UNION U_memory^(f)(I) "
            "UNION U_disc^(f)(I) UNION U_relation^(f)(I)] UNION U_phase"
        ),
        "coupling_rule": {
            "core_statement": "Inventory entries do not just lie inside M_feas; they parameterize the family-patched charts and overlap widths that define M_feas.",
            "entry_level_statement": "Family radius, concept offsets, and stress stability determine how wide each local chart and overlap band can be.",
        },
        "family_charts": coupled_charts,
        "stability_prior": {
            "stable_under_stress_ratio": mean_stable,
            "meaning": "higher inventory stability supports wider safe memory overlaps but does not automatically widen object-disc overlaps",
        },
        "mathematical_meaning": {
            "Mfeas_from_inventory": [
                "family patch defines local object chart location",
                "concept inventory defines chart occupancy",
                "restricted overlap widths define legal transport bands",
                "stress stability defines whether trajectories remain in those bands",
            ],
            "why_important": "This upgrades inventory from atlas database to viability-manifold parameterization.",
        },
        "verdict": {
            "core_answer": "The inventory can now be coupled directly to M_feas: family patches and stress profiles determine chart structure and overlap width.",
            "next_theory_target": "join the A(I) and M_feas(I) constructions into one unified inventory-conditioned encoding system",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
