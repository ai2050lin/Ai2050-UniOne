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
    ap = argparse.ArgumentParser(description="Theory-track inventory math structure formalization")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_math_structure_formalization_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load("theory_track_concept_encoding_inventory_20260312.json")
    attrs = load("theory_track_attribute_axis_analysis_20260312.json")
    relation_atlas = load("theory_track_concept_relation_attribute_atlas_synthesis_20260312.json")
    stress = load("theory_track_inventory_stress_profiling_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_math_structure_formalization",
        },
        "inventory_object_definition": {
            "name": "encoding_inventory_I",
            "high_level_form": "I = {E_c} over concepts c, grouped by family patches and augmented by attribute axes, relation templates, and stress fields",
            "entry_form": "E_c = (f_c, z_c, delta_c, N_same(c), N_cross(c), A_c, R_c, S_c)",
            "entry_fields": {
                "f_c": "family index / atlas patch id",
                "z_c": "concept state on the shared object atlas",
                "delta_c": "family-centered concept offset",
                "N_same(c)": "nearest same-family neighborhood",
                "N_cross(c)": "nearest cross-family neighborhood",
                "A_c": "attribute-axis attachment set",
                "R_c": "relation template attachment set",
                "S_c": "stress profile: novelty pressure, retention risk, relation-lift capacity",
            },
        },
        "layered_structure": {
            "family_patch_layer": relation_atlas["atlas_layers"]["layer_1_family_patch"],
            "concept_entry_layer": relation_atlas["atlas_layers"]["layer_2_concept_entry"],
            "attribute_axis_layer": relation_atlas["atlas_layers"]["layer_3_attribute_axis"],
            "relation_template_layer": relation_atlas["atlas_layers"]["layer_4_relation_template"],
            "stress_field_layer": "dynamic profile over each concept entry",
        },
        "candidate_equations": {
            "concept_state_decomposition": "z_c = b_(f_c) + delta_c",
            "local_attribute_expansion": "delta_c ~= SUM_k a_(c,k) u_(f_c,k) + epsilon_c",
            "neighborhood_condition": "d(z_c, N_same(c)) << d(z_c, N_cross(c))",
            "stress_field": "S_c = (sigma_novel(c), sigma_ret(c), sigma_rel(c))",
            "inventory_viability_rule": "E_c is dynamically admissible only if S_c stays inside the family-conditioned admissible region induced by A and M_feas",
        },
        "mathematical_principles": {
            "low_rank_family_patches": "family patches behave like low-rank local charts",
            "sparse_concept_offsets": "concept identity is carried by relatively small family-centered offsets",
            "reusable_attribute_directions": "attributes act like local basis directions attached to family patches",
            "restricted_relation_lifts": "relations should lift from concept entries without breaking family-local geometry",
            "stress_augmented_inventory": "inventory becomes dynamic once each concept entry carries a stress field",
        },
        "support_metrics": {
            "mean_within_to_cross_margin": inventory["headline_metrics"]["mean_within_to_cross_margin"],
            "mean_attribute_alignment": attrs["headline_metrics"]["mean_attribute_alignment"],
            "stable_under_stress_ratio": stress["headline_metrics"]["stable_under_stress_ratio"],
        },
        "verdict": {
            "core_answer": "The encoding inventory can now be formalized as a layered mathematical object: family patch plus concept entry plus attribute axes plus relation templates plus stress fields.",
            "next_theory_target": "connect this inventory object to family-conditioned operators and restricted overlap maps so the inventory directly parameterizes A and M_feas",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
