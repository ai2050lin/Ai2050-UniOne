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
    ap = argparse.ArgumentParser(description="Theory-track inventory higher-order geometry")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_higher_order_geometry_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    explicit = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    info = load("theory_track_inventory_information_gain_summary_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_higher_order_geometry",
        },
        "higher_order_object": {
            "name": "inventory_conditioned_stratified_path_bundle",
            "high_level_form": "H(I) = (B_family, E_concept, F_attr, F_rel, F_stress, P_path, O_overlap)",
            "meaning": "inventory should be viewed as a higher-order bundle object: base family charts plus multiple attached fibers and admissible path structure",
        },
        "bundle_layers": {
            "B_family": "family-patch base manifold",
            "E_concept": "concept-entry section over each family patch",
            "F_attr": "attribute-direction fiber",
            "F_rel": "relation-template / role-kernel fiber",
            "F_stress": "novelty-retention-relation stress fiber",
            "P_path": "admissible path bundle controlling transitions to memory/readout/relation",
            "O_overlap": "restricted overlap bundle controlling valid chart-to-chart transport",
        },
        "candidate_equations": {
            "bundle_section": "s_c : B_family(f_c) -> E_concept(c)",
            "attribute_fiber": "F_attr(c) = span{u_(f_c,k)}",
            "relation_fiber": "F_rel(c) = span{T_rel(c), K_role(c)}",
            "stress_fiber": "F_stress(c) = (sigma_novel(c), sigma_ret(c), sigma_rel(c))",
            "path_bundle": "P_path(c, mode_1 -> mode_2) = {gamma : gamma stays inside A(I) and M_feas(I)}",
            "overlap_bundle": "O_overlap(m,n,f) = U_m^(f) INTERSECT U_n^(f)",
        },
        "why_higher_order": {
            "core_answer": "The inventory is no longer well-described by a flat atlas alone. It behaves like a stratified base manifold with multiple attached fibers and path constraints.",
            "evidence": [
                info["new_information"]["low_rank_family_axes"]["meaning"],
                info["new_information"]["recurrent_dimensions"]["meaning"],
                info["new_information"]["restricted_overlap"]["meaning"],
                explicit["compressed_theory_answer"]["core_statement"],
            ],
        },
        "mathematical_consequences": {
            "not_just_points": "concepts are not only points z_c, but sections with attached fibers",
            "not_just_vectors": "updates are not only vectors, but admissible paths through bundle charts",
            "not_just_one_space": "readout, relation, and memory do not live in one flat shared space; they are coupled through overlap bundles",
        },
        "verdict": {
            "core_answer": "The next theoretical upgrade is to treat the encoding inventory as a stratified path bundle rather than only a layered atlas.",
            "next_theory_target": "derive a compact unified theory from this bundle view and use it to propose falsifiable operator-form changes",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
