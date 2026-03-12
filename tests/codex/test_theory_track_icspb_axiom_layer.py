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
    ap = argparse.ArgumentParser(description="Theory-track ICSPB axiom layer")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_icspb_axiom_layer_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    higher = load("theory_track_inventory_higher_order_geometry_20260312.json")
    new_math = load("theory_track_new_math_theory_candidate_20260312.json")
    special = load("theory_track_special_math_system_formalization_20260312.json")
    explicit = load("theory_track_explicit_A_Mfeas_formalization_20260312.json")
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")

    axioms = [
        {
            "id": "A1_family_stratification",
            "statement": "Encoding objects first land on family-patched base manifolds rather than one global uniform coordinate system.",
            "formal": "B = UNION_f B_family^(f)",
            "supports": ["Q1", "Q5"],
        },
        {
            "id": "A2_section_based_concepts",
            "statement": "Each concept is a section with a local offset on a family patch rather than an isolated point.",
            "formal": "z_c = b_(f_c) + delta_c",
            "supports": ["Q1", "Q5", "Q6"],
        },
        {
            "id": "A3_attached_fibers",
            "statement": "Attribute, relation, and stress are attached fibers over concept sections rather than external labels.",
            "formal": "H(I) = (B_family, E_concept, F_attr, F_rel, F_stress, P_path, O_overlap)",
            "supports": ["Q2", "Q3", "Q4", "Q5"],
        },
        {
            "id": "A4_intersected_admissibility",
            "statement": "Update legality is decided by intersected cone families rather than a single scalar threshold.",
            "formal": special["special_math_system"]["components"]["admissible_cones"],
            "supports": ["Q2", "Q3", "Q4"],
        },
        {
            "id": "A5_stratified_viability",
            "statement": "Long-run system viability is determined by stratified manifolds with restricted overlaps.",
            "formal": special["special_math_system"]["components"]["viability_strata"],
            "supports": ["Q3", "Q5", "Q6", "Q7"],
        },
        {
            "id": "A6_path_conditioned_computation",
            "statement": "Usable readout, bridge lift, and phase switching all proceed along admissible paths.",
            "formal": path_law["path_conditioned_encoding_law"]["high_level_form"],
            "supports": ["Q2", "Q4", "Q6"],
        },
        {
            "id": "A7_system_projection",
            "statement": "System-level neural properties are projections of the same encoding structure rather than separate add-on modules.",
            "formal": "Properties_brain = Phi(H(I), A(I), M_feas(I), F, Q, R)",
            "supports": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_ICSPB_axiom_layer",
        },
        "theory_name": new_math["theory_candidate"]["name"],
        "axiom_count": len(axioms),
        "axioms": axioms,
        "compressed_view": {
            "core_claim": "ICSPB can now be treated as an axiom system rather than only a narrative theory candidate.",
            "bundle_reference": higher["higher_order_object"]["high_level_form"],
            "admissibility_reference": explicit["explicit_A"]["high_level_form"],
            "viability_reference": explicit["explicit_M_feas"]["high_level_form"],
        },
        "verdict": {
            "core_answer": "The new mathematical theory candidate now has a first explicit axiom layer.",
            "next_theory_target": "derive operator families and falsifiable predictions from the ICSPB axioms.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
