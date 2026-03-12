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
    ap = argparse.ArgumentParser(description="Theory-track inventory to brain-probe coupling")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_brain_probe_coupling_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_math = load("theory_track_inventory_math_structure_formalization_20260312.json")
    inv_map = load("theory_track_inventory_seven_question_mapping_20260312.json")
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")
    cross = load("theory_track_cross_family_probe_analysis_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_brain_probe_coupling",
        },
        "brain_probe_design": {
            "probe_unit": "family-patched concept entry",
            "probe_layers": {
                "object_probe": "family basis and concept offset should map to partially shared but region-parameterized cortical patterns",
                "attribute_probe": "attribute axes should map to locally reusable feature gradients rather than isolated neurons only",
                "relation_probe": "bridge-role lifts should map to cross-area coordination patterns rather than only local codes",
                "stress_probe": "novelty and retention stress should map to changes in plasticity-sensitive versus stable subspaces",
            },
        },
        "candidate_projection_rules": {
            "object_to_brain": "Pi_obj(E_c) = region_parameterized_object_pattern(f_c, z_c)",
            "attribute_to_brain": "Pi_attr(E_c) = local_gradient_pattern(A_c, f_c)",
            "relation_to_brain": "Pi_rel(E_c) = coordination_pattern(R_c, f_c)",
            "stress_to_brain": "Pi_stress(E_c) = plasticity_vs_stability_pattern(S_c, f_c)",
        },
        "support_metrics": {
            "atlas_separation_score": atlas["headline_metrics"]["atlas_separation_score"],
            "mean_intrusion_gap": cross["headline_metrics"]["mean_intrusion_gap"],
            "brain_probe_readiness": inv_map["seven_question_mapping"]["Q7_brain_mapping_3d"]["current_strength"],
        },
        "mathematical_meaning": {
            "core_statement": "Brain-side mapping does not need to start from whole-brain reconstruction. It can start from inventory-conditioned probe families.",
            "formation_answer": "If inventory is the right abstract object, then brain probes should preserve family patch structure, local concept offsets, and local attribute gradients.",
            "runtime_answer": "Brain activity should reflect region-parameterized realizations of the same inventory-conditioned encoding system rather than unrelated codes per area.",
        },
        "verdict": {
            "core_answer": "The inventory can now serve as the abstract probe source for brain-side mapping and falsification.",
            "next_theory_target": "compose object, attribute, relation, and stress probes into a concrete P4-ready brain-side bundle",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
