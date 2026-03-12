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
    ap = argparse.ArgumentParser(description="Theory-track concept relation attribute atlas synthesis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_concept_relation_attribute_atlas_synthesis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load("theory_track_concept_encoding_inventory_20260312.json")
    attrs = load("theory_track_attribute_axis_analysis_20260312.json")
    atlas = load("theory_track_concept_family_atlas_analysis_20260312.json")

    relation_templates = {
        "same_family_neighbor": [
            "concept should have at least one nearby same-family neighbor with small offset distance",
            "family patch should preserve local continuity between sibling concepts",
        ],
        "cross_family_separation": [
            "concept should keep a large margin against foreign-family neighbors",
            "cross-family boundaries should act as atlas patch borders",
        ],
        "attribute_lift": [
            "attribute axes should explain part of concept-local offset structure",
            "relation or role lifts should reuse some of the same local coordinates rather than invent a disjoint code",
        ],
    }

    fruit_entry = inventory["concept_inventory"]["apple"]
    animal_entry = inventory["concept_inventory"]["cat"]
    abstract_entry = inventory["concept_inventory"]["truth"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_concept_relation_attribute_atlas_synthesis",
        },
        "atlas_layers": {
            "layer_1_family_patch": "shared family basis and local patch radius",
            "layer_2_concept_entry": "concept-specific family offset and neighbor structure",
            "layer_3_attribute_axis": "reusable local directions such as round, elongated, domestic, persistent",
            "layer_4_relation_template": "same-family, cross-family, and bridge-role style relational organization",
        },
        "representative_entries": {
            "fruit_entry": fruit_entry,
            "animal_entry": animal_entry,
            "abstract_entry": abstract_entry,
        },
        "representative_attribute_axes": {
            "round": attrs["attribute_axes"]["round"],
            "domestic": attrs["attribute_axes"]["domestic"],
            "persistent": attrs["attribute_axes"]["persistent"],
            "structured": attrs["attribute_axes"]["structured"],
        },
        "relation_templates": relation_templates,
        "system_hypothesis": {
            "formation_process": [
                "family basis defines coarse chart location",
                "concept offset defines local identity within the family patch",
                "attribute axes define reusable local directions on top of concept offsets",
                "relation templates connect concept entries into structured neighborhoods and bridge-role organization",
            ],
            "runtime_answer": [
                "encoding is not just token lookup but movement inside a layered atlas",
                "concept, attribute, and relation are different coordinate layers on the same atlas rather than separate symbolic systems",
            ],
        },
        "support_metrics": {
            "atlas_separation_score": atlas["headline_metrics"]["atlas_separation_score"],
            "mean_attribute_alignment": attrs["headline_metrics"]["mean_attribute_alignment"],
            "mean_within_to_cross_margin": inventory["headline_metrics"]["mean_within_to_cross_margin"],
        },
        "verdict": {
            "core_answer": "The encoding atlas can now be described as concept entries plus attribute axes plus relation templates, not only family patches.",
            "next_theory_target": "attach novelty and retention stress to each atlas layer to identify which layers are most stable and most fragile",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
