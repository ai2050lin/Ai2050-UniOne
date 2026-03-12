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
    ap = argparse.ArgumentParser(description="Theory-track inventory to seven-question mapping")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_seven_question_mapping_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load("theory_track_concept_encoding_inventory_20260312.json")
    attrs = load("theory_track_attribute_axis_analysis_20260312.json")
    atlas = load("theory_track_concept_relation_attribute_atlas_synthesis_20260312.json")
    exclusion = load("theory_track_atlas_to_A_Mfeas_exclusion_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_seven_question_mapping",
        },
        "seven_question_mapping": {
            "Q1_encoding_object_layer": {
                "inventory_support": "concept entries + family patch + family offsets",
                "current_strength": "strong",
                "why": "inventory already reconstructs object kernel, family basis, local neighbors, and cross-family margins",
            },
            "Q2_local_update_law": {
                "inventory_support": "inventory provides stable concept offsets and family-specific low-rank axes",
                "current_strength": "medium",
                "why": "safe update directions are not fully solved, but inventory now gives the local geometry updates must preserve",
            },
            "Q3_write_read_separation": {
                "inventory_support": "inventory exposes which concept-local coordinates remain stable under family-centered representation",
                "current_strength": "medium",
                "why": "it constrains what must be preserved, though write/read dynamics themselves still need direct stress tests",
            },
            "Q4_bridge_role_kernel": {
                "inventory_support": "relation templates can now be layered onto concept entries and attribute axes",
                "current_strength": "medium",
                "why": "bridge-role structure is not fully reconstructed, but the atlas now offers the object substrate they must lift from",
            },
            "Q5_crossmodal_consistency": {
                "inventory_support": "concept entries already unify visual, tactile, and language-like state into one atlas slot",
                "current_strength": "strong",
                "why": "inventory directly supports the shared object manifold hypothesis",
            },
            "Q6_discriminative_geometry": {
                "inventory_support": "cross-family margins and restricted overlaps sharply constrain readout geometry",
                "current_strength": "strong",
                "why": "inventory now explains why direct object-to-disc collapse fails and why restricted overlap readout is needed",
            },
            "Q7_brain_mapping_3d": {
                "inventory_support": "atlas entries can become natural probes for cortical mapping and falsification",
                "current_strength": "partial",
                "why": "the abstract probe set is ready, but brain-side execution is still pending",
            },
        },
        "support_metrics": {
            "mean_within_to_cross_margin": inventory["headline_metrics"]["mean_within_to_cross_margin"],
            "mean_attribute_alignment": attrs["headline_metrics"]["mean_attribute_alignment"],
            "atlas_separation_score": atlas["support_metrics"]["atlas_separation_score"],
        },
        "exclusion_support": exclusion["excluded_candidates"],
        "verdict": {
            "core_answer": "The encoding inventory now strongly supports Q1, Q5, and Q6, gives medium support to Q2, Q3, and Q4, and provides a partial probe-ready substrate for Q7.",
            "next_theory_target": "attach stress and relation-lift experiments to convert medium-support areas into stronger closure",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
