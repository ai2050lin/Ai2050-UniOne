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
    ap = argparse.ArgumentParser(description="Theory-track inventory bottleneck resolution analysis")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_bottleneck_resolution_analysis_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inv_map = load("theory_track_inventory_seven_question_mapping_20260312.json")
    mining = load("theory_track_encoding_inventory_feature_mining_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    p3_loop = load("theory_track_atlas_driven_p3_exclusion_loop_20260312.json")

    object_disc_overlap = [
        overlap["restricted_overlap_maps"][family]["object_disc_overlap"]
        for family in overlap["restricted_overlap_maps"]
    ]
    object_memory_overlap = [
        overlap["restricted_overlap_maps"][family]["object_memory_overlap"]
        for family in overlap["restricted_overlap_maps"]
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_bottleneck_resolution_analysis",
        },
        "current_bottlenecks": {
            "main_bottleneck": "shared object manifold to discriminative geometry compatibility",
            "secondary_bottlenecks": [
                "write/read law not yet fully tied to inventory stress profiles",
                "bridge-role lifts not yet densely layered over the atlas",
                "brain-side mapping still protocol-ready rather than executed",
            ],
        },
        "how_inventory_hits_bottlenecks": {
            "object_disc_bottleneck": "inventory plus overlap maps show object-disc overlap is narrow, so direct collapse candidates can be excluded",
            "update_law_bottleneck": "stable family axes and concept offsets define what safe updates must preserve",
            "bridge_role_bottleneck": "relation templates show where role lifts must anchor instead of building on free-floating symbolic states",
            "brain_mapping_bottleneck": "inventory yields more concrete probes for later cortical mapping and falsification",
        },
        "can_inventory_crack_key_details": {
            "yes_part": [
                "family basis and concept offset structure",
                "cross-family patch boundaries",
                "attribute directions on local charts",
                "restricted object-to-disc overlap constraints",
                "family-conditioned operator candidates",
            ],
            "not_yet": [
                "full admissible-update law under novelty stress",
                "full bridge-role lift mechanics",
                "brain-side 3D projection closure",
            ],
            "overall_answer": "inventory can already expose several of the most important coding details, but it still needs stress and brain-side coupling to close the full mechanism",
        },
        "inventory_statistics": {
            "cross_to_within_ratio": mining["headline_metrics"]["cross_to_within_ratio"],
            "mean_object_disc_overlap": float(sum(object_disc_overlap) / len(object_disc_overlap)),
            "mean_object_memory_overlap": float(sum(object_memory_overlap) / len(object_memory_overlap)),
            "excluded_p3_candidate_count": p3_loop["bridge_to_engineering"]["excluded_by_theory_count"],
        },
        "verdict": {
            "core_answer": "The encoding inventory is already solving the object-layer and readout-compatibility parts of the puzzle, and it is starting to constrain update law and bridge-role law, but it has not yet fully closed those latter two.",
            "next_theory_target": "attach novelty stress, retention stress, and relation-lift probes to each atlas entry so inventory can move from structural explanation to dynamic closure",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
