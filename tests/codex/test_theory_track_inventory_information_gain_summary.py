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
    ap = argparse.ArgumentParser(description="Theory-track inventory information gain summary")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_inventory_information_gain_summary_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load("theory_track_concept_encoding_inventory_20260312.json")
    mining = load("theory_track_encoding_inventory_feature_mining_20260312.json")
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    operators = load("theory_track_inventory_operator_family_closure_20260312.json")
    stress = load("theory_track_inventory_stress_profiling_20260312.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_inventory_information_gain_summary",
        },
        "new_information": {
            "family_patch_structure": {
                "num_concepts": inventory["headline_metrics"]["num_concepts"],
                "mean_within_to_cross_margin": inventory["headline_metrics"]["mean_within_to_cross_margin"],
                "meaning": "concepts cluster into stable family patches rather than one globally smooth cloud",
            },
            "low_rank_family_axes": {
                "family_rank_structure": mining["family_rank_structure"],
                "stable_family_axes": mining["stable_family_axes"],
                "meaning": "each family patch is low-rank and has its own reusable local basis",
            },
            "recurrent_dimensions": {
                "universal_recurrent_dims": mining["universal_recurrent_dims"],
                "meaning": "different families still reuse a smaller common scaffold of dimensions",
            },
            "restricted_overlap": {
                "restricted_overlap_maps": overlap["restricted_overlap_maps"],
                "meaning": "object-memory overlap is wide, object-disc overlap is narrow, object-relation overlap is medium",
            },
            "operator_families": {
                "closure_status": operators["closure_status"],
                "meaning": "inventory induces family-conditioned update/readout/bridge operators rather than one global operator",
            },
            "stress_profiles": {
                "headline_metrics": stress["headline_metrics"],
                "meaning": "each concept entry carries novelty/retention/relation-lift information, not only static geometry",
            },
        },
        "verdict": {
            "core_answer": "The inventory has moved from a concept list to a structured source of geometric, dynamic, and operator-level information.",
            "next_theory_target": "turn these information gains into explicit operator-form changes and tighter engineering filters",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
