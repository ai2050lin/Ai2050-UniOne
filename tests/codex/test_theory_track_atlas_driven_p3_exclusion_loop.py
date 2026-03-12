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
    ap = argparse.ArgumentParser(description="Theory-track atlas-driven P3 exclusion loop")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_atlas_driven_p3_exclusion_loop_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    overlap = load("theory_track_restricted_overlap_maps_20260312.json")
    operators = load("theory_track_family_conditioned_projection_operators_20260312.json")
    exclusion = load("theory_track_atlas_to_A_Mfeas_exclusion_20260312.json")

    candidate_classes = {
        "global_isotropic_transport": "excluded",
        "direct_object_to_disc_collapse": "excluded",
        "family_conditioned_transport": "kept",
        "restricted_overlap_readout": "kept",
        "family_agnostic_readout_head": "excluded",
    }

    kept_families = {}
    for family, overlap_map in overlap["restricted_overlap_maps"].items():
        kept_families[family] = {
            "P_obj_dims": operators["core_operators"][family]["P_obj_family"]["support_dims"],
            "P_disc_dims": operators["core_operators"][family]["P_disc_family"]["support_dims"],
            "object_disc_overlap": overlap_map["object_disc_overlap"],
            "object_memory_overlap": overlap_map["object_memory_overlap"],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_atlas_driven_P3_exclusion_loop",
        },
        "candidate_classes": candidate_classes,
        "kept_family_conditioned_transport_state": kept_families,
        "loop_principle": {
            "step_1": "Extract family atlas patch statistics.",
            "step_2": "Construct family-conditioned projection operators.",
            "step_3": "Construct restricted overlap maps.",
            "step_4": "Exclude P3 transport/readout candidates incompatible with those operators and overlaps.",
            "step_5": "Pass only the surviving candidate classes back to engineering P3 experiments.",
        },
        "bridge_to_engineering": {
            "core_statement": "P3 should stop testing family-agnostic transport geometries and instead test only family-conditioned transport/readout candidates.",
            "project_statement": "This converts theory-track outputs into a concrete pruning loop for engineering-track P3.",
            "excluded_by_theory_count": int(sum(1 for value in candidate_classes.values() if value == "excluded")),
        },
        "prior_support": {
            "global_exclusions": exclusion["excluded_candidates"],
        },
        "verdict": {
            "core_answer": "The theory track can now drive a concrete exclusion loop for P3 rather than only producing broad hypotheses.",
            "next_theory_target": "connect surviving family-conditioned transport candidates to the next engineering P3 search block",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
