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
    ap = argparse.ArgumentParser(description="Stage P3 path-conditioned transport pruned search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_path_conditioned_transport_pruned_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    path_law = load("theory_track_path_conditioned_encoding_law_20260312.json")
    family_transport = load("theory_track_family_level_transport_operator_20260312.json")
    switching = load("theory_track_switching_aware_readout_law_20260312.json")

    family_rows = family_transport["family_transport_operator"]["formal_family"]
    retained: dict[str, dict] = {}
    pruned: dict[str, dict] = {}

    for family, row in family_rows.items():
        budget = float(row["mean_transport_budget"])
        status = str(row["status"])
        disc_dims = row["disc_support_dims"]
        if status == "open" and budget >= 0.13:
            retained[family] = {
                "mean_transport_budget": budget,
                "disc_support_dims": disc_dims,
                "path_condition": "stabilize->read open AND switching-aware readout positive",
                "reason": "passes path-conditioned transport floor",
            }
        else:
            pruned[family] = {
                "mean_transport_budget": budget,
                "disc_support_dims": disc_dims,
                "path_condition": "fails path-conditioned transport floor",
                "reason": "insufficient family-level path budget",
            }

    candidate_classes = {
        "family_conditioned_transport": "kept",
        "restricted_overlap_readout": "kept",
        "switching_aware_readout": "kept",
        "path_conditioned_transport": "kept",
        "global_isotropic_transport": "excluded",
        "direct_object_to_disc_collapse": "excluded",
        "family_agnostic_readout_head": "excluded",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_path_conditioned_transport_pruned_search",
        },
        "acceptance_rule": path_law["path_conditioned_encoding_law"]["path_open_indicator"],
        "supporting_transport_law": switching["switching_aware_readout_law"]["formal_form"],
        "candidate_classes": candidate_classes,
        "retained_families": retained,
        "pruned_families": pruned,
        "bridge_to_engineering": {
            "core_statement": "The next P3 block should search only inside the path-conditioned switching-aware transport family space.",
            "kept_family_count": int(len(retained)),
            "pruned_family_count": int(len(pruned)),
        },
        "verdict": {
            "core_answer": "P3 can now be pruned by path openness: only candidate families that preserve an admissible, switching-aware readout path should remain.",
            "next_engineering_target": "run the next P3 candidate search only inside the retained path-conditioned transport family space",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
