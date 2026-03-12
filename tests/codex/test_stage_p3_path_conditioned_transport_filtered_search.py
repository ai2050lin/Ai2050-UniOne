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
    ap = argparse.ArgumentParser(description="Stage P3 path-conditioned transport filtered search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_path_conditioned_transport_filtered_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    pruned = load("stage_p3_path_conditioned_transport_pruned_search_20260312.json")
    readout = load("theory_track_path_conditioned_readout_law_20260312.json")

    filtered_space = {}
    for family, row in pruned["retained_families"].items():
        filtered_space[family] = {
            "mean_transport_budget": float(row["mean_transport_budget"]),
            "disc_support_dims": row["disc_support_dims"],
            "search_constraints": [
                "stabilize->read path must remain open",
                "novelty->read path must stay narrow rather than collapse",
                "switching-aware readout must remain positive",
                "direct object-to-disc collapse is disallowed",
            ],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_path_conditioned_transport_filtered_search",
        },
        "filtered_search_space": filtered_space,
        "law_reference": readout["path_conditioned_readout_law"]["expanded_form"],
        "phase_profile": readout["phase_profile"],
        "candidate_classes": pruned["candidate_classes"],
        "bridge_to_engineering": {
            "core_statement": "The next P3 search should operate only inside the path-conditioned, switching-aware transport space.",
            "family_count": int(len(filtered_space)),
        },
        "verdict": {
            "core_answer": "P3 now has a filtered search space rather than only a pruning rule.",
            "next_engineering_target": "run the next readout/transport benchmark only inside this filtered family-conditioned space",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
