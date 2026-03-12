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
    ap = argparse.ArgumentParser(description="Stage B path-conditioned bridge filtered search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_b_path_conditioned_bridge_filtered_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    pruned = load("stage_b_path_conditioned_bridge_pruned_search_20260312.json")
    bridge_law = load("theory_track_path_conditioned_bridge_lift_law_20260312.json")

    retained_families = pruned["retained_families"]
    filtered_space = {}
    for family, row in retained_families.items():
        filtered_space[family] = {
            "mean_relation_overlap": float(row["mean_relation_overlap"]),
            "conditional_ratio": float(row["conditional_ratio"]),
            "search_constraints": [
                "role kernel must remain family-anchored",
                "bridge lift must satisfy Pi_bridge(c) = 1",
                "free symbolic role expansion is disallowed",
            ],
        }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageB_path_conditioned_bridge_filtered_search",
        },
        "filtered_search_space": filtered_space,
        "law_reference": bridge_law["path_conditioned_bridge_lift_law"]["expanded_form"],
        "candidate_classes": pruned["candidate_classes"],
        "bridge_to_engineering": {
            "core_statement": "The next B-line search should operate only inside family-anchored, path-conditioned bridge space.",
            "family_count": int(len(filtered_space)),
        },
        "verdict": {
            "core_answer": "B-line now has a filtered search space rather than only a pruning rule.",
            "next_engineering_target": "run the next bridge-role benchmark only inside this filtered family-conditioned space",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
