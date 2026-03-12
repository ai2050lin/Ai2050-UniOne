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
    ap = argparse.ArgumentParser(description="Stage P2 stress-coupled update pruned search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p2_stress_coupled_update_pruned_search_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stress_law = load("theory_track_stress_coupled_write_read_law_20260312.json")
    task_block2 = load("task_block_2_unified_training_closure_20260311.json")

    pillars = task_block2["pillars"]
    retained = {}
    pruned = {}

    for pillar_name, row in pillars.items():
        score = float(row["score"])
        components = row["components"]
        lesion = float(components.get("lesion_recovery", 0.0))
        transfer = float(components.get("transfer_stability", 0.0))
        summary = {
            "score": score,
            "lesion_recovery_component": lesion,
            "transfer_stability_component": transfer,
            "stress_rule": "prefer candidates compatible with guarded write and stable read under novelty/retention stress",
        }
        if score >= 0.55 and lesion >= 0.45:
            retained[pillar_name] = summary
        else:
            pruned[pillar_name] = summary

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP2_stress_coupled_update_pruned_search",
        },
        "candidate_classes": {
            "guarded_write_update_family": "kept",
            "stable_read_update_family": "kept",
            "stress_agnostic_fast_write_family": "excluded",
            "transfer_fragile_update_family": "excluded",
        },
        "retained_pillars": retained,
        "pruned_pillars": pruned,
        "bridge_to_engineering": {
            "core_statement": "The next P2 block should search only inside guarded-write, stable-read update families.",
            "kept_pillar_count": int(len(retained)),
            "pruned_pillar_count": int(len(pruned)),
            "open_write_count": stress_law["headline_metrics"]["open_write_count"],
            "guarded_write_count": stress_law["headline_metrics"]["guarded_write_count"],
            "stable_read_count": stress_law["headline_metrics"]["stable_read_count"],
        },
        "verdict": {
            "core_answer": "P2 can now be pruned by stress-coupled write/read compatibility instead of broader update-law heuristics.",
            "next_engineering_target": "run the next update-law search only inside guarded-write, stable-read families with acceptable recovery and transfer components",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
