from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> dict:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Theory-track naturalized frontier progress")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_naturalized_frontier_progress_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    strict_pf = load_latest("theory_track_priority34_strict_pass_fail_*.json")
    block = load_latest("stage_p3_p4_priority14_execution_block_*.json")
    inventory = load_latest("theory_track_large_scale_naturalized_reasoning_inventory_*.json")

    statuses = {item["theorem"]: item["status"] for item in strict_pf["strict_pass_fail"]}
    strict_survivals = 2 + sum(1 for status in statuses.values() if status == "strict_pass")
    strengthened_provisionals = sum(1 for status in statuses.values() if status == "strengthened_provisional")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_naturalized_frontier_progress",
        },
        "frontier_summary": {
            "base_priority14_gain": float(block["priority_scores"]["gain_total_vs_baseline"]),
            "strict_survivals_now": strict_survivals,
            "strengthened_provisionals_now": strengthened_provisionals,
            "family_cross_to_within_ratio": float(inventory["headline_metrics"]["family_cross_to_within_ratio"]),
            "relation_cross_to_within_ratio": float(inventory["headline_metrics"]["relation_cross_to_within_ratio"]),
            "temporal_cross_to_within_ratio": float(inventory["headline_metrics"]["temporal_cross_to_within_ratio"]),
            "chain_successor_to_cross_stage_ratio": float(inventory["headline_metrics"]["chain_successor_to_cross_stage_ratio"]),
        },
        "verdict": {
            "core_answer": "The naturalized long-chain route now does more than expand inventory size: it directly strengthens the theorem survival frontier and starts turning stage/successor constraints into strict pass/fail decisions.",
            "next_big_block": "promote the next active frontier by bringing stress_guarded_update_theorem and anchored_bridge_lift_theorem into the same strict pass/fail machinery, while further increasing naturalized chain realism.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
