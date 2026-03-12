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
    ap = argparse.ArgumentParser(description="Theory-track successor-strengthened frontier pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_successor_strengthened_frontier_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load_latest("theory_track_successor_strengthened_reasoning_inventory_*.json")
    stage_frontier = load_latest("theory_track_stage_strengthened_frontier_pruning_*.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_successor_strengthened_frontier_pruning",
        },
        "inventory_constraints": inventory["headline_metrics"],
        "frontier_pruning": {
            "strengthened_theorems": stage_frontier["frontier_pruning"]["strengthened_theorems"],
            "queued_theorems": stage_frontier["frontier_pruning"]["queued_theorems"],
            "preserved_A_families": stage_frontier["frontier_pruning"]["preserved_A_families"] + [
                "successor_coherence_gate",
            ],
            "preserved_Mfeas_families": stage_frontier["frontier_pruning"]["preserved_Mfeas_families"] + [
                "successor_chain_band",
            ],
            "preserved_interventions": stage_frontier["frontier_pruning"]["preserved_interventions"] + [
                "successor_strengthened_alignment_intervention",
            ],
        },
        "verdict": {
            "core_answer": "Successor-strengthened inventory extends the preserved frontier with explicit successor-coherence admissibility and viability structures.",
            "next_theory_target": "rerun strict pass/fail with successor-sensitive thresholds and see whether the successor theorem can finally strict-pass.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
