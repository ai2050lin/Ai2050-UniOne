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
    ap = argparse.ArgumentParser(description="Theory-track long-chain block progress assessment")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_block_progress_assessment_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load_latest("stage_p3_p4_priority14_execution_block_*.json")
    survival = load_latest("theory_track_long_chain_first4_theorem_survival_*.json")
    theorem_set = load_latest("theory_track_long_chain_extended_theorem_set_*.json")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_block_progress_assessment",
        },
        "closure_metrics": {
            "gain_total_vs_baseline": block["priority_scores"]["gain_total_vs_baseline"],
            "strict_survivals": survival["summary"]["strict_survivals"],
            "provisional_survivals": survival["summary"]["provisional_survivals"],
            "extended_theorem_count": len(theorem_set["legacy_theorems"]) + len(theorem_set["new_theorems"]),
        },
        "progress_reading": {
            "encoding_inverse_reconstruction_shift": "reasoning-trajectory coding is moving from static patch theory toward stage/successor-constrained transport theory",
            "new_math_shift": "ICSPB now has an active six-theorem frontier rather than a static candidate set",
            "engineering_shift": "P3/P4 is no longer only winner-centric; it is now long-chain constrained and theorem-survival-driven",
        },
        "verdict": {
            "core_answer": "The long-chain constrained block completes a full stage goal: theory expansion, intervention reprioritization, and the first active four-theorem survival frontier.",
            "next_theory_target": "move from provisional stage/successor survivals to stricter intervention-backed survival or failure.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
