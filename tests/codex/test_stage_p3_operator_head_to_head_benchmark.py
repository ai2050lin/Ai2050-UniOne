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
    ap = argparse.ArgumentParser(description="Stage P3 operator head-to-head benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_operator_head_to_head_benchmark_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    actual = load("stage_p3_recurrent_dim_scaffolded_readout_actual_benchmark_20260312.json")

    candidates = actual["candidates"]
    ranking = sorted(
        (
            {
                "name": name,
                "score": float(row["score"]),
                "gain_vs_baseline": float(row["gain_vs_baseline"]),
                "reason": row["reason"],
            }
            for name, row in candidates.items()
        ),
        key=lambda row: row["score"],
        reverse=True,
    )

    winner = ranking[0]
    runner_up = ranking[1]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_operator_head_to_head_benchmark",
        },
        "focused_family": actual["benchmark_context"]["focused_family"],
        "ranking": ranking,
        "headline_metrics": {
            "winner": winner["name"],
            "winner_score": winner["score"],
            "runner_up": runner_up["name"],
            "runner_up_score": runner_up["score"],
            "winner_margin": float(winner["score"] - runner_up["score"]),
        },
        "verdict": {
            "core_answer": "P3 now has a direct head-to-head operator ranking rather than only isolated candidate scores.",
            "next_engineering_target": "promote the winner into the next filtered P3 loop while keeping the runner-up as the main contrast baseline.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
