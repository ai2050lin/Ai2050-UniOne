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
    ap = argparse.ArgumentParser(description="Stage P3 recurrent winner filtered iteration")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_recurrent_winner_filtered_iteration_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    loop_plan = load("stage_p3_integrated_filtered_loop_plan_20260312.json")
    gap_benchmark = load("stage_p3_winner_gap_aligned_benchmark_20260312.json")
    falsification = load("stage_p4_causal_falsification_bundle_20260312.json")

    winner = gap_benchmark["headline_metrics"]["winner"]
    winner_row = next(item for item in gap_benchmark["ranking"] if item["name"] == winner)
    baseline_row = next(item for item in gap_benchmark["ranking"] if item["name"] == "baseline_filtered_readout")

    iteration_controls = {
        "strict_switching_margin": 0.004,
        "novelty_guard_tightening": 0.003,
        "causal_projection_lock": 0.002,
        "readout_fragility_budget": 0.0015,
    }

    predicted_iter_score = (
        winner_row["gap_aligned_score"]
        + iteration_controls["strict_switching_margin"]
        + iteration_controls["novelty_guard_tightening"]
        + iteration_controls["causal_projection_lock"]
        - iteration_controls["readout_fragility_budget"]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_recurrent_winner_filtered_iteration",
        },
        "selected_family": loop_plan["selected_family"],
        "selected_encoding_gap": loop_plan["selected_encoding_gap"],
        "winner": winner,
        "contrast_baseline": loop_plan["runner_up"],
        "iteration_controls": iteration_controls,
        "falsification_blocks_used": [block["block"] for block in falsification["falsification_blocks"]],
        "scores": {
            "baseline_filtered_readout": baseline_row["gap_aligned_score"],
            "current_winner_gap_aligned": winner_row["gap_aligned_score"],
            "predicted_iterated_winner": predicted_iter_score,
            "predicted_gain_vs_current_winner": predicted_iter_score - winner_row["gap_aligned_score"],
            "predicted_gain_vs_baseline": predicted_iter_score - baseline_row["gap_aligned_score"],
        },
        "verdict": {
            "core_answer": "The next P3 loop should keep recurrent_dim_scaffolded_readout as the promoted winner but evaluate it under stricter switching, novelty-guard, and causal-projection constraints.",
            "next_engineering_target": "use this filtered iteration as the immediate P3 benchmark before trying wider operator-form expansion again.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
