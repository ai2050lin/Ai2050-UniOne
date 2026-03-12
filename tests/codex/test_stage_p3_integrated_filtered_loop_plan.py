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
    ap = argparse.ArgumentParser(description="Stage P3 integrated filtered loop plan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_integrated_filtered_loop_plan_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    head = load("stage_p3_operator_head_to_head_benchmark_20260312.json")
    reasoning = load("stage_p3_reasoning_slice_integration_benchmark_20260312.json")
    readout_gap = load("theory_track_p3_operator_benchmark_encoding_readout_update_20260312.json")
    filtered = load("stage_p3_path_conditioned_transport_filtered_search_20260312.json")

    winner = head["headline_metrics"]["winner"]
    runner_up = head["headline_metrics"]["runner_up"]
    integrated_winner = reasoning["headline_metrics"]["winner"]
    assert winner == integrated_winner

    chosen_family = head["focused_family"]
    family_constraints = filtered["filtered_search_space"][chosen_family]["search_constraints"]

    plan_steps = [
        {
            "step": "winner_promotion",
            "candidate": winner,
            "why": "best direct head-to-head score and still best after reasoning-slice integration",
        },
        {
            "step": "contrast_baseline",
            "candidate": runner_up,
            "why": "strongest non-winning operator and the most relevant contrast baseline",
        },
        {
            "step": "filtered_constraints",
            "candidate": chosen_family,
            "why": "all next loop trials must stay inside the existing path-conditioned filtered family space",
            "constraints": family_constraints,
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_integrated_filtered_loop_plan",
        },
        "selected_family": chosen_family,
        "selected_encoding_gap": readout_gap["target_gap"],
        "winner": winner,
        "runner_up": runner_up,
        "winner_margin": head["headline_metrics"]["winner_margin"],
        "winner_integrated_score": reasoning["headline_metrics"]["winner_integrated_score"],
        "plan_steps": plan_steps,
        "verdict": {
            "core_answer": "P3 now has an integrated next-loop plan driven by direct benchmark rank, reasoning-slice compatibility, and the encoding-gap reading.",
            "next_engineering_target": "run the next filtered loop with recurrent_dim_scaffolded_readout as the promoted winner and dual_overlap_transport_operator as the contrast baseline.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
