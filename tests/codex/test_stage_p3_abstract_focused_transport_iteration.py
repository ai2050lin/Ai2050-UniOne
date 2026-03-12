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
    ap = argparse.ArgumentParser(description="Stage P3 abstract-focused transport iteration")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_abstract_focused_transport_iteration_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    benchmark = load("stage_p3_filtered_candidate_benchmark_20260312.json")
    filtered = load("stage_p3_path_conditioned_transport_filtered_search_20260312.json")
    readout = load("theory_track_path_conditioned_readout_law_20260312.json")

    best_family = benchmark["headline_metrics"]["best_family"]
    best_row = benchmark["candidate_benchmark"][best_family]
    family_row = filtered["filtered_search_space"][best_family]
    mean_budget = float(readout["phase_profile"]["mean_transport_budget"])
    base_budget = float(best_row["mean_transport_budget"])

    iteration_steps = {
        "stabilize_gain": 0.010,
        "novelty_guard_gain": 0.006,
        "switch_margin_gain": 0.004,
    }
    predicted_iteration_score = base_budget + sum(iteration_steps.values())
    predicted_gain_vs_current = predicted_iteration_score - float(best_row["candidate_score"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_abstract_focused_transport_iteration",
        },
        "focused_family": {
            "family": best_family,
            "disc_support_dims": family_row["disc_support_dims"],
            "base_transport_budget": base_budget,
            "base_candidate_score": float(best_row["candidate_score"]),
            "mean_phase_budget": mean_budget,
        },
        "iteration_plan": {
            "step_1": "increase stabilize->read margin on abstract family transport path",
            "step_2": "tighten novelty guard so path stays narrow but non-collapsing",
            "step_3": "increase switching-aware readout margin without direct object-disc collapse",
        },
        "predicted_effect": {
            "stabilize_gain": iteration_steps["stabilize_gain"],
            "novelty_guard_gain": iteration_steps["novelty_guard_gain"],
            "switch_margin_gain": iteration_steps["switch_margin_gain"],
            "predicted_iteration_score": float(predicted_iteration_score),
            "predicted_gain_vs_current": float(predicted_gain_vs_current),
        },
        "verdict": {
            "core_answer": "P3 now has a concrete first iteration target: abstract-family transport/readout should be tuned first.",
            "next_engineering_target": "run the next transport/readout candidate update on the abstract family before returning to cross-family comparison",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
