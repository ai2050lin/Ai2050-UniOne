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
    ap = argparse.ArgumentParser(description="Stage P3 recurrent-dim scaffolded readout benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_recurrent_dim_scaffolded_readout_benchmark_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    operator_change = load("stage_p3_inventory_guided_operator_form_change_20260312.json")
    benchmark = load("stage_p3_filtered_candidate_benchmark_20260312.json")
    inventory_info = load("theory_track_inventory_information_gain_summary_20260312.json")

    best_family = operator_change["starting_point"]["best_family"]
    base_score = float(operator_change["starting_point"]["best_score"])
    recurrent_dims = operator_change["starting_point"]["recurrent_dims"]
    predicted_gain = float(
        operator_change["operator_form_change_candidates"]["recurrent_dim_scaffolded_readout"]["predicted_gain"]
    )
    improved_score = base_score + predicted_gain

    family_axes = inventory_info["new_information"]["low_rank_family_axes"]["stable_family_axes"][best_family]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_recurrent_dim_scaffolded_readout_benchmark",
        },
        "benchmark_setup": {
            "focused_family": best_family,
            "base_score": base_score,
            "candidate_name": "recurrent_dim_scaffolded_readout",
            "family_axes": family_axes,
            "recurrent_dims": recurrent_dims,
        },
        "predicted_benchmark": {
            "predicted_gain": predicted_gain,
            "predicted_score": float(improved_score),
            "gain_source": [
                "shared recurrent scaffold across families",
                "family-specific low-rank disc axes",
                "avoidance of direct object-disc collapse",
            ],
        },
        "verdict": {
            "core_answer": "The first operator-form change benchmark now has a concrete target and predicted gain.",
            "next_engineering_target": "test whether recurrent-dim scaffolded readout can move beyond the current abstract-family plateau",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
