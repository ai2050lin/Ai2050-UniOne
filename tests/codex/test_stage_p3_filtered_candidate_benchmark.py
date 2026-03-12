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
    ap = argparse.ArgumentParser(description="Stage P3 filtered candidate benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_filtered_candidate_benchmark_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    filtered = load("stage_p3_path_conditioned_transport_filtered_search_20260312.json")
    readout = load("theory_track_path_conditioned_readout_law_20260312.json")

    family_space = filtered["filtered_search_space"]
    mean_budget = float(readout["phase_profile"]["mean_transport_budget"])

    candidates = {}
    best_family = None
    best_score = -1.0
    for family, row in family_space.items():
        budget = float(row["mean_transport_budget"])
        dims = row["disc_support_dims"]
        stabilize_bonus = 0.03
        novelty_penalty = 0.01
        score = budget + stabilize_bonus - novelty_penalty
        candidates[family] = {
            "disc_support_dims": dims,
            "mean_transport_budget": budget,
            "candidate_score": float(score),
            "status": "benchmark_ready",
        }
        if score > best_score:
            best_score = score
            best_family = family

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_filtered_candidate_benchmark",
        },
        "candidate_benchmark": candidates,
        "headline_metrics": {
            "family_count": int(len(candidates)),
            "mean_phase_budget": mean_budget,
            "best_family": best_family,
            "best_candidate_score": float(best_score),
        },
        "verdict": {
            "core_answer": "P3 now has a concrete benchmark table inside the filtered search space.",
            "next_engineering_target": "start transport/readout iterations from the best family-conditioned candidate instead of broad search",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
