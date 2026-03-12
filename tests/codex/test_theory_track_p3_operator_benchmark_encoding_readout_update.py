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
    ap = argparse.ArgumentParser(description="Update encoding readout gap from P3 operator benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_p3_operator_benchmark_encoding_readout_update_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    bench = load("stage_p3_recurrent_dim_scaffolded_readout_actual_benchmark_20260312.json")
    gaps = load("theory_track_encoding_mechanism_core_gaps_20260312.json")

    current_gap = next(g for g in gaps["gaps"] if g["name"] == "object_to_readout_compatibility")
    best = bench["headline_metrics"]["best_candidate"]
    gain = float(bench["headline_metrics"]["best_gain_vs_baseline"])

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_P3_operator_benchmark_encoding_readout_update",
        },
        "target_gap": current_gap["name"],
        "previous_gap_statement": current_gap["why_open"],
        "benchmark_result": {
            "best_candidate": best,
            "best_gain_vs_baseline": gain,
            "best_score": bench["headline_metrics"]["best_score"],
        },
        "interpretation": {
            "core_answer": (
                "The readout gap is no longer just a static incompatibility statement. "
                "It now has an operator-sensitive benchmark signal."
            ),
            "current_status": (
                "If gain is positive, the gap is partially softened by operator-form change; "
                "if not, compatibility remains hard even after scaffolded readout."
            ),
            "operator_recommendation": best,
        },
        "verdict": {
            "core_answer": "P3 operator benchmarking can now directly update the encoding-gap reading for object-to-readout compatibility.",
            "next_theory_target": "integrate this benchmark signal with reasoning-slice and causal falsification tracks.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
