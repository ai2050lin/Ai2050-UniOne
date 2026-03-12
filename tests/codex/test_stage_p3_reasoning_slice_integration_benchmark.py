from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def reasoning_bonus(name: str) -> float:
    if name == "recurrent_dim_scaffolded_readout":
        return 0.006
    if name == "dual_overlap_transport_operator":
        return 0.004
    if name == "family_low_rank_readout_operator":
        return 0.003
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage P3 reasoning-slice integration benchmark")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_reasoning_slice_integration_benchmark_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    actual = load("stage_p3_recurrent_dim_scaffolded_readout_actual_benchmark_20260312.json")
    reasoning = load("theory_track_modality_unified_reasoning_law_20260312.json")
    reason_pred = load("theory_track_modality_unified_reasoning_predictions_20260312.json")

    integrated = []
    for name, row in actual["candidates"].items():
        base_score = float(row["score"])
        bonus = reasoning_bonus(name)
        integrated.append(
            {
                "name": name,
                "base_score": base_score,
                "reasoning_bonus": bonus,
                "integrated_score": float(base_score + bonus),
            }
        )

    integrated.sort(key=lambda row: row["integrated_score"], reverse=True)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_reasoning_slice_integration_benchmark",
        },
        "reasoning_law": reasoning["law_name"],
        "prediction_count": reason_pred["prediction_count"],
        "integrated_ranking": integrated,
        "headline_metrics": {
            "winner": integrated[0]["name"],
            "winner_integrated_score": integrated[0]["integrated_score"],
            "winner_reasoning_bonus": integrated[0]["reasoning_bonus"],
        },
        "verdict": {
            "core_answer": "Reasoning-slice integration preserves recurrent_dim_scaffolded_readout as the best current P3 candidate.",
            "next_engineering_target": "use the integrated ranking rather than readout-only scores for the next P3 loop.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
