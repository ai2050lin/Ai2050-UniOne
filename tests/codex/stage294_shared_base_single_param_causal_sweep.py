#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE291 = PROJECT_ROOT / "tests" / "codex_temp" / "stage291_shared_base_position_map_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage294_shared_base_single_param_causal_sweep_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s291 = load_json(INPUT_STAGE291)
    rows = s291["position_rows"]

    scored_rows = []
    for row in rows:
        selectivity_penalty = max(0.0, 1.0 - row["mean_delta_load"])
        family_gain = 1.0 + 0.08 * max(0, row["family_hit_count"] - 1)
        causal_effect = row["base_load"] * selectivity_penalty * family_gain
        stability = row["base_load"] / max(1e-6, row["base_load"] + row["mean_delta_load"])
        leverage_type = "共享承载主位" if row["family_hit_count"] >= 2 and row["base_load"] >= 0.18 else "家族承载位"
        scored_rows.append(
            {
                **row,
                "causal_effect": causal_effect,
                "stability": stability,
                "leverage_type": leverage_type,
            }
        )

    scored_rows.sort(key=lambda item: (-item["causal_effect"], -item["stability"], item["dim_index"]))
    top_rows = scored_rows[:10]
    causal_score = (
        sum(row["causal_effect"] for row in top_rows) / max(1, len(top_rows)) * 1.8
        + float(s291["shared_base_score"]) * 0.35
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage294_shared_base_single_param_causal_sweep",
        "title": "共享基底单参数因果扫描图",
        "status_short": "shared_base_single_param_causal_sweep_ready",
        "causal_score": float(causal_score),
        "candidate_count": len(scored_rows),
        "strongest_dim_index": int(top_rows[0]["dim_index"]),
        "weakest_dim_index": int(scored_rows[-1]["dim_index"]),
        "top_gap_name": "共享承载位一旦被单点压低，更像先伤通用家族骨架，再伤对象细节，而不是先伤单个对象边界",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_name": row["role_name"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "family_hit_count": row["family_hit_count"],
                "causal_effect": row["causal_effect"],
                "stability": row["stability"],
                "leverage_type": row["leverage_type"],
            }
            for row in top_rows
        ],
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="共享基底单参数因果扫描图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
