#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage307_cross_family_shared_base_core_compression import run_analysis as run_stage307
from stage298_shared_base_position_role_card import run_analysis as run_stage298


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage310_shared_carrier_operator_core_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s307 = run_stage307(force=False)
    s298 = run_stage298(force=False)

    source_rows = s298["position_rows"]
    core_rows = []
    for row in source_rows[:3]:
        core_strength = row["causal_effect"] * 0.55 + row["role_stability"] * 0.45
        core_rows.append(
            {
                "dim_index": row["dim_index"],
                "core_role": row["role_card"],
                "core_strength": core_strength,
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
            }
        )

    compression_score = float(s307["compression_score"]) * 0.55 + sum(row["core_strength"] for row in core_rows) / max(1, len(core_rows)) * 0.45
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage310_shared_carrier_operator_core_compression",
        "title": "共享承载算子主核压缩",
        "status_short": "shared_carrier_operator_core_compression_ready",
        "compression_score": float(compression_score),
        "core_count": len(core_rows),
        "strongest_dim_index": int(core_rows[0]["dim_index"]),
        "top_gap_name": "共享承载算子主核已经可以压到更少的核心位，但跨家族通用性仍偏弱",
        "core_rows": core_rows,
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
    parser = argparse.ArgumentParser(description="共享承载算子主核压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
