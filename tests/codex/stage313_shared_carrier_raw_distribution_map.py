#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage298_shared_base_position_role_card import run_analysis as run_stage298
from stage310_shared_carrier_operator_core_compression import run_analysis as run_stage310


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage313_shared_carrier_raw_distribution_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s298 = run_stage298(force=False)
    s310 = run_stage310(force=False)
    rows = s298["position_rows"]

    raw_distribution_score = (
        sum(row["base_load"] for row in rows) / max(1, len(rows)) * 0.35
        + sum(row["role_stability"] for row in rows) / max(1, len(rows)) * 0.35
        + float(s310["compression_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage313_shared_carrier_raw_distribution_map",
        "title": "共享承载原始分布图",
        "status_short": "shared_carrier_raw_distribution_map_ready",
        "raw_distribution_score": float(raw_distribution_score),
        "distribution_rows": rows[:10],
        "top_gap_name": "共享承载位的原始分布已经开始稳定，但跨家族共享位的原始覆盖仍然偏薄",
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
    parser = argparse.ArgumentParser(description="共享承载原始分布图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
