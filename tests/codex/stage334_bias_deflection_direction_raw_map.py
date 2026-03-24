#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage314_bias_deflection_raw_competition_map import run_analysis as run_stage314
from stage322_bias_deflection_task_competition_expansion import run_analysis as run_stage322


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage334_bias_deflection_direction_raw_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s314 = run_stage314(force=False)
    s322 = run_stage322(force=False)

    competition_rows = s314["competition_rows"]
    task_rows = s322["task_rows"]

    direction_rows = [
        {
            "direction_name": "对象细粒度偏转",
            "dim_indices": [int(row["dim_index"]) for row in competition_rows if row["role_card"] == "对象细粒度偏转位"],
        },
        {
            "direction_name": "类内竞争偏转",
            "dim_indices": [int(row["dim_index"]) for row in competition_rows if row["role_card"] == "类内竞争偏转位"],
        },
        {
            "direction_name": "对象域切换偏转",
            "dim_indices": [int(row["dim_index"]) for row in competition_rows if row["role_card"] == "对象域切换偏转位"],
        },
        {
            "direction_name": "品牌或跨类偏转",
            "dim_indices": [int(row["dim_index"]) for row in competition_rows if row["role_card"] == "品牌或跨类偏转位"],
        },
        {
            "direction_name": "任务偏转",
            "task_models": [
                {
                    "model_tag": row["model_tag"],
                    "strongest_axis": "操作偏转" if float(row["operation_strength"]) >= float(row["constraint_strength"]) else "约束偏转",
                }
                for row in task_rows
            ],
        },
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage334_bias_deflection_direction_raw_map",
        "title": "偏置偏转方向原始图",
        "status_short": "bias_deflection_direction_raw_map_ready",
        "direction_count": len(direction_rows),
        "direction_rows": direction_rows,
        "top_gap_name": "偏置偏转方向已经能拆成对象、类内竞争、对象域切换、品牌或跨类、任务五类，但任务偏转仍然是最薄的一层",
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
    parser = argparse.ArgumentParser(description="偏置偏转方向原始图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
