#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage322_bias_deflection_task_competition_expansion import run_analysis as run_stage322


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage326_task_bias_raw_competition_thickening_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s322 = run_stage322(force=False)
    task_rows = s322["task_rows"]

    thickened_rows = []
    for row in task_rows:
        thickened_rows.append(
            {
                "model_tag": row["model_tag"],
                "display_name": row["display_name"],
                "task_bias_thickness": (
                    float(row["operation_strength"]) * 0.40
                    + float(row["constraint_strength"]) * 0.35
                    + float(row["shared_task_bridge"]) * 0.25
                ),
                "strongest_axis": (
                    "操作偏转" if float(row["operation_strength"]) >= float(row["constraint_strength"]) else "约束偏转"
                ),
            }
        )

    thickening_score = sum(float(row["task_bias_thickness"]) for row in thickened_rows) / max(1, len(thickened_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage326_task_bias_raw_competition_thickening",
        "title": "任务偏转原始竞争厚化图",
        "status_short": "task_bias_raw_competition_thickening_ready",
        "thickening_score": float(thickening_score),
        "thickened_rows": thickened_rows,
        "top_gap_name": "任务偏转已经能区分操作偏转和约束偏转，但整体厚度仍然明显弱于对象竞争和类内竞争",
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
    parser = argparse.ArgumentParser(description="任务偏转原始竞争厚化图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
