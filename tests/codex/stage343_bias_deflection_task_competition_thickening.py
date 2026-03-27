#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage339_bias_deflection_raw_trajectory_review import run_analysis as run_stage339
from stage322_bias_deflection_task_competition_expansion import run_analysis as run_stage322


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage343_bias_deflection_task_competition_thickening_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s339 = run_stage339(force=False)
    s322 = run_stage322(force=False)

    trajectory_rows = {row["trajectory_name"]: row for row in s339["trajectory_rows"]}
    task_rows = list(s322["task_rows"])

    operation_strength = sum(float(row["operation_strength"]) for row in task_rows) / max(1, len(task_rows))
    constraint_strength = sum(float(row["constraint_strength"]) for row in task_rows) / max(1, len(task_rows))
    task_axis_strength = max(operation_strength, constraint_strength)

    thickening_rows = [
        {
            "axis_name": "对象竞争",
            "strength": float(trajectory_rows["对象偏转轨迹"]["mean_selectivity"]),
        },
        {
            "axis_name": "类内竞争",
            "strength": float(trajectory_rows["类内竞争轨迹"]["mean_selectivity"]),
        },
        {
            "axis_name": "任务竞争",
            "strength": float(task_axis_strength),
        },
    ]

    thickening_score = sum(float(row["strength"]) for row in thickening_rows) / max(1, len(thickening_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage343_bias_deflection_task_competition_thickening",
        "title": "偏置偏转任务竞争厚化图",
        "status_short": "bias_deflection_task_competition_thickening_ready",
        "thickening_score": float(thickening_score),
        "thickening_rows": thickening_rows,
        "top_gap_name": "任务竞争已经开始增厚，但当前仍然没有超过对象竞争和类内竞争，说明任务偏转还不是偏置层的主厚度来源。",
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
    parser = argparse.ArgumentParser(description="偏置偏转任务竞争厚化图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
