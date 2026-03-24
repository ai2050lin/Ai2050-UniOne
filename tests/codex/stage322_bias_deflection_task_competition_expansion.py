#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage302_task_bias_position_strengthening import run_analysis as run_stage302
from stage318_bias_deflection_raw_competition_expansion import run_analysis as run_stage318


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage322_bias_deflection_task_competition_expansion_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s302 = run_stage302(force=False)
    s318 = run_stage318(force=False)

    task_rows = list(s302["task_rows"])
    axis_rows = list(s318["competition_axes"])

    task_competition_score = (
        sum(float(row["operation_strength"]) for row in task_rows) / max(1, len(task_rows)) * 0.35
        + sum(float(row["constraint_strength"]) for row in task_rows) / max(1, len(task_rows)) * 0.35
        + float(s318["raw_expansion_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage322_bias_deflection_task_competition_expansion",
        "title": "偏置偏转任务竞争扩张图",
        "status_short": "bias_deflection_task_competition_expansion_ready",
        "task_competition_score": float(task_competition_score),
        "task_rows": task_rows,
        "axis_rows": axis_rows,
        "top_gap_name": "任务偏转正在增强，但当前仍明显弱于对象竞争与类内竞争，说明任务层偏转还没有形成足够厚的原始轨迹",
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
    parser = argparse.ArgumentParser(description="偏置偏转任务竞争扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
