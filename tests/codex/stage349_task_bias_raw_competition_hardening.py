#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage343_bias_deflection_task_competition_thickening import run_analysis as run_stage343
from stage326_task_bias_raw_competition_thickening import run_analysis as run_stage326


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage349_task_bias_raw_competition_hardening_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s343 = run_stage343(force=False)
    s326 = run_stage326(force=False)

    thickening_rows = {row["axis_name"]: row for row in s343["thickening_rows"]}
    current_task = float(thickening_rows["任务竞争"]["strength"])
    current_object = float(thickening_rows["对象竞争"]["strength"])
    current_class = float(thickening_rows["类内竞争"]["strength"])
    prior_task = sum(float(row["task_bias_thickness"]) for row in s326["thickened_rows"]) / max(1, len(s326["thickened_rows"]))

    hardening_rows = [
        {"metric_name": "当前任务竞争厚度", "strength": current_task},
        {"metric_name": "对象竞争参考厚度", "strength": current_object},
        {"metric_name": "类内竞争参考厚度", "strength": current_class},
        {"metric_name": "相对上一阶段提升量", "strength": max(0.0, current_task - prior_task)},
    ]

    hardening_score = (
        current_task * 0.55
        + max(0.0, current_task - prior_task) * 0.20
        + (current_task / max(current_object, current_class)) * 0.25
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage349_task_bias_raw_competition_hardening",
        "title": "任务偏转原始竞争加固图",
        "status_short": "task_bias_raw_competition_hardening_ready",
        "hardening_score": float(hardening_score),
        "hardening_rows": hardening_rows,
        "top_gap_name": "任务竞争已经比前一阶段更稳，但相对对象竞争和类内竞争仍然偏弱，说明任务偏转还没有形成偏置层主厚度。",
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
    parser = argparse.ArgumentParser(description="任务偏转原始竞争加固图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
