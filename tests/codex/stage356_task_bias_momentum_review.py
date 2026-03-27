#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage349_task_bias_raw_competition_hardening import run_analysis as run_stage349
from stage352_task_bias_high_competition_review import run_analysis as run_stage352


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage356_task_bias_momentum_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s349 = run_stage349(force=False)
    s352 = run_stage352(force=False)

    hardening_rows = {row["metric_name"]: row for row in s349["hardening_rows"]}
    review_rows = {row["metric_name"]: row for row in s352["review_rows"]}

    current_task = float(hardening_rows["当前任务竞争厚度"]["strength"])
    gain = float(hardening_rows["相对上一阶段提升量"]["strength"])
    obj_ratio = float(review_rows["任务竞争 / 对象竞争比值"]["strength"])
    class_ratio = float(review_rows["任务竞争 / 类内竞争比值"]["strength"])
    average_ratio = (obj_ratio + class_ratio) / 2.0
    momentum_score = current_task * 0.55 + average_ratio * 0.35 + min(gain / 0.05, 1.0) * 0.10

    momentum_rows = [
        {"metric_name": "当前任务竞争厚度", "strength": current_task},
        {"metric_name": "任务竞争相对厚度均值", "strength": average_ratio},
        {"metric_name": "阶段增厚动量", "strength": gain},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage356_task_bias_momentum_review",
        "title": "任务偏转增厚动量复核",
        "status_short": "task_bias_momentum_review_ready",
        "momentum_score": float(momentum_score),
        "momentum_rows": momentum_rows,
        "top_gap_name": "任务偏转已经足够进入高竞争层，但阶段增厚动量仍偏弱，说明后续要靠更高强度样本继续硬拉厚度。",
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
    parser = argparse.ArgumentParser(description="任务偏转增厚动量复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
