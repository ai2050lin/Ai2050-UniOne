#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage343_bias_deflection_task_competition_thickening import run_analysis as run_stage343
from stage349_task_bias_raw_competition_hardening import run_analysis as run_stage349


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage352_task_bias_high_competition_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s343 = run_stage343(force=False)
    s349 = run_stage349(force=False)

    base_rows = {row["axis_name"]: row for row in s343["thickening_rows"]}
    hard_rows = {row["metric_name"]: row for row in s349["hardening_rows"]}

    current_task = float(base_rows["任务竞争"]["strength"])
    object_ref = float(base_rows["对象竞争"]["strength"])
    class_ref = float(base_rows["类内竞争"]["strength"])
    prior_gain = float(hard_rows["相对上一阶段提升量"]["strength"])

    review_rows = [
        {"metric_name": "任务竞争 / 对象竞争比值", "strength": current_task / object_ref},
        {"metric_name": "任务竞争 / 类内竞争比值", "strength": current_task / class_ref},
        {"metric_name": "上一阶段到当前阶段增益", "strength": prior_gain},
    ]

    review_score = (
        float(review_rows[0]["strength"]) * 0.40
        + float(review_rows[1]["strength"]) * 0.40
        + min(1.0, float(review_rows[2]["strength"]) * 10.0) * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage352_task_bias_high_competition_review",
        "title": "任务偏转高竞争复核",
        "status_short": "task_bias_high_competition_review_ready",
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": "任务偏转已经能进入高竞争层，但相对对象竞争和类内竞争的比值仍偏低，说明任务偏转还不是偏置层主通道。",
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
    parser = argparse.ArgumentParser(description="任务偏转高竞争复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
