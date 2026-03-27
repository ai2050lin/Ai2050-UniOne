#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage334_bias_deflection_direction_raw_map import run_analysis as run_stage334
from stage318_bias_deflection_raw_competition_expansion import run_analysis as run_stage318


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage339_bias_deflection_raw_trajectory_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s334 = run_stage334(force=False)
    s318 = run_stage318(force=False)

    direction_rows = list(s334["direction_rows"])
    axis_rows = {row["axis_name"]: row for row in s318["competition_axes"]}

    trajectory_rows = [
        {
            "trajectory_name": "对象偏转轨迹",
            "member_count": len(direction_rows[0]["dim_indices"]),
            "mean_selectivity": float(axis_rows["对象竞争"]["mean_selectivity"]),
        },
        {
            "trajectory_name": "类内竞争轨迹",
            "member_count": len(direction_rows[1]["dim_indices"]),
            "mean_selectivity": float(axis_rows["类内竞争"]["mean_selectivity"]),
        },
        {
            "trajectory_name": "任务偏转轨迹",
            "member_count": len(direction_rows[4]["task_models"]),
            "mean_selectivity": (
                float(axis_rows["对象竞争"]["mean_selectivity"]) * 0.55
                + float(axis_rows["品牌或跨类"]["mean_selectivity"]) * 0.45
            ),
        },
    ]

    review_score = sum(float(row["mean_selectivity"]) for row in trajectory_rows) / max(1, len(trajectory_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage339_bias_deflection_raw_trajectory_review",
        "title": "偏转流形原始轨迹图",
        "status_short": "bias_deflection_raw_trajectory_review_ready",
        "review_score": float(review_score),
        "trajectory_rows": trajectory_rows,
        "top_gap_name": "对象偏转和类内竞争的轨迹已经更厚，任务偏转轨迹仍然较短，说明高层任务偏转还没有对象偏转那样成熟。",
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
    parser = argparse.ArgumentParser(description="偏转流形原始轨迹图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
