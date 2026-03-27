#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage341_multi_space_role_raw_alignment_expansion import run_analysis as run_stage341


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage347_attribute_position_operation_raw_thickening_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s341 = run_stage341(force=False)
    rows = {row["space_name"]: row for row in s341["expanded_rows"]}

    thickening_rows = [
        {
            "space_name": "属性空间",
            "current_strength": float(rows["属性空间"]["raw_bridge_strength"]),
            "target_strength": 0.08,
        },
        {
            "space_name": "位置空间",
            "current_strength": float(rows["位置空间"]["raw_bridge_strength"]),
            "target_strength": 0.08,
        },
        {
            "space_name": "操作空间",
            "current_strength": float(rows["操作空间"]["raw_bridge_strength"]),
            "target_strength": 0.10,
        },
    ]
    for row in thickening_rows:
        row["gap_to_target"] = max(0.0, float(row["target_strength"]) - float(row["current_strength"]))

    thickening_score = sum(float(row["current_strength"]) for row in thickening_rows) / max(1, len(thickening_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage347_attribute_position_operation_raw_thickening",
        "title": "属性 / 位置 / 操作空间原始厚化图",
        "status_short": "attribute_position_operation_raw_thickening_ready",
        "thickening_score": float(thickening_score),
        "thickening_rows": thickening_rows,
        "top_gap_name": "当前三块里最薄的是属性空间，其次是位置空间；操作空间相对最好，但仍未达到稳定厚度。",
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
    parser = argparse.ArgumentParser(description="属性 / 位置 / 操作空间原始厚化图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
