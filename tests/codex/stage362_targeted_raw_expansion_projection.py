#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage360_refined_raw_extractor import run_analysis as run_stage360


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage362_targeted_raw_expansion_projection_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s360 = run_stage360(force=True)
    category_counts = {row["category"]: int(row["row_count"]) for row in s360["category_rows"]}

    target_rows = {
        "共享承载": 60,
        "偏置偏转": 60,
        "逐层放大": 40,
        "多空间角色": 25,
        "跨模型": 20,
    }

    projection_rows = []
    for category, target in target_rows.items():
        current = int(category_counts.get(category, 0))
        projection_rows.append(
            {
                "category": category,
                "current_row_count": current,
                "target_row_count": target,
                "remaining_gap": max(0, target - current),
            }
        )

    projection_rows.sort(key=lambda row: row["remaining_gap"], reverse=True)
    projection_score = sum(
        min(row["current_row_count"] / max(row["target_row_count"], 1), 1.0) for row in projection_rows
    ) / max(1, len(projection_rows))

    remaining = [row for row in projection_rows if row["remaining_gap"] > 0]
    if remaining:
        top_gap_name = "当前扩充优先级依次是：" + "、".join(row["category"] for row in remaining)
    else:
        top_gap_name = "当前五类结构的原始行数量都已达到设定扩充门槛。"

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage362_targeted_raw_expansion_projection",
        "title": "定向原始数据扩充投影",
        "status_short": "targeted_raw_expansion_projection_ready",
        "projection_score": float(projection_score),
        "projection_rows": projection_rows,
        "top_gap_name": top_gap_name,
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
    parser = argparse.ArgumentParser(description="定向原始数据扩充投影")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
