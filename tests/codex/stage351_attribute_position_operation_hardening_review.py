#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage347_attribute_position_operation_raw_thickening import run_analysis as run_stage347


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage351_attribute_position_operation_hardening_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s347 = run_stage347(force=False)
    rows = list(s347["thickening_rows"])

    review_rows = []
    for row in rows:
        current = float(row["current_strength"])
        target = float(row["target_strength"])
        ratio = current / target if target > 0 else 0.0
        review_rows.append(
            {
                "space_name": row["space_name"],
                "current_strength": current,
                "target_strength": target,
                "coverage_ratio": ratio,
                "remaining_gap": max(0.0, target - current),
            }
        )

    review_score = sum(float(row["coverage_ratio"]) for row in review_rows) / max(1, len(review_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage351_attribute_position_operation_hardening_review",
        "title": "属性 / 位置 / 操作空间加固复核",
        "status_short": "attribute_position_operation_hardening_review_ready",
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": "属性、位置、操作三块仍远低于目标覆盖率，这说明多空间结构当前不是局部偏薄，而是系统性欠厚。",
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
    parser = argparse.ArgumentParser(description="属性 / 位置 / 操作空间加固复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
