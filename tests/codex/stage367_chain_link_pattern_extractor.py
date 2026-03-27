#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage359_refined_raw_inventory import run_analysis as run_stage359


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage367_chain_link_pattern_extractor_20260325"

CHAIN_ORDER = ["共享承载", "偏置偏转", "逐层放大"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s359 = run_stage359(force=True)
    category_counts = {row["category"]: int(row["raw_row_count"]) for row in s359["category_rows"]}

    chain_rows = []
    previous_count = None
    for category in CHAIN_ORDER:
        current_count = category_counts.get(category, 0)
        chain_rows.append(
            {
                "category": category,
                "raw_row_count": current_count,
                "relative_to_previous": None if previous_count is None else float(current_count / max(previous_count, 1)),
            }
        )
        previous_count = current_count

    link_complete = all(row["raw_row_count"] > 0 for row in chain_rows)
    link_score = sum(row["raw_row_count"] for row in chain_rows) / max(1, len(chain_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage367_chain_link_pattern_extractor",
        "title": "结构链模式提取器",
        "status_short": "chain_link_pattern_extractor_ready",
        "link_complete": link_complete,
        "link_score": float(link_score),
        "chain_rows": chain_rows,
        "top_gap_name": "当前结构链提取器只跟踪共享承载、偏置偏转、逐层放大三段原始链。",
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
    parser = argparse.ArgumentParser(description="结构链模式提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
