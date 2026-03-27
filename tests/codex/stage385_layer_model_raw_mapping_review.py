#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage359_refined_raw_inventory import run_analysis as run_stage359
from stage381_raw_row_layer_map import run_analysis as run_stage381
from stage382_raw_row_position_map import run_analysis as run_stage382
from stage383_raw_row_parameter_link_map import run_analysis as run_stage383


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage385_layer_model_raw_mapping_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s359 = run_stage359(force=True)
    s381 = run_stage381(force=True)
    s382 = run_stage382(force=True)
    s383 = run_stage383(force=True)

    total_raw_rows = int(s359["total_raw_row_count"])
    layer_mapped = int(s381["mapped_row_count"])
    position_mapped = int(s382["mapped_row_count"])
    parameter_mapped = int(s383["mapped_row_count"])

    review_rows = [
        {"name": "总原始行", "count": total_raw_rows, "ratio": 1.0},
        {"name": "层号映射行", "count": layer_mapped, "ratio": layer_mapped / total_raw_rows if total_raw_rows else 0.0},
        {"name": "位置映射行", "count": position_mapped, "ratio": position_mapped / total_raw_rows if total_raw_rows else 0.0},
        {"name": "参数位链接行", "count": parameter_mapped, "ratio": parameter_mapped / total_raw_rows if total_raw_rows else 0.0},
        {"name": "token_index 显式映射行", "count": 0, "ratio": 0.0},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage385_layer_model_raw_mapping_review",
        "title": "层级模型原始映射复核",
        "status_short": "layer_model_raw_mapping_review_ready",
        "review_rows": review_rows,
        "review_score": sum(row["ratio"] for row in review_rows[1:4]) / 3.0,
        "top_gap_name": "当前缺口是 token_index 显式映射仍为 0。",
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
    parser = argparse.ArgumentParser(description="层级模型原始映射复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
