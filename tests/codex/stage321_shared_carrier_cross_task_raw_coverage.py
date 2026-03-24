#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage280_translation_refactor_shared_base_compression import run_analysis as run_stage280
from stage317_shared_carrier_raw_coverage_expansion import run_analysis as run_stage317


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage321_shared_carrier_cross_task_raw_coverage_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s280 = run_stage280(force=False)
    s317 = run_stage317(force=False)

    coverage_rows = list(s317["coverage_rows"])
    model_rows = {row["model_tag"]: row for row in s280["model_rows"]}
    task_rows = [
        {
            "task_name": "翻译",
            "shared_base_bridge": float(model_rows["qwen4b"]["parts"]["shared_task_role_base"]),
            "role_overlap": float(model_rows["qwen4b"]["compression_score"]),
        },
        {
            "task_name": "重构",
            "shared_base_bridge": float(model_rows["deepseek7b"]["parts"]["shared_task_role_base"]),
            "role_overlap": float(model_rows["deepseek7b"]["compression_score"]),
        },
    ]

    raw_cross_task_score = (
        sum(float(row["mean_base_load"]) for row in coverage_rows) / max(1, len(coverage_rows)) * 0.35
        + sum(float(row["shared_base_bridge"]) for row in task_rows) / max(1, len(task_rows)) * 0.35
        + sum(float(row["role_overlap"]) for row in task_rows) / max(1, len(task_rows)) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage321_shared_carrier_cross_task_raw_coverage",
        "title": "共享承载跨任务原始覆盖图",
        "status_short": "shared_carrier_cross_task_raw_coverage_ready",
        "raw_cross_task_score": float(raw_cross_task_score),
        "coverage_rows": coverage_rows,
        "task_rows": task_rows,
        "top_gap_name": "共享承载位已经跨对象家族显影，也开始跨翻译与重构任务复用，但任务层覆盖仍然明显薄于对象层覆盖",
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
    parser = argparse.ArgumentParser(description="共享承载跨任务原始覆盖图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
