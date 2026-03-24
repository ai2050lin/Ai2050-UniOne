#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage302_task_bias_position_strengthening import run_analysis as run_stage302
from stage299_bias_position_role_card import run_analysis as run_stage299


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage308_task_bias_core_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s302 = run_stage302(force=False)
    s299 = run_stage299(force=False)

    task_core_rows = []
    task_like_dims = [row for row in s299["position_rows"] if row["role_card"] in {"品牌或跨类偏转位", "对象域切换偏转位"}]
    for row in task_like_dims[:4]:
        task_core_rows.append(
            {
                "dim_index": row["dim_index"],
                "core_role": "任务偏转主核" if len(task_core_rows) == 0 else "任务偏转候选核",
                "role_card": row["role_card"],
                "leverage": row["leverage"],
            }
        )

    compression_score = (
        float(s302["strengthening_score"]) * 0.55
        + min(1.0, len(task_core_rows) / 4.0) * 0.20
        + float(s299["role_score"]) * 0.25
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage308_task_bias_core_compression",
        "title": "任务偏转主核压缩",
        "status_short": "task_bias_core_compression_ready",
        "compression_score": float(compression_score),
        "core_count": len(task_core_rows),
        "strongest_model": s302["strongest_model"],
        "top_gap_name": "任务偏转主核已经开始显影，但仍混在跨类和对象域切换偏转位里，尚未形成干净任务核",
        "core_rows": task_core_rows,
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
    parser = argparse.ArgumentParser(description="任务偏转主核压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
