#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage321_shared_carrier_cross_task_raw_coverage import run_analysis as run_stage321


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage325_shared_carrier_cross_task_core_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s321 = run_stage321(force=False)

    task_rows = s321["task_rows"]
    coverage_rows = s321["coverage_rows"]

    strongest_task = max(task_rows, key=lambda row: float(row["shared_base_bridge"]))
    weakest_task = min(task_rows, key=lambda row: float(row["shared_base_bridge"]))

    compressed_rows = [
        {
            "core_name": "对象-任务共享承载核",
            "support_strength": (
                sum(float(row["shared_base_bridge"]) for row in task_rows) / max(1, len(task_rows)) * 0.55
                + sum(float(row["mean_base_load"]) for row in coverage_rows) / max(1, len(coverage_rows)) * 0.45
            ),
            "strongest_task": strongest_task["task_name"],
            "weakest_task": weakest_task["task_name"],
        }
    ]

    compression_score = sum(float(row["support_strength"]) for row in compressed_rows) / len(compressed_rows)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage325_shared_carrier_cross_task_core_compression",
        "title": "共享承载跨任务主核压缩",
        "status_short": "shared_carrier_cross_task_core_compression_ready",
        "compression_score": float(compression_score),
        "compressed_rows": compressed_rows,
        "top_gap_name": "共享承载已经能跨对象和任务形成共同底盘候选，但当前真正稳定的跨任务共享主核仍然只有很少一层",
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
    parser = argparse.ArgumentParser(description="共享承载跨任务主核压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
