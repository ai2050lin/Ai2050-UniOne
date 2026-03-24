#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324
from stage325_shared_carrier_cross_task_core_compression import run_analysis as run_stage325
from stage326_task_bias_raw_competition_thickening import run_analysis as run_stage326
from stage327_joint_amplification_independent_core_isolation import run_analysis as run_stage327


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage328_cross_model_common_core_compression_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s324 = run_stage324(force=False)
    s325 = run_stage325(force=False)
    s326 = run_stage326(force=False)
    s327 = run_stage327(force=False)

    common_core_score = (
        float(s324["cross_model_score"]) * 0.40
        + float(s325["compression_score"]) * 0.20
        + float(s326["thickening_score"]) * 0.20
        + float(s327["isolation_score"]) * 0.20
    )

    common_rows = [
        {
            "core_name": "共享承载共同核",
            "strength": float(s325["compression_score"]),
        },
        {
            "core_name": "任务偏转共同核",
            "strength": float(s326["thickening_score"]),
        },
        {
            "core_name": "联合放大共同核",
            "strength": float(s327["isolation_score"]),
        },
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage328_cross_model_common_core_compression",
        "title": "跨模型共同主核压缩",
        "status_short": "cross_model_common_core_compression_ready",
        "common_core_score": float(common_core_score),
        "common_rows": common_rows,
        "top_gap_name": "共同主核已经从单模型候选推进到跨模型压缩层，但任务偏转和独立放大核仍然偏薄，离硬主核还有距离",
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
    parser = argparse.ArgumentParser(description="跨模型共同主核压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
