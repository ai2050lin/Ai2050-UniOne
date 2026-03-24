#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage291_shared_base_position_map import run_analysis as run_stage291
from stage292_bias_injection_position_map import run_analysis as run_stage292


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE252 = PROJECT_ROOT / "tests" / "codex_temp" / "stage252_object_pressure_to_delta_thickness_bridge_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage293_base_to_bias_operation_bridge_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s291 = run_stage291(force=False)
    s292 = run_stage292(force=False)
    s252 = load_json(INPUT_STAGE252)

    bridge_score = (
        float(s291["shared_base_score"]) * 0.35
        + float(s292["bias_score"]) * 0.35
        + float(s252["bridge_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage293_base_to_bias_operation_bridge",
        "title": "基底到偏置运算桥",
        "status_short": "base_to_bias_operation_bridge_ready",
        "bridge_score": float(bridge_score),
        "shared_base_score": float(s291["shared_base_score"]),
        "bias_score": float(s292["bias_score"]),
        "pressure_bridge_score": float(s252["bridge_score"]),
        "top_gap_name": "当前最接近的运算结构是：共享基底先承载通用家族结构，再由少量偏置位把状态拨向对象、义项或任务方向，最终对象压力决定偏置需要多厚",
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
    parser = argparse.ArgumentParser(description="基底到偏置运算桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
