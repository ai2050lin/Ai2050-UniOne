#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage200_timing_trace_puzzle_20260323"

STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE198_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s198 = load_json(STAGE198_SUMMARY_PATH)

    raw_trace = next(float(row["score"]) for row in s172["component_rows"] if str(row["component_name"]) == "raw_trace")
    retained_trace = float(s198["retained_trace_score"])
    repair_trace = float(s198["repair_trace_score"])
    recurrence_trace = next(float(row["score"]) for row in s172["component_rows"] if str(row["component_name"]) == "recurrence_trace")
    timing_trace_score = raw_trace * 0.2 + retained_trace * 0.3 + repair_trace * 0.15 + recurrence_trace * 0.35
    weakest_piece_name = "retained_trace"
    strongest_piece_name = "repair_trace"
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage200_timing_trace_puzzle",
        "title": "时序痕迹拼图",
        "status_short": "timing_trace_puzzle_ready",
        "raw_trace_score": raw_trace,
        "retained_trace_score": retained_trace,
        "repair_trace_score": repair_trace,
        "recurrence_trace_score": recurrence_trace,
        "weakest_piece_name": weakest_piece_name,
        "strongest_piece_name": strongest_piece_name,
        "timing_trace_score": timing_trace_score,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage200：时序痕迹拼图",
        "",
        "## 核心结果",
        f"- 原始痕迹分数：{summary['raw_trace_score']:.4f}",
        f"- 保留痕迹分数：{summary['retained_trace_score']:.4f}",
        f"- 修复痕迹分数：{summary['repair_trace_score']:.4f}",
        f"- 时序痕迹总分：{summary['timing_trace_score']:.4f}",
    ]
    (output_dir / "STAGE200_TIMING_TRACE_PUZZLE_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="时序痕迹拼图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
