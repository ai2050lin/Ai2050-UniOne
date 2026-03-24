#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage209_natural_trace_retention_map_20260323"

STAGE198_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323" / "summary.json"
STAGE200_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage200_timing_trace_puzzle_20260323" / "summary.json"
STAGE203_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage203_retained_trace_hardening_20260323" / "summary.json"
STAGE206_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage206_retained_trace_transfer_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_segment(score: float) -> str:
    if score >= 0.7:
        return "稳定段"
    if score >= 0.45:
        return "过渡段"
    return "薄弱段"


def build_summary() -> dict:
    s198 = load_json(STAGE198_SUMMARY_PATH)
    s200 = load_json(STAGE200_SUMMARY_PATH)
    s203 = load_json(STAGE203_SUMMARY_PATH)
    s206 = load_json(STAGE206_SUMMARY_PATH)

    segment_rows = [
        {
            "segment_name": "原始痕迹段",
            "score": float(s200["raw_trace_score"]),
        },
        {
            "segment_name": "天然保留段",
            "score": float(s203["retained_trace_score"]),
        },
        {
            "segment_name": "复现痕迹段",
            "score": next(float(row["score"]) for row in s203["piece_rows"] if str(row["piece_name"]) == "复现痕迹"),
        },
        {
            "segment_name": "传递桥段",
            "score": next(float(row["score"]) for row in s206["transfer_rows"] if str(row["piece_name"]) == "时序痕迹桥"),
        },
        {
            "segment_name": "修复迁移段",
            "score": float(s203["repair_trace_score"]),
        },
    ]
    for row in segment_rows:
        row["status"] = classify_segment(float(row["score"]))
    ranked_rows = sorted(segment_rows, key=lambda row: float(row["score"]))
    retention_map_score = sum(float(row["score"]) for row in segment_rows) / float(len(segment_rows))
    weakest_segment_name = str(ranked_rows[0]["segment_name"])
    strongest_segment_name = str(ranked_rows[-1]["segment_name"])
    continuity_gap = float(s198["continuity_gap"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage209_natural_trace_retention_map",
        "title": "天然痕迹保留图",
        "status_short": "natural_trace_retention_map_ready",
        "segment_count": len(segment_rows),
        "continuity_gap": continuity_gap,
        "retention_map_score": retention_map_score,
        "weakest_segment_name": weakest_segment_name,
        "strongest_segment_name": strongest_segment_name,
        "segment_rows": segment_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage209：天然痕迹保留图",
        "",
        "## 核心结果",
        f"- 段数量：{summary['segment_count']}",
        f"- 持续断层：{summary['continuity_gap']:.4f}",
        f"- 保留图总分：{summary['retention_map_score']:.4f}",
        f"- 最弱段：{summary['weakest_segment_name']}",
        f"- 最强段：{summary['strongest_segment_name']}",
    ]
    (output_dir / "STAGE209_NATURAL_TRACE_RETENTION_MAP_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="天然痕迹保留图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
