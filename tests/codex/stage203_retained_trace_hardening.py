#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage203_retained_trace_hardening_20260323"

STAGE198_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323" / "summary.json"
STAGE200_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage200_timing_trace_puzzle_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s198 = load_json(STAGE198_SUMMARY_PATH)
    s200 = load_json(STAGE200_SUMMARY_PATH)

    retained_trace = float(s200["retained_trace_score"])
    repair_trace = float(s200["repair_trace_score"])
    raw_trace = float(s200["raw_trace_score"])
    recurrence_trace = float(s200["recurrence_trace_score"])
    continuity_gap = float(s198["continuity_gap"])

    hardening_gain_space = repair_trace - retained_trace
    hardening_score = retained_trace * 0.30 + raw_trace * 0.15 + recurrence_trace * 0.25 + hardening_gain_space * 0.30

    piece_rows = [
        {
            "piece_name": "天然保留",
            "score": retained_trace,
            "status": "薄弱硬化点",
        },
        {
            "piece_name": "原始痕迹",
            "score": raw_trace,
            "status": "薄弱硬化点",
        },
        {
            "piece_name": "复现痕迹",
            "score": recurrence_trace,
            "status": "过渡硬化点",
        },
        {
            "piece_name": "修复迁移",
            "score": repair_trace,
            "status": "稳定支撑点",
        },
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage203_retained_trace_hardening",
        "title": "天然痕迹保留强化块",
        "status_short": "retained_trace_hardening_ready",
        "retained_trace_score": retained_trace,
        "repair_trace_score": repair_trace,
        "hardening_gain_space": hardening_gain_space,
        "continuity_gap": continuity_gap,
        "hardening_score": hardening_score,
        "weakest_piece_name": "天然保留",
        "strongest_piece_name": "修复迁移",
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage203：天然痕迹保留强化块",
        "",
        "## 核心结果",
        f"- 天然保留分数：{summary['retained_trace_score']:.4f}",
        f"- 修复迁移分数：{summary['repair_trace_score']:.4f}",
        f"- 可释放增益空间：{summary['hardening_gain_space']:.4f}",
        f"- 强化总分：{summary['hardening_score']:.4f}",
        f"- 最弱点：{summary['weakest_piece_name']}",
    ]
    (output_dir / "STAGE203_RETAINED_TRACE_HARDENING_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="天然痕迹保留强化块")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
