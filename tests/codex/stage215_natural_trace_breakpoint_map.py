#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage215_natural_trace_breakpoint_map_20260324"

STAGE212_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage212_natural_trace_decay_map_20260323" / "summary.json"
STAGE209_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage209_natural_trace_retention_map_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s212 = load_json(STAGE212_SUMMARY_PATH)
    s209 = load_json(STAGE209_SUMMARY_PATH)

    decay_rows = list(s212["decay_rows"])
    sorted_rows = sorted(decay_rows, key=lambda row: float(row["decay_strength"]), reverse=True)

    breakpoint_rows = []
    previous_retention = None
    for row in decay_rows:
        retention_score = float(row["retention_score"])
        if previous_retention is None:
            drop_from_previous = 0.0
        else:
            drop_from_previous = max(0.0, previous_retention - retention_score)
        breakpoint_rows.append(
            {
                "segment_name": str(row["segment_name"]),
                "retention_score": retention_score,
                "drop_from_previous": drop_from_previous,
                "decay_strength": float(row["decay_strength"]),
            }
        )
        previous_retention = retention_score

    strongest_breakpoint_name = str(sorted_rows[0]["segment_name"])
    weakest_breakpoint_name = str(sorted_rows[-1]["segment_name"])
    breakpoint_score = sum(float(row["decay_strength"]) for row in decay_rows) / float(len(decay_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage215_natural_trace_breakpoint_map",
        "title": "天然痕迹断裂点图",
        "status_short": "natural_trace_breakpoint_map_ready",
        "segment_count": len(breakpoint_rows),
        "continuity_gap": float(s209["continuity_gap"]),
        "breakpoint_score": breakpoint_score,
        "strongest_breakpoint_name": strongest_breakpoint_name,
        "weakest_breakpoint_name": weakest_breakpoint_name,
        "breakpoint_rows": breakpoint_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage215：天然痕迹断裂点图",
        "",
        "## 核心结果",
        f"- 段数量：{summary['segment_count']}",
        f"- 连续性缺口：{summary['continuity_gap']:.4f}",
        f"- 断裂图总分：{summary['breakpoint_score']:.4f}",
        f"- 最强断裂点：{summary['strongest_breakpoint_name']}",
        f"- 最弱断裂点：{summary['weakest_breakpoint_name']}",
    ]
    (output_dir / "STAGE215_NATURAL_TRACE_BREAKPOINT_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="天然痕迹断裂点图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
