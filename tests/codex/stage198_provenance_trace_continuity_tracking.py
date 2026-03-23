#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage198_provenance_trace_continuity_tracking_20260323"

STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE192_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage192_time_unfolded_role_slicing_20260323" / "summary.json"
STAGE194_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage194_bottom_block_intervention_priority_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_score(rows: list[dict], key: str, value: str, score_key: str = "score") -> float:
    for row in rows:
        if str(row[key]) == value:
            return float(row[score_key])
    raise KeyError(value)


def build_summary() -> dict:
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s192 = load_json(STAGE192_SUMMARY_PATH)
    s194 = load_json(STAGE194_SUMMARY_PATH)

    retained_trace = find_score(s172["component_rows"], "component_name", "retained_trace")
    repair_trace = find_score(s172["component_rows"], "component_name", "repair_trace")
    trace_slice = find_score(s192["slice_rows"], "slice_name", "来源痕迹切片")
    continuity_gap = repair_trace - retained_trace
    top_priority = next(row for row in s194["target_rows"] if str(row["target_name"]) == "来源痕迹束")
    continuity_score = retained_trace * 0.5 + trace_slice * 0.3 + (1.0 - continuity_gap) * 0.2
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage198_provenance_trace_continuity_tracking",
        "title": "来源痕迹持续追踪",
        "status_short": "provenance_trace_continuity_tracking_ready",
        "retained_trace_score": retained_trace,
        "repair_trace_score": repair_trace,
        "trace_slice_score": trace_slice,
        "continuity_gap": continuity_gap,
        "priority_level": str(top_priority["priority"]),
        "trace_continuity_score": continuity_score,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage198：来源痕迹持续追踪",
        "",
        "## 核心结果",
        f"- 保留痕迹分数：{summary['retained_trace_score']:.4f}",
        f"- 修复痕迹分数：{summary['repair_trace_score']:.4f}",
        f"- 持续断层：{summary['continuity_gap']:.4f}",
        f"- 持续追踪总分：{summary['trace_continuity_score']:.4f}",
    ]
    (output_dir / "STAGE198_PROVENANCE_TRACE_CONTINUITY_TRACKING_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="来源痕迹持续追踪")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
