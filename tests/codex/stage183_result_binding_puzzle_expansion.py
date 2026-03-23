#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage183_result_binding_puzzle_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE183_RESULT_BINDING_PUZZLE_EXPANSION_REPORT.md"

STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE163_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage163_result_binding_failure_atlas_20260323" / "summary.json"
STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE173_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage173_multi_entity_recovery_stress_test_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s158 = load_json(STAGE158_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s163 = load_json(STAGE163_SUMMARY_PATH)
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s173 = load_json(STAGE173_SUMMARY_PATH)

    puzzle_rows = [
        {"piece_name": "原生绑定", "score": float(s158["apple_result_binding_score"])},
        {"piece_name": "修复能力", "score": float(s160["apple_result_repair_score"])},
        {"piece_name": "失效图谱", "score": 1.0 - (float(s163["hard_failure_count"]) / float(s163["case_count"]))},
        {"piece_name": "来源痕迹", "score": float(s172["provenance_trace_score"])},
        {"piece_name": "多实体承压", "score": float(s173["stress_survival_score"])},
    ]
    ranked_rows = sorted(puzzle_rows, key=lambda row: float(row["score"]))
    expansion_score = sum(float(row["score"]) for row in puzzle_rows) / float(len(puzzle_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage183_result_binding_puzzle_expansion",
        "title": "结果绑定拼图扩张",
        "status_short": "result_binding_puzzle_expansion_ready",
        "piece_count": len(puzzle_rows),
        "result_binding_expansion_score": expansion_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "puzzle_rows": puzzle_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage183: 结果绑定拼图扩张",
        "",
        "## 核心结果",
        f"- 拼图片数量: {summary['piece_count']}",
        f"- 结果绑定拼图平均分: {summary['result_binding_expansion_score']:.4f}",
        f"- 最弱拼图片: {summary['weakest_piece_name']}",
        f"- 最强拼图片: {summary['strongest_piece_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="结果绑定拼图扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
