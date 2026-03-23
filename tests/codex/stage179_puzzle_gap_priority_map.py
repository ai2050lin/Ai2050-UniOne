#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage178_puzzle_board_overview import run_analysis as run_stage178_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage179_puzzle_gap_priority_map_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE179_PUZZLE_GAP_PRIORITY_MAP_REPORT.md"
STAGE178_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage178_puzzle_board_overview_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_priority(score: float) -> str:
    if score < 0.45:
        return "一级缺口"
    if score < 0.6:
        return "二级缺口"
    if score < 0.75:
        return "三级缺口"
    return "稳定块"


def build_summary() -> dict:
    if not STAGE178_SUMMARY_PATH.exists():
        run_stage178_analysis(force=True)
    s178 = load_json(STAGE178_SUMMARY_PATH)
    ranked_rows = sorted(s178["block_rows"], key=lambda row: float(row["score"]))
    priority_rows = []
    for index, row in enumerate(ranked_rows, start=1):
        priority_rows.append(
            {
                "priority_rank": index,
                "block_name": str(row["block_name"]),
                "score": float(row["score"]),
                "priority_level": classify_priority(float(row["score"])),
                "status": str(row["status"]),
            }
        )
    critical_gap_count = sum(1 for row in priority_rows if str(row["priority_level"]) == "一级缺口")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage179_puzzle_gap_priority_map",
        "title": "拼图缺口优先级图",
        "status_short": "puzzle_gap_priority_map_ready",
        "priority_count": len(priority_rows),
        "critical_gap_count": critical_gap_count,
        "top_gap_block_name": str(priority_rows[0]["block_name"]),
        "top_stable_block_name": str(priority_rows[-1]["block_name"]),
        "priority_rows": priority_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage179: 拼图缺口优先级图",
        "",
        "## 核心结果",
        f"- 优先级项数量: {summary['priority_count']}",
        f"- 一级缺口数量: {summary['critical_gap_count']}",
        f"- 当前头号缺口: {summary['top_gap_block_name']}",
        f"- 当前最稳板块: {summary['top_stable_block_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="拼图缺口优先级图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
