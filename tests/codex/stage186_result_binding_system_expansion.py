#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage186_result_binding_system_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE186_RESULT_BINDING_SYSTEM_EXPANSION_REPORT.md"

STAGE183_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage183_result_binding_puzzle_expansion_20260323" / "summary.json"
STAGE174_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage174_recovery_closure_equation_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_piece(score: float) -> str:
    if score >= 0.75:
        return "强块"
    if score >= 0.55:
        return "中块"
    return "弱块"


def build_summary() -> dict:
    s183 = load_json(STAGE183_SUMMARY_PATH)
    s174 = load_json(STAGE174_SUMMARY_PATH)

    piece_rows = []
    for row in s183["puzzle_rows"]:
        score = float(row["score"])
        piece_rows.append(
            {
                "piece_name": str(row["piece_name"]),
                "score": score,
                "piece_level": classify_piece(score),
            }
        )
    piece_rows.append(
        {
            "piece_name": "回收闭合",
            "score": float(s174["closure_score"]),
            "piece_level": classify_piece(float(s174["closure_score"])),
        }
    )
    ranked_rows = sorted(piece_rows, key=lambda row: float(row["score"]))
    weak_piece_count = sum(1 for row in piece_rows if str(row["piece_level"]) == "弱块")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage186_result_binding_system_expansion",
        "title": "结果绑定系统扩张",
        "status_short": "result_binding_system_expansion_ready",
        "piece_count": len(piece_rows),
        "weak_piece_count": weak_piece_count,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage186: 结果绑定系统扩张",
        "",
        "## 核心结果",
        f"- 拼图片数量: {summary['piece_count']}",
        f"- 弱拼图片数量: {summary['weak_piece_count']}",
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
    parser = argparse.ArgumentParser(description="结果绑定系统扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
