#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage187_cross_model_shared_puzzle_strengthening_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE187_CROSS_MODEL_SHARED_PUZZLE_STRENGTHENING_REPORT.md"

STAGE184_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage184_cross_model_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_priority(score: float, spread: float) -> str:
    if score < 0.45:
        return "优先补强"
    if spread > 0.15:
        return "优先收敛"
    return "持续观察"


def build_summary() -> dict:
    s184 = load_json(STAGE184_SUMMARY_PATH)
    shared_map = {str(row["block_name"]): float(row["score"]) for row in s184["shared_rows"]}
    spread_map = {str(row["piece_name"]): float(row["spread"]) for row in s184["spread_rows"]}

    piece_rows = []
    for piece_name, score in shared_map.items():
        spread = float(spread_map.get(piece_name, 0.0))
        piece_rows.append(
            {
                "piece_name": piece_name,
                "shared_score": score,
                "spread": spread,
                "priority": classify_priority(score, spread),
            }
        )
    ranked_rows = sorted(piece_rows, key=lambda row: (str(row["priority"]), float(row["shared_score"])))
    strengthening_target_count = sum(1 for row in piece_rows if str(row["priority"]) != "持续观察")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage187_cross_model_shared_puzzle_strengthening",
        "title": "跨模型共同拼图补强",
        "status_short": "cross_model_shared_puzzle_strengthening_ready",
        "piece_count": len(piece_rows),
        "strengthening_target_count": strengthening_target_count,
        "top_target_name": str(ranked_rows[0]["piece_name"]),
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage187: 跨模型共同拼图补强",
        "",
        "## 核心结果",
        f"- 拼图片数量: {summary['piece_count']}",
        f"- 补强目标数量: {summary['strengthening_target_count']}",
        f"- 头号补强目标: {summary['top_target_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型共同拼图补强")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
