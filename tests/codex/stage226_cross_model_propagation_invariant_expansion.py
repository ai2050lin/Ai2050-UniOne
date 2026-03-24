#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage226_cross_model_propagation_invariant_expansion_20260324"

STAGE223_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage223_natural_vs_repair_closure_split_map_20260324" / "summary.json"
STAGE196_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323" / "summary.json"
STAGE184_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage184_cross_model_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s223 = load_json(STAGE223_SUMMARY_PATH)
    s196 = load_json(STAGE196_SUMMARY_PATH)
    s184 = load_json(STAGE184_SUMMARY_PATH)

    rows = [
        {"piece_name": "稳定共同块", "score": float(s196["stable_block_count"]) / 5.0},
        {"piece_name": "过渡共同块", "score": float(s196["transition_block_count"]) / 5.0},
        {"piece_name": "跨模型精炼", "score": float(s196["refinement_score"])},
        {"piece_name": "共同薄弱块地板", "score": min(float(row["shared_floor"]) for row in s184["spread_rows"])},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    expansion_score = (
        float(rows[0]["score"]) * 0.20
        + float(rows[1]["score"]) * 0.15
        + float(rows[2]["score"]) * 0.35
        + float(rows[3]["score"]) * 0.30
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage226_cross_model_propagation_invariant_expansion",
        "title": "跨模型传播不变量扩张",
        "status_short": "cross_model_propagation_invariant_expansion_ready",
        "piece_count": len(rows),
        "expansion_score": expansion_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "跨模型传播不变量偏少",
        "expansion_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage226：跨模型传播不变量扩张",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 扩张总分：{summary['expansion_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE226_CROSS_MODEL_PROPAGATION_INVARIANT_EXPANSION_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型传播不变量扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
