#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage223_natural_vs_repair_closure_split_map_20260324"

STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE217_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage217_source_fidelity_closure_block_20260324" / "summary.json"
STAGE196_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s158 = load_json(STAGE158_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s217 = load_json(STAGE217_SUMMARY_PATH)
    s196 = load_json(STAGE196_SUMMARY_PATH)

    natural_binding = float(s158["apple_result_binding_score"])
    repair_closure = float(s160["apple_result_repair_score"])
    source_fidelity = float(s217["block_score"])
    cross_model_refinement = float(s196["refinement_score"])

    rows = [
        {"piece_name": "原生闭合", "score": natural_binding},
        {"piece_name": "修复闭合", "score": repair_closure},
        {"piece_name": "来源保真闭合", "score": source_fidelity},
        {"piece_name": "跨模型传播不变量", "score": cross_model_refinement},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    split_score = (
        natural_binding * 0.30
        + repair_closure * 0.35
        + source_fidelity * 0.20
        + cross_model_refinement * 0.15
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage223_natural_vs_repair_closure_split_map",
        "title": "天然闭合与修复闭合分离图",
        "status_short": "natural_vs_repair_closure_split_ready",
        "piece_count": len(rows),
        "split_score": split_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "跨模型传播不变量偏少",
        "split_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage223：天然闭合与修复闭合分离图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 分离图总分：{summary['split_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE223_NATURAL_VS_REPAIR_CLOSURE_SPLIT_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="天然闭合与修复闭合分离图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
