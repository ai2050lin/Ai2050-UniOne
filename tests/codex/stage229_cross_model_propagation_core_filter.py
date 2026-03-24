#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage229_cross_model_propagation_core_filter_20260324"

STAGE226_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage226_cross_model_propagation_invariant_expansion_20260324" / "summary.json"
STAGE196_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage196_cross_model_invariant_block_refinement_20260323" / "summary.json"
STAGE184_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage184_cross_model_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s226 = load_json(STAGE226_SUMMARY_PATH)
    s196 = load_json(STAGE196_SUMMARY_PATH)
    s184 = load_json(STAGE184_SUMMARY_PATH)

    stable_names = list(s196["stable_block_names"])
    transition_names = list(s196["transition_block_names"])
    weak_names = list(s196["weak_block_names"])
    filtered_core_names = stable_names + transition_names
    filter_score = (
        (len(filtered_core_names) / 5.0) * 0.40
        + float(s196["refinement_score"]) * 0.30
        + float(s226["expansion_score"]) * 0.30
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage229_cross_model_propagation_core_filter",
        "title": "跨模型传播主核筛选",
        "status_short": "cross_model_propagation_core_filter_ready",
        "filtered_core_count": len(filtered_core_names),
        "filtered_core_names": filtered_core_names,
        "excluded_weak_names": weak_names,
        "filter_score": filter_score,
        "top_gap_name": "硬主核仍偏少",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage229：跨模型传播主核筛选",
        "",
        "## 核心结果",
        f"- 主核数量：{summary['filtered_core_count']}",
        f"- 主核筛选分：{summary['filter_score']:.4f}",
        f"- 保留主核：{', '.join(summary['filtered_core_names'])}",
        f"- 排除弱块：{', '.join(summary['excluded_weak_names'])}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE229_CROSS_MODEL_PROPAGATION_CORE_FILTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型传播主核筛选")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
