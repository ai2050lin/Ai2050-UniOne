#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage366_frequency_pattern_extractor import run_analysis as run_stage366
from stage367_chain_link_pattern_extractor import run_analysis as run_stage367
from stage368_cross_model_consensus_extractor import run_analysis as run_stage368


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage369_extractor_algorithm_compare_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s366 = run_stage366(force=True)
    s367 = run_stage367(force=True)
    s368 = run_stage368(force=True)

    compare_rows = [
        {
            "algorithm_name": "频率模式提取器",
            "coverage_count": len(s366["category_rows"]),
            "score": float(s366["frequency_score"]),
        },
        {
            "algorithm_name": "结构链模式提取器",
            "coverage_count": len(s367["chain_rows"]),
            "score": float(s367["link_score"]),
        },
        {
            "algorithm_name": "跨模型共识提取器",
            "coverage_count": len(s368["consensus_rows"]),
            "score": float(s368["consensus_score"]),
        },
    ]

    best_row = max(compare_rows, key=lambda row: (row["coverage_count"], row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage369_extractor_algorithm_compare",
        "title": "提取算法对照复核",
        "status_short": "extractor_algorithm_compare_ready",
        "compare_rows": compare_rows,
        "best_algorithm_name": best_row["algorithm_name"],
        "top_gap_name": "当前最佳算法按覆盖数量优先，再按得分排序。",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="提取算法对照复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
