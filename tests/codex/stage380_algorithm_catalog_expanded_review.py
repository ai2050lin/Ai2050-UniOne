#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage373_algorithm_catalog_review import run_analysis as run_stage373
from stage374_mechanism_candidate_merge import run_analysis as run_stage374
from stage375_shared_carrier_subchain_extractor import run_analysis as run_stage375
from stage376_bias_deflection_subchain_extractor import run_analysis as run_stage376
from stage377_amplification_subchain_extractor import run_analysis as run_stage377
from stage378_task_bias_dedicated_extractor import run_analysis as run_stage378
from stage379_cross_model_common_segment_extractor import run_analysis as run_stage379


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage380_algorithm_catalog_expanded_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s373 = run_stage373(force=True)
    s374 = run_stage374(force=True)
    s375 = run_stage375(force=True)
    s376 = run_stage376(force=True)
    s377 = run_stage377(force=True)
    s378 = run_stage378(force=True)
    s379 = run_stage379(force=True)

    algorithm_rows = list(s373["algorithm_rows"])
    algorithm_rows.extend(
        [
            {
                "algorithm_name": "候选结构合并器",
                "principle": "把频率、字段、行束、耦合四类输出压到共享承载、偏置偏转、逐层放大、多空间角色四个候选结构上。",
                "score_name": "merge_score",
                "score_value": float(s374["merge_score"]),
                "coverage_count": len(s374["candidate_rows"]),
                "output_scope": "候选结构清单",
            },
            {
                "algorithm_name": "共享承载子链提取器",
                "principle": "只用共享承载原始行抽取覆盖、分布、核心、位置等局部子链。",
                "score_name": "subchain_score",
                "score_value": float(s375["subchain_score"]),
                "coverage_count": len(s375["subchain_rows"]),
                "output_scope": "共享承载子链",
            },
            {
                "algorithm_name": "偏置偏转子链提取器",
                "principle": "只用偏置偏转原始行抽取对象差分、类内竞争、品牌/跨类等局部子链。",
                "score_name": "subchain_score",
                "score_value": float(s376["subchain_score"]),
                "coverage_count": len(s376["subchain_rows"]),
                "output_scope": "偏置偏转子链",
            },
            {
                "algorithm_name": "逐层放大子链提取器",
                "principle": "只用逐层放大原始行抽取早层、中层、后层的放大候选子链。",
                "score_name": "subchain_score",
                "score_value": float(s377["subchain_score"]),
                "coverage_count": len(s377["anchor_rows"]),
                "output_scope": "逐层放大子链",
            },
            {
                "algorithm_name": "任务偏转专用提取器",
                "principle": "从任务、操作、约束相关原始行中单独抽取任务偏转模式。",
                "score_name": "task_bias_score",
                "score_value": float(s378["task_bias_score"]),
                "coverage_count": len(s378["label_rows"]),
                "output_scope": "任务偏转模式",
            },
            {
                "algorithm_name": "跨模型共同原段提取器",
                "principle": "按指标前缀把跨模型原始行压成共同结构段。",
                "score_name": "segment_score",
                "score_value": float(s379["segment_score"]),
                "coverage_count": len(s379["segment_rows"]),
                "output_scope": "跨模型共同原段",
            },
        ]
    )
    algorithm_rows.sort(key=lambda row: (-row["coverage_count"], -row["score_value"], row["algorithm_name"]))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage380_algorithm_catalog_expanded_review",
        "title": "提取算法扩展目录复核",
        "status_short": "algorithm_catalog_expanded_ready",
        "algorithm_rows": algorithm_rows,
        "best_algorithm_name": algorithm_rows[0]["algorithm_name"] if algorithm_rows else "",
        "top_gap_name": "当前算法目录已扩展到候选结构合并、三类子链、任务偏转专用、跨模型共同原段等新视角。",
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
    parser = argparse.ArgumentParser(description="提取算法扩展目录复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
