#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage366_frequency_pattern_extractor import run_analysis as run_stage366
from stage367_chain_link_pattern_extractor import run_analysis as run_stage367
from stage368_cross_model_consensus_extractor import run_analysis as run_stage368
from stage369_extractor_algorithm_compare import run_analysis as run_stage369
from stage370_numeric_field_pattern_extractor import run_analysis as run_stage370
from stage371_bundle_structure_extractor import run_analysis as run_stage371
from stage372_label_field_coupling_extractor import run_analysis as run_stage372


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage373_algorithm_catalog_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s366 = run_stage366(force=True)
    s367 = run_stage367(force=True)
    s368 = run_stage368(force=True)
    s369 = run_stage369(force=True)
    s370 = run_stage370(force=True)
    s371 = run_stage371(force=True)
    s372 = run_stage372(force=True)

    algorithm_rows = [
        {
            "algorithm_name": "频率模式提取器",
            "principle": "按类别统计高频标签和高频原始行束，观察重复出现的结构词与行束类型。",
            "score_name": "frequency_score",
            "score_value": float(s366["frequency_score"]),
            "coverage_count": len(s366["category_rows"]),
            "output_scope": "类别高频标签、高频行束",
        },
        {
            "algorithm_name": "结构链模式提取器",
            "principle": "只跟踪共享承载、偏置偏转、逐层放大三段原始链的行数量与相邻比值。",
            "score_name": "link_score",
            "score_value": float(s367["link_score"]),
            "coverage_count": len(s367["chain_rows"]),
            "output_scope": "三段结构链及相邻比值",
        },
        {
            "algorithm_name": "跨模型共识提取器",
            "principle": "按指标前缀汇总模型覆盖数，提取跨模型重复出现的指标集合。",
            "score_name": "consensus_score",
            "score_value": float(s368["consensus_score"]),
            "coverage_count": len(s368["consensus_rows"]),
            "output_scope": "跨模型共识指标集合",
        },
        {
            "algorithm_name": "数值字段模式提取器",
            "principle": "按类别统计最常出现的数值字段，提取参数结构中重复出现的量纲模式。",
            "score_name": "field_pattern_score",
            "score_value": float(s370["field_pattern_score"]),
            "coverage_count": len(s370["category_rows"]),
            "output_scope": "类别高频数值字段",
        },
        {
            "algorithm_name": "原始行束结构提取器",
            "principle": "按类别统计最常出现的行束类型，提取原始数据组织结构模式。",
            "score_name": "bundle_structure_score",
            "score_value": float(s371["bundle_structure_score"]),
            "coverage_count": len(s371["category_rows"]),
            "output_scope": "类别高频行束类型",
        },
        {
            "algorithm_name": "标签字段耦合提取器",
            "principle": "按标签与字段签名的组合统计重复模式，提取稳定耦合结构。",
            "score_name": "coupling_score",
            "score_value": float(s372["coupling_score"]),
            "coverage_count": len(s372["category_rows"]),
            "output_scope": "标签-字段耦合模式",
        },
    ]

    algorithm_rows.sort(key=lambda row: (-row["coverage_count"], -row["score_value"], row["algorithm_name"]))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage373_algorithm_catalog_review",
        "title": "提取算法目录复核",
        "status_short": "algorithm_catalog_review_ready",
        "algorithm_rows": algorithm_rows,
        "best_algorithm_name": s369["best_algorithm_name"],
        "top_gap_name": "当前算法目录已覆盖频率、结构链、跨模型、数值字段、行束结构、标签字段耦合六类提取视角。",
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
    parser = argparse.ArgumentParser(description="提取算法目录复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
