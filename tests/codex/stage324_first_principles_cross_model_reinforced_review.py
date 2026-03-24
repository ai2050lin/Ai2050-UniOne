#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage303_shared_base_bias_cross_model_joint_review import run_analysis as run_stage303
from stage320_first_principles_eligibility_reinforced_review import run_analysis as run_stage320
from stage321_shared_carrier_cross_task_raw_coverage import run_analysis as run_stage321
from stage322_bias_deflection_task_competition_expansion import run_analysis as run_stage322
from stage323_joint_amplification_position_core_split import run_analysis as run_stage323


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage324_first_principles_cross_model_reinforced_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s303 = run_stage303(force=False)
    s320 = run_stage320(force=False)
    s321 = run_stage321(force=False)
    s322 = run_stage322(force=False)
    s323 = run_stage323(force=False)

    cross_model_score = (
        float(s303["review_score"]) * 0.35
        + float(s320["reinforced_score"]) * 0.25
        + float(s321["raw_cross_task_score"]) * 0.15
        + float(s322["task_competition_score"]) * 0.15
        + float(s323["position_split_score"]) * 0.10
    )

    checklist = {
        "最小性": "中等偏上",
        "跨模型稳定性": "中等" if cross_model_score >= 0.50 else "中等偏弱",
        "因果闭合性": "中等",
        "可扩展性": "中等偏上" if float(s321["raw_cross_task_score"]) >= 0.35 else "中等",
        "可判伪性": "较强",
        "原始数据独立显影": "中等偏上",
    }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage324_first_principles_cross_model_reinforced_review",
        "title": "第一性原理跨模型增强复核",
        "status_short": "first_principles_cross_model_reinforced_review_ready",
        "cross_model_score": float(cross_model_score),
        "checklist": checklist,
        "top_gap_name": "三算子的原始链条在单模型内部更清楚了，但跨模型共同主核仍然不够厚，因此还不能升级成硬主核第一性原理",
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
    parser = argparse.ArgumentParser(description="第一性原理跨模型增强复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
