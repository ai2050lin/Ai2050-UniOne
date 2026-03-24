#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage316_first_principles_eligibility_review import run_analysis as run_stage316
from stage317_shared_carrier_raw_coverage_expansion import run_analysis as run_stage317
from stage318_bias_deflection_raw_competition_expansion import run_analysis as run_stage318
from stage319_joint_amplification_layerwise_core_split import run_analysis as run_stage319


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage320_first_principles_eligibility_reinforced_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s316 = run_stage316(force=False)
    s317 = run_stage317(force=False)
    s318 = run_stage318(force=False)
    s319 = run_stage319(force=False)

    reinforced_score = (
        float(s316["eligibility_score"]) * 0.40
        + float(s317["raw_coverage_score"]) * 0.20
        + float(s318["raw_expansion_score"]) * 0.20
        + float(s319["layerwise_split_score"]) * 0.20
    )

    checklist = {
        "最小性": "中等偏上" if reinforced_score >= 0.45 else "中等",
        "跨模型稳定性": "中等偏弱",
        "因果闭合性": "中等",
        "可扩展性": "中等",
        "可判伪性": "较强",
        "原始数据独立显影": "中等偏上" if float(s317["raw_coverage_score"]) >= 0.30 else "中等",
    }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage320_first_principles_eligibility_reinforced_review",
        "title": "第一性原理资格增强复核",
        "status_short": "first_principles_eligibility_reinforced_review_ready",
        "reinforced_score": float(reinforced_score),
        "checklist": checklist,
        "top_gap_name": "三算子的原始覆盖、原始竞争和逐层放大都在增强，但跨模型稳定性仍然不够厚，距离硬主核理论还有一段距离",
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
    parser = argparse.ArgumentParser(description="第一性原理资格增强复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
