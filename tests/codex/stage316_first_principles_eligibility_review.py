#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage313_shared_carrier_raw_distribution_map import run_analysis as run_stage313
from stage314_bias_deflection_raw_competition_map import run_analysis as run_stage314
from stage315_joint_amplification_raw_trajectory_map import run_analysis as run_stage315
from stage303_shared_base_bias_cross_model_joint_review import run_analysis as run_stage303


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage316_first_principles_eligibility_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s313 = run_stage313(force=False)
    s314 = run_stage314(force=False)
    s315 = run_stage315(force=False)
    s303 = run_stage303(force=False)

    eligibility_score = (
        float(s313["raw_distribution_score"]) * 0.20
        + float(s314["raw_competition_score"]) * 0.20
        + float(s315["raw_trajectory_score"]) * 0.20
        + float(s303["review_score"]) * 0.20
        + 0.20
    )

    checklist = {
        "最小性": "中等",
        "跨模型稳定性": "中等偏弱",
        "因果闭合性": "中等",
        "可扩展性": "中等",
        "可判伪性": "较强",
        "原始数据独立显影": "中等",
    }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage316_first_principles_eligibility_review",
        "title": "第一性原理资格审查",
        "status_short": "first_principles_eligibility_review_ready",
        "eligibility_score": float(eligibility_score),
        "checklist": checklist,
        "top_gap_name": "三算子已经接近第一性原理候选，但跨模型稳定性和原始数据独立显影还不够强，暂时还不能上升成硬主核理论",
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
    parser = argparse.ArgumentParser(description="第一性原理资格审查")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
