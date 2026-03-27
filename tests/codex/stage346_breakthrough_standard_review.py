#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage341_multi_space_role_raw_alignment_expansion import run_analysis as run_stage341
from stage342_shared_carrier_cluster_stability_expansion import run_analysis as run_stage342
from stage343_bias_deflection_task_competition_thickening import run_analysis as run_stage343
from stage344_layerwise_amplification_independent_core_review import run_analysis as run_stage344
from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage346_breakthrough_standard_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def judge(current: float, threshold: float) -> str:
    return "达到" if current >= threshold else "未达到"


def build_summary() -> dict:
    s341 = run_stage341(force=False)
    s342 = run_stage342(force=False)
    s343 = run_stage343(force=False)
    s344 = run_stage344(force=False)
    s324 = run_stage324(force=False)

    review_rows = [
        {
            "criterion_name": "多空间原始厚度",
            "current_value": float(s341["expansion_score"]),
            "target_value": 0.30,
            "status": judge(float(s341["expansion_score"]), 0.30),
        },
        {
            "criterion_name": "共享承载跨对象 / 跨任务 / 跨模型稳定性",
            "current_value": float(s342["expansion_score"]),
            "target_value": 0.55,
            "status": judge(float(s342["expansion_score"]), 0.55),
        },
        {
            "criterion_name": "任务偏转厚度接近对象竞争厚度",
            "current_value": float(s343["thickening_rows"][2]["strength"]),
            "target_value": 0.68,
            "status": judge(float(s343["thickening_rows"][2]["strength"]), 0.68),
        },
        {
            "criterion_name": "独立放大核从接力整体中剥离",
            "current_value": float(s344["review_rows"][1]["strength"]),
            "target_value": 0.24,
            "status": judge(float(s344["review_rows"][1]["strength"]), 0.24),
        },
        {
            "criterion_name": "跨模型共同主核稳定性",
            "current_value": float(s324["cross_model_score"]),
            "target_value": 0.60,
            "status": judge(float(s324["cross_model_score"]), 0.60),
        },
    ]

    reached_count = sum(1 for row in review_rows if row["status"] == "达到")
    review_score = reached_count / max(1, len(review_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage346_breakthrough_standard_review",
        "title": "突破点标准复核",
        "status_short": "breakthrough_standard_review_ready",
        "review_score": float(review_score),
        "reached_count": reached_count,
        "criterion_count": len(review_rows),
        "review_rows": review_rows,
        "top_gap_name": "当前最接近突破点的是任务偏转厚化，其次是多空间原始厚度；真正拖住突破的是共享承载跨任务主核和独立放大核剥离。",
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
    parser = argparse.ArgumentParser(description="突破点标准复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
