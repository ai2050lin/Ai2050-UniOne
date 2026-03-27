#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage341_multi_space_role_raw_alignment_expansion import run_analysis as run_stage341
from stage348_shared_carrier_cross_task_core_review import run_analysis as run_stage348
from stage349_task_bias_raw_competition_hardening import run_analysis as run_stage349
from stage353_independent_amplification_core_isolation_review import run_analysis as run_stage353
from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage354_breakthrough_standard_rereview_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def judge(current: float, threshold: float) -> str:
    return "达到" if current >= threshold else "未达到"


def build_summary() -> dict:
    s341 = run_stage341(force=False)
    s348 = run_stage348(force=False)
    s349 = run_stage349(force=False)
    s353 = run_stage353(force=False)
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
            "current_value": float(s348["review_score"]),
            "target_value": 0.50,
            "status": judge(float(s348["review_score"]), 0.50),
        },
        {
            "criterion_name": "任务偏转厚度接近对象竞争厚度",
            "current_value": float(s349["hardening_rows"][0]["strength"]),
            "target_value": 0.68,
            "status": judge(float(s349["hardening_rows"][0]["strength"]), 0.68),
        },
        {
            "criterion_name": "独立放大核从接力整体中剥离",
            "current_value": float(s353["review_rows"][1]["strength"]),
            "target_value": 0.10,
            "status": judge(float(s353["review_rows"][1]["strength"]), 0.10),
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
        "experiment_id": "stage354_breakthrough_standard_rereview",
        "title": "突破点标准再复核",
        "status_short": "breakthrough_standard_rereview_ready",
        "review_score": float(review_score),
        "reached_count": reached_count,
        "criterion_count": len(review_rows),
        "review_rows": review_rows,
        "top_gap_name": "当前最可能先过线的是独立放大核净增益门槛，其次是任务偏转厚化；最拖后腿的仍然是多空间厚度和跨模型主核稳定性。",
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
    parser = argparse.ArgumentParser(description="突破点标准再复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
