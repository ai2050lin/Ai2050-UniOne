#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage354_breakthrough_standard_rereview import run_analysis as run_stage354
from stage355_attribute_position_operation_expansion_review import run_analysis as run_stage355
from stage356_task_bias_momentum_review import run_analysis as run_stage356
from stage357_independent_amplification_net_gain_review import run_analysis as run_stage357


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage358_bottleneck_priority_projection_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s354 = run_stage354(force=False)
    s355 = run_stage355(force=False)
    s356 = run_stage356(force=False)
    s357 = run_stage357(force=False)

    rows = []
    for row in s354["review_rows"]:
        gap = max(0.0, float(row["target_value"]) - float(row["current_value"]))
        rows.append(
            {
                "criterion_name": row["criterion_name"],
                "current_value": float(row["current_value"]),
                "target_value": float(row["target_value"]),
                "gap": gap,
            }
        )

    rows.sort(key=lambda item: item["gap"], reverse=True)

    projection_score = (
        float(s355["expansion_score"]) * 0.30
        + float(s356["momentum_score"]) * 0.35
        + float(s357["review_score"]) * 0.35
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage358_bottleneck_priority_projection",
        "title": "瓶颈优先级投影",
        "status_short": "bottleneck_priority_projection_ready",
        "projection_score": float(projection_score),
        "priority_rows": rows,
        "top_gap_name": "当前最先要打的仍然是多空间厚度，其次是任务偏转厚化和跨模型共同主核；独立放大核已经接近门槛，但还未真正越线。",
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
    parser = argparse.ArgumentParser(description="瓶颈优先级投影")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
