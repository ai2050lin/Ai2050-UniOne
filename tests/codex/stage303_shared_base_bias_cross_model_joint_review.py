#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage301_cross_family_shared_base_compression import run_analysis as run_stage301
from stage302_task_bias_position_strengthening import run_analysis as run_stage302


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE268 = PROJECT_ROOT / "tests" / "codex_temp" / "stage268_complete_test_to_parameter_principle_bridge_20260324" / "summary.json"
INPUT_STAGE300 = PROJECT_ROOT / "tests" / "codex_temp" / "stage300_shared_base_bias_joint_causal_map_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage303_shared_base_bias_cross_model_joint_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s301 = run_stage301(force=False)
    s302 = run_stage302(force=False)
    s268 = load_json(INPUT_STAGE268)
    s300 = load_json(INPUT_STAGE300)

    model_rows = []
    for row in s268["model_rows"]:
        review_score = (
            row["bridge_score"] * 0.45
            + float(s301["compression_score"]) * 0.20
            + float(s302["strengthening_score"]) * 0.20
            + float(s300["joint_score"]) * 0.15
        )
        model_rows.append(
            {
                "model_tag": row["model_tag"],
                "display_name": row["display_name"],
                "review_score": review_score,
                "parameter_bridge_score": row["bridge_score"],
                "behavior_weakest_category": row["behavior_weakest_category"],
            }
        )

    strongest_model = max(model_rows, key=lambda row: row["review_score"])["display_name"]
    weakest_model = min(model_rows, key=lambda row: row["review_score"])["display_name"]
    review_score = sum(row["review_score"] for row in model_rows) / max(1, len(model_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage303_shared_base_bias_cross_model_joint_review",
        "title": "共享承载与偏置偏转跨模型联合复核",
        "status_short": "shared_base_bias_cross_model_joint_review_ready",
        "review_score": float(review_score),
        "strongest_model": strongest_model,
        "weakest_model": weakest_model,
        "model_rows": model_rows,
        "top_gap_name": "共享承载与偏置偏转的联合结构已经能跨模型显影，但共同主核仍然偏少，天然来源保真仍会限制闭合层验证",
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
    parser = argparse.ArgumentParser(description="共享承载与偏置偏转跨模型联合复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
