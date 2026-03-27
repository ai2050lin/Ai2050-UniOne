#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage359_refined_raw_inventory import run_analysis as run_stage359
from stage360_refined_raw_extractor import run_analysis as run_stage360


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage361_parameter_mechanism_extractability_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s359 = run_stage359(force=True)
    s360 = run_stage360(force=True)

    category_counts = {row["category"]: int(row["row_count"]) for row in s360["category_rows"]}
    total_raw_rows = int(s359["total_raw_row_count"])
    shared_rows = int(category_counts.get("共享承载", 0))
    bias_rows = int(category_counts.get("偏置偏转", 0))
    amplification_rows = int(category_counts.get("逐层放大", 0))
    multispace_rows = int(category_counts.get("多空间角色", 0))
    cross_model_rows = int(category_counts.get("跨模型", 0))

    review_rows = [
        {
            "criterion_name": "总原始行数量是否足够",
            "current_value": total_raw_rows,
            "target_value": 180,
            "status": "达到" if total_raw_rows >= 180 else "未达到",
        },
        {
            "criterion_name": "共享承载原始行是否足够",
            "current_value": shared_rows,
            "target_value": 60,
            "status": "达到" if shared_rows >= 60 else "未达到",
        },
        {
            "criterion_name": "偏置偏转原始行是否足够",
            "current_value": bias_rows,
            "target_value": 60,
            "status": "达到" if bias_rows >= 60 else "未达到",
        },
        {
            "criterion_name": "多空间角色原始行是否足够",
            "current_value": multispace_rows,
            "target_value": 25,
            "status": "达到" if multispace_rows >= 25 else "未达到",
        },
        {
            "criterion_name": "逐层放大原始行是否足够",
            "current_value": amplification_rows,
            "target_value": 30,
            "status": "达到" if amplification_rows >= 30 else "未达到",
        },
        {
            "criterion_name": "跨模型原始行是否足够",
            "current_value": cross_model_rows,
            "target_value": 20,
            "status": "达到" if cross_model_rows >= 20 else "未达到",
        },
    ]

    reached_count = sum(1 for row in review_rows if row["status"] == "达到")
    extractable = reached_count == len(review_rows)
    review_score = reached_count / max(1, len(review_rows))

    unmet = [row["criterion_name"] for row in review_rows if row["status"] != "达到"]
    if unmet:
        top_gap_name = "当前数据还不足以直接提取参数级编码机制，未达标项是：" + "、".join(unmet)
    else:
        top_gap_name = "当前数据已经满足直接提取参数级编码机制的原始数据门槛。"

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage361_parameter_mechanism_extractability_review",
        "title": "参数级编码机制可提取性复核",
        "status_short": "parameter_mechanism_extractability_review_ready",
        "extractable": extractable,
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": top_gap_name,
        "category_snapshot": {
            "共享承载": shared_rows,
            "偏置偏转": bias_rows,
            "逐层放大": amplification_rows,
            "多空间角色": multispace_rows,
            "跨模型": cross_model_rows,
        },
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
    parser = argparse.ArgumentParser(description="参数级编码机制可提取性复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
