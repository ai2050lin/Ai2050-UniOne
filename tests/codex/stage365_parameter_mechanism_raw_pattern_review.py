#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stage360_refined_raw_extractor import run_analysis as run_stage360
from stage363_cross_model_raw_expansion import run_analysis as run_stage363
from stage364_layerwise_amplification_raw_expansion import run_analysis as run_stage364


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage365_parameter_mechanism_raw_pattern_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s360 = run_stage360(force=True)
    s363 = run_stage363(force=True)
    s364 = run_stage364(force=True)

    category_counts = {row["category"]: int(row["row_count"]) for row in s360["category_rows"]}
    thresholds = {
        "共享承载": 60,
        "偏置偏转": 60,
        "逐层放大": 40,
        "多空间角色": 25,
        "跨模型": 20,
    }
    chain_rows = []
    for category, target in thresholds.items():
        current = int(category_counts.get(category, 0))
        chain_rows.append(
            {
                "category": category,
                "current_row_count": current,
                "target_row_count": target,
                "coverage_ratio": float(current / target) if target else 0.0,
                "covered": current >= target,
            }
        )

    model_counter = Counter(row["model_name"] for row in s363["cross_model_rows"])
    model_rows = [
        {"model_name": model_name, "row_count": row_count}
        for model_name, row_count in model_counter.most_common()
    ]

    band_counter = Counter(row["layer_band"] for row in s364["amplification_rows"])
    band_rows = [
        {"layer_band": layer_band, "row_count": row_count}
        for layer_band, row_count in band_counter.most_common()
    ]

    metric_counter = Counter(row["metric_name"] for row in s364["amplification_rows"])
    metric_rows = [
        {"metric_name": metric_name, "row_count": row_count}
        for metric_name, row_count in metric_counter.most_common()
    ]

    chain_complete = all(row["covered"] for row in chain_rows)
    raw_pattern_score = sum(row["coverage_ratio"] for row in chain_rows) / max(1, len(chain_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage365_parameter_mechanism_raw_pattern_review",
        "title": "参数级编码机制原始模式复核",
        "status_short": "parameter_mechanism_raw_pattern_review_ready",
        "raw_pattern_score": float(raw_pattern_score),
        "chain_complete": chain_complete,
        "chain_rows": chain_rows,
        "cross_model_row_count": len(s363["cross_model_rows"]),
        "amplification_row_count": len(s364["amplification_rows"]),
        "model_rows": model_rows,
        "band_rows": band_rows,
        "metric_rows": metric_rows,
        "top_gap_name": "当前原始结构链的最薄类别是：" + min(chain_rows, key=lambda row: row["coverage_ratio"])["category"],
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
    parser = argparse.ArgumentParser(description="参数级编码机制原始模式复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
