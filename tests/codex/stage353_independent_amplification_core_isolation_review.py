#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage344_layerwise_amplification_independent_core_review import run_analysis as run_stage344
from stage350_independent_amplification_core_hardening import run_analysis as run_stage350


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage353_independent_amplification_core_isolation_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s344 = run_stage344(force=False)
    s350 = run_stage350(force=False)

    review_rows_344 = {row["part_name"]: row for row in s344["review_rows"]}
    review_rows_350 = {row["metric_name"]: row for row in s350["hardening_rows"]}

    independent_gain = float(review_rows_344["独立放大核增益"]["strength"])
    relay_strength = float(review_rows_344["接力整体强度"]["strength"])
    residual = float(review_rows_350["残余耦合均值"]["strength"])

    review_rows = [
        {"metric_name": "独立放大核 / 接力整体比值", "strength": independent_gain / relay_strength},
        {"metric_name": "残余耦合抑制后净增益", "strength": max(0.0, independent_gain - residual)},
        {"metric_name": "独立放大核原始增益", "strength": independent_gain},
    ]

    review_score = (
        float(review_rows[0]["strength"]) * 0.45
        + float(review_rows[1]["strength"]) * 0.30
        + float(review_rows[2]["strength"]) * 0.25
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage353_independent_amplification_core_isolation_review",
        "title": "独立放大核剥离复核",
        "status_short": "independent_amplification_core_isolation_review_ready",
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": "独立放大核已经不是纯粹附属信号，但其净增益仍然偏低，说明剥离尚未完成。",
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
    parser = argparse.ArgumentParser(description="独立放大核剥离复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
