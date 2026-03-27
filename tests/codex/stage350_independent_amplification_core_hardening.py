#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage344_layerwise_amplification_independent_core_review import run_analysis as run_stage344
from stage327_joint_amplification_independent_core_isolation import run_analysis as run_stage327


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage350_independent_amplification_core_hardening_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s344 = run_stage344(force=False)
    s327 = run_stage327(force=False)

    review_rows = {row["part_name"]: row for row in s344["review_rows"]}
    isolated_rows = list(s327["isolated_rows"])

    mean_residual = sum(float(row["residual_coupling"]) for row in isolated_rows) / max(1, len(isolated_rows))

    hardening_rows = [
        {
            "metric_name": "独立放大核增益",
            "strength": float(review_rows["独立放大核增益"]["strength"]),
        },
        {
            "metric_name": "接力整体强度",
            "strength": float(review_rows["接力整体强度"]["strength"]),
        },
        {
            "metric_name": "残余耦合均值",
            "strength": float(mean_residual),
        },
    ]

    hardening_score = (
        float(review_rows["独立放大核增益"]["strength"]) * 0.50
        + max(0.0, 1.0 - float(mean_residual)) * 0.25
        + (float(review_rows["独立放大核增益"]["strength"]) / max(float(review_rows["接力整体强度"]["strength"]), 1e-6)) * 0.25
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage350_independent_amplification_core_hardening",
        "title": "独立放大核加固图",
        "status_short": "independent_amplification_core_hardening_ready",
        "hardening_score": float(hardening_score),
        "hardening_rows": hardening_rows,
        "top_gap_name": "独立放大核已经显影，但当前仍然更多依赖接力整体，而不是作为一个足够强的独立主核存在。",
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
    parser = argparse.ArgumentParser(description="独立放大核加固图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
