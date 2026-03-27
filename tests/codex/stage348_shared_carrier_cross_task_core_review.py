#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage342_shared_carrier_cluster_stability_expansion import run_analysis as run_stage342
from stage325_shared_carrier_cross_task_core_compression import run_analysis as run_stage325


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage348_shared_carrier_cross_task_core_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s342 = run_stage342(force=False)
    s325 = run_stage325(force=False)

    rows = list(s342["expansion_rows"])
    compression_score = float(s325["compression_score"])

    review_rows = rows + [
        {
            "axis_name": "跨任务主核压缩度",
            "strength": compression_score,
        }
    ]

    review_score = sum(float(row["strength"]) for row in review_rows) / max(1, len(review_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage348_shared_carrier_cross_task_core_review",
        "title": "共享承载跨任务主核复核",
        "status_short": "shared_carrier_cross_task_core_review_ready",
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": "跨任务主核压缩度仍然低于跨模型稳定性，说明共享承载虽然已经跨任务显影，但还没有压成真正稳定的任务主核。",
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
    parser = argparse.ArgumentParser(description="共享承载跨任务主核复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
