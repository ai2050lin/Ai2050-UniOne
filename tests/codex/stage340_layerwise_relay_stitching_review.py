#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage331_layerwise_relay_independent_core_map import run_analysis as run_stage331
from stage332_local_operator_stitching_map import run_analysis as run_stage332


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage340_layerwise_relay_stitching_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s331 = run_stage331(force=False)
    s332 = run_stage332(force=False)

    layer_rows = list(s331["layer_rows"])
    stitch_rows = list(s332["stitch_rows"])

    review_score = float(s331["relay_score"]) * 0.55 + float(s332["stitching_score"]) * 0.45

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage340_layerwise_relay_stitching_review",
        "title": "层间接力拼接复核",
        "status_short": "layerwise_relay_stitching_review_ready",
        "review_score": float(review_score),
        "layer_rows": layer_rows,
        "stitch_rows": stitch_rows,
        "top_gap_name": "层间接力与局部拼接已经能进入同一张复核图，但当前仍然是工作性联接，不是统一稳定主核。",
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
    parser = argparse.ArgumentParser(description="层间接力拼接复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
