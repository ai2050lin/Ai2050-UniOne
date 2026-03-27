#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage340_layerwise_relay_stitching_review import run_analysis as run_stage340
from stage327_joint_amplification_independent_core_isolation import run_analysis as run_stage327


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage344_layerwise_amplification_independent_core_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s340 = run_stage340(force=False)
    s327 = run_stage327(force=False)

    layer_rows = list(s340["layer_rows"])
    isolated_rows = list(s327["isolated_rows"])

    isolated_mean = sum(float(row["independent_gain"]) for row in isolated_rows) / max(1, len(isolated_rows))
    relay_mean = sum(float(row["relay_strength"]) for row in layer_rows) / max(1, len(layer_rows))

    review_rows = [
        {"part_name": "接力整体强度", "strength": float(relay_mean)},
        {"part_name": "独立放大核增益", "strength": float(isolated_mean)},
        {"part_name": "独立放大核显影余量", "strength": max(0.0, float(relay_mean - isolated_mean))},
    ]

    review_score = (
        float(relay_mean) * 0.45
        + float(isolated_mean) * 0.35
        + max(0.0, 1.0 - (float(relay_mean - isolated_mean))) * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage344_layerwise_amplification_independent_core_review",
        "title": "逐层放大独立主核剥离复核",
        "status_short": "layerwise_amplification_independent_core_review_ready",
        "review_score": float(review_score),
        "review_rows": review_rows,
        "top_gap_name": "逐层放大的接力整体已经成立，但独立放大核增益仍然低于接力整体，说明独立主核还没有完全剥离出来。",
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
    parser = argparse.ArgumentParser(description="逐层放大独立主核剥离复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
