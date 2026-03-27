#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from stage363_cross_model_raw_expansion import run_analysis as run_stage363


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage368_cross_model_consensus_extractor_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_metric(metric_name: str) -> str:
    if "::" in metric_name:
        return metric_name.split("::", 1)[0]
    return metric_name


def build_summary() -> dict:
    s363 = run_stage363(force=True)

    metric_models: dict[str, set[str]] = defaultdict(set)
    for row in s363["cross_model_rows"]:
        metric_models[normalize_metric(row["metric_name"])].add(row["model_name"])

    consensus_rows = []
    for metric_name, models in sorted(metric_models.items(), key=lambda item: (-len(item[1]), item[0])):
        consensus_rows.append(
            {
                "metric_name": metric_name,
                "model_count": len(models),
                "model_names": sorted(models),
            }
        )

    consensus_score = sum(row["model_count"] for row in consensus_rows) / max(1, len(consensus_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage368_cross_model_consensus_extractor",
        "title": "跨模型共识提取器",
        "status_short": "cross_model_consensus_extractor_ready",
        "consensus_score": float(consensus_score),
        "consensus_rows": consensus_rows,
        "top_gap_name": "当前跨模型共识提取器按指标前缀汇总模型覆盖数量。",
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
    parser = argparse.ArgumentParser(description="跨模型共识提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
