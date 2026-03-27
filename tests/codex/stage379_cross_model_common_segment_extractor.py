#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from stage363_cross_model_raw_expansion import run_analysis as run_stage363


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage379_cross_model_common_segment_extractor_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_segment(metric_name: str) -> str:
    if "::" in metric_name:
        return metric_name.split("::", 1)[0]
    return metric_name


def build_summary() -> dict:
    s363 = run_stage363(force=True)
    grouped: dict[str, set[str]] = defaultdict(set)
    strength_sum: dict[str, float] = defaultdict(float)
    strength_count: dict[str, int] = defaultdict(int)

    for row in s363["cross_model_rows"]:
        segment = normalize_segment(str(row["metric_name"]))
        grouped[segment].add(str(row["model_name"]))
        strength_sum[segment] += float(row["strength"])
        strength_count[segment] += 1

    segment_rows = []
    for segment, models in grouped.items():
        avg_strength = strength_sum[segment] / max(strength_count[segment], 1)
        segment_rows.append(
            {
                "segment_name": segment,
                "model_count": len(models),
                "avg_strength": avg_strength,
                "models": sorted(models),
            }
        )
    segment_rows.sort(key=lambda row: (-row["model_count"], -row["avg_strength"], row["segment_name"]))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage379_cross_model_common_segment_extractor",
        "title": "跨模型共同原段提取器",
        "status_short": "cross_model_common_segment_ready",
        "segment_score": sum(row["model_count"] for row in segment_rows[:10]) / 10.0 if segment_rows else 0.0,
        "segment_rows": segment_rows[:20],
        "top_gap_name": "当前跨模型共同原段已能按指标前缀聚合，后续可继续压缩到更少的共同结构段。",
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
    parser = argparse.ArgumentParser(description="跨模型共同原段提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
