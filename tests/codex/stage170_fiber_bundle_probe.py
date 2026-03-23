#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from stage166_category_fiber_map import run_analysis as run_stage166_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage170_fiber_bundle_probe_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE170_FIBER_BUNDLE_PROBE_REPORT.md"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> Dict[str, object]:
    if not STAGE166_SUMMARY_PATH.exists():
        run_stage166_analysis(force=True)
    stage166 = load_json(STAGE166_SUMMARY_PATH)
    categories = [row["category_name"] for row in stage166["category_rows"]]
    edge_rows = stage166["edge_rows"]
    bundle_rows: List[Dict[str, object]] = []
    for category in categories:
        connected = [
            float(row["centroid_similarity"])
            for row in edge_rows
            if row["category_a"] == category or row["category_b"] == category
        ]
        bundle_strength = mean(connected)
        bundle_rows.append(
            {
                "category_name": str(category),
                "bundle_strength": bundle_strength,
                "bundle_separation": 1.0 - ((bundle_strength + 1.0) / 2.0),
            }
        )
    bundle_rows.sort(key=lambda row: float(row["bundle_strength"]), reverse=True)
    fiber_bundle_score = clamp01(mean(float(row["bundle_strength"]) for row in bundle_rows) * 0.5)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage170_fiber_bundle_probe",
        "title": "纤维束探针",
        "status_short": "fiber_bundle_probe_ready",
        "bundle_count": len(bundle_rows),
        "fiber_bundle_score": fiber_bundle_score,
        "strongest_bundle_name": str(bundle_rows[0]["category_name"]),
        "weakest_bundle_name": str(bundle_rows[-1]["category_name"]),
        "bundle_rows": bundle_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage170: 纤维束探针",
        "",
        "## 核心结果",
        f"- 束数: {summary['bundle_count']}",
        f"- 纤维束分数: {summary['fiber_bundle_score']:.4f}",
        f"- 最强纤维束: {summary['strongest_bundle_name']}",
        f"- 最弱纤维束: {summary['weakest_bundle_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="纤维束探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
