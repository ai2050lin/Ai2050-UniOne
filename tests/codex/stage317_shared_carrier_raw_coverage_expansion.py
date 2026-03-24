#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage301_cross_family_shared_base_compression import run_analysis as run_stage301
from stage307_cross_family_shared_base_core_compression import run_analysis as run_stage307
from stage313_shared_carrier_raw_distribution_map import run_analysis as run_stage313


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage317_shared_carrier_raw_coverage_expansion_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s301 = run_stage301(force=False)
    s307 = run_stage307(force=False)
    s313 = run_stage313(force=False)

    rows = s313["distribution_rows"]
    families = {
        "水果": lambda row: "水果" in row["role_card"] or "水果" in row["role_name"],
        "跨类": lambda row: "跨类" in row["role_name"],
        "器物": lambda row: "工具" in row["role_name"] or "器物" in row["role_name"],
        "混合": lambda row: "混合" in row["role_name"],
    }

    coverage_rows = []
    for family_name, matcher in families.items():
        matched = [row for row in rows if matcher(row)]
        if matched:
            mean_base_load = sum(float(row["base_load"]) for row in matched) / len(matched)
            mean_stability = sum(float(row["role_stability"]) for row in matched) / len(matched)
            family_hit_count = sum(int(row["family_hit_count"]) for row in matched)
        else:
            mean_base_load = 0.0
            mean_stability = 0.0
            family_hit_count = 0
        coverage_rows.append(
            {
                "family_name": family_name,
                "coverage_count": len(matched),
                "mean_base_load": mean_base_load,
                "mean_stability": mean_stability,
                "family_hit_count": family_hit_count,
            }
        )

    raw_coverage_score = (
        sum(row["mean_base_load"] for row in coverage_rows) / max(1, len(coverage_rows)) * 0.30
        + sum(row["mean_stability"] for row in coverage_rows) / max(1, len(coverage_rows)) * 0.30
        + float(s301["compression_score"]) * 0.20
        + float(s307["compression_score"]) * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage317_shared_carrier_raw_coverage_expansion",
        "title": "共享承载原始覆盖扩张图",
        "status_short": "shared_carrier_raw_coverage_expansion_ready",
        "raw_coverage_score": float(raw_coverage_score),
        "coverage_rows": coverage_rows,
        "top_gap_name": "共享承载位的原始覆盖已经从水果扩到跨类和器物层，但真正跨家族稳定复用的主核仍然偏少",
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
    parser = argparse.ArgumentParser(description="共享承载原始覆盖扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
