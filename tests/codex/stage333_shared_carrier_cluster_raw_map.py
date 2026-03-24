#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage313_shared_carrier_raw_distribution_map import run_analysis as run_stage313
from stage317_shared_carrier_raw_coverage_expansion import run_analysis as run_stage317


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage333_shared_carrier_cluster_raw_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s313 = run_stage313(force=False)
    s317 = run_stage317(force=False)

    distribution_rows = s313["distribution_rows"]
    coverage_rows = s317["coverage_rows"]

    cluster_rows = []
    cluster_defs = [
        ("水果共享簇", lambda row: "水果" in row["role_card"] or "水果" in row["role_name"]),
        ("跨类共享簇", lambda row: "跨类" in row["role_name"]),
        ("器物共享簇", lambda row: "工具" in row["role_name"] or "器物" in row["role_name"]),
        ("混合共享簇", lambda row: "混合" in row["role_name"]),
    ]

    for cluster_name, matcher in cluster_defs:
        matched = [row for row in distribution_rows if matcher(row)]
        cluster_rows.append(
            {
                "cluster_name": cluster_name,
                "member_count": len(matched),
                "dim_indices": [int(row["dim_index"]) for row in matched],
                "mean_base_load": 0.0 if not matched else sum(float(row["base_load"]) for row in matched) / len(matched),
                "mean_stability": 0.0 if not matched else sum(float(row["role_stability"]) for row in matched) / len(matched),
            }
        )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage333_shared_carrier_cluster_raw_map",
        "title": "共享承载簇原始图",
        "status_short": "shared_carrier_cluster_raw_map_ready",
        "cluster_count": len(cluster_rows),
        "cluster_rows": cluster_rows,
        "coverage_rows": coverage_rows,
        "top_gap_name": "共享承载已经形成水果和跨类两个较清楚的原始簇，但器物和混合簇仍然偏薄，跨家族通用簇还没有彻底长出来",
    }
    return summary


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
    parser = argparse.ArgumentParser(description="共享承载簇原始图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
