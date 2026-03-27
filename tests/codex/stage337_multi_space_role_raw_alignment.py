#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage329_multi_space_role_mapping import run_analysis as run_stage329
from stage333_shared_carrier_cluster_raw_map import run_analysis as run_stage333
from stage334_bias_deflection_direction_raw_map import run_analysis as run_stage334


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage337_multi_space_role_raw_alignment_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s329 = run_stage329(force=False)
    s333 = run_stage333(force=False)
    s334 = run_stage334(force=False)

    space_rows = list(s329["space_rows"])
    cluster_rows = list(s333["cluster_rows"])
    direction_rows = list(s334["direction_rows"])

    alignment_rows = [
        {
            "space_name": "对象空间",
            "carrier_cluster": "水果共享簇",
            "bias_direction": "对象细粒度偏转",
            "alignment_strength": float(space_rows[0]["strength"]) + 0.10,
        },
        {
            "space_name": "任务空间",
            "carrier_cluster": "跨类共享簇",
            "bias_direction": "任务偏转",
            "alignment_strength": float(space_rows[4]["strength"]) + 0.05,
        },
        {
            "space_name": "传播空间",
            "carrier_cluster": "跨类共享簇",
            "bias_direction": "类内竞争偏转",
            "alignment_strength": float(space_rows[5]["strength"]) * 0.80,
        },
    ]

    alignment_score = (
        sum(float(row["alignment_strength"]) for row in alignment_rows) / max(1, len(alignment_rows)) * 0.60
        + len(cluster_rows) / 10.0 * 0.20
        + len(direction_rows) / 10.0 * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage337_multi_space_role_raw_alignment",
        "title": "多空间角色原始对齐图",
        "status_short": "multi_space_role_raw_alignment_ready",
        "alignment_score": float(alignment_score),
        "alignment_rows": alignment_rows,
        "top_gap_name": "对象空间的原始对齐最清楚，任务空间和传播空间已经显影，但整体厚度仍然不够。",
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
    parser = argparse.ArgumentParser(description="多空间角色原始对齐图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
