#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage297_base_bias_local_operator_map import run_analysis as run_stage297
from stage333_shared_carrier_cluster_raw_map import run_analysis as run_stage333
from stage334_bias_deflection_direction_raw_map import run_analysis as run_stage334


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage335_local_operator_raw_cooccurrence_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s297 = run_stage297(force=False)
    s333 = run_stage333(force=False)
    s334 = run_stage334(force=False)

    operator_rows = s297["operator_rows"]
    cluster_rows = s333["cluster_rows"]
    direction_rows = s334["direction_rows"]

    cooccurrence_rows = []
    for operator_row in operator_rows:
        function_text = operator_row["operator_function"]
        if "共享" in function_text or "底盘" in function_text:
            matched_clusters = [row["cluster_name"] for row in cluster_rows if "共享" in row["cluster_name"]]
        else:
            matched_clusters = [row["cluster_name"] for row in cluster_rows]

        if "水果" in function_text or "动物" in function_text or "品牌" in function_text or "器物" in function_text:
            matched_directions = [row["direction_name"] for row in direction_rows if "偏转" in row["direction_name"] and row["direction_name"] != "任务偏转"]
        else:
            matched_directions = [row["direction_name"] for row in direction_rows if "任务偏转" in row["direction_name"] or "偏转" in row["direction_name"]]

        cooccurrence_rows.append(
            {
                "operator_name": operator_row["operator_name"],
                "source_dim_index": int(operator_row["source_dim_index"]),
                "operator_function": function_text,
                "carrier_clusters": matched_clusters[:3],
                "bias_directions": matched_directions[:3],
            }
        )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage335_local_operator_raw_cooccurrence_map",
        "title": "局部运算元原始共现图",
        "status_short": "local_operator_raw_cooccurrence_map_ready",
        "operator_count": len(cooccurrence_rows),
        "cooccurrence_rows": cooccurrence_rows,
        "top_gap_name": "局部运算元已经能与共享承载簇、偏置偏转方向发生稳定共现，但这些共现仍然是粗粒度的，还没细到逐位角色链",
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
    parser = argparse.ArgumentParser(description="局部运算元原始共现图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
