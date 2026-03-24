#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage297_base_bias_local_operator_map import run_analysis as run_stage297
from stage329_multi_space_role_mapping import run_analysis as run_stage329
from stage330_fuzzy_carrier_sparse_deflection_joint_map import run_analysis as run_stage330
from stage331_layerwise_relay_independent_core_map import run_analysis as run_stage331


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage332_local_operator_stitching_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s297 = run_stage297(force=False)
    s329 = run_stage329(force=False)
    s330 = run_stage330(force=False)
    s331 = run_stage331(force=False)

    stitching_score = (
        float(s297["operator_score"]) * 0.25
        + float(s329["mapping_score"]) * 0.25
        + float(s330["joint_score"]) * 0.25
        + float(s331["relay_score"]) * 0.25
    )

    stitch_rows = [
        {"part_name": "局部运算元", "strength": float(s297["operator_score"])},
        {"part_name": "多空间角色映射", "strength": float(s329["mapping_score"])},
        {"part_name": "模糊承载与稀疏偏转", "strength": float(s330["joint_score"])},
        {"part_name": "层间接力", "strength": float(s331["relay_score"])},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage332_local_operator_stitching_map",
        "title": "局部运算元拼接图",
        "status_short": "local_operator_stitching_map_ready",
        "stitching_score": float(stitching_score),
        "stitch_rows": stitch_rows,
        "top_gap_name": "局部运算元已经能与多空间角色、模糊承载、层间接力拼接，但当前拼接仍然是工作性框架，还不是统一理论",
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
    parser = argparse.ArgumentParser(description="局部运算元拼接图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
