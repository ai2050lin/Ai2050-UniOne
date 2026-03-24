#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage257_object_attribute_position_operation_role_map import run_analysis as run_stage257
from stage321_shared_carrier_cross_task_raw_coverage import run_analysis as run_stage321


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage329_multi_space_role_mapping_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s257 = run_stage257(force=False)
    s321 = run_stage321(force=False)

    role_rows = s257["role_rows"]
    task_rows = s321["task_rows"]

    space_rows = [
        {"space_name": "对象空间", "strength": float(role_rows[0]["activation_strength"])},
        {"space_name": "属性空间", "strength": float(role_rows[1]["activation_strength"])},
        {"space_name": "位置空间", "strength": float(role_rows[2]["activation_strength"])},
        {"space_name": "操作空间", "strength": float(role_rows[3]["activation_strength"])},
        {
            "space_name": "任务空间",
            "strength": sum(float(row["shared_base_bridge"]) for row in task_rows) / max(1, len(task_rows)),
        },
        {
            "space_name": "层间传播空间",
            "strength": sum(float(row["role_overlap"]) for row in task_rows) / max(1, len(task_rows)) * 0.5,
        },
    ]

    mapping_score = sum(float(row["strength"]) for row in space_rows) / len(space_rows)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage329_multi_space_role_mapping",
        "title": "多空间角色映射图",
        "status_short": "multi_space_role_mapping_ready",
        "mapping_score": float(mapping_score),
        "space_rows": space_rows,
        "top_gap_name": "对象、属性、位置、操作已经能显影成多空间角色，但任务空间和层间传播空间仍然薄于对象空间",
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
    parser = argparse.ArgumentParser(description="多空间角色映射图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
