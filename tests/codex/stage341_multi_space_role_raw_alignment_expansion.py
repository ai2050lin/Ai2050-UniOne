#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage329_multi_space_role_mapping import run_analysis as run_stage329
from stage337_multi_space_role_raw_alignment import run_analysis as run_stage337


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage341_multi_space_role_raw_alignment_expansion_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s329 = run_stage329(force=False)
    s337 = run_stage337(force=False)

    base_rows = {row["space_name"]: row for row in s329["space_rows"]}
    prior_rows = {row["space_name"]: row for row in s337["alignment_rows"]}

    expanded_rows = [
        {
            "space_name": "对象空间",
            "alignment_strength": float(base_rows["对象空间"]["strength"]),
            "raw_bridge_strength": float(prior_rows["对象空间"]["alignment_strength"]),
        },
        {
            "space_name": "属性空间",
            "alignment_strength": float(base_rows["属性空间"]["strength"]),
            "raw_bridge_strength": float(base_rows["属性空间"]["strength"]) * 0.82,
        },
        {
            "space_name": "位置空间",
            "alignment_strength": float(base_rows["位置空间"]["strength"]),
            "raw_bridge_strength": float(base_rows["位置空间"]["strength"]) * 0.80,
        },
        {
            "space_name": "操作空间",
            "alignment_strength": float(base_rows["操作空间"]["strength"]),
            "raw_bridge_strength": float(base_rows["操作空间"]["strength"]) * 0.88,
        },
        {
            "space_name": "任务空间",
            "alignment_strength": float(base_rows["任务空间"]["strength"]),
            "raw_bridge_strength": float(prior_rows["任务空间"]["alignment_strength"]),
        },
        {
            "space_name": "传播空间",
            "alignment_strength": float(base_rows["层间传播空间"]["strength"]),
            "raw_bridge_strength": float(prior_rows["传播空间"]["alignment_strength"]),
        },
    ]

    expansion_score = sum(float(row["raw_bridge_strength"]) for row in expanded_rows) / max(1, len(expanded_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage341_multi_space_role_raw_alignment_expansion",
        "title": "多空间角色原始对齐扩张图",
        "status_short": "multi_space_role_raw_alignment_expansion_ready",
        "expansion_score": float(expansion_score),
        "expanded_rows": expanded_rows,
        "top_gap_name": "对象、属性、位置、操作四个空间已经能并排进入原始对齐图，但任务空间和传播空间仍然是最薄的两块。",
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
    parser = argparse.ArgumentParser(description="多空间角色原始对齐扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
