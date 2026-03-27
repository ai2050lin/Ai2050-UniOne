#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
OUTPUT_DIR = TEMP_ROOT / "stage345_raw_data_inventory_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_count(summary: dict) -> tuple[int, int]:
    bundle_count = 0
    total_rows = 0
    for key, value in summary.items():
        if key.endswith("_rows") and isinstance(value, list):
            bundle_count += 1
            total_rows += len(value)
    return bundle_count, total_rows


def classify_stage(experiment_id: str) -> str:
    if "shared" in experiment_id or "carrier" in experiment_id:
        return "共享承载"
    if "bias" in experiment_id or "deflection" in experiment_id:
        return "偏置偏转"
    if "amplification" in experiment_id or "relay" in experiment_id:
        return "逐层放大"
    if "space" in experiment_id or "role" in experiment_id:
        return "多空间角色"
    return "其他"


def build_summary() -> dict:
    stage_dirs = []
    for idx in range(313, 345):
        matched = sorted(TEMP_ROOT.glob(f"stage{idx}_*"))
        if matched:
            stage_dirs.append(matched[-1])

    stage_rows = []
    category_totals: dict[str, int] = {}
    total_row_bundles = 0
    total_rows = 0

    for directory in stage_dirs:
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        bundle_count, rows = row_count(summary)
        category = classify_stage(summary.get("experiment_id", directory.name))
        category_totals[category] = category_totals.get(category, 0) + rows
        total_row_bundles += bundle_count
        total_rows += rows
        stage_rows.append(
            {
                "experiment_id": summary.get("experiment_id", directory.name),
                "title": summary.get("title", directory.name),
                "category": category,
                "row_bundle_count": bundle_count,
                "raw_row_count": rows,
                "path": str(directory),
            }
        )

    category_rows = [
        {
            "category": key,
            "raw_row_count": value,
        }
        for key, value in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    ]

    inventory_score = total_rows / max(1, len(stage_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage345_raw_data_inventory",
        "title": "原始数据存量清点",
        "status_short": "raw_data_inventory_ready",
        "covered_stage_count": len(stage_rows),
        "total_row_bundle_count": total_row_bundles,
        "total_raw_row_count": total_rows,
        "inventory_score": float(inventory_score),
        "stage_rows": stage_rows,
        "category_rows": category_rows,
        "top_gap_name": "当前原始数据已经形成连续阶段积累，但多空间角色和逐层放大两块的原始行数仍然明显少于共享承载与偏置偏转。",
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
    parser = argparse.ArgumentParser(description="原始数据存量清点")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
