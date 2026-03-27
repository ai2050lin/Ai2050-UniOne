#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
OUTPUT_DIR = TEMP_ROOT / "stage359_refined_raw_inventory_20260325"

EXCLUDED_NAME_PARTS = [
    "_3d_",
    "manifest",
    "client_scene",
    "operator_special_format_export",
    "raw_data_inventory",
    "refined_raw_inventory",
    "refined_raw_extractor",
    "extractability_review",
    "expansion_projection",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_stage(experiment_id: str) -> str:
    text = experiment_id.lower()
    if "shared" in text or "carrier" in text or "base" in text:
        return "共享承载"
    if "bias" in text or "deflection" in text:
        return "偏置偏转"
    if "amplification" in text or "relay" in text:
        return "逐层放大"
    if "space" in text or "role" in text or "semantic" in text:
        return "多空间角色"
    if "cross_model" in text or "qwen" in text or "deepseek" in text:
        return "跨模型"
    return "其他"


def row_bundles(summary: dict) -> list[tuple[str, list]]:
    bundles = []
    for key, value in summary.items():
        if key.endswith("_rows") and isinstance(value, list) and value:
            bundles.append((key, value))
    return bundles


def build_summary() -> dict:
    stage_dirs = []
    for idx in range(257, 400):
        stage_dirs.extend(sorted(TEMP_ROOT.glob(f"stage{idx}_*")))

    stage_rows = []
    category_totals: dict[str, int] = {}
    total_bundle_count = 0
    total_row_count = 0

    for directory in stage_dirs:
        if any(part in directory.name for part in EXCLUDED_NAME_PARTS):
            continue
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue

        summary = load_json(summary_path)
        bundles = row_bundles(summary)
        if not bundles:
            continue

        bundle_count = len(bundles)
        row_count = sum(len(rows) for _, rows in bundles)
        experiment_id = summary.get("experiment_id", directory.name)
        category = classify_stage(experiment_id)

        category_totals[category] = category_totals.get(category, 0) + row_count
        total_bundle_count += bundle_count
        total_row_count += row_count
        stage_rows.append(
            {
                "experiment_id": experiment_id,
                "title": summary.get("title", directory.name),
                "category": category,
                "row_bundle_count": bundle_count,
                "raw_row_count": row_count,
                "path": str(directory),
            }
        )

    category_rows = [
        {"category": key, "raw_row_count": value}
        for key, value in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    weakest_categories = [row["category"] for row in category_rows[-2:]] if len(category_rows) >= 2 else [row["category"] for row in category_rows]

    inventory_score = total_row_count / max(1, len(stage_rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage359_refined_raw_inventory",
        "title": "扩展区间原始数据精炼清点",
        "status_short": "refined_raw_inventory_ready",
        "covered_stage_count": len(stage_rows),
        "total_row_bundle_count": total_bundle_count,
        "total_raw_row_count": total_row_count,
        "inventory_score": float(inventory_score),
        "stage_rows": stage_rows,
        "category_rows": category_rows,
        "top_gap_name": "当前精炼清点已经覆盖更长阶段区间，当前原始行最薄的类别是：" + "、".join(weakest_categories),
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
    parser = argparse.ArgumentParser(description="扩展区间原始数据精炼清点")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
