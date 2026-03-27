#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = PROJECT_ROOT / "tests" / "codex_temp"
OUTPUT_DIR = TEMP_ROOT / "stage360_refined_raw_extractor_20260325"

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

LABEL_KEYS = [
    "space_name",
    "metric_name",
    "criterion_name",
    "part_name",
    "cluster_name",
    "axis_name",
    "role_name",
    "stage_name",
    "layer_name",
    "category",
    "task_name",
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


def extract_label(row: dict) -> str:
    for key in LABEL_KEYS:
        if key in row:
            return str(row[key])
    return "未命名行"


def extract_numeric_fields(row: dict) -> dict:
    numeric = {}
    for key, value in row.items():
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def build_summary() -> dict:
    records = []
    for idx in range(257, 400):
        for directory in sorted(TEMP_ROOT.glob(f"stage{idx}_*")):
            if any(part in directory.name for part in EXCLUDED_NAME_PARTS):
                continue

            summary_path = directory / "summary.json"
            if not summary_path.exists():
                continue

            summary = load_json(summary_path)
            experiment_id = summary.get("experiment_id", directory.name)
            category = classify_stage(experiment_id)

            for key, value in summary.items():
                if not (key.endswith("_rows") and isinstance(value, list) and value):
                    continue
                for row_index, row in enumerate(value):
                    if not isinstance(row, dict):
                        continue
                    numeric_fields = extract_numeric_fields(row)
                    if not numeric_fields:
                        continue
                    records.append(
                        {
                            "experiment_id": experiment_id,
                            "category": category,
                            "bundle_key": key,
                            "row_index": row_index,
                            "label": extract_label(row),
                            "numeric_fields": numeric_fields,
                            "path": str(directory),
                        }
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "refined_rows.jsonl"
    with jsonl_path.open("w", encoding="utf-8-sig") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    category_counts: dict[str, int] = {}
    for record in records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1

    category_rows = [
        {"category": key, "row_count": value}
        for key, value in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    weakest_categories = [row["category"] for row in category_rows[-2:]] if len(category_rows) >= 2 else [row["category"] for row in category_rows]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage360_refined_raw_extractor",
        "title": "精炼原始行提取器",
        "status_short": "refined_raw_extractor_ready",
        "record_count": len(records),
        "category_rows": category_rows,
        "jsonl_path": str(jsonl_path),
        "top_gap_name": "当前提取规则已经排除了展示层和无效阶段，当前可用原始行最少的类别是：" + "、".join(weakest_categories),
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
    parser = argparse.ArgumentParser(description="精炼原始行提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
