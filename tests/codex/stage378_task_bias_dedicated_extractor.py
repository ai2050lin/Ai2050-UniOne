#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage378_task_bias_dedicated_extractor_20260325"
TASK_KEYWORDS = ["translation", "refactor", "image_edit", "task", "operation", "constraint", "修改", "翻译", "重构"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_task_row(row: dict) -> bool:
    text = " ".join([str(row.get("experiment_id", "")), str(row.get("label", "")), str(row.get("bundle_key", ""))]).lower()
    return any(keyword.lower() in text for keyword in TASK_KEYWORDS)


def build_summary() -> dict:
    rows = [row for row in load_rows(INPUT_JSONL) if is_task_row(row)]
    source_counter = Counter(row["category"] for row in rows)
    label_counter = Counter(row["label"] for row in rows)
    bundle_counter = Counter(row["bundle_key"] for row in rows)
    field_counter = Counter()
    for row in rows:
        field_counter.update(row.get("numeric_fields", {}).keys())

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage378_task_bias_dedicated_extractor",
        "title": "任务偏转专用提取器",
        "status_short": "task_bias_dedicated_ready",
        "task_bias_score": sum(source_counter.values()) / max(len(load_rows(INPUT_JSONL)), 1),
        "source_rows": [{"category": key, "count": value} for key, value in source_counter.most_common()],
        "label_rows": [{"label": key, "count": value} for key, value in label_counter.most_common(12)],
        "bundle_rows": [{"bundle_key": key, "count": value} for key, value in bundle_counter.most_common(8)],
        "field_rows": [{"field_name": key, "count": value} for key, value in field_counter.most_common(8)],
        "top_gap_name": "当前任务偏转专用提取已覆盖翻译、重构、操作、约束四类标签，后续可继续扩充任务竞争样本。",
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
    parser = argparse.ArgumentParser(description="任务偏转专用提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
