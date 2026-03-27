#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage382_raw_row_position_map_20260325"


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


def infer_position_kind(row: dict) -> str | None:
    label = str(row.get("label", ""))
    bundle_key = str(row.get("bundle_key", ""))
    experiment_id = str(row.get("experiment_id", "")).lower()
    if bundle_key == "position_rows":
        return "位置行束"
    if label in {"object", "attribute", "position", "operation"}:
        return "原始角色"
    if any(key in label for key in ["对象空间", "属性空间", "位置空间", "操作空间", "任务空间", "传播空间"]):
        return "空间标签"
    if "position" in experiment_id:
        return "位置相关实验"
    return None


def build_summary() -> dict:
    rows = load_rows(INPUT_JSONL)
    mapped_rows = []
    counter = Counter()
    for row in rows:
        kind = infer_position_kind(row)
        if kind is None:
            continue
        mapped_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "category": row["category"],
                "bundle_key": row["bundle_key"],
                "label": row["label"],
                "position_kind": kind,
                "path": row["path"],
            }
        )
        counter[kind] += 1

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage382_raw_row_position_map",
        "title": "原始行位置映射图",
        "status_short": "raw_row_position_map_ready",
        "mapped_row_count": len(mapped_rows),
        "position_rows": [{"position_kind": key, "count": value} for key, value in counter.most_common()],
        "mapped_rows": mapped_rows[:120],
        "top_gap_name": "当前位置映射来自位置行束、角色标签和空间标签，还不包含 token_index。",
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
    parser = argparse.ArgumentParser(description="原始行位置映射图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
