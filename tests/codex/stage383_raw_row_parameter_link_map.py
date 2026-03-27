#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage383_raw_row_parameter_link_map_20260325"
PARAMETER_KEYS = ["dim_index", "source_dim_index", "base_dim_index", "bias_dim_index", "carrier_dim", "bias_dim"]


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


def build_summary() -> dict:
    rows = load_rows(INPUT_JSONL)
    mapped_rows = []
    key_counter = Counter()
    for row in rows:
        numeric_fields = row.get("numeric_fields", {})
        present_keys = [key for key in PARAMETER_KEYS if key in numeric_fields]
        if not present_keys:
            continue
        mapped_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "category": row["category"],
                "bundle_key": row["bundle_key"],
                "label": row["label"],
                "parameter_keys": present_keys,
                "parameter_values": {key: numeric_fields[key] for key in present_keys},
                "path": row["path"],
            }
        )
        for key in present_keys:
            key_counter[key] += 1

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage383_raw_row_parameter_link_map",
        "title": "原始行参数位链接图",
        "status_short": "raw_row_parameter_link_map_ready",
        "mapped_row_count": len(mapped_rows),
        "parameter_key_rows": [{"parameter_key": key, "count": value} for key, value in key_counter.most_common()],
        "mapped_rows": mapped_rows[:120],
        "top_gap_name": "当前参数位链接已覆盖 dim_index、source_dim_index、base_dim_index、bias_dim_index、carrier_dim、bias_dim 六类字段。",
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
    parser = argparse.ArgumentParser(description="原始行参数位链接图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
