#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from stage360_refined_raw_extractor import run_analysis as run_stage360


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage370_numeric_field_pattern_extractor_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_summary() -> dict:
    s360 = run_stage360(force=True)
    rows = load_jsonl(Path(s360["jsonl_path"]))

    field_counter: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        category = row["category"]
        for field_name in row["numeric_fields"].keys():
            field_counter[category][field_name] += 1

    category_rows = []
    matched_total = 0
    for category in sorted(field_counter.keys()):
        top_fields = [
            {"field_name": field_name, "count": count}
            for field_name, count in field_counter[category].most_common(8)
        ]
        matched_total += sum(item["count"] for item in top_fields)
        category_rows.append(
            {
                "category": category,
                "top_fields": top_fields,
            }
        )

    field_pattern_score = matched_total / max(1, len(rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage370_numeric_field_pattern_extractor",
        "title": "数值字段模式提取器",
        "status_short": "numeric_field_pattern_extractor_ready",
        "field_pattern_score": float(field_pattern_score),
        "category_rows": category_rows,
        "top_gap_name": "当前数值字段模式提取器按类别统计最常出现的数值字段。",
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
    parser = argparse.ArgumentParser(description="数值字段模式提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
