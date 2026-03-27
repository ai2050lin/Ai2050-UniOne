#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage375_shared_carrier_subchain_extractor_20260325"


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
    rows = [row for row in load_rows(INPUT_JSONL) if row.get("category") == "共享承载"]
    bundle_counter = Counter(row["bundle_key"] for row in rows)
    label_counter = Counter(row["label"] for row in rows)
    field_counter = Counter()
    for row in rows:
        field_counter.update(row.get("numeric_fields", {}).keys())

    subchain_rows = []
    for rank, (bundle_key, count) in enumerate(bundle_counter.most_common(5), start=1):
        subchain_rows.append(
            {
                "rank": rank,
                "bundle_key": bundle_key,
                "count": count,
                "top_label": Counter(row["label"] for row in rows if row["bundle_key"] == bundle_key).most_common(1)[0][0],
            }
        )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage375_shared_carrier_subchain_extractor",
        "title": "共享承载子链提取器",
        "status_short": "shared_carrier_subchain_ready",
        "subchain_score": sum(count for _, count in bundle_counter.most_common(3)) / max(len(rows), 1),
        "bundle_rows": [{"bundle_key": key, "count": value} for key, value in bundle_counter.most_common(8)],
        "label_rows": [{"label": key, "count": value} for key, value in label_counter.most_common(8)],
        "field_rows": [{"field_name": key, "count": value} for key, value in field_counter.most_common(8)],
        "subchain_rows": subchain_rows,
        "top_gap_name": "当前共享承载子链的高频行束已提取，可继续对覆盖、分布、核心、位置四类行束做局部对齐。",
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
    parser = argparse.ArgumentParser(description="共享承载子链提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
