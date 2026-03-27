#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from stage360_refined_raw_extractor import run_analysis as run_stage360


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage372_label_field_coupling_extractor_20260325"


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

    coupling_counter: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        fields = sorted(row["numeric_fields"].keys())
        field_signature = "+".join(fields[:4]) if fields else "无字段"
        coupling_counter[row["category"]][f"{row['label']}::{field_signature}"] += 1

    category_rows = []
    matched_total = 0
    for category in sorted(coupling_counter.keys()):
        top_couplings = [
            {"coupling_name": coupling_name, "count": count}
            for coupling_name, count in coupling_counter[category].most_common(6)
        ]
        matched_total += sum(item["count"] for item in top_couplings)
        category_rows.append(
            {
                "category": category,
                "top_couplings": top_couplings,
            }
        )

    coupling_score = matched_total / max(1, len(rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage372_label_field_coupling_extractor",
        "title": "标签字段耦合提取器",
        "status_short": "label_field_coupling_extractor_ready",
        "coupling_score": float(coupling_score),
        "category_rows": category_rows,
        "top_gap_name": "当前标签字段耦合提取器按标签与数值字段组合统计重复模式。",
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
    parser = argparse.ArgumentParser(description="标签字段耦合提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
