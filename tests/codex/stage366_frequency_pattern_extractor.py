#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from stage360_refined_raw_extractor import run_analysis as run_stage360


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage366_frequency_pattern_extractor_20260325"


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

    category_label_counter: dict[str, Counter] = defaultdict(Counter)
    category_bundle_counter: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        category = row["category"]
        category_label_counter[category][row["label"]] += 1
        category_bundle_counter[category][row["bundle_key"]] += 1

    category_rows = []
    for category in sorted(category_label_counter.keys()):
        top_labels = [
            {"label": label, "count": count}
            for label, count in category_label_counter[category].most_common(5)
        ]
        top_bundles = [
            {"bundle_key": bundle_key, "count": count}
            for bundle_key, count in category_bundle_counter[category].most_common(5)
        ]
        category_rows.append(
            {
                "category": category,
                "top_labels": top_labels,
                "top_bundles": top_bundles,
            }
        )

    frequency_score = sum(sum(item["count"] for item in row["top_labels"]) for row in category_rows) / max(1, len(rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage366_frequency_pattern_extractor",
        "title": "频率模式提取器",
        "status_short": "frequency_pattern_extractor_ready",
        "frequency_score": float(frequency_score),
        "category_rows": category_rows,
        "top_gap_name": "当前频率模式提取器优先输出各类别最常出现的标签和原始行束。",
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
    parser = argparse.ArgumentParser(description="频率模式提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
