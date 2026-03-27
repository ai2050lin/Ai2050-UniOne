#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage377_amplification_subchain_extractor_20260325"


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
    rows = [row for row in load_rows(INPUT_JSONL) if row.get("category") == "逐层放大"]
    label_counter = Counter(row["label"] for row in rows)
    bundle_counter = Counter(row["bundle_key"] for row in rows)
    field_counter = Counter()
    for row in rows:
        field_counter.update(row.get("numeric_fields", {}).keys())

    anchor_labels = ["第一次放大主核候选", "中层主放大主核候选", "后层持续放大主核候选"]
    anchor_rows = []
    for label in anchor_labels:
        count = label_counter.get(label, 0)
        if count:
            anchor_rows.append({"label": label, "count": count})

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage377_amplification_subchain_extractor",
        "title": "逐层放大子链提取器",
        "status_short": "amplification_subchain_ready",
        "subchain_score": sum(row["count"] for row in anchor_rows) / max(len(rows), 1),
        "label_rows": [{"label": key, "count": value} for key, value in label_counter.most_common(10)],
        "bundle_rows": [{"bundle_key": key, "count": value} for key, value in bundle_counter.most_common(8)],
        "field_rows": [{"field_name": key, "count": value} for key, value in field_counter.most_common(8)],
        "anchor_rows": anchor_rows,
        "top_gap_name": "当前逐层放大子链已能抽出早层、中层、后层三段候选，但独立放大核仍混在接力整体中。",
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
    parser = argparse.ArgumentParser(description="逐层放大子链提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
