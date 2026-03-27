#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = PROJECT_ROOT / "tests" / "codex_temp" / "stage360_refined_raw_extractor_20260325" / "refined_rows.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage381_raw_row_layer_map_20260325"


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


def infer_layer_kind(row: dict) -> tuple[str | None, float | None]:
    label = str(row.get("label", ""))
    nf = row.get("numeric_fields", {})
    if "route_peak_layer" in nf:
        return "显式层号", float(nf["route_peak_layer"])
    if "早层" in label or "第一次放大" in label:
        return "早层", None
    if "中层" in label or "主放大" in label:
        return "中层", None
    if "后层" in label or "持续放大" in label:
        return "后层", None
    if "global" in label.lower():
        return "全局", None
    return None, None


def build_summary() -> dict:
    rows = load_rows(INPUT_JSONL)
    mapped_rows = []
    layer_counter = Counter()
    for row in rows:
        layer_kind, layer_value = infer_layer_kind(row)
        if layer_kind is None:
            continue
        mapped_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "category": row["category"],
                "bundle_key": row["bundle_key"],
                "label": row["label"],
                "layer_kind": layer_kind,
                "layer_value": layer_value,
                "path": row["path"],
            }
        )
        layer_counter[layer_kind] += 1

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage381_raw_row_layer_map",
        "title": "原始行层号映射图",
        "status_short": "raw_row_layer_map_ready",
        "mapped_row_count": len(mapped_rows),
        "layer_rows": [{"layer_kind": key, "count": value} for key, value in layer_counter.most_common()],
        "mapped_rows": mapped_rows[:120],
        "top_gap_name": "当前原始行层号映射只包含显式层号和早中后层标签，不包含统一 token 层号映射。",
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
    parser = argparse.ArgumentParser(description="原始行层号映射图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
