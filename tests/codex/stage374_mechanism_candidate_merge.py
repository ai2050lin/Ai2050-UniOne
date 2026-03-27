#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stage366_frequency_pattern_extractor import run_analysis as run_stage366
from stage370_numeric_field_pattern_extractor import run_analysis as run_stage370
from stage371_bundle_structure_extractor import run_analysis as run_stage371
from stage372_label_field_coupling_extractor import run_analysis as run_stage372


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage374_mechanism_candidate_merge_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _to_map(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in rows}


def _top_values(rows: list[dict], key: str, value_key: str, top_k: int = 3) -> list[str]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get(value_key, 0.0)), reverse=True)
    return [str(row[key]) for row in sorted_rows[:top_k]]


def build_summary() -> dict:
    s366 = run_stage366(force=True)
    s370 = run_stage370(force=True)
    s371 = run_stage371(force=True)
    s372 = run_stage372(force=True)

    freq_map = _to_map(s366["category_rows"], "category")
    field_map = _to_map(s370["category_rows"], "category")
    bundle_map = _to_map(s371["category_rows"], "category")
    coupling_map = _to_map(s372["category_rows"], "category")

    candidate_specs = [
        ("共享承载原段", "共享承载"),
        ("偏置偏转原段", "偏置偏转"),
        ("逐层放大原段", "逐层放大"),
        ("多空间角色原段", "多空间角色"),
    ]

    candidate_rows = []
    for candidate_name, category in candidate_specs:
        freq_row = freq_map.get(category, {"top_labels": [], "top_bundles": []})
        field_row = field_map.get(category, {"field_rows": [], "field_density": 0.0})
        bundle_row = bundle_map.get(category, {"bundle_rows": [], "bundle_density": 0.0})
        coupling_row = coupling_map.get(category, {"coupling_rows": [], "coupling_density": 0.0})

        score = (
            (1.0 if freq_row.get("top_labels") else 0.0)
            + float(field_row.get("field_density", 0.0))
            + float(bundle_row.get("bundle_density", 0.0))
            + float(coupling_row.get("coupling_density", 0.0))
        ) / 4.0

        candidate_rows.append(
            {
                "candidate_name": candidate_name,
                "category": category,
                "score": score,
                "top_labels": _top_values(freq_row.get("top_labels", []), "label", "count"),
                "top_fields": _top_values(field_row.get("field_rows", []), "field_name", "count"),
                "top_bundles": _top_values(bundle_row.get("bundle_rows", []), "bundle_key", "count"),
                "top_couplings": _top_values(coupling_row.get("coupling_rows", []), "coupling_key", "count"),
            }
        )

    top_gap_name = "当前候选结构中得分最低的是：" + min(candidate_rows, key=lambda row: row["score"])["candidate_name"]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage374_mechanism_candidate_merge",
        "title": "参数级编码机制候选结构合并",
        "status_short": "mechanism_candidate_merge_ready",
        "merge_score": sum(row["score"] for row in candidate_rows) / len(candidate_rows),
        "candidate_rows": candidate_rows,
        "top_gap_name": top_gap_name,
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
    parser = argparse.ArgumentParser(description="参数级编码机制候选结构合并")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
