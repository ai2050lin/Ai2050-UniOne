#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage279_refactor_structure_rewrite_causal_map import run_analysis as run_stage279


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage286_refactor_structure_function_map_20260324"


def build_summary() -> dict:
    s279 = run_stage279(force=False)
    rows = []
    for row in s279["model_rows"]:
        dim_counts: dict[int, int] = {}
        for prompt in row["prompt_rows"]:
            for dim in prompt["hot_dims"]:
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
        shared = sorted((dim for dim, count in dim_counts.items() if count > 1))
        single = sorted((dim for dim, count in dim_counts.items() if count == 1))
        rows.append(
            {
                "model_tag": row["model_tag"],
                "display_name": row["display_name"],
                "function_score": float((len(shared) * 2 + row["causal_score"] * 10) / 10.0),
                "shared_function_dims": shared[:16],
                "prompt_local_dims": single[:16],
                "strongest_function_name": "shared_rewrite_function" if shared else "prompt_local_rewrite_function",
            }
        )
    strongest = max(rows, key=lambda item: item["function_score"])
    weakest = min(rows, key=lambda item: item["function_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage286_refactor_structure_function_map",
        "title": "重构结构改写功能图",
        "status_short": "refactor_structure_function_map_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "重构的结构改写位不只是出现频繁，它们还会跨提示重复承担同一类结构功能，开始形成可复用功能位",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="重构结构改写功能图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
