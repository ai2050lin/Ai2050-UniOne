#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage347_attribute_position_operation_raw_thickening import run_analysis as run_stage347
from stage351_attribute_position_operation_hardening_review import run_analysis as run_stage351


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage355_attribute_position_operation_expansion_review_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s347 = run_stage347(force=False)
    s351 = run_stage351(force=False)

    thick_rows = {row["space_name"]: row for row in s347["thickening_rows"]}
    review_rows = {row["space_name"]: row for row in s351["review_rows"]}

    rows = []
    for space_name in ["属性空间", "位置空间", "操作空间"]:
        current_strength = float(thick_rows[space_name]["current_strength"])
        target_strength = float(thick_rows[space_name]["target_strength"])
        gap = max(0.0, target_strength - current_strength)
        coverage = float(review_rows[space_name]["coverage_ratio"])
        systemic_thinness = gap / max(target_strength, 1e-6)
        rows.append(
            {
                "space_name": space_name,
                "current_strength": current_strength,
                "target_strength": target_strength,
                "coverage_ratio": coverage,
                "remaining_gap": gap,
                "systemic_thinness": systemic_thinness,
            }
        )

    expansion_score = sum(1.0 - row["systemic_thinness"] for row in rows) / max(1, len(rows))
    top_gap_name = max(rows, key=lambda row: row["systemic_thinness"])["space_name"]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage355_attribute_position_operation_expansion_review",
        "title": "属性 / 位置 / 操作空间扩张复核",
        "status_short": "attribute_position_operation_expansion_review_ready",
        "expansion_score": float(expansion_score),
        "expanded_rows": rows,
        "top_gap_name": f"{top_gap_name} 当前仍是系统性最薄空间，说明多空间结构缺口还没有被拉平。",
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
    parser = argparse.ArgumentParser(description="属性 / 位置 / 操作空间扩张复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
