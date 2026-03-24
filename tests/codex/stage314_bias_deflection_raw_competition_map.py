#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage299_bias_position_role_card import run_analysis as run_stage299
from stage311_bias_deflection_operator_core_compression import run_analysis as run_stage311


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage314_bias_deflection_raw_competition_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s299 = run_stage299(force=False)
    s311 = run_stage311(force=False)
    rows = s299["position_rows"]

    raw_competition_score = (
        sum(row["selectivity"] for row in rows) / max(1, len(rows)) * 0.35
        + sum(row["causal_effect"] for row in rows) / max(1, len(rows)) * 0.30
        + float(s311["compression_score"]) * 0.35
    )

    role_counter = {}
    for row in rows:
        role_counter[row["role_card"]] = role_counter.get(row["role_card"], 0) + 1

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage314_bias_deflection_raw_competition_map",
        "title": "偏置偏转原始竞争图",
        "status_short": "bias_deflection_raw_competition_map_ready",
        "raw_competition_score": float(raw_competition_score),
        "role_counter": role_counter,
        "competition_rows": rows[:12],
        "top_gap_name": "偏置偏转位在对象和类内竞争上已经显影，但任务竞争和品牌竞争的原始轨迹仍然不够厚",
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
    parser = argparse.ArgumentParser(description="偏置偏转原始竞争图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
