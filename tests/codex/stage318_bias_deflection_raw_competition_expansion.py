#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage302_task_bias_position_strengthening import run_analysis as run_stage302
from stage308_task_bias_core_compression import run_analysis as run_stage308
from stage314_bias_deflection_raw_competition_map import run_analysis as run_stage314


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage318_bias_deflection_raw_competition_expansion_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s302 = run_stage302(force=False)
    s308 = run_stage308(force=False)
    s314 = run_stage314(force=False)

    rows = s314["competition_rows"]
    families = {
        "对象竞争": lambda row: "对象细粒度偏转位" in row["role_card"],
        "类内竞争": lambda row: "类内竞争偏转位" in row["role_card"],
        "对象域切换": lambda row: "对象域切换偏转位" in row["role_card"],
        "品牌或跨类": lambda row: "品牌或跨类偏转位" in row["role_card"],
    }

    competition_axes = []
    for axis_name, matcher in families.items():
        matched = [row for row in rows if matcher(row)]
        if matched:
            mean_selectivity = sum(float(row["selectivity"]) for row in matched) / len(matched)
            mean_leverage = sum(float(row["leverage"]) for row in matched) / len(matched)
            mean_causal = sum(float(row["causal_effect"]) for row in matched) / len(matched)
        else:
            mean_selectivity = 0.0
            mean_leverage = 0.0
            mean_causal = 0.0
        competition_axes.append(
            {
                "axis_name": axis_name,
                "count": len(matched),
                "mean_selectivity": mean_selectivity,
                "mean_leverage": mean_leverage,
                "mean_causal": mean_causal,
            }
        )

    raw_expansion_score = (
        sum(row["mean_selectivity"] for row in competition_axes) / max(1, len(competition_axes)) * 0.30
        + sum(row["mean_leverage"] for row in competition_axes) / max(1, len(competition_axes)) * 0.30
        + float(s302["strengthening_score"]) * 0.20
        + float(s308["compression_score"]) * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage318_bias_deflection_raw_competition_expansion",
        "title": "偏置偏转原始竞争扩张图",
        "status_short": "bias_deflection_raw_competition_expansion_ready",
        "raw_expansion_score": float(raw_expansion_score),
        "competition_axes": competition_axes,
        "top_gap_name": "偏置偏转位在对象和类内竞争上最厚，任务偏转和品牌竞争虽然在增强，但原始轨迹仍然不够稳定",
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
    parser = argparse.ArgumentParser(description="偏置偏转原始竞争扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
