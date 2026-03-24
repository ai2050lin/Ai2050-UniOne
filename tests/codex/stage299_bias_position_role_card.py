#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE295 = PROJECT_ROOT / "tests" / "codex_temp" / "stage295_bias_single_param_causal_sweep_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage299_bias_position_role_card_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def infer_bias_role(row: dict) -> str:
    if row["brand_like"]:
        return "品牌或跨类偏转位"
    if row["role_family"] == "水果偏转位":
        return "对象细粒度偏转位"
    if row["role_family"] == "动物偏转位":
        return "类内竞争偏转位"
    if row["role_family"] == "器物偏转位":
        return "对象域切换偏转位"
    return "任务或义项偏转位"


def build_summary() -> dict:
    s295 = load_json(INPUT_STAGE295)
    rows = []
    for row in s295["position_rows"]:
        role_card = infer_bias_role(row)
        leverage = row["causal_effect"] * (1.0 + 0.08 * row["selectivity"])
        rows.append(
            {
                **row,
                "role_card": role_card,
                "leverage": leverage,
            }
        )
    rows.sort(key=lambda item: (-item["leverage"], -item["causal_effect"], item["dim_index"]))
    top_rows = rows[:12]
    role_counter = {}
    for row in top_rows:
        role_counter[row["role_card"]] = role_counter.get(row["role_card"], 0) + 1

    role_score = (
        float(s295["causal_score"]) * 0.50
        + sum(row["leverage"] for row in top_rows) / max(1, len(top_rows)) * 0.50
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage299_bias_position_role_card",
        "title": "偏置偏转位逐位角色卡",
        "status_short": "bias_position_role_card_ready",
        "role_score": float(role_score),
        "role_counter": role_counter,
        "strongest_dim_index": int(top_rows[0]["dim_index"]),
        "top_gap_name": "偏置位已经可以区分成对象、类内竞争、品牌或跨类等偏转角色，但任务偏转位仍然偏少",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_card": row["role_card"],
                "role_name": row["role_name"],
                "role_family": row["role_family"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "selectivity": row["selectivity"],
                "causal_effect": row["causal_effect"],
                "leverage": row["leverage"],
            }
            for row in top_rows
        ],
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
    parser = argparse.ArgumentParser(description="偏置偏转位逐位角色卡")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
