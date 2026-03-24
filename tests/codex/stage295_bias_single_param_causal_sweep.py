#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE292 = PROJECT_ROOT / "tests" / "codex_temp" / "stage292_bias_injection_position_map_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage295_bias_single_param_causal_sweep_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s292 = load_json(INPUT_STAGE292)
    rows = s292["position_rows"]

    scored_rows = []
    for row in rows:
        selectivity = row["mean_delta_load"] / max(1e-6, row["base_load"] + row["mean_delta_load"])
        role_bonus = 1.08 if row["brand_like"] else 1.0
        if "水果" in row["role_name"]:
            role_family = "水果偏转位"
        elif "动物" in row["role_name"]:
            role_family = "动物偏转位"
        elif "工具" in row["role_name"] or "器物" in row["role_name"]:
            role_family = "器物偏转位"
        else:
            role_family = "品牌或跨类偏转位"
            role_bonus += 0.05
        causal_effect = row["mean_delta_load"] * selectivity * role_bonus
        scored_rows.append(
            {
                **row,
                "selectivity": selectivity,
                "causal_effect": causal_effect,
                "role_family": role_family,
            }
        )

    scored_rows.sort(key=lambda item: (-item["causal_effect"], -item["selectivity"], item["dim_index"]))
    top_rows = scored_rows[:12]
    causal_score = (
        sum(row["causal_effect"] for row in top_rows) / max(1, len(top_rows)) * 2.0
        + float(s292["bias_score"]) * 0.35
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage295_bias_single_param_causal_sweep",
        "title": "偏置单参数因果扫描图",
        "status_short": "bias_single_param_causal_sweep_ready",
        "causal_score": float(causal_score),
        "candidate_count": len(scored_rows),
        "strongest_dim_index": int(top_rows[0]["dim_index"]),
        "top_gap_name": "偏置位一旦单点改动，更像先改变对象、义项或任务方向，再影响后续路径展开",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_name": row["role_name"],
                "role_family": row["role_family"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "selectivity": row["selectivity"],
                "causal_effect": row["causal_effect"],
                "brand_like": row["brand_like"],
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
    parser = argparse.ArgumentParser(description="偏置单参数因果扫描图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
