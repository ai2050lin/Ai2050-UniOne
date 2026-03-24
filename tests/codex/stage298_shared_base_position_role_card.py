#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE294 = PROJECT_ROOT / "tests" / "codex_temp" / "stage294_shared_base_single_param_causal_sweep_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage298_shared_base_position_role_card_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def infer_role(row: dict) -> str:
    if row["family_hit_count"] >= 3 and row["base_load"] >= 0.18:
        return "跨家族共享承载位"
    if row["family_hit_count"] >= 2 and "水果" in row["role_name"]:
        return "水果家族共享承载位"
    if row["family_hit_count"] >= 2 and "动物" in row["role_name"]:
        return "动物家族共享承载位"
    return "局部共享承载位"


def build_summary() -> dict:
    s294 = load_json(INPUT_STAGE294)
    rows = []
    for row in s294["position_rows"]:
        role_card = infer_role(row)
        role_stability = row["stability"] * (1.0 + 0.05 * max(0, row["family_hit_count"] - 1))
        rows.append(
            {
                **row,
                "role_card": role_card,
                "role_stability": role_stability,
            }
        )
    rows.sort(key=lambda item: (-item["role_stability"], -item["causal_effect"], item["dim_index"]))
    top_rows = rows[:10]
    role_counter = {}
    for row in top_rows:
        role_counter[row["role_card"]] = role_counter.get(row["role_card"], 0) + 1

    role_score = (
        float(s294["causal_score"]) * 0.45
        + sum(row["role_stability"] for row in top_rows) / max(1, len(top_rows)) * 0.55
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage298_shared_base_position_role_card",
        "title": "共享承载位逐位角色卡",
        "status_short": "shared_base_position_role_card_ready",
        "role_score": float(role_score),
        "role_counter": role_counter,
        "strongest_dim_index": int(top_rows[0]["dim_index"]),
        "top_gap_name": "共享承载位正在从位置簇推进到逐位角色层，但跨家族共享承载位仍然偏少",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_card": row["role_card"],
                "role_name": row["role_name"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "family_hit_count": row["family_hit_count"],
                "causal_effect": row["causal_effect"],
                "role_stability": row["role_stability"],
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
    parser = argparse.ArgumentParser(description="共享承载位逐位角色卡")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
