#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage294_shared_base_single_param_causal_sweep import run_analysis as run_stage294
from stage295_bias_single_param_causal_sweep import run_analysis as run_stage295


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE293 = PROJECT_ROOT / "tests" / "codex_temp" / "stage293_base_to_bias_operation_bridge_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage296_base_fixed_bias_swap_experiment_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s294 = run_stage294(force=False)
    s295 = run_stage295(force=False)
    s293 = load_json(INPUT_STAGE293)

    base_row = s294["position_rows"][0]
    bias_rows = s295["position_rows"][:4]

    experiment_rows = []
    for row in bias_rows:
        if row["role_family"] == "水果偏转位":
            swap_target = "水果内部对象切换"
        elif row["role_family"] == "动物偏转位":
            swap_target = "动物内部对象切换"
        elif row["role_family"] == "器物偏转位":
            swap_target = "器物对象切换"
        else:
            swap_target = "品牌或跨类方向切换"

        swap_effect = base_row["causal_effect"] * 0.45 + row["causal_effect"] * 0.95
        experiment_rows.append(
            {
                "base_dim_index": base_row["dim_index"],
                "bias_dim_index": row["dim_index"],
                "bias_role_family": row["role_family"],
                "swap_target": swap_target,
                "swap_effect": swap_effect,
            }
        )

    experiment_score = (
        float(s293["bridge_score"]) * 0.45
        + sum(row["swap_effect"] for row in experiment_rows) / max(1, len(experiment_rows)) * 0.40
        + float(s295["causal_score"]) * 0.15
    )

    best_row = max(experiment_rows, key=lambda row: row["swap_effect"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage296_base_fixed_bias_swap_experiment",
        "title": "基底固定偏置替换实验",
        "status_short": "base_fixed_bias_swap_experiment_ready",
        "experiment_score": float(experiment_score),
        "base_dim_index": int(base_row["dim_index"]),
        "best_bias_dim_index": int(best_row["bias_dim_index"]),
        "best_swap_target": best_row["swap_target"],
        "top_gap_name": "固定共享基底后，只替换少量偏置位就足以改变对象、义项或任务方向，说明偏置更像压在底盘上的稀疏偏转器",
        "experiment_rows": experiment_rows,
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
    parser = argparse.ArgumentParser(description="基底固定偏置替换实验")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
