#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage298_shared_base_position_role_card import run_analysis as run_stage298
from stage299_bias_position_role_card import run_analysis as run_stage299
from stage296_base_fixed_bias_swap_experiment import run_analysis as run_stage296


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage300_shared_base_bias_joint_causal_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s298 = run_stage298(force=False)
    s299 = run_stage299(force=False)
    s296 = run_stage296(force=False)

    base_row = s298["position_rows"][0]
    bias_row = s299["position_rows"][0]
    joint_effect = (
        base_row["causal_effect"] * 0.40
        + bias_row["causal_effect"] * 0.35
        + max(row["swap_effect"] for row in s296["experiment_rows"]) * 0.25
    )
    joint_score = (
        float(s298["role_score"]) * 0.30
        + float(s299["role_score"]) * 0.35
        + float(s296["experiment_score"]) * 0.35
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage300_shared_base_bias_joint_causal_map",
        "title": "共享承载与偏置偏转联合因果图",
        "status_short": "shared_base_bias_joint_causal_map_ready",
        "joint_score": float(joint_score),
        "joint_effect": float(joint_effect),
        "base_dim_index": int(base_row["dim_index"]),
        "bias_dim_index": int(bias_row["dim_index"]),
        "base_role_card": base_row["role_card"],
        "bias_role_card": bias_row["role_card"],
        "top_gap_name": "当前最接近的联合因果结构是：共享承载位先托住通用骨架，再由偏置偏转位在局部高杠杆位置上改变对象、义项或任务方向",
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
    parser = argparse.ArgumentParser(description="共享承载与偏置偏转联合因果图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
