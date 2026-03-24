#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage300_shared_base_bias_joint_causal_map import run_analysis as run_stage300
from stage309_operator_to_architecture_bridge import run_analysis as run_stage309


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage315_joint_amplification_raw_trajectory_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s300 = run_stage300(force=False)
    s309 = run_stage309(force=False)

    trajectory_rows = [
        {"stage_name": "共享承载段", "strength": 0.16813853764247405},
        {"stage_name": "偏置偏转段", "strength": 0.2281730119151726},
        {"stage_name": "联合放大段", "strength": float(s300["joint_effect"])},
    ]

    raw_trajectory_score = (
        float(s300["joint_score"]) * 0.55
        + float(s309["bridge_score"]) * 0.45
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage315_joint_amplification_raw_trajectory_map",
        "title": "联合放大原始层间轨迹图",
        "status_short": "joint_amplification_raw_trajectory_map_ready",
        "raw_trajectory_score": float(raw_trajectory_score),
        "trajectory_rows": trajectory_rows,
        "top_gap_name": "联合放大已经能被压成连续轨迹，但真实层间放大主核仍未逐位拆清",
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
    parser = argparse.ArgumentParser(description="联合放大原始层间轨迹图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
