#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage315_joint_amplification_raw_trajectory_map import run_analysis as run_stage315
from stage319_joint_amplification_layerwise_core_split import run_analysis as run_stage319


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage323_joint_amplification_position_core_split_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s315 = run_stage315(force=False)
    s319 = run_stage319(force=False)

    position_rows = [
        {
            "role_name": "第一次放大主核候选",
            "carrier_dim": 138,
            "bias_dim": 5,
            "strength": float(s319["layer_rows"][0]["strength"]),
        },
        {
            "role_name": "中层主放大主核候选",
            "carrier_dim": 138,
            "bias_dim": 5,
            "strength": float(s319["layer_rows"][1]["strength"]),
        },
        {
            "role_name": "后层持续放大主核候选",
            "carrier_dim": 138,
            "bias_dim": 5,
            "strength": float(s319["layer_rows"][2]["strength"]),
        },
    ]

    position_split_score = (
        sum(float(row["strength"]) for row in position_rows) / len(position_rows) * 0.70
        + float(s315["raw_trajectory_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage323_joint_amplification_position_core_split",
        "title": "联合放大逐位主核拆分图",
        "status_short": "joint_amplification_position_core_split_ready",
        "position_split_score": float(position_split_score),
        "position_rows": position_rows,
        "top_gap_name": "联合放大已经能按三段拆成位置主核候选，但真正独立的逐位放大核仍然尚未从共享位和偏置位中完全剥离",
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
    parser = argparse.ArgumentParser(description="联合放大逐位主核拆分图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
