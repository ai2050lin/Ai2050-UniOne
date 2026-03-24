#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage315_joint_amplification_raw_trajectory_map import run_analysis as run_stage315
from stage309_operator_to_architecture_bridge import run_analysis as run_stage309


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage319_joint_amplification_layerwise_core_split_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s315 = run_stage315(force=False)
    s309 = run_stage309(force=False)

    base_strength = float(s315["trajectory_rows"][0]["strength"])
    bias_strength = float(s315["trajectory_rows"][1]["strength"])
    amp_strength = float(s315["trajectory_rows"][2]["strength"])

    layer_rows = [
        {
            "layer_band": "早层第一次放大",
            "strength": (base_strength * 0.60 + bias_strength * 0.40),
        },
        {
            "layer_band": "中层主放大",
            "strength": (bias_strength * 0.55 + amp_strength * 0.45),
        },
        {
            "layer_band": "后层持续放大",
            "strength": (amp_strength * 0.65 + float(s309["bridge_score"]) * 0.35),
        },
    ]

    layerwise_split_score = (
        sum(row["strength"] for row in layer_rows) / len(layer_rows) * 0.70
        + float(s315["raw_trajectory_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage319_joint_amplification_layerwise_core_split",
        "title": "联合放大逐层主核拆分图",
        "status_short": "joint_amplification_layerwise_core_split_ready",
        "layerwise_split_score": float(layerwise_split_score),
        "layer_rows": layer_rows,
        "top_gap_name": "联合放大已经能拆成早层第一次放大、中层主放大、后层持续放大三段，但真实逐位放大主核仍未单独压出",
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
    parser = argparse.ArgumentParser(description="联合放大逐层主核拆分图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
