#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage245_large_scale_noun_shared_delta_tensor import run_analysis as run_stage245
from stage292_bias_injection_position_map import run_analysis as run_stage292


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage330_fuzzy_carrier_sparse_deflection_joint_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s245 = run_stage245(force=False)
    s292 = run_stage292(force=False)

    fuzzy_carrier_strength = float(s245["shared_base_mean"])
    sparse_deflection_strength = float(s245["sparse_delta_mean"])
    bias_injection_strength = float(s292["bias_score"])

    joint_score = (
        fuzzy_carrier_strength * 0.35
        + sparse_deflection_strength * 0.30
        + bias_injection_strength * 0.35
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage330_fuzzy_carrier_sparse_deflection_joint_map",
        "title": "模糊承载与稀疏偏转联合图",
        "status_short": "fuzzy_carrier_sparse_deflection_joint_map_ready",
        "joint_score": float(joint_score),
        "fuzzy_carrier_strength": fuzzy_carrier_strength,
        "sparse_deflection_strength": sparse_deflection_strength,
        "bias_injection_strength": bias_injection_strength,
        "top_gap_name": "共享底盘已经表现出模糊承载特征，偏置也表现出稀疏偏转特征，但两者仍然更多是工作性联合，不是统一主核",
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
    parser = argparse.ArgumentParser(description="模糊承载与稀疏偏转联合图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
