#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage298_shared_base_position_role_card import run_analysis as run_stage298
from stage299_bias_position_role_card import run_analysis as run_stage299
from stage300_shared_base_bias_joint_causal_map import run_analysis as run_stage300


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage305_neuron_level_operator_decomposition_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s298 = run_stage298(force=False)
    s299 = run_stage299(force=False)
    s300 = run_stage300(force=False)

    carrier = s298["position_rows"][0]
    bias = s299["position_rows"][0]

    operator_rows = [
        {
            "operator_name": "共享承载算子",
            "source_dim_index": carrier["dim_index"],
            "operator_score": carrier["causal_effect"],
            "operator_function": "维持家族公共骨架和通用状态底盘",
        },
        {
            "operator_name": "偏置偏转算子",
            "source_dim_index": bias["dim_index"],
            "operator_score": bias["causal_effect"],
            "operator_function": "在局部高杠杆位置上改变对象、义项或任务方向",
        },
        {
            "operator_name": "联合放大算子",
            "source_dim_index": s300["bias_dim_index"],
            "operator_score": s300["joint_effect"],
            "operator_function": "把共享承载与偏置偏转接成连续运算链并把局部偏转继续放大",
        },
    ]

    decomposition_score = (
        float(s298["role_score"]) * 0.25
        + float(s299["role_score"]) * 0.30
        + float(s300["joint_score"]) * 0.45
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage305_neuron_level_operator_decomposition",
        "title": "神经元级局部运算分解图",
        "status_short": "neuron_level_operator_decomposition_ready",
        "decomposition_score": float(decomposition_score),
        "operator_count": len(operator_rows),
        "operator_rows": operator_rows,
        "top_gap_name": "当前已经能把局部编码原段分解成共享承载算子、偏置偏转算子和联合放大算子，但还没压到真正逐神经元闭式层",
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
    parser = argparse.ArgumentParser(description="神经元级局部运算分解图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
