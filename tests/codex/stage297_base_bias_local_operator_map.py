#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage294_shared_base_single_param_causal_sweep import run_analysis as run_stage294
from stage295_bias_single_param_causal_sweep import run_analysis as run_stage295
from stage296_base_fixed_bias_swap_experiment import run_analysis as run_stage296


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage297_base_bias_local_operator_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s294 = run_stage294(force=False)
    s295 = run_stage295(force=False)
    s296 = run_stage296(force=False)

    operator_rows = [
        {
            "operator_name": "共享承载运算元",
            "source_dim_index": s294["position_rows"][0]["dim_index"],
            "operator_score": s294["position_rows"][0]["causal_effect"],
            "operator_function": "维持通用家族骨架和跨对象公共底盘",
        },
        {
            "operator_name": "家族偏转运算元",
            "source_dim_index": s295["position_rows"][0]["dim_index"],
            "operator_score": s295["position_rows"][0]["causal_effect"],
            "operator_function": "把共享底盘拨向水果、动物、品牌或器物方向",
        },
        {
            "operator_name": "底盘偏置联动运算元",
            "source_dim_index": s296["best_bias_dim_index"],
            "operator_score": max(row["swap_effect"] for row in s296["experiment_rows"]),
            "operator_function": "固定基底后，通过少量偏置替换实现对象或任务切换",
        },
    ]

    operator_score = (
        float(s294["causal_score"]) * 0.30
        + float(s295["causal_score"]) * 0.30
        + float(s296["experiment_score"]) * 0.40
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage297_base_bias_local_operator_map",
        "title": "基底偏置局部运算元图",
        "status_short": "base_bias_local_operator_map_ready",
        "operator_score": float(operator_score),
        "operator_count": len(operator_rows),
        "top_gap_name": "当前最接近的局部编码原段不是单概念参数，而是共享承载位与稀疏偏转位共同组成的局部运算元",
        "operator_rows": operator_rows,
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
    parser = argparse.ArgumentParser(description="基底偏置局部运算元图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
