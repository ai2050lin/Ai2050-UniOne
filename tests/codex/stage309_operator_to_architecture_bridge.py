#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage305_neuron_level_operator_decomposition import run_analysis as run_stage305
from stage306_neuron_level_math_principle_summary import run_analysis as run_stage306
from stage307_cross_family_shared_base_core_compression import run_analysis as run_stage307
from stage308_task_bias_core_compression import run_analysis as run_stage308


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage309_operator_to_architecture_bridge_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s305 = run_stage305(force=False)
    s306 = run_stage306(force=False)
    s307 = run_stage307(force=False)
    s308 = run_stage308(force=False)

    bridge_score = (
        float(s305["decomposition_score"]) * 0.30
        + float(s306["principle_score"]) * 0.30
        + float(s307["compression_score"]) * 0.20
        + float(s308["compression_score"]) * 0.20
    )

    architecture_template = [
        "共享承载层：少量高复用主核位，负责托住通用家族骨架",
        "偏置偏转层：少量高杠杆偏转位，负责对象、义项、任务和输出方向切换",
        "联合放大层：把局部偏转扩展成完整功能差异",
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage309_operator_to_architecture_bridge",
        "title": "局部运算元到网络架构桥",
        "status_short": "operator_to_architecture_bridge_ready",
        "bridge_score": float(bridge_score),
        "architecture_template": architecture_template,
        "top_gap_name": "当前已经能把三算子压成网络原型模板，但离真正可替代现有网络的完整架构仍缺少闭合层和跨模型硬主核",
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
    parser = argparse.ArgumentParser(description="局部运算元到网络架构桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
