#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage304_neuron_level_shared_bias_pattern_extractor import run_analysis as run_stage304
from stage305_neuron_level_operator_decomposition import run_analysis as run_stage305
from stage303_shared_base_bias_cross_model_joint_review import run_analysis as run_stage303


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage306_neuron_level_math_principle_summary_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s304 = run_stage304(force=False)
    s305 = run_stage305(force=False)
    s303 = run_stage303(force=False)

    principle_score = (
        float(s304["extraction_score"]) * 0.35
        + float(s305["decomposition_score"]) * 0.35
        + float(s303["review_score"]) * 0.30
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage306_neuron_level_math_principle_summary",
        "title": "神经元级数学原理工作性摘要",
        "status_short": "neuron_level_math_principle_summary_ready",
        "principle_score": float(principle_score),
        "working_principle": "共享承载位先维持通用家族骨架，偏置偏转位再在少量高杠杆位置上改变对象、义项或任务方向，联合放大算子把局部偏转扩展成完整处理差异",
        "operator_triplet": [
            "共享承载算子",
            "偏置偏转算子",
            "联合放大算子",
        ],
        "top_gap_name": "当前已经形成神经元级工作性数学原理，但还不是最终第一性原理，因为跨家族共享主核和任务偏转位仍未完全压缩",
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
    parser = argparse.ArgumentParser(description="神经元级数学原理工作性摘要")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
