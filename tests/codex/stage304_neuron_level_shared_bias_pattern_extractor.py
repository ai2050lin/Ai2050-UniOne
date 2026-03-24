#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage298_shared_base_position_role_card import run_analysis as run_stage298
from stage299_bias_position_role_card import run_analysis as run_stage299
from stage300_shared_base_bias_joint_causal_map import run_analysis as run_stage300
from stage301_cross_family_shared_base_compression import run_analysis as run_stage301
from stage302_task_bias_position_strengthening import run_analysis as run_stage302
from stage303_shared_base_bias_cross_model_joint_review import run_analysis as run_stage303


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage304_neuron_level_shared_bias_pattern_extractor_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s298 = run_stage298(force=False)
    s299 = run_stage299(force=False)
    s300 = run_stage300(force=False)
    s301 = run_stage301(force=False)
    s302 = run_stage302(force=False)
    s303 = run_stage303(force=False)

    shared_core_dims = [row["dim_index"] for row in s298["position_rows"][:4]]
    bias_core_dims = [row["dim_index"] for row in s299["position_rows"][:6]]

    extraction_score = (
        float(s298["role_score"]) * 0.18
        + float(s299["role_score"]) * 0.18
        + float(s300["joint_score"]) * 0.18
        + float(s301["compression_score"]) * 0.14
        + float(s302["strengthening_score"]) * 0.14
        + float(s303["review_score"]) * 0.18
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage304_neuron_level_shared_bias_pattern_extractor",
        "title": "神经元级共享承载与偏置偏转规律提取器",
        "status_short": "neuron_level_shared_bias_pattern_extractor_ready",
        "extraction_score": float(extraction_score),
        "shared_core_count": len(shared_core_dims),
        "bias_core_count": len(bias_core_dims),
        "shared_core_dims": shared_core_dims,
        "bias_core_dims": bias_core_dims,
        "algorithm_steps": [
            "先从共享承载位角色卡中筛选高复用、高稳定的共享承载候选位",
            "再从偏置偏转位角色卡中筛选高选择性、高杠杆的偏转候选位",
            "对共享位和偏置位分别做单参数因果扫描",
            "执行基底固定、偏置替换实验，观察对象、义项或任务方向是否被单独改写",
            "把共享位、偏置位和联合因果链压缩到更少的局部运算元",
            "最后做跨模型复核，只保留两边都稳定成立的候选主核",
        ],
        "top_gap_name": "当前已经能把共享承载位和偏置偏转位提取成候选规律，但距离真正逐神经元硬主核还差跨家族共享位压缩和任务偏转位补强",
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
    parser = argparse.ArgumentParser(description="神经元级共享承载与偏置偏转规律提取器")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
