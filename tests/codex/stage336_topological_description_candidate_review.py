#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage329_multi_space_role_mapping import run_analysis as run_stage329
from stage330_fuzzy_carrier_sparse_deflection_joint_map import run_analysis as run_stage330
from stage331_layerwise_relay_independent_core_map import run_analysis as run_stage331
from stage332_local_operator_stitching_map import run_analysis as run_stage332


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage336_topological_description_candidate_review_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s329 = run_stage329(force=False)
    s330 = run_stage330(force=False)
    s331 = run_stage331(force=False)
    s332 = run_stage332(force=False)

    candidate_score = (
        float(s329["mapping_score"]) * 0.20
        + float(s330["joint_score"]) * 0.30
        + float(s331["relay_score"]) * 0.20
        + float(s332["stitching_score"]) * 0.30
    )

    candidate_rows = [
        {"candidate_name": "多空间结构", "strength": float(s329["mapping_score"])},
        {"candidate_name": "模糊承载与稀疏偏转", "strength": float(s330["joint_score"])},
        {"candidate_name": "层间接力", "strength": float(s331["relay_score"])},
        {"candidate_name": "局部拼接", "strength": float(s332["stitching_score"])},
    ]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage336_topological_description_candidate_review",
        "title": "拓扑式描述候选复核",
        "status_short": "topological_description_candidate_review_ready",
        "candidate_score": float(candidate_score),
        "candidate_rows": candidate_rows,
        "top_gap_name": "当前数据已经支持多空间、模糊承载、层间接力和局部拼接四类结构，因此拓扑式描述很有潜力，但仍然缺少更厚的跨模型共同主核",
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
    parser = argparse.ArgumentParser(description="拓扑式描述候选复核")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
