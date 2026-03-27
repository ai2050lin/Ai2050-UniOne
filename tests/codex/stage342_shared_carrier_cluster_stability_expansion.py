#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage338_fuzzy_carrier_cluster_stability_review import run_analysis as run_stage338
from stage321_shared_carrier_cross_task_raw_coverage import run_analysis as run_stage321
from stage324_first_principles_cross_model_reinforced_review import run_analysis as run_stage324


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage342_shared_carrier_cluster_stability_expansion_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s338 = run_stage338(force=False)
    s321 = run_stage321(force=False)
    s324 = run_stage324(force=False)

    stability_rows = list(s338["stability_rows"])
    task_rows = list(s321["task_rows"])

    task_bridge_mean = sum(float(row["shared_base_bridge"]) for row in task_rows) / max(1, len(task_rows))
    model_stability = float(s324["cross_model_score"])
    object_stability = sum(float(row["cluster_stability"]) for row in stability_rows) / max(1, len(stability_rows))

    expansion_rows = [
        {"axis_name": "跨对象稳定性", "strength": float(object_stability)},
        {"axis_name": "跨任务稳定性", "strength": float(task_bridge_mean)},
        {"axis_name": "跨模型稳定性", "strength": float(model_stability)},
    ]

    expansion_score = sum(float(row["strength"]) for row in expansion_rows) / max(1, len(expansion_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage342_shared_carrier_cluster_stability_expansion",
        "title": "共享承载簇稳定性扩张图",
        "status_short": "shared_carrier_cluster_stability_expansion_ready",
        "expansion_score": float(expansion_score),
        "expansion_rows": expansion_rows,
        "top_gap_name": "共享承载簇已经能跨对象稳定，但跨任务和跨模型稳定性仍然明显弱于跨对象稳定性。",
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
    parser = argparse.ArgumentParser(description="共享承载簇稳定性扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
