#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage322_bias_deflection_task_competition_expansion import run_analysis as run_stage322
from stage339_bias_deflection_raw_trajectory_review import run_analysis as run_stage339


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage342_bias_deflection_3d_layer_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s322 = run_stage322(force=False)
    s339 = run_stage339(force=False)

    direction_nodes = []
    for idx, row in enumerate(s339["trajectory_rows"]):
        direction_nodes.append(
            {
                "id": f"bias_direction_{idx}",
                "trajectory_name": row["trajectory_name"],
                "member_count": int(row["member_count"]),
                "mean_selectivity": float(row["mean_selectivity"]),
                "position": {
                    "x": float(idx) * 2.0,
                    "y": float(row["mean_selectivity"]) * 10.0,
                    "z": float(row["member_count"]) * 1.5,
                },
                "visual": {
                    "shape": "arrow_cluster",
                    "color_role": "bias_deflection",
                    "thickness": 0.5 + float(row["mean_selectivity"]),
                },
            }
        )

    task_bias_rows = []
    for row in s322["task_rows"]:
        strongest_axis = "操作偏转" if float(row["operation_strength"]) >= float(row["constraint_strength"]) else "约束偏转"
        task_bias_thickness = max(float(row["operation_strength"]), float(row["constraint_strength"]))
        task_bias_rows.append(
            {
                "display_name": row["display_name"],
                "strongest_axis": strongest_axis,
                "task_bias_thickness": task_bias_thickness,
            }
        )

    layer_score = float(s339["review_score"]) * 0.65 + float(s322["task_competition_score"]) * 0.35

    return {
        "schema_version": "agi_3d_scene_layer.v1",
        "experiment_id": "stage342_bias_deflection_3d_layer",
        "title": "偏置偏转层 3D 场景",
        "status_short": "bias_deflection_3d_layer_ready",
        "layer_name": "bias_deflection_layer",
        "layer_score": layer_score,
        "scene_axes": {
            "x": "偏转轨迹编号",
            "y": "平均选择性",
            "z": "轨迹成员数",
        },
        "direction_nodes": direction_nodes,
        "task_bias_rows": task_bias_rows,
        "top_gap_name": "对象和类内竞争的偏转方向已经适合直接可视化，任务偏转虽然出现，但厚度仍然偏薄。",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "scene_layer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="偏置偏转层 3D 场景")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
