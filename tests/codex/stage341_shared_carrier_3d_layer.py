#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage321_shared_carrier_cross_task_raw_coverage import run_analysis as run_stage321
from stage338_fuzzy_carrier_cluster_stability_review import run_analysis as run_stage338


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage341_shared_carrier_3d_layer_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s321 = run_stage321(force=False)
    s338 = run_stage338(force=False)

    cluster_nodes = []
    for idx, row in enumerate(s338["stability_rows"]):
        cluster_nodes.append(
            {
                "id": f"carrier_cluster_{idx}",
                "cluster_name": row["cluster_name"],
                "member_count": int(row["member_count"]),
                "stability": float(row["cluster_stability"]),
                "position": {
                    "x": float(idx) * 2.0,
                    "y": float(row["cluster_stability"]) * 10.0,
                    "z": float(row["member_count"]) * 1.5,
                },
                "visual": {
                    "shape": "cloud",
                    "color_role": "shared_carrier",
                    "size": float(row["member_count"]) + 1.0,
                    "opacity": 0.35 + float(row["cluster_stability"]),
                },
            }
        )

    task_bridges = []
    for idx, row in enumerate(s321["task_rows"]):
        task_bridges.append(
            {
                "id": f"task_bridge_{idx}",
                "task_name": row["task_name"],
                "shared_base_bridge": float(row["shared_base_bridge"]),
                "role_overlap": float(row["role_overlap"]),
            }
        )

    layer_score = (
        float(s338["review_score"]) * 0.6
        + float(s321["raw_cross_task_score"]) * 0.4
    )

    return {
        "schema_version": "agi_3d_scene_layer.v1",
        "experiment_id": "stage341_shared_carrier_3d_layer",
        "title": "共享承载层 3D 场景",
        "status_short": "shared_carrier_3d_layer_ready",
        "layer_name": "shared_carrier_layer",
        "layer_score": layer_score,
        "scene_axes": {
            "x": "共享簇编号",
            "y": "稳定性",
            "z": "成员密度",
        },
        "cluster_nodes": cluster_nodes,
        "task_bridges": task_bridges,
        "top_gap_name": "共享承载簇已经能在 3D 场景中稳定显示，但跨任务共享桥仍然薄于对象簇本身。",
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
    parser = argparse.ArgumentParser(description="共享承载层 3D 场景")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
