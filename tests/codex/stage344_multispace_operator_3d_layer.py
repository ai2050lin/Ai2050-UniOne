#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage337_multi_space_role_raw_alignment import run_analysis as run_stage337
from stage332_local_operator_stitching_map import run_analysis as run_stage332


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage344_multispace_operator_3d_layer_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s337 = run_stage337(force=False)
    s332 = run_stage332(force=False)

    role_nodes = []
    for idx, row in enumerate(s337["alignment_rows"]):
        role_nodes.append(
            {
                "id": f"role_node_{idx}",
                "space_name": row["space_name"],
                "carrier_cluster": row["carrier_cluster"],
                "bias_direction": row["bias_direction"],
                "alignment_strength": float(row["alignment_strength"]),
                "position": {
                    "x": float(idx) * 2.0,
                    "y": float(row["alignment_strength"]) * 10.0,
                    "z": float(idx),
                },
            }
        )

    operator_parts = []
    for row in s332["stitch_rows"]:
        operator_parts.append(
            {
                "part_name": row["part_name"],
                "strength": float(row["strength"]),
            }
        )

    layer_score = float(s337["alignment_score"]) * 0.5 + float(s332["stitching_score"]) * 0.5

    return {
        "schema_version": "agi_3d_scene_layer.v1",
        "experiment_id": "stage344_multispace_operator_3d_layer",
        "title": "多空间角色与局部运算元层 3D 场景",
        "status_short": "multispace_operator_3d_layer_ready",
        "layer_name": "multispace_operator_layer",
        "layer_score": layer_score,
        "scene_axes": {
            "x": "空间编号",
            "y": "对齐强度",
            "z": "局部运算元拼接深度",
        },
        "role_nodes": role_nodes,
        "operator_parts": operator_parts,
        "top_gap_name": "多空间角色和局部运算元已经能放进同一层显示，但统一拼接关系仍然偏薄。",
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
    parser = argparse.ArgumentParser(description="多空间角色与局部运算元层 3D 场景")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
