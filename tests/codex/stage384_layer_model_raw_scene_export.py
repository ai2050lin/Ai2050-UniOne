#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage381_raw_row_layer_map import run_analysis as run_stage381
from stage382_raw_row_position_map import run_analysis as run_stage382
from stage383_raw_row_parameter_link_map import run_analysis as run_stage383


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage384_layer_model_raw_scene_export_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s381 = run_stage381(force=True)
    s382 = run_stage382(force=True)
    s383 = run_stage383(force=True)

    scene = {
        "version": "agi_layer_raw_scene_v1",
        "layer_nodes": s381["mapped_rows"][:40],
        "position_nodes": s382["mapped_rows"][:40],
        "parameter_nodes": s383["mapped_rows"][:40],
    }
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage384_layer_model_raw_scene_export",
        "title": "层级模型原始场景导出",
        "status_short": "layer_model_raw_scene_ready",
        "layer_node_count": len(scene["layer_nodes"]),
        "position_node_count": len(scene["position_nodes"]),
        "parameter_node_count": len(scene["parameter_nodes"]),
        "scene_path": str(OUTPUT_DIR / "agi_layer_raw_scene_v1.json"),
        "top_gap_name": "当前导出已具备层号、位置类、参数位三类原始节点，可继续接入前端层级模型视图。",
        "scene_preview": {
            "layer_nodes": scene["layer_nodes"][:5],
            "position_nodes": scene["position_nodes"][:5],
            "parameter_nodes": scene["parameter_nodes"][:5],
        },
        "_scene_object": scene,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_object = summary.pop("_scene_object")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "agi_layer_raw_scene_v1.json").write_text(json.dumps(scene_object, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return load_json(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="层级模型原始场景导出")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
