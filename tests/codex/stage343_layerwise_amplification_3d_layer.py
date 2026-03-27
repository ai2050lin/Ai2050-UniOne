#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage340_layerwise_relay_stitching_review import run_analysis as run_stage340


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage343_layerwise_amplification_3d_layer_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s340 = run_stage340(force=False)

    relay_bands = []
    for idx, row in enumerate(s340["layer_rows"]):
        relay_bands.append(
            {
                "id": f"relay_band_{idx}",
                "layer_band": row["layer_band"],
                "relay_strength": float(row["relay_strength"]),
                "independent_gain": float(row["independent_gain"]),
                "residual_coupling": float(row["residual_coupling"]),
                "path_points": [
                    {"x": float(idx) * 2.0, "y": 0.0, "z": 0.0},
                    {"x": float(idx) * 2.0, "y": float(row["relay_strength"]) * 10.0, "z": float(row["independent_gain"]) * 10.0},
                ],
            }
        )

    return {
        "schema_version": "agi_3d_scene_layer.v1",
        "experiment_id": "stage343_layerwise_amplification_3d_layer",
        "title": "逐层放大层 3D 场景",
        "status_short": "layerwise_amplification_3d_layer_ready",
        "layer_name": "layerwise_amplification_layer",
        "layer_score": float(s340["review_score"]),
        "scene_axes": {
            "x": "层带编号",
            "y": "接力强度",
            "z": "独立增益",
        },
        "relay_bands": relay_bands,
        "top_gap_name": "放大已经能拆成三段，但独立增益仍然没有完全从接力整体中剥离出来。",
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
    parser = argparse.ArgumentParser(description="逐层放大层 3D 场景")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
