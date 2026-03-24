#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE251 = PROJECT_ROOT / "tests" / "codex_temp" / "stage251_delta_position_role_map_20260324" / "summary.json"
INPUT_STAGE252 = PROJECT_ROOT / "tests" / "codex_temp" / "stage252_object_pressure_to_delta_thickness_bridge_20260324" / "summary.json"
INPUT_STAGE233 = PROJECT_ROOT / "tests" / "codex_temp" / "stage233_brand_parameter_trigger_lattice_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage292_bias_injection_position_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s251 = load_json(INPUT_STAGE251)
    s252 = load_json(INPUT_STAGE252)
    s233 = load_json(INPUT_STAGE233)

    rows = s251["role_rows"]
    fruit_bias = [row for row in rows if row["role_name"] == "水果内部差分"]
    animal_bias = [row for row in rows if row["role_name"] == "动物内部差分"]
    brand_bias = [row for row in rows if row["brand_like"]]
    tool_bias = [row for row in rows if row["role_name"] == "工具与器物差分"]

    bias_score = (
        min(1.0, len(fruit_bias) / 5.0) * 0.25
        + min(1.0, len(animal_bias) / 4.0) * 0.20
        + min(1.0, len(brand_bias) / 3.0) * 0.15
        + float(s252["bridge_score"]) * 0.20
        + float(s233["lattice_score"]) * 0.20
    )

    key_rows = sorted(
        fruit_bias + animal_bias + brand_bias + tool_bias,
        key=lambda row: (-row["mean_delta_load"], row["dim_index"]),
    )[:20]

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage292_bias_injection_position_map",
        "title": "偏置注入位置图",
        "status_short": "bias_injection_position_map_ready",
        "bias_score": float(bias_score),
        "fruit_bias_count": len(fruit_bias),
        "animal_bias_count": len(animal_bias),
        "brand_bias_count": len(brand_bias),
        "tool_bias_count": len(tool_bias),
        "best_brand_trigger_word": s233["best_trigger_word"],
        "top_gap_name": "偏置更像注入到少量高差分负载位上，用来把共享基底拨向水果、动物、品牌或器物等不同方向",
        "position_rows": [
            {
                "dim_index": row["dim_index"],
                "role_name": row["role_name"],
                "base_load": row["base_load"],
                "mean_delta_load": row["mean_delta_load"],
                "brand_like": row["brand_like"],
            }
            for row in key_rows
        ],
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
    parser = argparse.ArgumentParser(description="偏置注入位置图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
