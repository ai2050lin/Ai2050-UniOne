#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage319_joint_amplification_layerwise_core_split import run_analysis as run_stage319
from stage327_joint_amplification_independent_core_isolation import run_analysis as run_stage327


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage331_layerwise_relay_independent_core_map_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s319 = run_stage319(force=False)
    s327 = run_stage327(force=False)

    layer_rows = []
    isolated = {row["role_name"]: row for row in s327["isolated_rows"]}
    for row in s319["layer_rows"]:
        role_name = row["layer_band"].replace("早层", "第一次").replace("中层", "中层").replace("后层", "后层")
        matched = None
        for key, value in isolated.items():
            if ("第一次" in key and "第一次" in role_name) or ("中层" in key and "中层" in role_name) or ("后层" in key and "后层" in role_name):
                matched = value
                break
        layer_rows.append(
            {
                "layer_band": row["layer_band"],
                "relay_strength": float(row["strength"]),
                "independent_gain": 0.0 if matched is None else float(matched["independent_gain"]),
                "residual_coupling": 0.0 if matched is None else float(matched["residual_coupling"]),
            }
        )

    relay_score = (
        sum(float(row["relay_strength"]) for row in layer_rows) / len(layer_rows) * 0.50
        + sum(float(row["independent_gain"]) for row in layer_rows) / len(layer_rows) * 0.30
        + (1.0 - sum(float(row["residual_coupling"]) for row in layer_rows) / len(layer_rows)) * 0.20
    )

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage331_layerwise_relay_independent_core_map",
        "title": "层间接力独立主核图",
        "status_short": "layerwise_relay_independent_core_map_ready",
        "relay_score": float(relay_score),
        "layer_rows": layer_rows,
        "top_gap_name": "层间接力已经能分段看见，但独立增益仍明显弱于接力整体，说明独立放大核还没完全从耦合结构中剥离",
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
    parser = argparse.ArgumentParser(description="层间接力独立主核图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
