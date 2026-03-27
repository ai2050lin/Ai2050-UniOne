#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage315_joint_amplification_raw_trajectory_map import run_analysis as run_stage315
from stage319_joint_amplification_layerwise_core_split import run_analysis as run_stage319
from stage323_joint_amplification_position_core_split import run_analysis as run_stage323
from stage327_joint_amplification_independent_core_isolation import run_analysis as run_stage327
from stage331_layerwise_relay_independent_core_map import run_analysis as run_stage331
from stage340_layerwise_relay_stitching_review import run_analysis as run_stage340
from stage344_layerwise_amplification_independent_core_review import run_analysis as run_stage344
from stage350_independent_amplification_core_hardening import run_analysis as run_stage350
from stage353_independent_amplification_core_isolation_review import run_analysis as run_stage353
from stage357_independent_amplification_net_gain_review import run_analysis as run_stage357


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage364_layerwise_amplification_raw_expansion_20260325"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s315 = run_stage315(force=False)
    s319 = run_stage319(force=False)
    s323 = run_stage323(force=False)
    s327 = run_stage327(force=False)
    s331 = run_stage331(force=False)
    s340 = run_stage340(force=False)
    s344 = run_stage344(force=False)
    s350 = run_stage350(force=False)
    s353 = run_stage353(force=False)
    s357 = run_stage357(force=False)

    amplification_rows = []

    for row in s315["trajectory_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage315",
                "layer_band": row["stage_name"],
                "metric_name": "trajectory_strength",
                "strength": float(row["strength"]),
            }
        )

    for row in s319["layer_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage319",
                "layer_band": row["layer_band"],
                "metric_name": "core_split_strength",
                "strength": float(row["strength"]),
            }
        )

    for row in s323["position_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage323",
                "layer_band": row["role_name"],
                "metric_name": "position_core_split_strength",
                "strength": float(row["strength"]),
            }
        )

    for row in s327["isolated_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage327",
                "layer_band": row["role_name"],
                "metric_name": "isolated_gain",
                "strength": float(row["independent_gain"]),
            }
        )
        amplification_rows.append(
            {
                "source_stage": "stage327",
                "layer_band": row["role_name"],
                "metric_name": "residual_coupling",
                "strength": float(row["residual_coupling"]),
            }
        )

    for row in s331["layer_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage331",
                "layer_band": row["layer_band"],
                "metric_name": "relay_strength",
                "strength": float(row["relay_strength"]),
            }
        )

    for row in s340["layer_rows"]:
        amplification_rows.extend(
            [
                {
                    "source_stage": "stage340",
                    "layer_band": row["layer_band"],
                    "metric_name": "relay_strength",
                    "strength": float(row["relay_strength"]),
                },
                {
                    "source_stage": "stage340",
                    "layer_band": row["layer_band"],
                    "metric_name": "independent_gain",
                    "strength": float(row["independent_gain"]),
                },
                {
                    "source_stage": "stage340",
                    "layer_band": row["layer_band"],
                    "metric_name": "residual_coupling",
                    "strength": float(row["residual_coupling"]),
                },
            ]
        )

    for row in s344["review_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage344",
                "layer_band": row["part_name"],
                "metric_name": "review_strength",
                "strength": float(row["strength"]),
            }
        )

    for row in s350["hardening_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage350",
                "layer_band": "global",
                "metric_name": row["metric_name"],
                "strength": float(row["strength"]),
            }
        )

    for row in s353["review_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage353",
                "layer_band": "global",
                "metric_name": row["metric_name"],
                "strength": float(row["strength"]),
            }
        )

    for row in s357["review_rows"]:
        amplification_rows.append(
            {
                "source_stage": "stage357",
                "layer_band": "global",
                "metric_name": row["metric_name"],
                "strength": float(row["strength"]),
            }
        )

    expansion_score = sum(float(row["strength"]) for row in amplification_rows) / max(1, len(amplification_rows))

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage364_layerwise_amplification_raw_expansion",
        "title": "逐层放大原始数据扩张图",
        "status_short": "layerwise_amplification_raw_expansion_ready",
        "expansion_score": float(expansion_score),
        "amplification_rows": amplification_rows,
        "top_gap_name": "当前阶段已将逐层放大原始行扩展到轨迹层、逐层拆分层、独立增益层和净增益层。",
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
    parser = argparse.ArgumentParser(description="逐层放大原始数据扩张图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
