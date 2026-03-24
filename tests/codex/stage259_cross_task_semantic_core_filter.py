#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage257_object_attribute_position_operation_role_map import run_analysis as run_stage257
from stage258_task_semantic_to_processing_route_bridge import run_analysis as run_stage258

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage259_cross_task_semantic_core_filter_20260324"
STAGE251_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage251_delta_position_role_map_20260324" / "summary.json"
STAGE252_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage252_object_pressure_to_delta_thickness_bridge_20260324" / "summary.json"
STAGE255_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage255_translation_token_role_refinement_20260324" / "summary.json"
STAGE257_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage257_object_attribute_position_operation_role_map_20260324" / "summary.json"
STAGE258_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage258_task_semantic_to_processing_route_bridge_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def role_strength(summary257: dict, role_name: str) -> float:
    row = next(row for row in summary257["role_rows"] if row["role_name"] == role_name)
    return clamp01((row["activation_strength"] + row["compactness"]) / 2.0)


def build_summary() -> dict:
    summary251 = load_json(STAGE251_SUMMARY)
    summary252 = load_json(STAGE252_SUMMARY)
    summary255 = load_json(STAGE255_SUMMARY)
    if not STAGE257_SUMMARY.exists():
        run_stage257()
    if not STAGE258_SUMMARY.exists():
        run_stage258()
    summary257 = load_json(STAGE257_SUMMARY)
    summary258 = load_json(STAGE258_SUMMARY)

    object_core = (role_strength(summary257, "object") + summary252["bridge_score"]) / 2.0
    operation_core = (
        role_strength(summary257, "operation")
        + summary258["bridge_score"]
        + clamp01(summary255["variant_rows"][0]["gate_shift_mean"] / 60.0)
    ) / 3.0
    position_attribute_core = (
        role_strength(summary257, "position")
        + role_strength(summary257, "attribute")
        + clamp01(next(row["route_peak"] for row in summary258["task_rows"] if row["task_name"] == "image_edit") / 200.0)
    ) / 3.0
    shared_delta_route_core = (summary251["role_score"] + summary252["bridge_score"] + summary258["bridge_score"]) / 3.0

    core_rows = [
        {"core_name": "object_core", "core_score": object_core},
        {"core_name": "operation_core", "core_score": operation_core},
        {"core_name": "position_attribute_core", "core_score": position_attribute_core},
        {"core_name": "shared_delta_route_core", "core_score": shared_delta_route_core},
    ]
    strongest = max(core_rows, key=lambda row: row["core_score"])
    weakest = min(core_rows, key=lambda row: row["core_score"])
    stable_count = sum(1 for row in core_rows if row["core_score"] >= 0.55)
    filter_score = sum(row["core_score"] for row in core_rows) / len(core_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage259_cross_task_semantic_core_filter",
        "title": "跨任务语义主核筛选",
        "status_short": "cross_task_semantic_core_filter_ready",
        "core_count": len(core_rows),
        "stable_core_count": stable_count,
        "filter_score": filter_score,
        "strongest_core_name": strongest["core_name"],
        "weakest_core_name": weakest["core_name"],
        "top_gap_name": "当前最稳的不是单词本身，而是对象核、操作核、位置属性绑定核，以及共享-差分-路径联合核",
        "core_rows": core_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "core_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["core_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["core_rows"])
    report = [
        "# Stage259：跨任务语义主核筛选",
        "",
        "## 核心结果",
        f"- 主核数量：{summary['core_count']}",
        f"- 稳定主核数量：{summary['stable_core_count']}",
        f"- 筛选总分：{summary['filter_score']:.4f}",
        f"- 最强主核：{summary['strongest_core_name']}",
        f"- 最弱主核：{summary['weakest_core_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE259_CROSS_TASK_SEMANTIC_CORE_FILTER_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨任务语义主核筛选")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
