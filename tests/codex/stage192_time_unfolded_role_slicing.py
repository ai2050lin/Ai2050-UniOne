#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage192_time_unfolded_role_slicing_20260323"

STAGE123_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json"
STAGE124_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json"
STAGE127_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage127_noun_context_neuron_probe_20260323" / "summary.json"
STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE174_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage174_recovery_closure_equation_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_slice(score: float) -> str:
    if score >= 0.7:
        return "稳定切片"
    if score >= 0.5:
        return "过渡切片"
    return "薄弱切片"


def build_summary() -> dict:
    s123 = load_json(STAGE123_SUMMARY_PATH)
    s124 = load_json(STAGE124_SUMMARY_PATH)
    s127 = load_json(STAGE127_SUMMARY_PATH)
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s174 = load_json(STAGE174_SUMMARY_PATH)

    slice_rows = [
        {
            "slice_name": "早层定锚切片",
            "layer_index": int(s127["dominant_general_layer_index"]),
            "score": float(s127["dominant_general_layer_score"]),
        },
        {
            "slice_name": "中段选路切片",
            "layer_index": int(s123["dominant_layer_index"]),
            "score": float(s123["route_shift_layer_localization_score"]),
        },
        {
            "slice_name": "后层聚合切片",
            "layer_index": int(s124["dominant_general_layer_index"]),
            "score": float(s124["dominant_general_layer_score"]),
        },
        {
            "slice_name": "来源痕迹切片",
            "layer_index": -1,
            "score": float(s172["provenance_trace_score"]),
        },
        {
            "slice_name": "回收闭合切片",
            "layer_index": -1,
            "score": float(s174["closure_score"]),
        },
    ]
    for row in slice_rows:
        row["status"] = classify_slice(float(row["score"]))
    slice_continuity_score = sum(float(row["score"]) for row in slice_rows) / float(len(slice_rows))
    strongest_slice_name = max(slice_rows, key=lambda row: float(row["score"]))["slice_name"]
    weakest_slice_name = min(slice_rows, key=lambda row: float(row["score"]))["slice_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage192_time_unfolded_role_slicing",
        "title": "时间展开角色切片",
        "status_short": "time_unfolded_role_slicing_ready",
        "slice_count": len(slice_rows),
        "earliest_slice_name": "早层定锚切片",
        "middle_slice_name": "中段选路切片",
        "latest_slice_name": "后层聚合切片",
        "strongest_slice_name": strongest_slice_name,
        "weakest_slice_name": weakest_slice_name,
        "slice_continuity_score": slice_continuity_score,
        "slice_rows": slice_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage192：时间展开角色切片",
        "",
        "## 核心结果",
        f"- 切片数量：{summary['slice_count']}",
        f"- 最强切片：{summary['strongest_slice_name']}",
        f"- 最弱切片：{summary['weakest_slice_name']}",
        f"- 切片连续性分数：{summary['slice_continuity_score']:.4f}",
    ]
    (output_dir / "STAGE192_TIME_UNFOLDED_ROLE_SLICING_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="时间展开角色切片")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
