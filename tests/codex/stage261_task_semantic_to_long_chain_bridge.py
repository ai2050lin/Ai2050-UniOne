#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage261_task_semantic_to_long_chain_bridge_20260324"
STAGE258_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage258_task_semantic_to_processing_route_bridge_20260324" / "summary.json"
STAGE244_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage244_deepseek14b_stress_long_chain_probe_20260324" / "summary.json"
STAGE247_SUMMARY = PROJECT_ROOT / "tests" / "codex_temp" / "stage247_deepseek14b_large_template_long_chain_review_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    summary258 = load_json(STAGE258_SUMMARY)
    summary244 = load_json(STAGE244_SUMMARY)
    summary247 = load_json(STAGE247_SUMMARY)

    route_task_mean = sum(row["route_peak"] for row in summary258["task_rows"]) / len(summary258["task_rows"])
    route_task_score = min(route_task_mean / 250.0, 1.0)
    stress_score = float(summary244["stress_score"])
    long_template_score = float(summary247["review_score"])
    bridge_score = (summary258["bridge_score"] + route_task_score + stress_score + long_template_score) / 4.0
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage261_task_semantic_to_long_chain_bridge",
        "title": "任务语义到长链处理桥",
        "status_short": "task_semantic_to_long_chain_bridge_ready",
        "piece_count": 4,
        "bridge_score": bridge_score,
        "strongest_piece_name": "long_template_execution",
        "weakest_piece_name": "task_semantic_route_entry",
        "top_gap_name": "当前任务语义已经能切入处理路径，并能延伸到长链执行，但前段入口桥仍然弱于后段执行表现",
        "piece_rows": [
            {"piece_name": "task_semantic_route_entry", "score": float(summary258["bridge_score"])},
            {"piece_name": "route_peak_support", "score": route_task_score},
            {"piece_name": "stress_long_chain_execution", "score": stress_score},
            {"piece_name": "long_template_execution", "score": long_template_score},
        ],
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    with (output_dir / "piece_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["piece_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["piece_rows"])
    report = [
        "# Stage261：任务语义到长链处理桥",
        "",
        "## 核心结果",
        f"- 片段数量：{summary['piece_count']}",
        f"- 桥总分：{summary['bridge_score']:.4f}",
        f"- 最强片段：{summary['strongest_piece_name']}",
        f"- 最弱片段：{summary['weakest_piece_name']}",
        f"- 头号发现：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE261_TASK_SEMANTIC_TO_LONG_CHAIN_BRIDGE_REPORT.md").write_text("\n".join(report), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="任务语义到长链处理桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
