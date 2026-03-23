#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage166_category_fiber_map import run_analysis as run_stage166_analysis
from stage167_difference_boundary_equation import run_analysis as run_stage167_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage168_fiber_route_recovery_closure_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE168_FIBER_ROUTE_RECOVERY_CLOSURE_REPORT.md"

STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"
STAGE167_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage167_difference_boundary_equation_20260323" / "summary.json"
STAGE165_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage165_apple_to_language_kernel_bridge_20260323" / "summary.json"
STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    if not STAGE166_SUMMARY_PATH.exists():
        run_stage166_analysis(force=True)
    if not STAGE167_SUMMARY_PATH.exists():
        run_stage167_analysis(force=True)
    s157 = load_json(STAGE157_SUMMARY_PATH)
    s158 = load_json(STAGE158_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s165 = load_json(STAGE165_SUMMARY_PATH)
    s166 = load_json(STAGE166_SUMMARY_PATH)
    s167 = load_json(STAGE167_SUMMARY_PATH)

    fiber_score = float(s166["category_fiber_score"])
    boundary_score = float(s167["equation_score"])
    route_score = float(s157["apple_action_route_score"])
    recovery_score = float(s158["apple_result_binding_score"])
    repair_score = float(s160["apple_result_repair_score"])
    bridge_score = float(s165["bridge_score"])

    best_formula = "fiber_route_recovery = 0.18*fiber + 0.14*boundary + 0.24*route + 0.14*recovery + 0.18*repair + 0.12*bridge"
    closure_score = clamp01(
        0.18 * fiber_score
        + 0.14 * boundary_score
        + 0.24 * route_score
        + 0.14 * recovery_score
        + 0.18 * repair_score
        + 0.12 * bridge_score
    )
    component_rows = [
        {"component_name": "fiber", "score": fiber_score},
        {"component_name": "boundary", "score": boundary_score},
        {"component_name": "route", "score": route_score},
        {"component_name": "recovery", "score": recovery_score},
        {"component_name": "repair", "score": repair_score},
        {"component_name": "bridge", "score": bridge_score},
    ]
    component_rows.sort(key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage168_fiber_route_recovery_closure",
        "title": "纤维-路径-回收闭合",
        "status_short": "fiber_route_recovery_closure_ready",
        "best_formula": best_formula,
        "closure_score": closure_score,
        "weakest_component_name": str(component_rows[0]["component_name"]),
        "strongest_component_name": str(component_rows[-1]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage168: 纤维-路径-回收闭合",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 闭合分数: {summary['closure_score']:.4f}",
        f"- 最弱组件: {summary['weakest_component_name']}",
        f"- 最强组件: {summary['strongest_component_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="纤维-路径-回收闭合")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
