#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage169_category_delta_tensor import run_analysis as run_stage169_analysis
from stage170_fiber_bundle_probe import run_analysis as run_stage170_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage171_delta_route_recovery_bridge_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE171_DELTA_ROUTE_RECOVERY_BRIDGE_REPORT.md"

STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE169_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage169_category_delta_tensor_20260323" / "summary.json"
STAGE170_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage170_fiber_bundle_probe_20260323" / "summary.json"
STAGE168_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage168_fiber_route_recovery_closure_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    if not STAGE169_SUMMARY_PATH.exists():
        run_stage169_analysis(force=True)
    if not STAGE170_SUMMARY_PATH.exists():
        run_stage170_analysis(force=True)

    s157 = load_json(STAGE157_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s168 = load_json(STAGE168_SUMMARY_PATH)
    s169 = load_json(STAGE169_SUMMARY_PATH)
    s170 = load_json(STAGE170_SUMMARY_PATH)

    delta_score = float(s169["delta_tensor_score"])
    bundle_score = float(s170["fiber_bundle_score"])
    route_score = float(s157["apple_action_route_score"])
    repair_score = float(s160["apple_result_repair_score"])
    closure_score = float(s168["closure_score"])

    best_formula = "delta_route_recovery = 0.18*delta + 0.18*bundle + 0.26*route + 0.18*repair + 0.20*closure"
    bridge_score = clamp01(
        0.18 * delta_score
        + 0.18 * bundle_score
        + 0.26 * route_score
        + 0.18 * repair_score
        + 0.20 * closure_score
    )
    component_rows = [
        {"component_name": "delta", "score": delta_score},
        {"component_name": "bundle", "score": bundle_score},
        {"component_name": "route", "score": route_score},
        {"component_name": "repair", "score": repair_score},
        {"component_name": "closure", "score": closure_score},
    ]
    component_rows.sort(key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage171_delta_route_recovery_bridge",
        "title": "差分-路径-回收桥",
        "status_short": "delta_route_recovery_bridge_ready",
        "best_formula": best_formula,
        "bridge_score": bridge_score,
        "weakest_component_name": str(component_rows[0]["component_name"]),
        "strongest_component_name": str(component_rows[-1]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage171: 差分-路径-回收桥",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 桥接分数: {summary['bridge_score']:.4f}",
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
    parser = argparse.ArgumentParser(description="差分-路径-回收桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
