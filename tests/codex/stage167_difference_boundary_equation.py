#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage166_category_fiber_map import run_analysis as run_stage166_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage167_difference_boundary_equation_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE167_DIFFERENCE_BOUNDARY_EQUATION_REPORT.md"

STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"
STAGE166_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    if not STAGE166_SUMMARY_PATH.exists():
        run_stage166_analysis(force=True)
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s155 = load_json(STAGE155_SUMMARY_PATH)
    s166 = load_json(STAGE166_SUMMARY_PATH)

    same_reuse = float(s154["shared_core_score"])
    boundary_stability = 1.0 - float(s155["collision_rate"])
    cross_category_separation = float(s166["cross_category_separation"])
    category_fiber = float(s166["category_fiber_score"])

    best_formula = "difference_boundary = 0.30*same_reuse + 0.30*cross_category_separation + 0.20*boundary_stability + 0.20*category_fiber"
    equation_score = clamp01(
        0.30 * same_reuse
        + 0.30 * cross_category_separation
        + 0.20 * boundary_stability
        + 0.20 * category_fiber
    )
    component_rows = [
        {"component_name": "same_reuse", "score": same_reuse},
        {"component_name": "cross_category_separation", "score": cross_category_separation},
        {"component_name": "boundary_stability", "score": boundary_stability},
        {"component_name": "category_fiber", "score": category_fiber},
    ]
    component_rows.sort(key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage167_difference_boundary_equation",
        "title": "差分边界方程",
        "status_short": "difference_boundary_equation_ready",
        "best_formula": best_formula,
        "equation_score": equation_score,
        "weakest_component_name": str(component_rows[0]["component_name"]),
        "strongest_component_name": str(component_rows[-1]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage167: 差分边界方程",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 方程分数: {summary['equation_score']:.4f}",
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
    parser = argparse.ArgumentParser(description="差分边界方程")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
