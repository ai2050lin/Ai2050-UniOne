#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage172_provenance_trace_probe import run_analysis as run_stage172_analysis
from stage173_multi_entity_recovery_stress_test import run_analysis as run_stage173_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage174_recovery_closure_equation_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE174_RECOVERY_CLOSURE_EQUATION_REPORT.md"

STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE171_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage171_delta_route_recovery_bridge_20260323" / "summary.json"
STAGE172_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json"
STAGE173_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage173_multi_entity_recovery_stress_test_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    if not STAGE172_SUMMARY_PATH.exists():
        run_stage172_analysis(force=True)
    if not STAGE173_SUMMARY_PATH.exists():
        run_stage173_analysis(force=True)

    s158 = load_json(STAGE158_SUMMARY_PATH)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s171 = load_json(STAGE171_SUMMARY_PATH)
    s172 = load_json(STAGE172_SUMMARY_PATH)
    s173 = load_json(STAGE173_SUMMARY_PATH)

    trace_score = float(s172["provenance_trace_score"])
    binding_score = float(s158["apple_result_binding_score"])
    repair_score = float(s160["apple_result_repair_score"])
    stress_score = float(s173["stress_survival_score"])
    bridge_score = float(s171["bridge_score"])

    component_rows = [
        {"component_name": "trace", "score": trace_score},
        {"component_name": "binding", "score": binding_score},
        {"component_name": "repair", "score": repair_score},
        {"component_name": "stress", "score": stress_score},
        {"component_name": "bridge", "score": bridge_score},
    ]
    best_formula = "recovery_closure = 0.22*trace + 0.20*binding + 0.20*repair + 0.18*stress + 0.20*bridge"
    closure_score = clamp01(
        0.22 * trace_score
        + 0.20 * binding_score
        + 0.20 * repair_score
        + 0.18 * stress_score
        + 0.20 * bridge_score
    )
    ranked_rows = sorted(component_rows, key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage174_recovery_closure_equation",
        "title": "回收闭合方程",
        "status_short": "recovery_closure_equation_ready",
        "best_formula": best_formula,
        "closure_score": closure_score,
        "weakest_component_name": str(ranked_rows[0]["component_name"]),
        "strongest_component_name": str(ranked_rows[-1]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage174: 回收闭合方程",
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
    parser = argparse.ArgumentParser(description="回收闭合方程")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
