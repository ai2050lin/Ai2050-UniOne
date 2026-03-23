#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage160_apple_result_repair_map import run_analysis as run_stage160_analysis
from stage161_apple_cross_model_noise_split import run_analysis as run_stage161_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage162_apple_full_chain_equation_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE162_APPLE_FULL_CHAIN_EQUATION_REPORT.md"

STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"
STAGE156_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage156_apple_context_bias_shift_20260323" / "summary.json"
STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE158_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE161_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage161_apple_cross_model_noise_split_20260323" / "summary.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s155 = load_json(STAGE155_SUMMARY_PATH)
    s156 = load_json(STAGE156_SUMMARY_PATH)
    s157 = load_json(STAGE157_SUMMARY_PATH)
    s158 = load_json(STAGE158_SUMMARY_PATH)
    if not STAGE160_SUMMARY_PATH.exists():
        run_stage160_analysis(force=True)
    if not STAGE161_SUMMARY_PATH.exists():
        run_stage161_analysis(force=True)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s161 = load_json(STAGE161_SUMMARY_PATH)

    shared_core = float(s154["shared_core_score"])
    boundary_stability = 1.0 - float(s155["collision_rate"])
    context_bias = float(s156["context_bias_shift_score"])
    action_route = float(s157["apple_action_route_score"])
    result_binding = float(s158["apple_result_binding_score"])
    result_repair = float(s160["apple_result_repair_score"])
    cross_model_clean = float(s161["mean_clean_ratio"])

    chain_equation = (
        "apple_chain = 0.20*S + 0.10*B + 0.20*C + 0.20*R + 0.15*T + 0.10*P + 0.05*M"
    )
    chain_score = clamp01(
        0.20 * shared_core
        + 0.10 * boundary_stability
        + 0.20 * context_bias
        + 0.20 * action_route
        + 0.15 * result_binding
        + 0.10 * result_repair
        + 0.05 * cross_model_clean
    )
    component_rows = [
        {"component_name": "shared_core", "symbol": "S", "score": shared_core},
        {"component_name": "boundary_stability", "symbol": "B", "score": boundary_stability},
        {"component_name": "context_bias", "symbol": "C", "score": context_bias},
        {"component_name": "action_route", "symbol": "R", "score": action_route},
        {"component_name": "result_binding", "symbol": "T", "score": result_binding},
        {"component_name": "result_repair", "symbol": "P", "score": result_repair},
        {"component_name": "cross_model_clean", "symbol": "M", "score": cross_model_clean},
    ]
    component_rows.sort(key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage162_apple_full_chain_equation",
        "title": "苹果全链方程",
        "status_short": "apple_full_chain_equation_ready",
        "best_formula": chain_equation,
        "apple_chain_score": chain_score,
        "weakest_component_name": str(component_rows[0]["component_name"]),
        "strongest_component_name": str(component_rows[-1]["component_name"]),
        "component_rows": component_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage162: 苹果全链方程",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 全链分数: {summary['apple_chain_score']:.4f}",
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
    parser = argparse.ArgumentParser(description="苹果全链方程")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
