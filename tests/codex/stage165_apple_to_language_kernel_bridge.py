#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage160_apple_result_repair_map import run_analysis as run_stage160_analysis
from stage161_apple_cross_model_noise_split import run_analysis as run_stage161_analysis
from stage163_result_binding_failure_atlas import run_analysis as run_stage163_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage165_apple_to_language_kernel_bridge_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE165_APPLE_TO_LANGUAGE_KERNEL_BRIDGE_REPORT.md"

STAGE154_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json"
STAGE156_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage156_apple_context_bias_shift_20260323" / "summary.json"
STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE160_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json"
STAGE161_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage161_apple_cross_model_noise_split_20260323" / "summary.json"
STAGE163_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage163_result_binding_failure_atlas_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_summary() -> dict:
    s154 = load_json(STAGE154_SUMMARY_PATH)
    s156 = load_json(STAGE156_SUMMARY_PATH)
    s157 = load_json(STAGE157_SUMMARY_PATH)
    if not STAGE160_SUMMARY_PATH.exists():
        run_stage160_analysis(force=True)
    if not STAGE161_SUMMARY_PATH.exists():
        run_stage161_analysis(force=True)
    if not STAGE163_SUMMARY_PATH.exists():
        run_stage163_analysis(force=True)
    s160 = load_json(STAGE160_SUMMARY_PATH)
    s161 = load_json(STAGE161_SUMMARY_PATH)
    s163 = load_json(STAGE163_SUMMARY_PATH)

    a_proxy = float(s154["shared_core_score"])
    r_proxy = clamp01(1.0 - float(s163["hard_failure_count"]) / max(1, int(s163["case_count"])))
    f_proxy = float(s161["mean_clean_ratio"])
    g_proxy = float(s157["apple_action_route_score"])
    q_proxy = clamp01(float(s160["positive_repair_rate"]))
    b_proxy = float(s156["context_bias_shift_score"])

    best_formula = "apple_kernel_bridge = 0.18*a + 0.12*r + 0.18*f + 0.22*g + 0.12*q + 0.18*b"
    bridge_score = clamp01(
        0.18 * a_proxy + 0.12 * r_proxy + 0.18 * f_proxy + 0.22 * g_proxy + 0.12 * q_proxy + 0.18 * b_proxy
    )
    proxy_rows = [
        {"proxy_name": "a_proxy", "score": a_proxy},
        {"proxy_name": "r_proxy", "score": r_proxy},
        {"proxy_name": "f_proxy", "score": f_proxy},
        {"proxy_name": "g_proxy", "score": g_proxy},
        {"proxy_name": "q_proxy", "score": q_proxy},
        {"proxy_name": "b_proxy", "score": b_proxy},
    ]
    proxy_rows.sort(key=lambda row: float(row["score"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage165_apple_to_language_kernel_bridge",
        "title": "苹果到语言主核桥",
        "status_short": "apple_to_language_kernel_bridge_ready",
        "best_formula": best_formula,
        "bridge_score": bridge_score,
        "weakest_proxy_name": str(proxy_rows[0]["proxy_name"]),
        "strongest_proxy_name": str(proxy_rows[-1]["proxy_name"]),
        "proxy_rows": proxy_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage165: 苹果到语言主核桥",
        "",
        "## 核心结果",
        f"- 最优公式: {summary['best_formula']}",
        f"- 桥接分数: {summary['bridge_score']:.4f}",
        f"- 最弱代理量: {summary['weakest_proxy_name']}",
        f"- 最强代理量: {summary['strongest_proxy_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果到语言主核桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
