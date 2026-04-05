#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage490_cpu_gpu_dual_path_formal_protocol_{time.strftime('%Y%m%d')}"
)

CPU_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403" / "summary.json"
GPU_ORIGINAL_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403_gpu" / "summary.json"
GPU_RETEST_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage484_apple_switch_signed_residual_basis_20260403_gpu_retest" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare_unit(left_unit: Dict[str, object], right_unit: Dict[str, object]) -> Dict[str, object]:
    left = left_unit["tracking"]
    right = right_unit["tracking"]
    return {
        "unit_id": left_unit["unit_id"],
        "role": left_unit["role"],
        "forward_peak_layer_left": int(left["forward_peak_layer"]),
        "forward_peak_layer_right": int(right["forward_peak_layer"]),
        "reverse_peak_layer_left": int(left["reverse_peak_layer"]),
        "reverse_peak_layer_right": int(right["reverse_peak_layer"]),
        "late_mean_abs_diff": abs(float(left["late_mean_signed_contrast_switch_coupling"]) - float(right["late_mean_signed_contrast_switch_coupling"])),
        "forward_peak_abs_diff": abs(float(left["forward_peak_signed_contrast_switch_coupling"]) - float(right["forward_peak_signed_contrast_switch_coupling"])),
        "reverse_peak_abs_diff": abs(float(left["reverse_peak_signed_contrast_switch_coupling"]) - float(right["reverse_peak_signed_contrast_switch_coupling"])),
    }


def compare_pair(left_summary: Dict[str, object], right_summary: Dict[str, object], left_label: str, right_label: str) -> Dict[str, object]:
    models: Dict[str, object] = {}
    max_late = 0.0
    max_forward = 0.0
    max_reverse = 0.0
    match_count = 0
    total_count = 0
    for model_key in ["qwen3", "deepseek7b"]:
        left_row = left_summary["models"][model_key]
        right_row = right_summary["models"][model_key]
        right_units = {unit["unit_id"]: unit for unit in right_row["units"]}
        unit_rows: List[Dict[str, object]] = []
        for left_unit in left_row["units"]:
            row = compare_unit(left_unit, right_units[left_unit["unit_id"]])
            unit_rows.append(row)
            max_late = max(max_late, float(row["late_mean_abs_diff"]))
            max_forward = max(max_forward, float(row["forward_peak_abs_diff"]))
            max_reverse = max(max_reverse, float(row["reverse_peak_abs_diff"]))
            if int(row["forward_peak_layer_left"]) == int(row["forward_peak_layer_right"]):
                match_count += 1
            if int(row["reverse_peak_layer_left"]) == int(row["reverse_peak_layer_right"]):
                match_count += 1
            total_count += 2
        models[model_key] = {"unit_rows": unit_rows}
    return {
        "pair_name": f"{left_label}_vs_{right_label}",
        "left_label": left_label,
        "right_label": right_label,
        "models": models,
        "aggregate": {
            "max_late_mean_abs_diff": max_late,
            "max_forward_peak_abs_diff": max_forward,
            "max_reverse_peak_abs_diff": max_reverse,
            "peak_layer_match_rate": (match_count / total_count) if total_count else 0.0,
        },
    }


def classify_protocol(pair_row: Dict[str, object]) -> Dict[str, object]:
    agg = pair_row["aggregate"]
    trend_pass = float(agg["max_late_mean_abs_diff"]) <= 0.01
    magnitude_pass = float(agg["max_forward_peak_abs_diff"]) <= 0.01 and float(agg["max_reverse_peak_abs_diff"]) <= 0.01
    peak_pass = float(agg["peak_layer_match_rate"]) >= 0.75
    if trend_pass and magnitude_pass and peak_pass:
        recommendation = "可用于趋势与峰值双重验证"
    elif trend_pass and magnitude_pass:
        recommendation = "可用于趋势验证，但峰值层位必须保留 CPU 复核"
    else:
        recommendation = "当前不应单独信任 GPU 路径"
    return {
        "trend_pass": trend_pass,
        "magnitude_pass": magnitude_pass,
        "peak_pass": peak_pass,
        "recommendation": recommendation,
    }


def build_summary() -> Dict[str, object]:
    cpu_summary = load_json(CPU_SUMMARY_PATH)
    gpu_original = load_json(GPU_ORIGINAL_SUMMARY_PATH)
    gpu_retest = load_json(GPU_RETEST_SUMMARY_PATH)

    pairs = [
        compare_pair(cpu_summary, gpu_original, "cpu", "gpu_original"),
        compare_pair(cpu_summary, gpu_retest, "cpu", "gpu_retest"),
        compare_pair(gpu_original, gpu_retest, "gpu_original", "gpu_retest"),
    ]
    classifications = {row["pair_name"]: classify_protocol(row) for row in pairs}

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage490_cpu_gpu_dual_path_formal_protocol",
        "title": "CPU GPU 双路径正式协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {
            "trend_max_late_mean_abs_diff": 0.01,
            "magnitude_max_peak_abs_diff": 0.01,
            "peak_layer_match_rate": 0.75,
        },
        "pairs": pairs,
        "classifications": classifications,
        "formal_protocol": {
            "step_1": "先在 GPU 上跑趋势级实验，用于快速筛选方向。",
            "step_2": "如果趋势级差异通过阈值，再把关键结论在 CPU 上复核。",
            "step_3": "所有峰值层位、最强单元、细粒度方向耦合结论，默认必须保留 CPU 复核。",
            "step_4": "如果 GPU 与 GPU 重跑之间漂移超过阈值，则暂停把 GPU 结果当作研究基准。",
        },
        "core_answer": "当前正式协议不再把 CPU 与 GPU 当作可随意替换的同一路径。它们在趋势级结论上可以互相支撑，但在峰值层位和小信号上仍然存在明显漂移，因此研究工作必须采用双路径治理。",
        "sources": {
            "cpu": str(CPU_SUMMARY_PATH),
            "gpu_original": str(GPU_ORIGINAL_SUMMARY_PATH),
            "gpu_retest": str(GPU_RETEST_SUMMARY_PATH),
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心回答", summary["core_answer"], "", "## 阈值"]
    for key, value in summary["thresholds"].items():
        lines.append(f"- {key} = {value}")
    lines.extend(["", "## 对照结果"])
    for row in summary["pairs"]:
        agg = row["aggregate"]
        cls = summary["classifications"][row["pair_name"]]
        lines.extend(
            [
                f"- {row['pair_name']}:",
                f"  - max_late_mean_abs_diff = {agg['max_late_mean_abs_diff']:.6f}",
                f"  - max_forward_peak_abs_diff = {agg['max_forward_peak_abs_diff']:.6f}",
                f"  - max_reverse_peak_abs_diff = {agg['max_reverse_peak_abs_diff']:.6f}",
                f"  - peak_layer_match_rate = {agg['peak_layer_match_rate']:.4f}",
                f"  - recommendation = {cls['recommendation']}",
            ]
        )
    lines.extend(["", "## 正式协议"])
    for key in ["step_1", "step_2", "step_3", "step_4"]:
        lines.append(f"- {summary['formal_protocol'][key]}")
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU GPU 双路径正式协议")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status_short": "stage490_ready", "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
