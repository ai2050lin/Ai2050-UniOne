#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CPU_SUMMARY = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage484_apple_switch_signed_residual_basis_20260403"
    / "summary.json"
)
DEFAULT_GPU_SUMMARY = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage484_apple_switch_signed_residual_basis_20260403_gpu"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage485_cpu_gpu_consistency_compare_{time.strftime('%Y%m%d')}"
)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare_unit(cpu_unit: Dict[str, object], gpu_unit: Dict[str, object]) -> Dict[str, object]:
    cpu_track = cpu_unit["tracking"]
    gpu_track = gpu_unit["tracking"]
    return {
        "unit_id": cpu_unit["unit_id"],
        "role": cpu_unit["role"],
        "forward_peak_layer_cpu": int(cpu_track["forward_peak_layer"]),
        "forward_peak_layer_gpu": int(gpu_track["forward_peak_layer"]),
        "reverse_peak_layer_cpu": int(cpu_track["reverse_peak_layer"]),
        "reverse_peak_layer_gpu": int(gpu_track["reverse_peak_layer"]),
        "late_mean_signed_contrast_switch_coupling_cpu": float(cpu_track["late_mean_signed_contrast_switch_coupling"]),
        "late_mean_signed_contrast_switch_coupling_gpu": float(gpu_track["late_mean_signed_contrast_switch_coupling"]),
        "late_mean_signed_contrast_switch_coupling_abs_diff": abs(
            float(cpu_track["late_mean_signed_contrast_switch_coupling"]) - float(gpu_track["late_mean_signed_contrast_switch_coupling"])
        ),
        "forward_peak_signed_contrast_switch_coupling_cpu": float(cpu_track["forward_peak_signed_contrast_switch_coupling"]),
        "forward_peak_signed_contrast_switch_coupling_gpu": float(gpu_track["forward_peak_signed_contrast_switch_coupling"]),
        "forward_peak_signed_contrast_switch_coupling_abs_diff": abs(
            float(cpu_track["forward_peak_signed_contrast_switch_coupling"]) - float(gpu_track["forward_peak_signed_contrast_switch_coupling"])
        ),
        "reverse_peak_signed_contrast_switch_coupling_cpu": float(cpu_track["reverse_peak_signed_contrast_switch_coupling"]),
        "reverse_peak_signed_contrast_switch_coupling_gpu": float(gpu_track["reverse_peak_signed_contrast_switch_coupling"]),
        "reverse_peak_signed_contrast_switch_coupling_abs_diff": abs(
            float(cpu_track["reverse_peak_signed_contrast_switch_coupling"]) - float(gpu_track["reverse_peak_signed_contrast_switch_coupling"])
        ),
    }


def build_summary(cpu_summary: Dict[str, object], gpu_summary: Dict[str, object]) -> Dict[str, object]:
    models: Dict[str, object] = {}
    max_late_diff = 0.0
    max_forward_diff = 0.0
    max_reverse_diff = 0.0
    layer_match_count = 0
    total_units = 0

    for model_key in ["qwen3", "deepseek7b"]:
        cpu_row = cpu_summary["models"][model_key]
        gpu_row = gpu_summary["models"][model_key]
        cpu_units = {unit["unit_id"]: unit for unit in cpu_row["units"]}
        gpu_units = {unit["unit_id"]: unit for unit in gpu_row["units"]}

        unit_rows: List[Dict[str, object]] = []
        for unit_id in cpu_units:
            row = compare_unit(cpu_units[unit_id], gpu_units[unit_id])
            unit_rows.append(row)
            max_late_diff = max(max_late_diff, float(row["late_mean_signed_contrast_switch_coupling_abs_diff"]))
            max_forward_diff = max(max_forward_diff, float(row["forward_peak_signed_contrast_switch_coupling_abs_diff"]))
            max_reverse_diff = max(max_reverse_diff, float(row["reverse_peak_signed_contrast_switch_coupling_abs_diff"]))
            if int(row["forward_peak_layer_cpu"]) == int(row["forward_peak_layer_gpu"]):
                layer_match_count += 1
            if int(row["reverse_peak_layer_cpu"]) == int(row["reverse_peak_layer_gpu"]):
                layer_match_count += 1
            total_units += 2

        models[model_key] = {
            "unit_rows": unit_rows,
        }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage485_cpu_gpu_consistency_compare",
        "title": "CPU GPU 一致性对照",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_summary_path": str(DEFAULT_CPU_SUMMARY),
        "gpu_summary_path": str(DEFAULT_GPU_SUMMARY),
        "models": models,
        "aggregate": {
            "max_late_mean_abs_diff": max_late_diff,
            "max_forward_peak_abs_diff": max_forward_diff,
            "max_reverse_peak_abs_diff": max_reverse_diff,
            "peak_layer_match_rate": (layer_match_count / total_units) if total_units else 0.0,
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    agg = summary["aggregate"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 汇总",
        f"- max_late_mean_abs_diff = {agg['max_late_mean_abs_diff']:.6f}",
        f"- max_forward_peak_abs_diff = {agg['max_forward_peak_abs_diff']:.6f}",
        f"- max_reverse_peak_abs_diff = {agg['max_reverse_peak_abs_diff']:.6f}",
        f"- peak_layer_match_rate = {agg['peak_layer_match_rate']:.4f}",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        lines.append(f"## 模型 {model_key}")
        for row in summary["models"][model_key]["unit_rows"]:
            lines.extend(
                [
                    f"- 单元 {row['unit_id']} ({row['role']}):",
                    f"  forward_layer cpu/gpu = L{row['forward_peak_layer_cpu']} / L{row['forward_peak_layer_gpu']}",
                    f"  reverse_layer cpu/gpu = L{row['reverse_peak_layer_cpu']} / L{row['reverse_peak_layer_gpu']}",
                    f"  late_mean_abs_diff = {row['late_mean_signed_contrast_switch_coupling_abs_diff']:.6f}",
                    f"  forward_peak_abs_diff = {row['forward_peak_signed_contrast_switch_coupling_abs_diff']:.6f}",
                    f"  reverse_peak_abs_diff = {row['reverse_peak_signed_contrast_switch_coupling_abs_diff']:.6f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU GPU 一致性对照")
    parser.add_argument("--cpu-summary", default=str(DEFAULT_CPU_SUMMARY), help="CPU 汇总文件")
    parser.add_argument("--gpu-summary", default=str(DEFAULT_GPU_SUMMARY), help="GPU 汇总文件")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_summary = load_json(Path(args.cpu_summary))
    gpu_summary = load_json(Path(args.gpu_summary))
    summary = build_summary(cpu_summary, gpu_summary)
    write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status_short": "stage485_ready", "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
