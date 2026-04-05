#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage482_apple_switch_direction_tracking import (
    MODEL_CONFIGS,
    build_all_cases,
    collect_ablated_rows,
    collect_baseline_rows,
    free_model,
)
from stage483_apple_switch_residual_basis import cosine, pca_first_direction, safe_ratio
from qwen3_language_shared import discover_layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage484_apple_switch_signed_residual_basis_{time.strftime('%Y%m%d')}"
)


def summarize_signed_basis(
    baseline_rows: List[Dict[str, object]],
    ablated_rows: List[Dict[str, object]],
    layer_count: int,
) -> Dict[str, object]:
    fruit_base = [row for row in baseline_rows if int(row["sense_label"]) == 0]
    brand_base = [row for row in baseline_rows if int(row["sense_label"]) == 1]
    fruit_abl = [row for row in ablated_rows if int(row["sense_label"]) == 0]
    brand_abl = [row for row in ablated_rows if int(row["sense_label"]) == 1]

    layer_rows: List[Dict[str, object]] = []
    for layer_idx in range(layer_count):
        fruit_base_stack = torch.stack([row["layer_vectors"][layer_idx].float() for row in fruit_base], dim=0)
        brand_base_stack = torch.stack([row["layer_vectors"][layer_idx].float() for row in brand_base], dim=0)
        fruit_abl_stack = torch.stack([row["layer_vectors"][layer_idx].float() for row in fruit_abl], dim=0)
        brand_abl_stack = torch.stack([row["layer_vectors"][layer_idx].float() for row in brand_abl], dim=0)

        switch_vec = brand_base_stack.mean(dim=0) - fruit_base_stack.mean(dim=0)
        switch_norm = float(torch.linalg.norm(switch_vec).item())
        switch_axis = switch_vec / switch_norm if switch_norm > 1e-8 else torch.zeros_like(switch_vec)

        fruit_delta = fruit_abl_stack - fruit_base_stack
        brand_delta = brand_abl_stack - brand_base_stack
        all_delta = torch.cat([fruit_delta, brand_delta], dim=0)

        fruit_delta_mean = fruit_delta.mean(dim=0)
        brand_delta_mean = brand_delta.mean(dim=0)
        contrast_delta = brand_delta_mean - fruit_delta_mean
        contrast_norm = float(torch.linalg.norm(contrast_delta).item())
        mean_delta = all_delta.mean(dim=0)
        mean_delta_norm = float(torch.linalg.norm(mean_delta).item())

        pc1_dir, pc1_explained = pca_first_direction(all_delta)
        signed_contrast_cos = cosine(contrast_delta, switch_axis) if contrast_norm > 1e-8 else 0.0
        signed_mean_cos = cosine(mean_delta, switch_axis) if mean_delta_norm > 1e-8 else 0.0
        signed_pc1_cos = cosine(pc1_dir, switch_axis)

        contrast_norm_ratio = safe_ratio(contrast_norm, switch_norm)
        mean_norm_ratio = safe_ratio(mean_delta_norm, switch_norm)
        signed_contrast_coupling = signed_contrast_cos * contrast_norm_ratio
        signed_pc1_coupling = signed_pc1_cos * contrast_norm_ratio

        layer_rows.append(
            {
                "layer_index": layer_idx,
                "switch_norm": switch_norm,
                "contrast_delta_norm": contrast_norm,
                "mean_delta_norm": mean_delta_norm,
                "contrast_norm_ratio": contrast_norm_ratio,
                "mean_norm_ratio": mean_norm_ratio,
                "signed_contrast_switch_cos": float(signed_contrast_cos),
                "signed_mean_switch_cos": float(signed_mean_cos),
                "signed_pc1_switch_cos": float(signed_pc1_cos),
                "signed_contrast_switch_coupling": float(signed_contrast_coupling),
                "signed_pc1_switch_coupling": float(signed_pc1_coupling),
                "pc1_explained_variance_ratio": float(pc1_explained),
            }
        )

    forward_peak = max(layer_rows, key=lambda row: float(row["signed_contrast_switch_coupling"]))
    reverse_peak = min(layer_rows, key=lambda row: float(row["signed_contrast_switch_coupling"]))
    forward_pc1_peak = max(layer_rows, key=lambda row: float(row["signed_pc1_switch_coupling"]))
    reverse_pc1_peak = min(layer_rows, key=lambda row: float(row["signed_pc1_switch_coupling"]))
    late_window = layer_rows[max(0, layer_count - 4) :]

    return {
        "layer_rows": layer_rows,
        "forward_peak_layer": int(forward_peak["layer_index"]),
        "forward_peak_signed_contrast_switch_coupling": float(forward_peak["signed_contrast_switch_coupling"]),
        "forward_peak_signed_contrast_switch_cos": float(forward_peak["signed_contrast_switch_cos"]),
        "reverse_peak_layer": int(reverse_peak["layer_index"]),
        "reverse_peak_signed_contrast_switch_coupling": float(reverse_peak["signed_contrast_switch_coupling"]),
        "reverse_peak_signed_contrast_switch_cos": float(reverse_peak["signed_contrast_switch_cos"]),
        "forward_pc1_peak_layer": int(forward_pc1_peak["layer_index"]),
        "forward_peak_signed_pc1_switch_coupling": float(forward_pc1_peak["signed_pc1_switch_coupling"]),
        "forward_peak_signed_pc1_switch_cos": float(forward_pc1_peak["signed_pc1_switch_cos"]),
        "reverse_pc1_peak_layer": int(reverse_pc1_peak["layer_index"]),
        "reverse_peak_signed_pc1_switch_coupling": float(reverse_pc1_peak["signed_pc1_switch_coupling"]),
        "reverse_peak_signed_pc1_switch_cos": float(reverse_pc1_peak["signed_pc1_switch_cos"]),
        "late_mean_signed_contrast_switch_coupling": float(
            safe_ratio(sum(float(row["signed_contrast_switch_coupling"]) for row in late_window), len(late_window))
        ),
        "late_mean_signed_pc1_switch_coupling": float(
            safe_ratio(sum(float(row["signed_pc1_switch_coupling"]) for row in late_window), len(late_window))
        ),
        "top_forward_layers": sorted(
            layer_rows,
            key=lambda row: float(row["signed_contrast_switch_coupling"]),
            reverse=True,
        )[:6],
        "top_reverse_layers": sorted(
            layer_rows,
            key=lambda row: float(row["signed_contrast_switch_coupling"]),
        )[:6],
    }


def analyze_model(model_key: str, *, use_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        cases = build_all_cases()
        baseline_rows = collect_baseline_rows(model, tokenizer, cases)
        layer_count = len(discover_layers(model))
        unit_rows = []
        for unit in MODEL_CONFIGS[model_key]["focus_units"]:
            ablated_rows = collect_ablated_rows(model, tokenizer, cases, unit)
            tracking = summarize_signed_basis(baseline_rows, ablated_rows, layer_count)
            unit_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "kind": unit["kind"],
                    "layer_index": int(unit["layer_index"]),
                    "head_index": int(unit["head_index"]) if unit["kind"] == "attention_head" else None,
                    "neuron_index": int(unit["neuron_index"]) if unit["kind"] == "mlp_neuron" else None,
                    "role": unit["role"],
                    "tracking": tracking,
                }
            )
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(use_cuda),
            "case_count": len(cases),
            "layer_count": layer_count,
            "units": unit_rows,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for model_key, row in model_rows.items():
        out[model_key] = {
            "forward_peak_layers": {
                unit["unit_id"]: unit["tracking"]["forward_peak_layer"] for unit in row["units"]
            },
            "reverse_peak_layers": {
                unit["unit_id"]: unit["tracking"]["reverse_peak_layer"] for unit in row["units"]
            },
            "late_mean_signed_contrast_switch_coupling": {
                unit["unit_id"]: unit["tracking"]["late_mean_signed_contrast_switch_coupling"] for unit in row["units"]
            },
        }
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        "- 目标: 判断苹果切换核心单元的主残差方向，是顺着切换轴推进，还是反着切换轴抵消",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        row = summary["models"][model_key]
        lines.append(f"## 模型 {model_key}")
        for unit in row["units"]:
            track = unit["tracking"]
            lines.extend(
                [
                    f"- 单元 {unit['unit_id']} ({unit['role']}):",
                    f"  forward_peak_layer = L{track['forward_peak_layer']}",
                    f"  forward_peak_signed_contrast_switch_coupling = {track['forward_peak_signed_contrast_switch_coupling']:+.4f}",
                    f"  forward_peak_signed_contrast_switch_cos = {track['forward_peak_signed_contrast_switch_cos']:+.4f}",
                    f"  reverse_peak_layer = L{track['reverse_peak_layer']}",
                    f"  reverse_peak_signed_contrast_switch_coupling = {track['reverse_peak_signed_contrast_switch_coupling']:+.4f}",
                    f"  reverse_peak_signed_contrast_switch_cos = {track['reverse_peak_signed_contrast_switch_cos']:+.4f}",
                    f"  late_mean_signed_contrast_switch_coupling = {track['late_mean_signed_contrast_switch_coupling']:+.4f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果切换有符号主残差方向追踪")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    start_time = time.time()
    model_rows = {}
    for model_key in ["qwen3", "deepseek7b"]:
        model_rows[model_key] = analyze_model(model_key, use_cuda=use_cuda)
    elapsed = time.time() - start_time

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage484_apple_switch_signed_residual_basis",
        "title": "苹果切换有符号主残差方向追踪",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "models": model_rows,
        "cross_model_summary": build_cross_model_summary(model_rows),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage484_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
