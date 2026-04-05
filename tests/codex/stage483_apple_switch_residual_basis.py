#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from qwen3_language_shared import discover_layers
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage482_apple_switch_direction_tracking import (
    MODEL_CONFIGS,
    build_all_cases,
    collect_ablated_rows,
    collect_baseline_rows,
    free_model,
    mean_tensors,
    safe_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage483_apple_switch_residual_basis_{time.strftime('%Y%m%d')}"
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def pca_first_direction(matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise RuntimeError("输入矩阵维度错误，无法计算主方向")
    if matrix.shape[0] == 1:
        vec = matrix[0].float()
        norm = float(torch.linalg.norm(vec).item())
        if norm <= 1e-8:
            return torch.zeros_like(vec), 0.0
        return vec / norm, 1.0

    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    total_energy = float((centered**2).sum().item())
    if total_energy <= 1e-8:
        return torch.zeros(centered.shape[1], dtype=centered.dtype), 0.0

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    direction = vh[0].float()
    explained = float((singular_values[0].item() ** 2) / (singular_values.pow(2).sum().item()))
    return direction, explained


def summarize_residual_basis(
    baseline_rows: Sequence[Dict[str, object]],
    ablated_rows: Sequence[Dict[str, object]],
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
        pc1_switch_cos = abs(cosine(pc1_dir, switch_axis))
        contrast_switch_cos = cosine(contrast_delta, switch_axis) if contrast_norm > 1e-8 else 0.0
        mean_switch_cos = cosine(mean_delta, switch_axis) if mean_delta_norm > 1e-8 else 0.0

        layer_rows.append(
            {
                "layer_index": layer_idx,
                "switch_norm": switch_norm,
                "contrast_delta_norm": contrast_norm,
                "mean_delta_norm": mean_delta_norm,
                "contrast_norm_ratio": safe_ratio(contrast_norm, switch_norm),
                "mean_norm_ratio": safe_ratio(mean_delta_norm, switch_norm),
                "contrast_switch_cos": float(contrast_switch_cos),
                "contrast_switch_abs_cos": abs(float(contrast_switch_cos)),
                "mean_switch_cos": float(mean_switch_cos),
                "mean_switch_abs_cos": abs(float(mean_switch_cos)),
                "pc1_switch_abs_cos": float(pc1_switch_cos),
                "pc1_explained_variance_ratio": float(pc1_explained),
                "contrast_switch_coupling": abs(float(contrast_switch_cos)) * safe_ratio(contrast_norm, switch_norm),
                "pc1_switch_coupling": float(pc1_switch_cos) * safe_ratio(contrast_norm, switch_norm),
            }
        )

    peak_contrast = max(layer_rows, key=lambda row: float(row["contrast_switch_coupling"]))
    peak_pc1 = max(layer_rows, key=lambda row: float(row["pc1_switch_coupling"]))
    late_window = layer_rows[max(0, layer_count - 4) :]

    return {
        "layer_rows": layer_rows,
        "peak_contrast_alignment_layer": int(peak_contrast["layer_index"]),
        "peak_contrast_switch_coupling": float(peak_contrast["contrast_switch_coupling"]),
        "peak_contrast_switch_abs_cos": float(peak_contrast["contrast_switch_abs_cos"]),
        "peak_contrast_norm_ratio": float(peak_contrast["contrast_norm_ratio"]),
        "peak_pc1_alignment_layer": int(peak_pc1["layer_index"]),
        "peak_pc1_switch_coupling": float(peak_pc1["pc1_switch_coupling"]),
        "peak_pc1_switch_abs_cos": float(peak_pc1["pc1_switch_abs_cos"]),
        "peak_pc1_explained_variance_ratio": float(peak_pc1["pc1_explained_variance_ratio"]),
        "late_mean_contrast_switch_abs_cos": float(
            safe_ratio(sum(float(row["contrast_switch_abs_cos"]) for row in late_window), len(late_window))
        ),
        "late_mean_pc1_switch_abs_cos": float(
            safe_ratio(sum(float(row["pc1_switch_abs_cos"]) for row in late_window), len(late_window))
        ),
        "late_mean_contrast_switch_coupling": float(
            safe_ratio(sum(float(row["contrast_switch_coupling"]) for row in late_window), len(late_window))
        ),
        "late_mean_pc1_switch_coupling": float(
            safe_ratio(sum(float(row["pc1_switch_coupling"]) for row in late_window), len(late_window))
        ),
        "top_contrast_layers": sorted(
            layer_rows,
            key=lambda row: float(row["contrast_switch_coupling"]),
            reverse=True,
        )[:6],
        "top_pc1_layers": sorted(
            layer_rows,
            key=lambda row: float(row["pc1_switch_coupling"]),
            reverse=True,
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
            tracking = summarize_residual_basis(baseline_rows, ablated_rows, layer_count)
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
            "peak_contrast_layers": {
                unit["unit_id"]: unit["tracking"]["peak_contrast_alignment_layer"] for unit in row["units"]
            },
            "peak_pc1_layers": {
                unit["unit_id"]: unit["tracking"]["peak_pc1_alignment_layer"] for unit in row["units"]
            },
            "late_mean_contrast_switch_coupling": {
                unit["unit_id"]: unit["tracking"]["late_mean_contrast_switch_coupling"] for unit in row["units"]
            },
            "late_mean_pc1_switch_coupling": {
                unit["unit_id"]: unit["tracking"]["late_mean_pc1_switch_coupling"] for unit in row["units"]
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
        "- 目标: 提取苹果切换核心单元改写的主残差方向，并比较其与水果义/品牌义切换轴的对齐程度",
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
                    f"  peak_contrast_layer = L{track['peak_contrast_alignment_layer']}",
                    f"  peak_contrast_switch_coupling = {track['peak_contrast_switch_coupling']:+.4f}",
                    f"  peak_contrast_switch_abs_cos = {track['peak_contrast_switch_abs_cos']:+.4f}",
                    f"  peak_pc1_layer = L{track['peak_pc1_alignment_layer']}",
                    f"  peak_pc1_switch_coupling = {track['peak_pc1_switch_coupling']:+.4f}",
                    f"  peak_pc1_switch_abs_cos = {track['peak_pc1_switch_abs_cos']:+.4f}",
                    f"  peak_pc1_explained_variance_ratio = {track['peak_pc1_explained_variance_ratio']:+.4f}",
                    f"  late_mean_contrast_switch_coupling = {track['late_mean_contrast_switch_coupling']:+.4f}",
                    f"  late_mean_pc1_switch_coupling = {track['late_mean_pc1_switch_coupling']:+.4f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果切换主残差方向追踪")
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
        "experiment_id": "stage483_apple_switch_residual_basis",
        "title": "苹果切换主残差方向追踪",
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
                "status_short": "stage483_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_units": [unit["unit_id"] for unit in model_rows["qwen3"]["units"]],
                "deepseek7b_units": [unit["unit_id"] for unit in model_rows["deepseek7b"]["units"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
