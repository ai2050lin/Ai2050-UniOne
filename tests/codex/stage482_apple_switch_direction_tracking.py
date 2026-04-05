#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import discover_layers, move_batch_to_model_device
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import (
    register_attention_head_ablation,
    register_mlp_neuron_ablation,
)
from stage478_apple_switch_minimal_subcircuit import (
    build_sense_prompt,
    find_last_subsequence,
    locate_target_span,
    split_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage482_apple_switch_direction_tracking_{time.strftime('%Y%m%d')}"
)

MODEL_CONFIGS = {
    "qwen3": {
        "focus_units": [
            {"unit_id": "H:5:2", "kind": "attention_head", "layer_index": 5, "head_index": 2, "role": "skeleton_head_1"},
            {"unit_id": "H:5:29", "kind": "attention_head", "layer_index": 5, "head_index": 29, "role": "skeleton_head_2"},
            {"unit_id": "H:5:9", "kind": "attention_head", "layer_index": 5, "head_index": 9, "role": "bridge_head"},
            {"unit_id": "H:5:8", "kind": "attention_head", "layer_index": 5, "head_index": 8, "role": "heldout_booster"},
        ],
    },
    "deepseek7b": {
        "focus_units": [
            {"unit_id": "N:2:16785", "kind": "mlp_neuron", "layer_index": 2, "neuron_index": 16785, "role": "anchor_neuron"},
            {"unit_id": "H:2:22", "kind": "attention_head", "layer_index": 2, "head_index": 22, "role": "main_booster_1"},
            {"unit_id": "H:2:10", "kind": "attention_head", "layer_index": 2, "head_index": 10, "role": "main_booster_2"},
            {"unit_id": "H:2:26", "kind": "attention_head", "layer_index": 2, "head_index": 26, "role": "heldout_booster"},
        ],
    },
}


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def register_single_unit_ablation(model, unit: Dict[str, object]) -> List[object]:
    if unit["kind"] == "attention_head":
        return register_attention_head_ablation(model, [unit])
    return register_mlp_neuron_ablation(model, [unit])


def capture_prompt_layer_vectors(
    model,
    tokenizer,
    case: Dict[str, object],
    *,
    handles: Sequence[object] | None = None,
) -> Dict[int, torch.Tensor]:
    prompt = build_sense_prompt(case)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    start, end = locate_target_span(tokenizer, prompt, str(case["target"]))
    try:
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
        out: Dict[int, torch.Tensor] = {}
        layer_count = len(discover_layers(model))
        for layer_idx in range(layer_count):
            out[layer_idx] = outputs.hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
        return out
    finally:
        if handles:
            remove_hooks(handles)


def build_all_cases() -> List[Dict[str, object]]:
    search_cases, heldout_cases = split_cases()
    return list(search_cases) + list(heldout_cases)


def collect_baseline_rows(model, tokenizer, cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for case in cases:
        rows.append(
            {
                "sentence": case["sentence"],
                "target": case["target"],
                "label": case["label"],
                "sense_label": int(case["sense_label"]),
                "layer_vectors": capture_prompt_layer_vectors(model, tokenizer, case),
            }
        )
    return rows


def collect_ablated_rows(model, tokenizer, cases: Sequence[Dict[str, object]], unit: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for case in cases:
        handles = register_single_unit_ablation(model, unit)
        rows.append(
            {
                "sentence": case["sentence"],
                "target": case["target"],
                "label": case["label"],
                "sense_label": int(case["sense_label"]),
                "layer_vectors": capture_prompt_layer_vectors(model, tokenizer, case, handles=handles),
            }
        )
    return rows


def build_layer_axes(rows: Sequence[Dict[str, object]], layer_count: int) -> Dict[int, torch.Tensor]:
    out = {}
    fruit_rows = [row for row in rows if int(row["sense_label"]) == 0]
    brand_rows = [row for row in rows if int(row["sense_label"]) == 1]
    for layer_idx in range(layer_count):
        fruit_mean = mean_tensors([row["layer_vectors"][layer_idx] for row in fruit_rows])
        brand_mean = mean_tensors([row["layer_vectors"][layer_idx] for row in brand_rows])
        axis = brand_mean - fruit_mean
        norm = float(torch.linalg.norm(axis).item())
        if norm <= 1e-8:
            out[layer_idx] = torch.zeros_like(axis)
        else:
            out[layer_idx] = axis / norm
    return out


def project_row(row: Dict[str, object], axes: Dict[int, torch.Tensor], layer_count: int) -> Dict[int, float]:
    out = {}
    for layer_idx in range(layer_count):
        axis = axes[layer_idx]
        vec = row["layer_vectors"][layer_idx]
        if float(torch.linalg.norm(axis).item()) <= 1e-8:
            out[layer_idx] = 0.0
        else:
            out[layer_idx] = float(torch.dot(vec.float(), axis.float()).item())
    return out


def summarize_layer_tracking(
    baseline_rows: Sequence[Dict[str, object]],
    ablated_rows: Sequence[Dict[str, object]],
    layer_count: int,
) -> Dict[str, object]:
    axes = build_layer_axes(baseline_rows, layer_count)
    baseline_proj = [project_row(row, axes, layer_count) for row in baseline_rows]
    ablated_proj = [project_row(row, axes, layer_count) for row in ablated_rows]

    baseline_fruit = [proj for proj, row in zip(baseline_proj, baseline_rows) if int(row["sense_label"]) == 0]
    baseline_brand = [proj for proj, row in zip(baseline_proj, baseline_rows) if int(row["sense_label"]) == 1]
    ablated_fruit = [proj for proj, row in zip(ablated_proj, ablated_rows) if int(row["sense_label"]) == 0]
    ablated_brand = [proj for proj, row in zip(ablated_proj, ablated_rows) if int(row["sense_label"]) == 1]

    layer_rows = []
    for layer_idx in range(layer_count):
        baseline_sep = safe_ratio(
            sum(float(proj[layer_idx]) for proj in baseline_brand),
            len(baseline_brand),
        ) - safe_ratio(
            sum(float(proj[layer_idx]) for proj in baseline_fruit),
            len(baseline_fruit),
        )
        ablated_sep = safe_ratio(
            sum(float(proj[layer_idx]) for proj in ablated_brand),
            len(ablated_brand),
        ) - safe_ratio(
            sum(float(proj[layer_idx]) for proj in ablated_fruit),
            len(ablated_fruit),
        )
        separation_drop = baseline_sep - ablated_sep
        relative_drop = safe_ratio(separation_drop, baseline_sep) if abs(baseline_sep) > 1e-8 else 0.0
        layer_rows.append(
            {
                "layer_index": layer_idx,
                "baseline_separation": float(baseline_sep),
                "ablated_separation": float(ablated_sep),
                "separation_drop": float(separation_drop),
                "relative_drop": float(relative_drop),
            }
        )

    peak_row = max(layer_rows, key=lambda row: abs(float(row["separation_drop"])))
    late_window = layer_rows[max(0, layer_count - 4) :]
    late_mean_drop = safe_ratio(sum(float(row["separation_drop"]) for row in late_window), len(late_window))
    late_mean_relative = safe_ratio(sum(float(row["relative_drop"]) for row in late_window), len(late_window))
    top_drop_layers = sorted(layer_rows, key=lambda row: abs(float(row["separation_drop"])), reverse=True)[:6]

    return {
        "layer_rows": layer_rows,
        "peak_effect_layer": int(peak_row["layer_index"]),
        "peak_separation_drop": float(peak_row["separation_drop"]),
        "peak_relative_drop": float(peak_row["relative_drop"]),
        "late_mean_drop": float(late_mean_drop),
        "late_mean_relative_drop": float(late_mean_relative),
        "top_drop_layers": top_drop_layers,
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
            tracking = summarize_layer_tracking(baseline_rows, ablated_rows, layer_count)
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
    out = {}
    for model_key, row in model_rows.items():
        out[model_key] = {
            "peak_layers": {unit["unit_id"]: unit["tracking"]["peak_effect_layer"] for unit in row["units"]},
            "late_mean_relative_drop": {unit["unit_id"]: unit["tracking"]["late_mean_relative_drop"] for unit in row["units"]},
        }
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        "- 目标: 追踪苹果切换核心单元对后续层切换轴分离度的影响",
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
                    f"  peak_layer = L{track['peak_effect_layer']}",
                    f"  peak_drop = {track['peak_separation_drop']:+.4f}",
                    f"  peak_relative_drop = {track['peak_relative_drop']:+.4f}",
                    f"  late_mean_relative_drop = {track['late_mean_relative_drop']:+.4f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果切换方向追踪")
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
        "experiment_id": "stage482_apple_switch_direction_tracking",
        "title": "苹果切换方向追踪",
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
                "status_short": "stage482_ready",
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
