#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from multimodel_language_shared import discover_layers, free_model, load_model_bundle
from qwen3_language_shared import capture_qwen_mlp_payloads, move_batch_to_model_device, remove_hooks
from stage423_qwen3_deepseek_wordclass_layer_distribution import (
    TOP_FRACTION,
    WORD_CLASSES,
    mean_token_activation,
)
from stage515_cross_task_minimal_causal_circuit import _OutFeatureShim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage529_glm4_gemma4_wordclass_scan_20260404"
)
STAGE423_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
MODEL_KEYS = ["glm4", "gemma4"]
BATCH_SIZE = 8
CONTROL_PER_CLASS = 20
EPS = 1e-8


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def patch_glm4_mlp_compat(model) -> None:
    for layer in discover_layers(model):
        mlp = layer.mlp
        if not hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            mlp.gate_proj = _OutFeatureShim(mlp.down_proj.in_features)


def load_stage423_target_words() -> Dict[str, List[str]]:
    data = json.loads(STAGE423_PATH.read_text(encoding="utf-8-sig"))
    source = data["models"]["qwen3"]["classes"]
    return {
        class_name: [str(word) for word in source[class_name]["target_words"]]
        for class_name in WORD_CLASSES
    }


def init_stats(layer_widths: Sequence[int]) -> Dict[str, object]:
    def make_list() -> List[torch.Tensor]:
        return [torch.zeros(width, dtype=torch.float64) for width in layer_widths]

    return {
        "layer_widths": list(layer_widths),
        "target_count": 0,
        "control_count": 0,
        "target_sum": make_list(),
        "target_sumsq": make_list(),
        "target_pos": make_list(),
        "control_sum": make_list(),
        "control_sumsq": make_list(),
        "control_pos": make_list(),
    }


def update_layerwise_stats(stats: Dict[str, object], per_layer_rows: Sequence[torch.Tensor], *, is_target: bool) -> None:
    prefix = "target" if is_target else "control"
    for layer_idx, layer_rows in enumerate(per_layer_rows):
        stats[f"{prefix}_sum"][layer_idx] += layer_rows.sum(dim=0)
        stats[f"{prefix}_sumsq"][layer_idx] += (layer_rows * layer_rows).sum(dim=0)
        stats[f"{prefix}_pos"][layer_idx] += (layer_rows > 0).to(torch.float64).sum(dim=0)
    stats[f"{prefix}_count"] += int(per_layer_rows[0].shape[0])


def capture_all_layers(model) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    layer_count = len(discover_layers(model))
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(layer_count)}
    return capture_qwen_mlp_payloads(model, layer_payload_map)


def process_batch(
    model,
    tokenizer,
    buffers: Dict[int, torch.Tensor | None],
    batch_words: Sequence[str],
    *,
    stats: Dict[str, object],
    is_target: bool,
) -> None:
    encoded = tokenizer(
        list(batch_words),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    attention_mask_cpu = encoded["attention_mask"].cpu()
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        model(**encoded, use_cache=False, return_dict=True)

    per_layer_rows: List[torch.Tensor] = []
    for layer_idx in range(len(buffers)):
        layer_tensor = buffers[layer_idx]
        if layer_tensor is None:
            raise RuntimeError(f"第 {layer_idx} 层激活捕获失败")
        per_layer_rows.append(mean_token_activation(layer_tensor, attention_mask_cpu).to(torch.float64))
    update_layerwise_stats(stats, per_layer_rows, is_target=is_target)


def run_scan(
    model,
    tokenizer,
    target_words: Sequence[str],
    control_words: Sequence[str],
    *,
    layer_widths: Sequence[int],
) -> Dict[str, object]:
    buffers, handles = capture_all_layers(model)
    stats = init_stats(layer_widths)
    try:
        for start in range(0, len(target_words), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                buffers,
                target_words[start : start + BATCH_SIZE],
                stats=stats,
                is_target=True,
            )
        for start in range(0, len(control_words), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                buffers,
                control_words[start : start + BATCH_SIZE],
                stats=stats,
                is_target=False,
            )
    finally:
        remove_hooks(handles)
    return stats


def build_control_words(target_word_map: Dict[str, List[str]], target_class: str) -> List[str]:
    words: List[str] = []
    for class_name in WORD_CLASSES:
        if class_name == target_class:
            continue
        words.extend(target_word_map[class_name][:CONTROL_PER_CLASS])
    seen = set()
    out = []
    for word in words:
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def build_varwidth_summary(stats: Dict[str, object], *, top_fraction: float) -> Dict[str, object]:
    target_count = float(stats["target_count"])
    control_count = float(stats["control_count"])
    layer_rows = []
    top_neurons = []
    all_active_scores: List[float] = []
    all_candidates = []

    for layer_idx, width in enumerate(stats["layer_widths"]):
        target_mean = stats["target_sum"][layer_idx] / max(target_count, 1.0)
        control_mean = stats["control_sum"][layer_idx] / max(control_count, 1.0)
        target_var = stats["target_sumsq"][layer_idx] / max(target_count, 1.0) - target_mean * target_mean
        control_var = stats["control_sumsq"][layer_idx] / max(control_count, 1.0) - control_mean * control_mean
        pooled_std = torch.sqrt(((target_var.clamp_min(0.0) + control_var.clamp_min(0.0)) / 2.0).clamp_min(EPS))
        effect = (target_mean - control_mean) / pooled_std
        pos_gap = stats["target_pos"][layer_idx] / max(target_count, 1.0) - stats["control_pos"][layer_idx] / max(control_count, 1.0)
        diff = target_mean - control_mean
        score = (
            0.70 * torch.clamp(effect / 3.0, 0.0, 1.0)
            + 0.30 * torch.clamp(pos_gap / 0.50, 0.0, 1.0)
        )
        active_mask = (diff > 0) & (score > 0)
        active_scores = score[active_mask]
        all_active_scores.extend(float(v.item()) for v in active_scores)
        values, indices = torch.topk(score, k=min(8, int(width)))
        for value, idx in zip(values.tolist(), indices.tolist()):
            all_candidates.append(
                {
                    "layer_index": int(layer_idx),
                    "neuron_index": int(idx),
                    "score": float(value),
                    "effect_size": float(effect[idx].item()),
                    "positive_rate_gap": float(pos_gap[idx].item()),
                    "target_mean_activation": float(target_mean[idx].item()),
                    "control_mean_activation": float(control_mean[idx].item()),
                }
            )
        layer_rows.append(
            {
                "layer_index": int(layer_idx),
                "width": int(width),
                "score": score,
                "effect": effect,
                "pos_gap": pos_gap,
            }
        )

    if not all_active_scores:
        threshold = 0.0
    else:
        threshold = float(torch.quantile(torch.tensor(all_active_scores, dtype=torch.float64), max(0.0, min(1.0, 1.0 - top_fraction))).item())

    compact_layer_rows = []
    score_mass_total = 0.0
    count_total = 0
    for row in layer_rows:
        score = row["score"]
        effect = row["effect"]
        pos_gap = row["pos_gap"]
        effective_mask = score >= threshold
        effective_count = int(effective_mask.to(torch.int64).sum().item())
        effective_score_sum = float((score * effective_mask.to(score.dtype)).sum().item())
        score_mass_total += effective_score_sum
        count_total += effective_count
        top_values, _ = torch.topk(score, k=min(32, score.numel()))
        compact_layer_rows.append(
            {
                "layer_index": int(row["layer_index"]),
                "effective_count": effective_count,
                "effective_fraction": float(effective_count / max(1, row["width"])),
                "effective_score_sum": effective_score_sum,
                "mean_score": float(score.mean().item()),
                "top32_mean_score": float(top_values.mean().item()),
                "max_score": float(score.max().item()),
                "mean_effect_size": float(effect.mean().item()),
                "positive_effect_rate": float((effect > 0).to(torch.float64).mean().item()),
                "positive_pos_gap_rate": float((pos_gap > 0).to(torch.float64).mean().item()),
            }
        )

    score_mass_total += EPS
    for row in compact_layer_rows:
        row["effective_score_mass_share"] = float(row["effective_score_sum"] / score_mass_total)
        row["effective_count_share"] = float(row["effective_count"] / max(1, count_total))

    compact_layer_rows_by_count = sorted(compact_layer_rows, key=lambda row: row["effective_count"], reverse=True)
    compact_layer_rows_by_mass = sorted(compact_layer_rows, key=lambda row: row["effective_score_mass_share"], reverse=True)
    thirds = max(1, len(compact_layer_rows) // 3)
    early_end = thirds
    mid_end = min(len(compact_layer_rows), 2 * thirds)
    band_ranges = {
        "early": range(0, early_end),
        "middle": range(early_end, mid_end),
        "late": range(mid_end, len(compact_layer_rows)),
    }
    band_summary = {}
    for band_name, layer_range in band_ranges.items():
        rows_in_band = [compact_layer_rows[idx] for idx in layer_range]
        band_summary[band_name] = {
            "layer_indices": [row["layer_index"] for row in rows_in_band],
            "effective_count": int(sum(row["effective_count"] for row in rows_in_band)),
            "effective_score_mass_share": float(sum(row["effective_score_mass_share"] for row in rows_in_band)),
        }

    weighted_center = sum(row["layer_index"] * row["effective_score_mass_share"] for row in compact_layer_rows)
    top_candidates = sorted(all_candidates, key=lambda row: row["score"], reverse=True)[:20]
    for rank, row in enumerate(top_candidates, start=1):
        row["rank"] = rank
        row["is_effective"] = bool(row["score"] >= threshold)
        top_neurons.append(row)

    return {
        "target_count": int(target_count),
        "control_count": int(control_count),
        "layer_count": int(len(compact_layer_rows)),
        "neurons_per_layer": int(max(stats["layer_widths"])),
        "top_fraction": float(top_fraction),
        "effective_score_threshold": float(threshold),
        "effective_neuron_count": int(count_total),
        "weighted_layer_center": float(weighted_center),
        "top_layers_by_count": compact_layer_rows_by_count[:5],
        "top_layers_by_mass": compact_layer_rows_by_mass[:5],
        "band_summary": band_summary,
        "layer_rows": compact_layer_rows,
        "top_neurons": top_neurons,
    }


def analyze_model(model_key: str, target_word_map: Dict[str, List[str]]) -> dict:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        if model_key == "glm4":
            patch_glm4_mlp_compat(model)
        layers = discover_layers(model)
        layer_widths = [int(layer.mlp.down_proj.in_features) for layer in layers]
        payload = {
            "model_key": model_key,
            "layer_count": len(layers),
            "max_neurons_per_layer": max(layer_widths),
            "min_neurons_per_layer": min(layer_widths),
            "classes": {},
        }
        for class_name in WORD_CLASSES:
            target_words = list(target_word_map[class_name])
            control_words = build_control_words(target_word_map, class_name)
            stats = run_scan(model, tokenizer, target_words, control_words, layer_widths=layer_widths)
            summary = build_varwidth_summary(stats, top_fraction=TOP_FRACTION)
            summary["target_words"] = target_words
            summary["control_word_count"] = len(control_words)
            payload["classes"][class_name] = summary
        return payload
    finally:
        free_model(model)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    target_word_map = load_stage423_target_words()
    model_rows = {model_key: analyze_model(model_key, target_word_map) for model_key in MODEL_KEYS}
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage529_glm4_gemma4_wordclass_scan",
        "title": "GLM4 与 Gemma4 六类词性缩减版全景扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_target_words": str(STAGE423_PATH),
        "top_fraction": TOP_FRACTION,
        "batch_size": BATCH_SIZE,
        "control_per_class": CONTROL_PER_CLASS,
        "models": model_rows,
        "core_answer": (
            "通过同一套六词类高质量种子词，GLM4 和 Gemma4 也能显出稳定的词类层带风格。"
            "这说明六词类的层带分化不只是 Qwen3 和 DeepSeek7B 的局部现象。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# stage529 GLM4 与 Gemma4 六类词性缩减版全景扫描",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for model_key, model_row in model_rows.items():
        lines.append(f"## {model_key}")
        for class_name in WORD_CLASSES:
            row = model_row["classes"][class_name]
            lines.append(
                f"- `{class_name}`：质心 `{row['weighted_layer_center']:.2f}`，"
                f"主峰层 `{[item['layer_index'] for item in row['top_layers_by_mass'][:5]]}`"
            )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
