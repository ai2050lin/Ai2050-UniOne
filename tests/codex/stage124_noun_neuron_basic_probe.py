#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage124: noun（名词） neuron（神经元） basic probe（基础探针）分析。

目标：
1. 对 Stage119 中全部 noun（名词）词项做 GPT-2 MLP（多层感知机）神经元响应扫描。
2. 对比全部非名词词项，寻找更一般的 noun-selective（名词选择性）编码规则。
3. 继续拆出 general noun neurons（通用名词神经元）与 macro/meso split neurons（宏观/中观分裂神经元）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage119_gpt2_embedding_full_vocab_scan import MODEL_PATH
from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323"
BATCH_SIZE = 128
EPS = 1e-8


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def select_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    control_rows = [row for row in rows if row["lexical_type"] != "noun"]
    if not noun_rows or not control_rows:
        raise RuntimeError("名词或控制组为空，无法继续进行神经元探针")
    return noun_rows, control_rows


def unique_noun_groups(noun_rows: Sequence[Dict[str, object]]) -> List[str]:
    counts = Counter(str(row["group"]) for row in noun_rows)
    # 过滤过于稀疏的组，避免全局规则被极少量噪声组拖歪。
    groups = [name for name, count in sorted(counts.items()) if count >= 25]
    if not groups:
        groups = sorted(counts)
    return groups


def init_stats(layer_count: int, neuron_count: int, group_names: Sequence[str]) -> Dict[str, object]:
    shape = (layer_count, neuron_count)
    group_shape = (len(group_names), layer_count, neuron_count)
    return {
        "noun_count": 0,
        "control_count": 0,
        "noun_sum": torch.zeros(shape, dtype=torch.float64),
        "noun_sumsq": torch.zeros(shape, dtype=torch.float64),
        "noun_pos": torch.zeros(shape, dtype=torch.float64),
        "control_sum": torch.zeros(shape, dtype=torch.float64),
        "control_sumsq": torch.zeros(shape, dtype=torch.float64),
        "control_pos": torch.zeros(shape, dtype=torch.float64),
        "group_sum": torch.zeros(group_shape, dtype=torch.float64),
        "group_count": torch.zeros(len(group_names), dtype=torch.float64),
        "meso_sum": torch.zeros(shape, dtype=torch.float64),
        "meso_count": 0,
        "macro_sum": torch.zeros(shape, dtype=torch.float64),
        "macro_count": 0,
    }


def capture_mlp_activations(model) -> Tuple[List[torch.Tensor | None], List[object]]:
    layer_outputs: List[torch.Tensor | None] = [None for _ in range(len(model.transformer.h))]
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            layer_outputs[layer_idx] = output.detach().cpu()

        return hook

    for layer_idx, block in enumerate(model.transformer.h):
        hooks.append(block.mlp.act.register_forward_hook(make_hook(layer_idx)))
    return layer_outputs, hooks


def remove_hooks(hooks: Sequence[object]) -> None:
    for hook in hooks:
        hook.remove()


def mean_token_activation(layer_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(layer_tensor.dtype)
    lengths = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(layer_tensor.dtype)
    return (layer_tensor * mask).sum(dim=1) / lengths


def update_global_stats(
    stats: Dict[str, object],
    sample_tensor: torch.Tensor,
    is_noun: bool,
) -> None:
    sum_key = "noun_sum" if is_noun else "control_sum"
    sq_key = "noun_sumsq" if is_noun else "control_sumsq"
    pos_key = "noun_pos" if is_noun else "control_pos"
    count_key = "noun_count" if is_noun else "control_count"

    stats[sum_key] += sample_tensor.sum(dim=0)
    stats[sq_key] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[pos_key] += (sample_tensor > 0).to(torch.float64).sum(dim=0)
    stats[count_key] += sample_tensor.shape[0]


def update_group_stats(
    stats: Dict[str, object],
    sample_tensor: torch.Tensor,
    batch_rows: Sequence[Dict[str, object]],
    group_index: Dict[str, int],
) -> None:
    for sample_idx, row in enumerate(batch_rows):
        group_name = str(row["group"])
        if group_name in group_index:
            group_idx = group_index[group_name]
            stats["group_sum"][group_idx] += sample_tensor[sample_idx]
            stats["group_count"][group_idx] += 1
        band_name = str(row["band"])
        if band_name == "meso":
            stats["meso_sum"] += sample_tensor[sample_idx]
            stats["meso_count"] += 1
        elif band_name == "macro":
            stats["macro_sum"] += sample_tensor[sample_idx]
            stats["macro_count"] += 1


def process_batch(
    model,
    tokenizer,
    layer_outputs: List[torch.Tensor | None],
    batch_rows: Sequence[Dict[str, object]],
    *,
    is_noun: bool,
    stats: Dict[str, object],
    group_index: Dict[str, int],
) -> None:
    batch_texts = [str(row["word"]) for row in batch_rows]
    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    attention_mask = encoded["attention_mask"]
    with torch.inference_mode():
        model(**encoded, use_cache=False, return_dict=True)

    per_layer_rows = []
    for layer_tensor in layer_outputs:
        if layer_tensor is None:
            raise RuntimeError("神经元激活捕获失败：存在空层输出")
        per_layer_rows.append(mean_token_activation(layer_tensor, attention_mask).to(torch.float64))

    sample_tensor = torch.stack(per_layer_rows, dim=1)
    update_global_stats(stats, sample_tensor, is_noun=is_noun)
    if is_noun:
        update_group_stats(stats, sample_tensor, batch_rows, group_index)


def run_scan(
    model,
    tokenizer,
    noun_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    group_names: Sequence[str],
) -> Dict[str, object]:
    layer_outputs, hooks = capture_mlp_activations(model)
    layer_count = len(model.transformer.h)
    neuron_count = model.transformer.h[0].mlp.c_fc.nf
    stats = init_stats(layer_count, neuron_count, group_names)
    group_index = {name: idx for idx, name in enumerate(group_names)}

    try:
        for start in range(0, len(noun_rows), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                layer_outputs,
                noun_rows[start : start + BATCH_SIZE],
                is_noun=True,
                stats=stats,
                group_index=group_index,
            )
        for start in range(0, len(control_rows), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                layer_outputs,
                control_rows[start : start + BATCH_SIZE],
                is_noun=False,
                stats=stats,
                group_index=group_index,
            )
    finally:
        remove_hooks(hooks)

    return stats


def entropy_normalized(weights: torch.Tensor) -> torch.Tensor:
    probs = weights / weights.sum(dim=0, keepdim=True).clamp_min(EPS)
    entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=0)
    denom = math.log(max(2, weights.shape[0]))
    return entropy / denom


def build_neuron_tables(
    stats: Dict[str, object],
    group_names: Sequence[str],
) -> Dict[str, object]:
    noun_count = float(stats["noun_count"])
    control_count = float(stats["control_count"])
    noun_mean = stats["noun_sum"] / max(noun_count, 1.0)
    control_mean = stats["control_sum"] / max(control_count, 1.0)
    noun_var = stats["noun_sumsq"] / max(noun_count, 1.0) - noun_mean * noun_mean
    control_var = stats["control_sumsq"] / max(control_count, 1.0) - control_mean * control_mean
    pooled_std = torch.sqrt(((noun_var.clamp_min(0.0) + control_var.clamp_min(0.0)) / 2.0).clamp_min(EPS))

    effect = (noun_mean - control_mean) / pooled_std
    pos_gap = stats["noun_pos"] / max(noun_count, 1.0) - stats["control_pos"] / max(control_count, 1.0)
    diff = noun_mean - control_mean

    valid_group_counts = stats["group_count"].clone()
    group_mean = stats["group_sum"] / valid_group_counts.view(-1, 1, 1).clamp_min(1.0)
    group_gain = (group_mean - control_mean.unsqueeze(0)).clamp_min(0.0)
    active_threshold = control_mean.unsqueeze(0) + 0.25 * diff.unsqueeze(0).clamp_min(0.0)
    group_support = (group_mean > active_threshold).to(torch.float64).mean(dim=0)
    entropy = entropy_normalized(group_gain + EPS)
    dominant_group_share = group_gain.max(dim=0).values / group_gain.sum(dim=0).clamp_min(EPS)

    general_rule_score = (
        0.45 * torch.clamp(effect / 3.0, 0.0, 1.0)
        + 0.25 * torch.clamp(pos_gap / 0.60, 0.0, 1.0)
        + 0.15 * group_support
        + 0.15 * entropy
    )

    meso_mean = stats["meso_sum"] / max(float(stats["meso_count"]), 1.0)
    macro_mean = stats["macro_sum"] / max(float(stats["macro_count"]), 1.0)
    macro_bias = macro_mean - meso_mean
    meso_bias = meso_mean - macro_mean

    group_best_values, group_best_indices = group_gain.max(dim=0)

    layer_rows = []
    top_general_neurons = []
    top_macro_bias_neurons = []
    top_meso_bias_neurons = []

    for layer_idx in range(general_rule_score.shape[0]):
        layer_scores = general_rule_score[layer_idx]
        top_values, top_indices = torch.topk(layer_scores, k=min(20, layer_scores.numel()))
        layer_rows.append(
            {
                "layer_index": layer_idx,
                "mean_general_rule_score": float(layer_scores.mean().item()),
                "top20_mean_general_rule_score": float(top_values.mean().item()),
                "positive_effect_rate": float((effect[layer_idx] > 0).to(torch.float64).mean().item()),
                "positive_pos_gap_rate": float((pos_gap[layer_idx] > 0).to(torch.float64).mean().item()),
            }
        )

    def neuron_row(layer_idx: int, neuron_idx: int) -> Dict[str, object]:
        group_idx = int(group_best_indices[layer_idx, neuron_idx].item())
        return {
            "layer_index": int(layer_idx),
            "neuron_index": int(neuron_idx),
            "general_rule_score": float(general_rule_score[layer_idx, neuron_idx].item()),
            "effect_size": float(effect[layer_idx, neuron_idx].item()),
            "positive_rate_gap": float(pos_gap[layer_idx, neuron_idx].item()),
            "noun_mean_activation": float(noun_mean[layer_idx, neuron_idx].item()),
            "control_mean_activation": float(control_mean[layer_idx, neuron_idx].item()),
            "group_support_ratio": float(group_support[layer_idx, neuron_idx].item()),
            "group_entropy": float(entropy[layer_idx, neuron_idx].item()),
            "dominant_group_name": group_names[group_idx],
            "dominant_group_share": float(dominant_group_share[layer_idx, neuron_idx].item()),
            "dominant_group_gain": float(group_best_values[layer_idx, neuron_idx].item()),
            "macro_bias": float(macro_bias[layer_idx, neuron_idx].item()),
            "meso_bias": float(meso_bias[layer_idx, neuron_idx].item()),
        }

    flat_general = general_rule_score.flatten()
    flat_macro = (macro_bias * torch.clamp(effect, min=0.0)).flatten()
    flat_meso = (meso_bias * torch.clamp(effect, min=0.0)).flatten()

    top_general_values, top_general_indices = torch.topk(flat_general, k=24)
    top_macro_values, top_macro_indices = torch.topk(flat_macro, k=16)
    top_meso_values, top_meso_indices = torch.topk(flat_meso, k=16)

    neuron_count = general_rule_score.shape[1]
    for flat_idx in top_general_indices.tolist():
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        top_general_neurons.append(neuron_row(layer_idx, neuron_idx))
    for flat_idx in top_macro_indices.tolist():
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        row = neuron_row(layer_idx, neuron_idx)
        row["macro_bias_score"] = float(top_macro_values[len(top_macro_bias_neurons)].item())
        top_macro_bias_neurons.append(row)
    for flat_idx in top_meso_indices.tolist():
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        row = neuron_row(layer_idx, neuron_idx)
        row["meso_bias_score"] = float(top_meso_values[len(top_meso_bias_neurons)].item())
        top_meso_bias_neurons.append(row)

    dominant_general_layer = max(layer_rows, key=lambda row: row["top20_mean_general_rule_score"])
    probe_score = (
        0.35 * clamp01(dominant_general_layer["top20_mean_general_rule_score"] / 0.55)
        + 0.30 * clamp01(top_general_neurons[0]["general_rule_score"] / 0.75)
        + 0.20 * clamp01(top_general_neurons[0]["group_support_ratio"] / 0.55)
        + 0.15 * clamp01(top_macro_bias_neurons[0]["macro_bias"] / 0.08 if top_macro_bias_neurons else 0.0)
    )

    return {
        "noun_count": int(noun_count),
        "control_count": int(control_count),
        "layer_count": int(general_rule_score.shape[0]),
        "neurons_per_layer": int(general_rule_score.shape[1]),
        "noun_group_count": len(group_names),
        "dominant_general_layer_index": int(dominant_general_layer["layer_index"]),
        "dominant_general_layer_score": float(dominant_general_layer["top20_mean_general_rule_score"]),
        "noun_neuron_basic_probe_score": float(probe_score),
        "layer_rows": layer_rows,
        "top_general_neurons": top_general_neurons,
        "top_macro_bias_neurons": top_macro_bias_neurons,
        "top_meso_bias_neurons": top_meso_bias_neurons,
    }


def build_summary(
    stage119_summary: Dict[str, object],
    noun_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    neuron_tables: Dict[str, object],
) -> Dict[str, object]:
    noun_group_counts = Counter(str(row["group"]) for row in noun_rows)
    noun_band_counts = Counter(str(row["band"]) for row in noun_rows)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage124_noun_neuron_basic_probe",
        "title": "Noun 神经元基础探针",
        "status_short": "gpt2_noun_neuron_probe_ready",
        "model_name": "gpt2",
        "model_path": str(MODEL_PATH),
        "source_stage": "stage119_gpt2_embedding_full_vocab_scan",
        "source_word_count": stage119_summary["clean_unique_word_count"],
        "noun_group_counts_top12": dict(noun_group_counts.most_common(12)),
        "noun_band_counts": dict(noun_band_counts),
        **neuron_tables,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage124: Noun 神经元基础探针",
        "",
        "## 核心结果",
        f"- 名词数量: {summary['noun_count']}",
        f"- 控制组数量: {summary['control_count']}",
        f"- 层数: {summary['layer_count']}",
        f"- 每层神经元数: {summary['neurons_per_layer']}",
        f"- 名词语义组数量: {summary['noun_group_count']}",
        f"- 主导通用层: L{summary['dominant_general_layer_index']}",
        f"- 主导通用层分数: {summary['dominant_general_layer_score']:.4f}",
        f"- 名词神经元基础探针分数: {summary['noun_neuron_basic_probe_score']:.4f}",
        "",
        "## 顶级通用名词神经元",
    ]

    for row in summary["top_general_neurons"][:12]:
        lines.append(
            "- "
            f"L{row['layer_index']} N{row['neuron_index']}: "
            f"rule={row['general_rule_score']:.4f}, "
            f"effect={row['effect_size']:.4f}, "
            f"support={row['group_support_ratio']:.4f}, "
            f"dominant_group={row['dominant_group_name']}"
        )

    lines.extend(["", "## 宏观 / 中观分裂神经元"])
    for row in summary["top_macro_bias_neurons"][:6]:
        lines.append(
            "- "
            f"macro L{row['layer_index']} N{row['neuron_index']}: "
            f"macro_bias={row['macro_bias']:.4f}, effect={row['effect_size']:.4f}, "
            f"dominant_group={row['dominant_group_name']}"
        )
    for row in summary["top_meso_bias_neurons"][:6]:
        lines.append(
            "- "
            f"meso L{row['layer_index']} N{row['neuron_index']}: "
            f"meso_bias={row['meso_bias']:.4f}, effect={row['effect_size']:.4f}, "
            f"dominant_group={row['dominant_group_name']}"
        )

    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果顶级神经元同时具有高 effect（效应量）和高 group support（组覆盖），说明它们更接近“通用名词编码”而不是家族专属词元。",
            "- 如果 macro / meso 分裂明显，说明名词内部至少还包含“抽象系统名词”和“具体对象名词”两条不同编码链。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "STAGE124_NOUN_NEURON_BASIC_PROBE_REPORT.md"
    layer_path = output_dir / "layer_rows.json"
    general_path = output_dir / "top_general_neurons.json"
    macro_path = output_dir / "top_macro_bias_neurons.json"
    meso_path = output_dir / "top_meso_bias_neurons.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")
    layer_path.write_text(json.dumps(summary["layer_rows"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    general_path.write_text(
        json.dumps(summary["top_general_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    macro_path.write_text(
        json.dumps(summary["top_macro_bias_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    meso_path.write_text(
        json.dumps(summary["top_meso_bias_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    return {
        "summary": summary_path,
        "report": report_path,
        "layer_rows": layer_path,
        "top_general_neurons": general_path,
        "top_macro_bias_neurons": macro_path,
        "top_meso_bias_neurons": meso_path,
    }


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    stage119_summary, rows = ensure_stage119_rows(input_dir)
    noun_rows, control_rows = select_rows(rows)
    group_names = unique_noun_groups(noun_rows)
    model, tokenizer = load_model()
    stats = run_scan(model, tokenizer, noun_rows, control_rows, group_names)
    neuron_tables = build_neuron_tables(stats, group_names)
    summary = build_summary(stage119_summary, noun_rows, control_rows, neuron_tables)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Noun 神经元基础探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage124 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "dominant_general_layer_index": summary["dominant_general_layer_index"],
                "noun_neuron_basic_probe_score": summary["noun_neuron_basic_probe_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
