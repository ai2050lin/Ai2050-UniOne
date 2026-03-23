#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

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


BATCH_SIZE = 128
EPS = 1e-8

LEXICAL_TYPE_LABELS = {
    "noun": "Noun",
    "adjective": "Adjective",
    "verb": "Verb",
    "adverb": "Adverb",
    "function": "Function",
}


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


def select_rows(
    rows: Sequence[Dict[str, object]],
    target_lexical_type: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    target_rows = [row for row in rows if row["lexical_type"] == target_lexical_type]
    control_rows = [row for row in rows if row["lexical_type"] != target_lexical_type]
    if not target_rows or not control_rows:
        raise RuntimeError(f"{target_lexical_type} 目标组或控制组为空，无法继续进行神经元探针")
    return target_rows, control_rows


def unique_target_groups(target_rows: Sequence[Dict[str, object]]) -> List[str]:
    counts = Counter(str(row["group"]) for row in target_rows)
    min_group_count = max(5, min(25, len(target_rows) // 50))
    groups = [name for name, count in sorted(counts.items()) if count >= min_group_count]
    if not groups:
        groups = sorted(counts)
    return groups


def top_two_bands(target_rows: Sequence[Dict[str, object]]) -> Tuple[str, str, Dict[str, int]]:
    counts = Counter(str(row["band"]) for row in target_rows)
    ordered = [name for name, _count in counts.most_common()]
    if len(ordered) == 1:
        ordered.append(ordered[0])
    return ordered[0], ordered[1], dict(counts)


def init_stats(layer_count: int, neuron_count: int, group_names: Sequence[str]) -> Dict[str, object]:
    shape = (layer_count, neuron_count)
    group_shape = (len(group_names), layer_count, neuron_count)
    return {
        "target_count": 0,
        "control_count": 0,
        "target_sum": torch.zeros(shape, dtype=torch.float64),
        "target_sumsq": torch.zeros(shape, dtype=torch.float64),
        "target_pos": torch.zeros(shape, dtype=torch.float64),
        "control_sum": torch.zeros(shape, dtype=torch.float64),
        "control_sumsq": torch.zeros(shape, dtype=torch.float64),
        "control_pos": torch.zeros(shape, dtype=torch.float64),
        "group_sum": torch.zeros(group_shape, dtype=torch.float64),
        "group_count": torch.zeros(len(group_names), dtype=torch.float64),
        "band_sum": {},
        "band_count": Counter(),
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
    is_target: bool,
) -> None:
    sum_key = "target_sum" if is_target else "control_sum"
    sq_key = "target_sumsq" if is_target else "control_sumsq"
    pos_key = "target_pos" if is_target else "control_pos"
    count_key = "target_count" if is_target else "control_count"

    stats[sum_key] += sample_tensor.sum(dim=0)
    stats[sq_key] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[pos_key] += (sample_tensor > 0).to(torch.float64).sum(dim=0)
    stats[count_key] += sample_tensor.shape[0]


def update_target_stats(
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
        if band_name not in stats["band_sum"]:
            stats["band_sum"][band_name] = torch.zeros_like(sample_tensor[0])
        stats["band_sum"][band_name] += sample_tensor[sample_idx]
        stats["band_count"][band_name] += 1


def process_batch(
    model,
    tokenizer,
    layer_outputs: List[torch.Tensor | None],
    batch_rows: Sequence[Dict[str, object]],
    *,
    is_target: bool,
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
    update_global_stats(stats, sample_tensor, is_target=is_target)
    if is_target:
        update_target_stats(stats, sample_tensor, batch_rows, group_index)


def run_scan(
    model,
    tokenizer,
    target_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    group_names: Sequence[str],
) -> Dict[str, object]:
    layer_outputs, hooks = capture_mlp_activations(model)
    layer_count = len(model.transformer.h)
    neuron_count = model.transformer.h[0].mlp.c_fc.nf
    stats = init_stats(layer_count, neuron_count, group_names)
    group_index = {name: idx for idx, name in enumerate(group_names)}

    try:
        for start in range(0, len(target_rows), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                layer_outputs,
                target_rows[start : start + BATCH_SIZE],
                is_target=True,
                stats=stats,
                group_index=group_index,
            )
        for start in range(0, len(control_rows), BATCH_SIZE):
            process_batch(
                model,
                tokenizer,
                layer_outputs,
                control_rows[start : start + BATCH_SIZE],
                is_target=False,
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
    primary_band_name: str,
    secondary_band_name: str,
) -> Dict[str, object]:
    target_count = float(stats["target_count"])
    control_count = float(stats["control_count"])
    target_mean = stats["target_sum"] / max(target_count, 1.0)
    control_mean = stats["control_sum"] / max(control_count, 1.0)
    target_var = stats["target_sumsq"] / max(target_count, 1.0) - target_mean * target_mean
    control_var = stats["control_sumsq"] / max(control_count, 1.0) - control_mean * control_mean
    pooled_std = torch.sqrt(((target_var.clamp_min(0.0) + control_var.clamp_min(0.0)) / 2.0).clamp_min(EPS))

    effect = (target_mean - control_mean) / pooled_std
    pos_gap = stats["target_pos"] / max(target_count, 1.0) - stats["control_pos"] / max(control_count, 1.0)
    diff = target_mean - control_mean

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

    primary_band_mean = stats["band_sum"][primary_band_name] / max(float(stats["band_count"][primary_band_name]), 1.0)
    secondary_band_mean = stats["band_sum"][secondary_band_name] / max(
        float(stats["band_count"][secondary_band_name]),
        1.0,
    )
    primary_band_bias = primary_band_mean - secondary_band_mean
    secondary_band_bias = secondary_band_mean - primary_band_mean

    group_best_values, group_best_indices = group_gain.max(dim=0)

    layer_rows = []
    top_general_neurons = []
    top_primary_band_bias_neurons = []
    top_secondary_band_bias_neurons = []

    for layer_idx in range(general_rule_score.shape[0]):
        layer_scores = general_rule_score[layer_idx]
        top_values, _top_indices = torch.topk(layer_scores, k=min(20, layer_scores.numel()))
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
            "target_mean_activation": float(target_mean[layer_idx, neuron_idx].item()),
            "control_mean_activation": float(control_mean[layer_idx, neuron_idx].item()),
            "group_support_ratio": float(group_support[layer_idx, neuron_idx].item()),
            "group_entropy": float(entropy[layer_idx, neuron_idx].item()),
            "dominant_group_name": group_names[group_idx],
            "dominant_group_share": float(dominant_group_share[layer_idx, neuron_idx].item()),
            "dominant_group_gain": float(group_best_values[layer_idx, neuron_idx].item()),
            "primary_band_bias": float(primary_band_bias[layer_idx, neuron_idx].item()),
            "secondary_band_bias": float(secondary_band_bias[layer_idx, neuron_idx].item()),
        }

    flat_general = general_rule_score.flatten()
    flat_primary = (primary_band_bias * torch.clamp(effect, min=0.0)).flatten()
    flat_secondary = (secondary_band_bias * torch.clamp(effect, min=0.0)).flatten()

    top_general_values, top_general_indices = torch.topk(flat_general, k=24)
    top_primary_values, top_primary_indices = torch.topk(flat_primary, k=16)
    top_secondary_values, top_secondary_indices = torch.topk(flat_secondary, k=16)

    neuron_count = general_rule_score.shape[1]
    for idx_in_top, flat_idx in enumerate(top_general_indices.tolist()):
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        row = neuron_row(layer_idx, neuron_idx)
        row["general_rank_score"] = float(top_general_values[idx_in_top].item())
        top_general_neurons.append(row)
    for idx_in_top, flat_idx in enumerate(top_primary_indices.tolist()):
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        row = neuron_row(layer_idx, neuron_idx)
        row["primary_band_bias_score"] = float(top_primary_values[idx_in_top].item())
        top_primary_band_bias_neurons.append(row)
    for idx_in_top, flat_idx in enumerate(top_secondary_indices.tolist()):
        layer_idx = flat_idx // neuron_count
        neuron_idx = flat_idx % neuron_count
        row = neuron_row(layer_idx, neuron_idx)
        row["secondary_band_bias_score"] = float(top_secondary_values[idx_in_top].item())
        top_secondary_band_bias_neurons.append(row)

    dominant_general_layer = max(layer_rows, key=lambda row: row["top20_mean_general_rule_score"])
    probe_score = (
        0.35 * clamp01(dominant_general_layer["top20_mean_general_rule_score"] / 0.55)
        + 0.30 * clamp01(top_general_neurons[0]["general_rule_score"] / 0.75)
        + 0.20 * clamp01(top_general_neurons[0]["group_support_ratio"] / 0.55)
        + 0.15 * clamp01(top_primary_band_bias_neurons[0]["primary_band_bias"] / 0.08 if top_primary_band_bias_neurons else 0.0)
    )

    return {
        "target_count": int(target_count),
        "control_count": int(control_count),
        "layer_count": int(general_rule_score.shape[0]),
        "neurons_per_layer": int(general_rule_score.shape[1]),
        "target_group_count": len(group_names),
        "primary_band_name": primary_band_name,
        "secondary_band_name": secondary_band_name,
        "dominant_general_layer_index": int(dominant_general_layer["layer_index"]),
        "dominant_general_layer_score": float(dominant_general_layer["top20_mean_general_rule_score"]),
        "wordclass_neuron_basic_probe_score": float(probe_score),
        "layer_rows": layer_rows,
        "top_general_neurons": top_general_neurons,
        "top_primary_band_bias_neurons": top_primary_band_bias_neurons,
        "top_secondary_band_bias_neurons": top_secondary_band_bias_neurons,
    }


def build_summary(
    *,
    experiment_id: str,
    title: str,
    status_short: str,
    target_lexical_type: str,
    stage119_summary: Dict[str, object],
    target_rows: Sequence[Dict[str, object]],
    neuron_tables: Dict[str, object],
    band_counts: Dict[str, int],
) -> Dict[str, object]:
    target_group_counts = Counter(str(row["group"]) for row in target_rows)
    target_band_counts = Counter(str(row["band"]) for row in target_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": experiment_id,
        "title": title,
        "status_short": status_short,
        "model_name": "gpt2",
        "model_path": str(MODEL_PATH),
        "source_stage": "stage119_gpt2_embedding_full_vocab_scan",
        "source_word_count": stage119_summary["clean_unique_word_count"],
        "target_lexical_type": target_lexical_type,
        "target_group_counts_top12": dict(target_group_counts.most_common(12)),
        "target_band_counts": dict(target_band_counts),
        "band_counts_reference": dict(band_counts),
        **neuron_tables,
    }


def build_report(summary: Dict[str, object]) -> str:
    lexical_type = summary["target_lexical_type"]
    title_label = LEXICAL_TYPE_LABELS.get(lexical_type, lexical_type.title())
    lines = [
        f"# {summary['experiment_id']}: {title_label} 神经元基础探针",
        "",
        "## 核心结果",
        f"- 目标词类: {lexical_type}",
        f"- 目标数量: {summary['target_count']}",
        f"- 控制组数量: {summary['control_count']}",
        f"- 层数: {summary['layer_count']}",
        f"- 每层神经元数: {summary['neurons_per_layer']}",
        f"- 目标语义组数量: {summary['target_group_count']}",
        f"- 主导通用层: L{summary['dominant_general_layer_index']}",
        f"- 主导通用层分数: {summary['dominant_general_layer_score']:.4f}",
        f"- 主要 band（尺度带）: {summary['primary_band_name']}",
        f"- 次要 band（尺度带）: {summary['secondary_band_name']}",
        f"- 词类神经元基础探针分数: {summary['wordclass_neuron_basic_probe_score']:.4f}",
        "",
        "## 顶级通用神经元",
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

    lines.extend(
        [
            "",
            f"## {summary['primary_band_name']} / {summary['secondary_band_name']} 分裂神经元",
        ]
    )
    for row in summary["top_primary_band_bias_neurons"][:6]:
        lines.append(
            "- "
            f"{summary['primary_band_name']} L{row['layer_index']} N{row['neuron_index']}: "
            f"bias={row['primary_band_bias']:.4f}, effect={row['effect_size']:.4f}, "
            f"dominant_group={row['dominant_group_name']}"
        )
    for row in summary["top_secondary_band_bias_neurons"][:6]:
        lines.append(
            "- "
            f"{summary['secondary_band_name']} L{row['layer_index']} N{row['neuron_index']}: "
            f"bias={row['secondary_band_bias']:.4f}, effect={row['effect_size']:.4f}, "
            f"dominant_group={row['dominant_group_name']}"
        )

    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果顶级神经元同时具有较高 effect（效应量）和较高 group support（组覆盖），说明该词类内部存在跨家族共享的较一般编码规则。",
            "- 如果主要 band 与次要 band 的分裂明显，说明该词类内部至少包含两条不同尺度的编码链，而不是单一平面。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / f"{summary['experiment_id'].upper()}_REPORT.md"
    layer_path = output_dir / "layer_rows.json"
    general_path = output_dir / "top_general_neurons.json"
    primary_path = output_dir / "top_primary_band_bias_neurons.json"
    secondary_path = output_dir / "top_secondary_band_bias_neurons.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    report_path.write_text(build_report(summary), encoding="utf-8-sig")
    layer_path.write_text(json.dumps(summary["layer_rows"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    general_path.write_text(json.dumps(summary["top_general_neurons"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    primary_path.write_text(
        json.dumps(summary["top_primary_band_bias_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    secondary_path.write_text(
        json.dumps(summary["top_secondary_band_bias_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    return {
        "summary": summary_path,
        "report": report_path,
        "layer_rows": layer_path,
        "top_general_neurons": general_path,
        "top_primary_band_bias_neurons": primary_path,
        "top_secondary_band_bias_neurons": secondary_path,
    }


def run_wordclass_analysis(
    *,
    target_lexical_type: str,
    experiment_id: str,
    title: str,
    status_short: str,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path,
) -> Dict[str, object]:
    stage119_summary, rows = ensure_stage119_rows(input_dir)
    target_rows, control_rows = select_rows(rows, target_lexical_type)
    group_names = unique_target_groups(target_rows)
    primary_band_name, secondary_band_name, band_counts = top_two_bands(target_rows)
    model, tokenizer = load_model()
    stats = run_scan(model, tokenizer, target_rows, control_rows, group_names)
    neuron_tables = build_neuron_tables(
        stats,
        group_names,
        primary_band_name=primary_band_name,
        secondary_band_name=secondary_band_name,
    )
    summary = build_summary(
        experiment_id=experiment_id,
        title=title,
        status_short=status_short,
        target_lexical_type=target_lexical_type,
        stage119_summary=stage119_summary,
        target_rows=target_rows,
        neuron_tables=neuron_tables,
        band_counts=band_counts,
    )
    write_outputs(summary, output_dir)
    return summary
