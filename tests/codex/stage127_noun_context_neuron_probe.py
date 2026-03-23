#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from stage124_noun_neuron_basic_probe import OUTPUT_DIR as STAGE124_OUTPUT_DIR
from wordclass_neuron_basic_probe_lib import (
    build_neuron_tables,
    build_summary,
    capture_mlp_activations,
    clamp01,
    load_model,
    remove_hooks,
    top_two_bands,
    unique_target_groups,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage127_noun_context_neuron_probe_20260323"
BATCH_SIZE = 96
TEMPLATE_SPECS = [
    ("The", " is nearby."),
    ("A", " can matter."),
    ("This", " remains visible."),
]


def select_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    control_rows = [row for row in rows if row["lexical_type"] != "noun"]
    if not noun_rows or not control_rows:
        raise RuntimeError("名词上下文探针缺少目标组或控制组")
    return noun_rows, control_rows


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
        "band_count": {},
    }


def update_global_stats(stats: Dict[str, object], sample_tensor: torch.Tensor, is_target: bool) -> None:
    prefix = "target" if is_target else "control"
    stats[f"{prefix}_sum"] += sample_tensor.sum(dim=0)
    stats[f"{prefix}_sumsq"] += (sample_tensor * sample_tensor).sum(dim=0)
    stats[f"{prefix}_pos"] += (sample_tensor > 0).to(torch.float64).sum(dim=0)
    stats[f"{prefix}_count"] += sample_tensor.shape[0]


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
            stats["band_count"][band_name] = 0
        stats["band_sum"][band_name] += sample_tensor[sample_idx]
        stats["band_count"][band_name] += 1


def build_batch_inputs(
    tokenizer,
    batch_rows: Sequence[Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    prefix_text, suffix_text = TEMPLATE_SPECS[0]
    # 模板轮转，避免某一个句壳主导全部结果。
    input_rows = []
    target_ranges = []
    pad_token_id = tokenizer.pad_token_id
    for offset, row in enumerate(batch_rows):
        prefix_text, suffix_text = TEMPLATE_SPECS[offset % len(TEMPLATE_SPECS)]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        word_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not word_ids:
            word_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        full_ids = prefix_ids + word_ids + suffix_ids
        input_rows.append(full_ids)
        target_ranges.append((len(prefix_ids), len(prefix_ids) + len(word_ids)))

    max_len = max(len(ids) for ids in input_rows)
    input_ids = []
    attention_mask = []
    for ids in input_rows:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        target_ranges,
    )


def mean_target_activation(layer_tensor: torch.Tensor, target_ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    rows = []
    for sample_idx, (start, end) in enumerate(target_ranges):
        rows.append(layer_tensor[sample_idx, start:end, :].mean(dim=0))
    return torch.stack(rows, dim=0)


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
    input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows)
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

    per_layer_rows = []
    for layer_tensor in layer_outputs:
        if layer_tensor is None:
            raise RuntimeError("句中名词探针未捕获到层输出")
        per_layer_rows.append(mean_target_activation(layer_tensor, target_ranges).to(torch.float64))
    sample_tensor = torch.stack(per_layer_rows, dim=1)
    update_global_stats(stats, sample_tensor, is_target=is_target)
    if is_target:
        update_target_stats(stats, sample_tensor, batch_rows, group_index)


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


def load_stage124_summary() -> Dict[str, object]:
    summary_path = STAGE124_OUTPUT_DIR / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8-sig"))


def build_context_summary(
    stage119_summary: Dict[str, object],
    noun_rows: Sequence[Dict[str, object]],
    neuron_tables: Dict[str, object],
    band_counts: Dict[str, int],
) -> Dict[str, object]:
    summary = build_summary(
        experiment_id="stage127_noun_context_neuron_probe",
        title="Noun 上下文神经元探针",
        status_short="gpt2_noun_context_neuron_probe_ready",
        target_lexical_type="noun",
        stage119_summary=stage119_summary,
        target_rows=noun_rows,
        neuron_tables=neuron_tables,
        band_counts=band_counts,
    )
    stage124_summary = load_stage124_summary()
    l11_preserved = int(summary["dominant_general_layer_index"] == stage124_summary["dominant_general_layer_index"] == 11)
    preservation_ratio = summary["dominant_general_layer_score"] / max(stage124_summary["dominant_general_layer_score"], 1e-8)
    context_score = (
        0.50 * l11_preserved
        + 0.30 * clamp01(preservation_ratio)
        + 0.20 * clamp01(summary["top_general_neurons"][0]["group_support_ratio"] / 0.85)
    )
    summary["context_template_count"] = len(TEMPLATE_SPECS)
    summary["stage124_dominant_general_layer_index"] = stage124_summary["dominant_general_layer_index"]
    summary["l11_rule_preserved"] = bool(l11_preserved)
    summary["l11_preservation_ratio"] = float(preservation_ratio)
    summary["noun_context_neuron_probe_score"] = float(context_score)
    return summary


def write_context_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    write_outputs(summary, output_dir)


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    stage119_summary, rows = ensure_stage119_rows(input_dir)
    noun_rows, control_rows = select_rows(rows)
    group_names = unique_target_groups(noun_rows)
    primary_band_name, secondary_band_name, band_counts = top_two_bands(noun_rows)
    model, tokenizer = load_model()
    stats = run_scan(model, tokenizer, noun_rows, control_rows, group_names)
    neuron_tables = build_neuron_tables(
        stats,
        group_names,
        primary_band_name=primary_band_name,
        secondary_band_name=secondary_band_name,
    )
    summary = build_context_summary(stage119_summary, noun_rows, neuron_tables, band_counts)
    write_context_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Noun 上下文神经元探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage127 输出目录")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "dominant_general_layer_index": summary["dominant_general_layer_index"],
                "noun_context_neuron_probe_score": summary["noun_context_neuron_probe_score"],
                "l11_rule_preserved": summary["l11_rule_preserved"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
