#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from wordclass_neuron_basic_probe_lib import clamp01, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE124_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323"
STAGE130_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage133_complex_discourse_noun_propagation_20260323"

BATCH_SIZE = 96
NOUN_SAMPLE_LIMIT = 2048
EARLY_LAYER = 1
LATE_LAYER = 11

DISCOURSE_FAMILIES = [
    {
        "name": "discourse_remention",
        "prefix": "The",
        "middle": " appeared in the archive. Later the team rechecked the",
        "suffix": " before sunset.",
    },
    {
        "name": "causal_remention",
        "prefix": "Because the",
        "middle": " changed overnight, the memo described the",
        "suffix": " carefully.",
    },
    {
        "name": "nested_memory",
        "prefix": "When the analyst said the",
        "middle": " that the archive stored was fragile, the report flagged the",
        "suffix": " again.",
    },
    {
        "name": "contrastive_remention",
        "prefix": "Although the",
        "middle": " looked ordinary, the review later called the",
        "suffix": " critical.",
    },
    {
        "name": "cross_sentence_bridge",
        "prefix": "We studied the",
        "middle": " yesterday. Today the planners still discussed the",
        "suffix": " in detail.",
    },
]


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def choose_noun_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    noun_rows.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    return noun_rows[:NOUN_SAMPLE_LIMIT]


def select_early_neurons() -> List[int]:
    summary = load_json(STAGE130_DIR / "summary.json")
    rows = [row for row in summary["recurrent_early_neurons"] if int(row["layer_index"]) == EARLY_LAYER]
    rows.sort(key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])), reverse=True)
    return [int(row["neuron_index"]) for row in rows[:12]]


def select_late_neurons() -> List[int]:
    summary = load_json(STAGE124_DIR / "summary.json")
    rows = [row for row in summary["top_general_neurons"] if int(row["layer_index"]) == LATE_LAYER]
    return [int(row["neuron_index"]) for row in rows[:12]]


def capture_layers(model) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    buffers: Dict[int, torch.Tensor | None] = {EARLY_LAYER: None, LATE_LAYER: None}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            buffers[layer_idx] = output.detach().cpu()

        return hook

    for layer_idx in buffers:
        handles.append(model.transformer.h[layer_idx].mlp.act.register_forward_hook(make_hook(layer_idx)))
    return buffers, handles


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def find_all_subsequence(sequence: Sequence[int], target: Sequence[int]) -> List[int]:
    if not target:
        return []
    hits = []
    for idx in range(len(sequence) - len(target) + 1):
        if list(sequence[idx : idx + len(target)]) == list(target):
            hits.append(idx)
    return hits


def build_batch_inputs(
    tokenizer,
    batch_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int, int]]]:
    input_rows = []
    target_ranges = []
    pad_token_id = tokenizer.pad_token_id
    prefix_ids = tokenizer.encode(family["prefix"], add_special_tokens=False)
    middle_ids = tokenizer.encode(family["middle"], add_special_tokens=False)
    suffix_ids = tokenizer.encode(family["suffix"], add_special_tokens=False)

    for row in batch_rows:
        word_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not word_ids:
            word_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)
        full_ids = prefix_ids + word_ids + middle_ids + word_ids + suffix_ids
        hits = find_all_subsequence(full_ids, word_ids)
        if len(hits) < 2:
            raise RuntimeError("复杂语篇名词传播探针未找到两次名词位置")
        first_start = hits[0]
        last_start = hits[-1]
        input_rows.append(full_ids)
        target_ranges.append((first_start, first_start + len(word_ids), last_start, last_start + len(word_ids)))

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


def mean_selected(layer_tensor: torch.Tensor, start: int, end: int, neurons: Sequence[int]) -> float:
    neuron_tensor = torch.tensor(neurons, dtype=torch.long)
    token_mean = layer_tensor[start:end, :].mean(dim=0)
    return float(token_mean[neuron_tensor].mean().item())


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    return cov / math.sqrt(x_var * y_var)


def sign_consistency(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return 0.0
    hits = 0
    for x, y in zip(xs, ys):
        if x == 0.0 or y == 0.0 or x * y > 0:
            hits += 1
    return hits / len(xs)


def run_family(
    model,
    tokenizer,
    sample_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
    early_neurons: Sequence[int],
    late_neurons: Sequence[int],
) -> Dict[str, object]:
    buffers, handles = capture_layers(model)
    early_first_values: List[float] = []
    early_last_values: List[float] = []
    late_first_values: List[float] = []
    late_last_values: List[float] = []

    try:
        for start in range(0, len(sample_rows), BATCH_SIZE):
            batch_rows = sample_rows[start : start + BATCH_SIZE]
            input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows, family)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

            early_layer = buffers[EARLY_LAYER]
            late_layer = buffers[LATE_LAYER]
            if early_layer is None or late_layer is None:
                raise RuntimeError("复杂语篇名词传播探针未捕获到目标层输出")

            for sample_idx, (first_start, first_end, last_start, last_end) in enumerate(target_ranges):
                early_first_values.append(mean_selected(early_layer[sample_idx], first_start, first_end, early_neurons))
                early_last_values.append(mean_selected(early_layer[sample_idx], last_start, last_end, early_neurons))
                late_first_values.append(mean_selected(late_layer[sample_idx], first_start, first_end, late_neurons))
                late_last_values.append(mean_selected(late_layer[sample_idx], last_start, last_end, late_neurons))
    finally:
        remove_hooks(handles)

    early_corr = correlation(early_first_values, early_last_values)
    late_corr = correlation(late_first_values, late_last_values)
    early_consistency = sign_consistency(early_first_values, early_last_values)
    late_consistency = sign_consistency(late_first_values, late_last_values)
    family_score = (
        0.30 * clamp01((early_corr + 1.0) / 2.0)
        + 0.30 * clamp01((late_corr + 1.0) / 2.0)
        + 0.20 * early_consistency
        + 0.20 * late_consistency
    )
    return {
        "family_name": family["name"],
        "sample_count": len(sample_rows),
        "early_first_mean": sum(early_first_values) / len(early_first_values),
        "early_last_mean": sum(early_last_values) / len(early_last_values),
        "late_first_mean": sum(late_first_values) / len(late_first_values),
        "late_last_mean": sum(late_last_values) / len(late_last_values),
        "early_remention_corr": early_corr,
        "late_remention_corr": late_corr,
        "early_sign_consistency_rate": early_consistency,
        "late_sign_consistency_rate": late_consistency,
        "discourse_family_score": family_score,
    }


def build_summary(
    sample_rows: Sequence[Dict[str, object]],
    early_neurons: Sequence[int],
    late_neurons: Sequence[int],
    family_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    mean_early_corr = sum(float(row["early_remention_corr"]) for row in family_rows) / len(family_rows)
    mean_late_corr = sum(float(row["late_remention_corr"]) for row in family_rows) / len(family_rows)
    mean_early_consistency = sum(float(row["early_sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    mean_late_consistency = sum(float(row["late_sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    mean_family_score = sum(float(row["discourse_family_score"]) for row in family_rows) / len(family_rows)
    early_positive_rate = sum(1 for row in family_rows if float(row["early_remention_corr"]) > 0.0) / len(family_rows)
    late_positive_rate = sum(1 for row in family_rows if float(row["late_remention_corr"]) > 0.0) / len(family_rows)
    propagation_score = (
        0.30 * clamp01((mean_early_corr + 1.0) / 2.0)
        + 0.30 * clamp01((mean_late_corr + 1.0) / 2.0)
        + 0.20 * mean_early_consistency
        + 0.20 * mean_late_consistency
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage133_complex_discourse_noun_propagation",
        "title": "复杂语篇名词传播块",
        "status_short": "gpt2_complex_discourse_noun_ready",
        "family_count": len(family_rows),
        "sample_count": len(sample_rows),
        "early_layer_index": EARLY_LAYER,
        "late_layer_index": LATE_LAYER,
        "early_neuron_count": len(early_neurons),
        "late_neuron_count": len(late_neurons),
        "mean_early_remention_corr": mean_early_corr,
        "mean_late_remention_corr": mean_late_corr,
        "mean_early_sign_consistency_rate": mean_early_consistency,
        "mean_late_sign_consistency_rate": mean_late_consistency,
        "early_positive_family_rate": early_positive_rate,
        "late_positive_family_rate": late_positive_rate,
        "mean_family_score": mean_family_score,
        "complex_discourse_noun_propagation_score": propagation_score,
        "family_rows": list(family_rows),
        "selected_early_neurons": list(early_neurons),
        "selected_late_neurons": list(late_neurons),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage133: 复杂语篇名词传播块",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 样本数量: {summary['sample_count']}",
        f"- 早层重提相关均值: {summary['mean_early_remention_corr']:.4f}",
        f"- 后层重提相关均值: {summary['mean_late_remention_corr']:.4f}",
        f"- 早层符号一致率均值: {summary['mean_early_sign_consistency_rate']:.4f}",
        f"- 后层符号一致率均值: {summary['mean_late_sign_consistency_rate']:.4f}",
        f"- 复杂语篇名词传播分数: {summary['complex_discourse_noun_propagation_score']:.4f}",
        "",
        "## 各语篇家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"early_corr={row['early_remention_corr']:.4f}, "
            f"late_corr={row['late_remention_corr']:.4f}, "
            f"family_score={row['discourse_family_score']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果复杂语篇里第一次名词和后续重提名词在早层仍保持正相关，说明早层定锚不是一次性闪现，而是可被语篇重新调起。",
            "- 如果后层相关也保持为正，说明后层聚合并没有在复杂语篇里完全散掉，而是在重提位置上继续参与闭合。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE133_COMPLEX_DISCOURSE_NOUN_PROPAGATION_REPORT.md").write_text(
        build_report(summary),
        encoding="utf-8-sig",
    )
    (output_dir / "family_rows.json").write_text(
        json.dumps(summary["family_rows"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def run_analysis(
    *,
    input_dir: Path = STAGE119_OUTPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> Dict[str, object]:
    if not force:
        cached = load_cached_summary(output_dir)
        if cached is not None:
            return cached

    _stage119_summary, rows = ensure_stage119_rows(input_dir)
    sample_rows = choose_noun_rows(rows)
    early_neurons = select_early_neurons()
    late_neurons = select_late_neurons()
    model, tokenizer = load_model()
    family_rows = [run_family(model, tokenizer, sample_rows, family, early_neurons, late_neurons) for family in DISCOURSE_FAMILIES]
    summary = build_summary(sample_rows, early_neurons, late_neurons, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="复杂语篇名词传播块")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage133 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "mean_early_remention_corr": summary["mean_early_remention_corr"],
                "complex_discourse_noun_propagation_score": summary["complex_discourse_noun_propagation_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
