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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage136_anaphora_ellipsis_propagation_20260323"

BATCH_SIZE = 96
NOUN_SAMPLE_LIMIT = 2048
EARLY_LAYER = 1
LATE_LAYER = 11

REFERENCE_FAMILIES = [
    {
        "name": "discourse_remention",
        "prefix": "The",
        "middle": " appeared in the archive. Later the team rechecked",
        "noun_suffix": " the",
        "pronoun_suffix": " it",
        "ellipsis_suffix": " that",
        "tail": " before sunset.",
    },
    {
        "name": "causal_remention",
        "prefix": "Because the",
        "middle": " changed overnight, the memo described",
        "noun_suffix": " the",
        "pronoun_suffix": " it",
        "ellipsis_suffix": " that",
        "tail": " carefully.",
    },
    {
        "name": "nested_memory",
        "prefix": "When the analyst said the",
        "middle": " that the archive stored was fragile, the report flagged",
        "noun_suffix": " the",
        "pronoun_suffix": " it",
        "ellipsis_suffix": " that",
        "tail": " again.",
    },
    {
        "name": "contrastive_remention",
        "prefix": "Although the",
        "middle": " looked ordinary, the review later called",
        "noun_suffix": " the",
        "pronoun_suffix": " it",
        "ellipsis_suffix": " that",
        "tail": " critical.",
    },
    {
        "name": "cross_sentence_bridge",
        "prefix": "We studied the",
        "middle": " yesterday. Today the planners still discussed",
        "noun_suffix": " the",
        "pronoun_suffix": " it",
        "ellipsis_suffix": " that",
        "tail": " in detail.",
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


def find_subsequence(sequence: Sequence[int], target: Sequence[int]) -> int | None:
    if not target:
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if list(sequence[idx : idx + len(target)]) == list(target):
            return idx
    return None


def build_family_inputs(
    tokenizer,
    batch_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Tuple[int, int]]]]:
    input_rows = []
    ranges = []
    pad_token_id = tokenizer.pad_token_id

    prefix_ids = tokenizer.encode(family["prefix"], add_special_tokens=False)
    middle_ids = tokenizer.encode(family["middle"], add_special_tokens=False)
    noun_suffix_ids = tokenizer.encode(family["noun_suffix"], add_special_tokens=False)
    pronoun_suffix_ids = tokenizer.encode(family["pronoun_suffix"], add_special_tokens=False)
    ellipsis_suffix_ids = tokenizer.encode(family["ellipsis_suffix"], add_special_tokens=False)
    tail_ids = tokenizer.encode(family["tail"], add_special_tokens=False)
    pronoun_ids = tokenizer.encode(" it", add_special_tokens=False)
    that_ids = tokenizer.encode(" that", add_special_tokens=False)

    for row in batch_rows:
        noun_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not noun_ids:
            noun_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)

        noun_prompt = prefix_ids + noun_ids + middle_ids + noun_suffix_ids + noun_ids + tail_ids
        pronoun_prompt = prefix_ids + noun_ids + middle_ids + pronoun_suffix_ids + tail_ids
        ellipsis_prompt = prefix_ids + noun_ids + middle_ids + ellipsis_suffix_ids + tail_ids

        noun_first = find_subsequence(noun_prompt, noun_ids)
        noun_last = find_subsequence(noun_prompt[noun_first + len(noun_ids):], noun_ids) if noun_first is not None else None
        pronoun_pos = find_subsequence(pronoun_prompt, pronoun_ids)
        ellipsis_pos = find_subsequence(ellipsis_prompt, that_ids)
        if noun_first is None or noun_last is None or pronoun_pos is None or ellipsis_pos is None:
            raise RuntimeError("回指与省略传播探针未定位到目标位置")
        noun_last = noun_first + len(noun_ids) + noun_last

        prompts = [noun_prompt, pronoun_prompt, ellipsis_prompt]
        for prompt_ids in prompts:
            input_rows.append(prompt_ids)

        ranges.extend(
            [
                {"source": (noun_first, noun_first + len(noun_ids)), "target": (noun_last, noun_last + len(noun_ids))},
                {"source": (noun_first, noun_first + len(noun_ids)), "target": (pronoun_pos, pronoun_pos + len(pronoun_ids))},
                {"source": (noun_first, noun_first + len(noun_ids)), "target": (ellipsis_pos, ellipsis_pos + len(that_ids))},
            ]
        )

    max_len = max(len(ids) for ids in input_rows)
    input_ids = []
    attention_mask = []
    for ids in input_rows:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), ranges


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
    noun_source_early: List[float] = []
    noun_target_early: List[float] = []
    pronoun_target_early: List[float] = []
    ellipsis_target_early: List[float] = []
    noun_source_late: List[float] = []
    noun_target_late: List[float] = []
    pronoun_target_late: List[float] = []
    ellipsis_target_late: List[float] = []

    try:
        for start in range(0, len(sample_rows), BATCH_SIZE):
            batch_rows = sample_rows[start : start + BATCH_SIZE]
            input_ids, attention_mask, ranges = build_family_inputs(tokenizer, batch_rows, family)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

            early_layer = buffers[EARLY_LAYER]
            late_layer = buffers[LATE_LAYER]
            if early_layer is None or late_layer is None:
                raise RuntimeError("回指与省略传播探针未捕获到目标层输出")

            for sample_idx, target_row in enumerate(ranges):
                source_start, source_end = target_row["source"]
                target_start, target_end = target_row["target"]
                early_source = mean_selected(early_layer[sample_idx], source_start, source_end, early_neurons)
                early_target = mean_selected(early_layer[sample_idx], target_start, target_end, early_neurons)
                late_source = mean_selected(late_layer[sample_idx], source_start, source_end, late_neurons)
                late_target = mean_selected(late_layer[sample_idx], target_start, target_end, late_neurons)

                kind = sample_idx % 3
                if kind == 0:
                    noun_source_early.append(early_source)
                    noun_target_early.append(early_target)
                    noun_source_late.append(late_source)
                    noun_target_late.append(late_target)
                elif kind == 1:
                    pronoun_target_early.append(early_target)
                    pronoun_target_late.append(late_target)
                else:
                    ellipsis_target_early.append(early_target)
                    ellipsis_target_late.append(late_target)
    finally:
        remove_hooks(handles)

    noun_pronoun_early_corr = correlation(noun_source_early, pronoun_target_early)
    noun_ellipsis_early_corr = correlation(noun_source_early, ellipsis_target_early)
    noun_pronoun_late_corr = correlation(noun_source_late, pronoun_target_late)
    noun_ellipsis_late_corr = correlation(noun_source_late, ellipsis_target_late)
    pronoun_consistency = sign_consistency(noun_source_early, pronoun_target_early)
    ellipsis_consistency = sign_consistency(noun_source_early, ellipsis_target_early)
    family_score = (
        0.25 * clamp01((noun_pronoun_early_corr + 1.0) / 2.0)
        + 0.25 * clamp01((noun_ellipsis_early_corr + 1.0) / 2.0)
        + 0.20 * clamp01((noun_pronoun_late_corr + 1.0) / 2.0)
        + 0.10 * clamp01((noun_ellipsis_late_corr + 1.0) / 2.0)
        + 0.10 * pronoun_consistency
        + 0.10 * ellipsis_consistency
    )
    return {
        "family_name": family["name"],
        "noun_pronoun_early_corr": noun_pronoun_early_corr,
        "noun_ellipsis_early_corr": noun_ellipsis_early_corr,
        "noun_pronoun_late_corr": noun_pronoun_late_corr,
        "noun_ellipsis_late_corr": noun_ellipsis_late_corr,
        "pronoun_sign_consistency_rate": pronoun_consistency,
        "ellipsis_sign_consistency_rate": ellipsis_consistency,
        "family_score": family_score,
    }


def build_summary(
    sample_rows: Sequence[Dict[str, object]],
    early_neurons: Sequence[int],
    late_neurons: Sequence[int],
    family_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    mean_noun_pronoun_early_corr = sum(float(row["noun_pronoun_early_corr"]) for row in family_rows) / len(family_rows)
    mean_noun_ellipsis_early_corr = sum(float(row["noun_ellipsis_early_corr"]) for row in family_rows) / len(family_rows)
    mean_noun_pronoun_late_corr = sum(float(row["noun_pronoun_late_corr"]) for row in family_rows) / len(family_rows)
    mean_noun_ellipsis_late_corr = sum(float(row["noun_ellipsis_late_corr"]) for row in family_rows) / len(family_rows)
    mean_pronoun_consistency = sum(float(row["pronoun_sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    mean_ellipsis_consistency = sum(float(row["ellipsis_sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    score = (
        0.25 * clamp01((mean_noun_pronoun_early_corr + 1.0) / 2.0)
        + 0.25 * clamp01((mean_noun_ellipsis_early_corr + 1.0) / 2.0)
        + 0.20 * clamp01((mean_noun_pronoun_late_corr + 1.0) / 2.0)
        + 0.10 * clamp01((mean_noun_ellipsis_late_corr + 1.0) / 2.0)
        + 0.10 * mean_pronoun_consistency
        + 0.10 * mean_ellipsis_consistency
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage136_anaphora_ellipsis_propagation",
        "title": "跨句回指与省略传播块",
        "status_short": "gpt2_anaphora_ellipsis_ready",
        "family_count": len(family_rows),
        "sample_count": len(sample_rows),
        "early_layer_index": EARLY_LAYER,
        "late_layer_index": LATE_LAYER,
        "early_neuron_count": len(early_neurons),
        "late_neuron_count": len(late_neurons),
        "mean_noun_pronoun_early_corr": mean_noun_pronoun_early_corr,
        "mean_noun_ellipsis_early_corr": mean_noun_ellipsis_early_corr,
        "mean_noun_pronoun_late_corr": mean_noun_pronoun_late_corr,
        "mean_noun_ellipsis_late_corr": mean_noun_ellipsis_late_corr,
        "mean_pronoun_sign_consistency_rate": mean_pronoun_consistency,
        "mean_ellipsis_sign_consistency_rate": mean_ellipsis_consistency,
        "anaphora_ellipsis_propagation_score": score,
        "family_rows": list(family_rows),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage136: 跨句回指与省略传播块",
        "",
        "## 核心结果",
        f"- 家族数量: {summary['family_count']}",
        f"- 样本数量: {summary['sample_count']}",
        f"- 名词到代词早层相关均值: {summary['mean_noun_pronoun_early_corr']:.4f}",
        f"- 名词到弱化提法早层相关均值: {summary['mean_noun_ellipsis_early_corr']:.4f}",
        f"- 名词到代词后层相关均值: {summary['mean_noun_pronoun_late_corr']:.4f}",
        f"- 名词到弱化提法后层相关均值: {summary['mean_noun_ellipsis_late_corr']:.4f}",
        f"- 回指与省略传播分数: {summary['anaphora_ellipsis_propagation_score']:.4f}",
        "",
        "## 各家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"pron_early={row['noun_pronoun_early_corr']:.4f}, "
            f"ell_early={row['noun_ellipsis_early_corr']:.4f}, "
            f"score={row['family_score']:.4f}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE136_ANAPHORA_ELLIPSIS_PROPAGATION_REPORT.md").write_text(
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
    family_rows = [run_family(model, tokenizer, sample_rows, family, early_neurons, late_neurons) for family in REFERENCE_FAMILIES]
    summary = build_summary(sample_rows, early_neurons, late_neurons, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨句回指与省略传播块")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage136 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "mean_noun_pronoun_early_corr": summary["mean_noun_pronoun_early_corr"],
                "anaphora_ellipsis_propagation_score": summary["anaphora_ellipsis_propagation_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
