#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from wordclass_neuron_basic_probe_lib import clamp01, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE128_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage128_noun_static_route_bridge_20260323"
STAGE130_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323"

BATCH_SIZE = 96
EARLY_LAYER = 1
ROUTE_LAYER = 3

JOINT_FAMILIES = [
    {
        "name": "discourse_remention",
        "prefix": "The",
        "middle": " resurfaced yesterday. Later they will",
        "suffix": " the report about the case.",
    },
    {
        "name": "causal_remention",
        "prefix": "Because the",
        "middle": " shifted overnight, they will",
        "suffix": " the record today.",
    },
    {
        "name": "nested_memory",
        "prefix": "When the analyst said the",
        "middle": " that the archive stored was fragile, they would",
        "suffix": " the file.",
    },
    {
        "name": "contrastive_remention",
        "prefix": "Although the",
        "middle": " looked ordinary, they would",
        "suffix": " the case after review.",
    },
    {
        "name": "cross_sentence_bridge",
        "prefix": "We studied the",
        "middle": " yesterday. Today they will",
        "suffix": " the plan.",
    },
]


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def select_early_neurons() -> List[int]:
    summary = load_json(STAGE130_DIR / "summary.json")
    rows = [row for row in summary["recurrent_early_neurons"] if int(row["layer_index"]) == EARLY_LAYER]
    rows.sort(key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])), reverse=True)
    return [int(row["neuron_index"]) for row in rows[:12]]


def select_route_neurons() -> List[int]:
    summary = load_json(STAGE128_DIR / "summary.json")
    return [int(row["neuron_index"]) for row in summary["selected_route_neurons"][:12]]


def balanced_noun_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["lexical_type"] == "noun":
            buckets[str(row["band"])].append(row)
    for band_name in buckets:
        buckets[band_name].sort(
            key=lambda row: (
                float(row.get("lexical_type_score", 0.0)),
                float(row.get("effective_encoding_score", 0.0)),
            ),
            reverse=True,
        )
    selected = []
    selected.extend(buckets.get("meso", [])[:160])
    selected.extend(buckets.get("macro", [])[:128])
    selected.extend(buckets.get("micro", [])[:32])
    return selected


def select_verbs(rows: Sequence[Dict[str, object]]) -> List[str]:
    verb_rows = [
        row
        for row in rows
        if row["lexical_type"] == "verb"
        and row["group"] == "macro_action"
        and float(row.get("lexical_type_score", 0.0)) >= 0.55
    ]
    verb_rows.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    verbs = []
    seen = set()
    for row in verb_rows:
        word = str(row["word"])
        if word not in seen:
            seen.add(word)
            verbs.append(word)
        if len(verbs) >= 6:
            break
    return verbs


def capture_layers(model) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    buffers: Dict[int, torch.Tensor | None] = {EARLY_LAYER: None, ROUTE_LAYER: None}
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


def build_case_rows(noun_rows: Sequence[Dict[str, object]], verbs: Sequence[str]) -> List[Dict[str, object]]:
    cases = []
    for noun_row in noun_rows:
        for verb in verbs:
            cases.append(
                {
                    "word": str(noun_row["word"]),
                    "band": str(noun_row["band"]),
                    "group": str(noun_row["group"]),
                    "verb": verb,
                }
            )
    return cases


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
        noun_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not noun_ids:
            noun_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)
        verb_ids = tokenizer.encode(" " + str(row["verb"]), add_special_tokens=False)
        if not verb_ids:
            verb_ids = tokenizer.encode(str(row["verb"]), add_special_tokens=False)
        full_ids = prefix_ids + noun_ids + middle_ids + verb_ids + suffix_ids
        noun_start = find_subsequence(full_ids, noun_ids)
        verb_start = find_subsequence(full_ids, verb_ids)
        if noun_start is None or verb_start is None:
            raise RuntimeError("名词-动词联合传播探针未定位到 noun 或 verb 位置")
        input_rows.append(full_ids)
        target_ranges.append((noun_start, noun_start + len(noun_ids), verb_start, verb_start + len(verb_ids)))

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


def run_family(
    model,
    tokenizer,
    case_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
    early_neurons: Sequence[int],
    route_neurons: Sequence[int],
) -> Dict[str, object]:
    buffers, handles = capture_layers(model)
    noun_anchor_values: List[float] = []
    verb_route_values: List[float] = []
    band_route_values: Dict[str, List[float]] = defaultdict(list)
    sign_hits = 0

    try:
        for start in range(0, len(case_rows), BATCH_SIZE):
            batch_rows = case_rows[start : start + BATCH_SIZE]
            input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows, family)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

            early_layer = buffers[EARLY_LAYER]
            route_layer = buffers[ROUTE_LAYER]
            if early_layer is None or route_layer is None:
                raise RuntimeError("名词-动词联合传播探针未捕获到目标层输出")

            for sample_idx, (noun_start, noun_end, verb_start, verb_end) in enumerate(target_ranges):
                noun_anchor = mean_selected(early_layer[sample_idx], noun_start, noun_end, early_neurons)
                verb_route = mean_selected(route_layer[sample_idx], verb_start, verb_end, route_neurons)
                noun_anchor_values.append(noun_anchor)
                verb_route_values.append(verb_route)
                band_route_values[str(batch_rows[sample_idx]["band"])].append(verb_route)
                if noun_anchor == 0.0 or verb_route == 0.0 or noun_anchor * verb_route > 0:
                    sign_hits += 1
    finally:
        remove_hooks(handles)

    noun_route_corr = correlation(noun_anchor_values, verb_route_values)
    sign_consistency = sign_hits / len(noun_anchor_values)
    macro_mean = sum(band_route_values.get("macro", [0.0])) / max(1, len(band_route_values.get("macro", [])))
    meso_mean = sum(band_route_values.get("meso", [0.0])) / max(1, len(band_route_values.get("meso", [])))
    micro_mean = sum(band_route_values.get("micro", [0.0])) / max(1, len(band_route_values.get("micro", [])))
    route_band_gap = abs(macro_mean - meso_mean)
    family_score = (
        0.45 * clamp01((noun_route_corr + 1.0) / 2.0)
        + 0.30 * sign_consistency
        + 0.25 * clamp01(route_band_gap / 0.08)
    )
    return {
        "family_name": family["name"],
        "case_count": len(case_rows),
        "noun_anchor_mean": sum(noun_anchor_values) / len(noun_anchor_values),
        "verb_route_mean": sum(verb_route_values) / len(verb_route_values),
        "noun_route_corr": noun_route_corr,
        "sign_consistency_rate": sign_consistency,
        "macro_route_mean": macro_mean,
        "meso_route_mean": meso_mean,
        "micro_route_mean": micro_mean,
        "route_band_gap": route_band_gap,
        "joint_family_score": family_score,
    }


def build_summary(
    noun_rows: Sequence[Dict[str, object]],
    verbs: Sequence[str],
    early_neurons: Sequence[int],
    route_neurons: Sequence[int],
    family_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    mean_corr = sum(float(row["noun_route_corr"]) for row in family_rows) / len(family_rows)
    mean_sign = sum(float(row["sign_consistency_rate"]) for row in family_rows) / len(family_rows)
    mean_gap = sum(float(row["route_band_gap"]) for row in family_rows) / len(family_rows)
    mean_family_score = sum(float(row["joint_family_score"]) for row in family_rows) / len(family_rows)
    positive_family_rate = sum(1 for row in family_rows if float(row["noun_route_corr"]) > 0.0) / len(family_rows)
    score = (
        0.45 * clamp01((mean_corr + 1.0) / 2.0)
        + 0.25 * mean_sign
        + 0.15 * clamp01(mean_gap / 0.08)
        + 0.15 * positive_family_rate
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage134_noun_verb_joint_propagation",
        "title": "名词-动词联合传播块",
        "status_short": "gpt2_noun_verb_joint_ready",
        "family_count": len(family_rows),
        "noun_sample_count": len(noun_rows),
        "verb_count": len(verbs),
        "case_count_per_family": len(noun_rows) * len(verbs),
        "early_layer_index": EARLY_LAYER,
        "route_layer_index": ROUTE_LAYER,
        "early_neuron_count": len(early_neurons),
        "route_neuron_count": len(route_neurons),
        "mean_noun_route_corr": mean_corr,
        "mean_sign_consistency_rate": mean_sign,
        "mean_route_band_gap": mean_gap,
        "positive_family_rate": positive_family_rate,
        "mean_family_score": mean_family_score,
        "noun_verb_joint_propagation_score": score,
        "verbs": list(verbs),
        "family_rows": list(family_rows),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage134: 名词-动词联合传播块",
        "",
        "## 核心结果",
        f"- 语篇家族数量: {summary['family_count']}",
        f"- 名词样本数: {summary['noun_sample_count']}",
        f"- 动词数量: {summary['verb_count']}",
        f"- 每家族案例数: {summary['case_count_per_family']}",
        f"- 名词-动词相关均值: {summary['mean_noun_route_corr']:.4f}",
        f"- 符号一致率均值: {summary['mean_sign_consistency_rate']:.4f}",
        f"- 路由带差均值: {summary['mean_route_band_gap']:.4f}",
        f"- 联合传播分数: {summary['noun_verb_joint_propagation_score']:.4f}",
        "",
        "## 各语篇家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"corr={row['noun_route_corr']:.4f}, "
            f"gap={row['route_band_gap']:.4f}, "
            f"score={row['joint_family_score']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果名词早层定锚与动词路由在同一句内保持正相关，说明名词并不只是被动承载内容，而会向后续动作选路链施压。",
            "- 如果不同名词尺度带会拉开动词路由均值，说明上下文名词类型已经在参与动词位置的条件偏置。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE134_NOUN_VERB_JOINT_PROPAGATION_REPORT.md").write_text(
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
    noun_rows = balanced_noun_rows(rows)
    verbs = select_verbs(rows)
    case_rows = build_case_rows(noun_rows, verbs)
    early_neurons = select_early_neurons()
    route_neurons = select_route_neurons()
    model, tokenizer = load_model()
    family_rows = [run_family(model, tokenizer, case_rows, family, early_neurons, route_neurons) for family in JOINT_FAMILIES]
    summary = build_summary(noun_rows, verbs, early_neurons, route_neurons, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="名词-动词联合传播块")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage134 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "mean_noun_route_corr": summary["mean_noun_route_corr"],
                "noun_verb_joint_propagation_score": summary["noun_verb_joint_propagation_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
