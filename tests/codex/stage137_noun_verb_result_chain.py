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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage137_noun_verb_result_chain_20260323"

BATCH_SIZE = 96
EARLY_LAYER = 1
ROUTE_LAYER = 3
RESULT_LAYER = 11

CHAIN_FAMILIES = [
    {
        "name": "discourse_remention",
        "prefix": "The",
        "middle": " resurfaced yesterday. Later they will",
        "postverb": " the report, so the outcome seems",
        "result_word": "clear",
        "tail": ".",
    },
    {
        "name": "causal_remention",
        "prefix": "Because the",
        "middle": " shifted overnight, they will",
        "postverb": " the record, so the result stays",
        "result_word": "stable",
        "tail": ".",
    },
    {
        "name": "nested_memory",
        "prefix": "When the analyst said the",
        "middle": " that the archive stored was fragile, they would",
        "postverb": " the file, leaving the ending",
        "result_word": "resolved",
        "tail": ".",
    },
    {
        "name": "contrastive_remention",
        "prefix": "Although the",
        "middle": " looked ordinary, they would",
        "postverb": " the case until the conclusion felt",
        "result_word": "consistent",
        "tail": ".",
    },
    {
        "name": "cross_sentence_bridge",
        "prefix": "We studied the",
        "middle": " yesterday. Today they will",
        "postverb": " the plan and make the ending",
        "result_word": "useful",
        "tail": ".",
    },
]

POSITIVE_RESULT_WORDS = ["stable", "clear", "useful", "resolved", "consistent", "complete"]
NEUTRAL_RESULT_WORDS = ["ordinary", "normal", "general", "routine", "typical"]


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
    buffers: Dict[int, torch.Tensor | None] = {
        EARLY_LAYER: None,
        ROUTE_LAYER: None,
        RESULT_LAYER: None,
    }
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            buffers[layer_idx] = output.detach().cpu()
        return hook

    handles.append(model.transformer.h[EARLY_LAYER].mlp.act.register_forward_hook(make_hook(EARLY_LAYER)))
    handles.append(model.transformer.h[ROUTE_LAYER].mlp.act.register_forward_hook(make_hook(ROUTE_LAYER)))
    # 结果位使用 c_proj（投影输出），回到 residual（残差）维度，便于与 embedding（嵌入）原型比较。
    handles.append(model.transformer.h[RESULT_LAYER].mlp.c_proj.register_forward_hook(make_hook(RESULT_LAYER)))
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


def word_embedding_centroid(tokenizer, embed_weight: torch.Tensor, words: Sequence[str]) -> torch.Tensor:
    vectors = []
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word, add_special_tokens=False)
        vectors.append(embed_weight[ids].float().mean(dim=0))
    return torch.stack(vectors, dim=0).mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a).clamp_min(1e-8) * torch.linalg.norm(b).clamp_min(1e-8)
    return float(torch.dot(a, b).item() / float(denom))


def build_batch_inputs(
    tokenizer,
    batch_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int, int, int, int]]]:
    input_rows = []
    target_ranges = []
    pad_token_id = tokenizer.pad_token_id
    prefix_ids = tokenizer.encode(family["prefix"], add_special_tokens=False)
    middle_ids = tokenizer.encode(family["middle"], add_special_tokens=False)
    postverb_ids = tokenizer.encode(family["postverb"], add_special_tokens=False)
    result_ids = tokenizer.encode(" " + family["result_word"], add_special_tokens=False)
    tail_ids = tokenizer.encode(family["tail"], add_special_tokens=False)

    for row in batch_rows:
        noun_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not noun_ids:
            noun_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)
        verb_ids = tokenizer.encode(" " + str(row["verb"]), add_special_tokens=False)
        if not verb_ids:
            verb_ids = tokenizer.encode(str(row["verb"]), add_special_tokens=False)
        full_ids = prefix_ids + noun_ids + middle_ids + verb_ids + postverb_ids + result_ids + tail_ids
        noun_start = find_subsequence(full_ids, noun_ids)
        verb_start = find_subsequence(full_ids, verb_ids)
        result_start = find_subsequence(full_ids, result_ids)
        if noun_start is None or verb_start is None or result_start is None:
            raise RuntimeError("名词-动词-结果链探针未定位到目标位置")
        input_rows.append(full_ids)
        target_ranges.append(
            (
                noun_start,
                noun_start + len(noun_ids),
                verb_start,
                verb_start + len(verb_ids),
                result_start,
                result_start + len(result_ids),
            )
        )

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
    embed_weight: torch.Tensor,
    case_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
    early_neurons: Sequence[int],
    route_neurons: Sequence[int],
) -> Dict[str, object]:
    positive_centroid = word_embedding_centroid(tokenizer, embed_weight, POSITIVE_RESULT_WORDS)
    neutral_centroid = word_embedding_centroid(tokenizer, embed_weight, NEUTRAL_RESULT_WORDS)

    buffers, handles = capture_layers(model)
    noun_anchor_values: List[float] = []
    verb_route_values: List[float] = []
    result_values: List[float] = []

    try:
        for start in range(0, len(case_rows), BATCH_SIZE):
            batch_rows = case_rows[start : start + BATCH_SIZE]
            input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows, family)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

            early_layer = buffers[EARLY_LAYER]
            route_layer = buffers[ROUTE_LAYER]
            result_layer = buffers[RESULT_LAYER]
            if early_layer is None or route_layer is None or result_layer is None:
                raise RuntimeError("名词-动词-结果链探针未捕获到目标层输出")

            for sample_idx, (noun_start, noun_end, verb_start, verb_end, result_start, result_end) in enumerate(target_ranges):
                noun_anchor = mean_selected(early_layer[sample_idx], noun_start, noun_end, early_neurons)
                verb_route = mean_selected(route_layer[sample_idx], verb_start, verb_end, route_neurons)
                result_vec = result_layer[sample_idx, result_start:result_end, :].mean(dim=0).float()
                result_score = cosine(result_vec, positive_centroid) - cosine(result_vec, neutral_centroid)
                noun_anchor_values.append(noun_anchor)
                verb_route_values.append(verb_route)
                result_values.append(result_score)
    finally:
        remove_hooks(handles)

    noun_verb_corr = correlation(noun_anchor_values, verb_route_values)
    verb_result_corr = correlation(verb_route_values, result_values)
    noun_result_corr = correlation(noun_anchor_values, result_values)
    family_score = (
        0.35 * clamp01((noun_verb_corr + 1.0) / 2.0)
        + 0.35 * clamp01((verb_result_corr + 1.0) / 2.0)
        + 0.30 * clamp01((noun_result_corr + 1.0) / 2.0)
    )
    return {
        "family_name": family["name"],
        "case_count": len(case_rows),
        "noun_anchor_mean": sum(noun_anchor_values) / len(noun_anchor_values),
        "verb_route_mean": sum(verb_route_values) / len(verb_route_values),
        "result_score_mean": sum(result_values) / len(result_values),
        "noun_verb_corr": noun_verb_corr,
        "verb_result_corr": verb_result_corr,
        "noun_result_corr": noun_result_corr,
        "chain_family_score": family_score,
    }


def build_summary(
    noun_rows: Sequence[Dict[str, object]],
    verbs: Sequence[str],
    family_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    mean_noun_verb_corr = sum(float(row["noun_verb_corr"]) for row in family_rows) / len(family_rows)
    mean_verb_result_corr = sum(float(row["verb_result_corr"]) for row in family_rows) / len(family_rows)
    mean_noun_result_corr = sum(float(row["noun_result_corr"]) for row in family_rows) / len(family_rows)
    mean_family_score = sum(float(row["chain_family_score"]) for row in family_rows) / len(family_rows)
    positive_bridge_rate = sum(1 for row in family_rows if float(row["verb_result_corr"]) > 0.0) / len(family_rows)
    score = (
        0.35 * clamp01((mean_noun_verb_corr + 1.0) / 2.0)
        + 0.35 * clamp01((mean_verb_result_corr + 1.0) / 2.0)
        + 0.20 * clamp01((mean_noun_result_corr + 1.0) / 2.0)
        + 0.10 * positive_bridge_rate
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage137_noun_verb_result_chain",
        "title": "名词到动词到句尾结果链",
        "status_short": "gpt2_noun_verb_result_ready",
        "family_count": len(family_rows),
        "noun_sample_count": len(noun_rows),
        "verb_count": len(verbs),
        "case_count_per_family": len(noun_rows) * len(verbs),
        "mean_noun_verb_corr": mean_noun_verb_corr,
        "mean_verb_result_corr": mean_verb_result_corr,
        "mean_noun_result_corr": mean_noun_result_corr,
        "positive_bridge_rate": positive_bridge_rate,
        "mean_family_score": mean_family_score,
        "noun_verb_result_chain_score": score,
        "verbs": list(verbs),
        "family_rows": list(family_rows),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage137: 名词到动词到句尾结果链",
        "",
        "## 核心结果",
        f"- 语篇家族数量: {summary['family_count']}",
        f"- 名词样本数: {summary['noun_sample_count']}",
        f"- 动词数量: {summary['verb_count']}",
        f"- 每家族案例数: {summary['case_count_per_family']}",
        f"- noun->verb 相关均值: {summary['mean_noun_verb_corr']:.4f}",
        f"- verb->result 相关均值: {summary['mean_verb_result_corr']:.4f}",
        f"- noun->result 相关均值: {summary['mean_noun_result_corr']:.4f}",
        f"- 结果链分数: {summary['noun_verb_result_chain_score']:.4f}",
        "",
        "## 各家族",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"nv={row['noun_verb_corr']:.4f}, "
            f"vr={row['verb_result_corr']:.4f}, "
            f"nr={row['noun_result_corr']:.4f}"
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE137_NOUN_VERB_RESULT_CHAIN_REPORT.md").write_text(
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
    embed_weight = model.get_input_embeddings().weight.detach().cpu()
    family_rows = [
        run_family(model, tokenizer, embed_weight, case_rows, family, early_neurons, route_neurons)
        for family in CHAIN_FAMILIES
    ]
    summary = build_summary(noun_rows, verbs, family_rows)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="名词到动词到句尾结果链")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage137 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "mean_verb_result_corr": summary["mean_verb_result_corr"],
                "noun_verb_result_chain_score": summary["noun_verb_result_chain_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
