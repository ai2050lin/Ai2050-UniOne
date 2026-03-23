#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage119_gpt2_embedding_full_vocab_scan import OUTPUT_DIR as STAGE119_OUTPUT_DIR
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows
from stage124_noun_neuron_basic_probe import OUTPUT_DIR as STAGE124_OUTPUT_DIR
from wordclass_neuron_basic_probe_lib import (
    build_neuron_tables,
    build_summary as build_wordclass_summary,
    clamp01,
    load_model,
    top_two_bands,
    unique_target_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323"
BATCH_SIZE = 96

SYNTAX_FAMILIES = [
    {"name": "subject_copula", "prefix": "The", "suffix": " is nearby."},
    {"name": "object_transitive", "prefix": "We observed the", "suffix": " today."},
    {"name": "preposition_about", "prefix": "They spoke about the", "suffix": " yesterday."},
    {"name": "relative_clause", "prefix": "The", "suffix": " that arrived early mattered."},
    {"name": "possessive_frame", "prefix": "Their", "suffix": " remained useful."},
    {"name": "evaluation_frame", "prefix": "People considered the", "suffix": " important."},
]


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return None


def select_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    control_rows = [row for row in rows if row["lexical_type"] != "noun"]
    if not noun_rows or not control_rows:
        raise RuntimeError("多句法名词探针缺少目标组或控制组")
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
    family: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    input_rows = []
    target_ranges = []
    pad_token_id = tokenizer.pad_token_id
    prefix_ids = tokenizer.encode(family["prefix"], add_special_tokens=False)
    suffix_ids = tokenizer.encode(family["suffix"], add_special_tokens=False)

    for row in batch_rows:
        word_ids = tokenizer.encode(" " + str(row["word"]), add_special_tokens=False)
        if not word_ids:
            word_ids = tokenizer.encode(str(row["word"]), add_special_tokens=False)
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
    family: Dict[str, str],
    is_target: bool,
    stats: Dict[str, object],
    group_index: Dict[str, int],
) -> None:
    input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows, family)
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

    per_layer_rows = []
    for layer_tensor in layer_outputs:
        if layer_tensor is None:
            raise RuntimeError("多句法名词探针未捕获到层输出")
        per_layer_rows.append(mean_target_activation(layer_tensor, target_ranges).to(torch.float64))

    sample_tensor = torch.stack(per_layer_rows, dim=1)
    update_global_stats(stats, sample_tensor, is_target=is_target)
    if is_target:
        update_target_stats(stats, sample_tensor, batch_rows, group_index)


def run_family_scan(
    model,
    tokenizer,
    noun_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    group_names: Sequence[str],
    family: Dict[str, str],
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
                family=family,
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
                family=family,
                is_target=False,
                stats=stats,
                group_index=group_index,
            )
    finally:
        remove_hooks(hooks)

    return stats


def build_family_summaries(
    model,
    tokenizer,
    noun_rows: Sequence[Dict[str, object]],
    control_rows: Sequence[Dict[str, object]],
    group_names: Sequence[str],
    primary_band_name: str,
    secondary_band_name: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    family_rows = []
    top_neuron_counter: Counter[Tuple[int, int]] = Counter()
    neuron_payloads: Dict[Tuple[int, int], Dict[str, object]] = {}

    for family in SYNTAX_FAMILIES:
        stats = run_family_scan(model, tokenizer, noun_rows, control_rows, group_names, family)
        tables = build_neuron_tables(
            stats,
            group_names,
            primary_band_name=primary_band_name,
            secondary_band_name=secondary_band_name,
        )
        family_rows.append(
            {
                "family_name": family["name"],
                "dominant_general_layer_index": tables["dominant_general_layer_index"],
                "dominant_general_layer_score": tables["dominant_general_layer_score"],
                "top20_mean_general_rule_score": tables["layer_rows"][tables["dominant_general_layer_index"]]["top20_mean_general_rule_score"],
                "top_general_neuron": tables["top_general_neurons"][0],
            }
        )
        for row in tables["top_general_neurons"][:12]:
            if int(row["layer_index"]) <= 2:
                key = (int(row["layer_index"]), int(row["neuron_index"]))
                top_neuron_counter[key] += 1
                neuron_payloads[key] = row

    recurrent_early_neurons = []
    for (layer_idx, neuron_idx), count in top_neuron_counter.most_common(24):
        row = dict(neuron_payloads[(layer_idx, neuron_idx)])
        row["syntax_hit_count"] = int(count)
        recurrent_early_neurons.append(row)

    return family_rows, recurrent_early_neurons


def load_stage124_summary() -> Dict[str, object]:
    return json.loads((STAGE124_OUTPUT_DIR / "summary.json").read_text(encoding="utf-8-sig"))


def build_summary_fn(
    *,
    stage119_summary: Dict[str, object],
    noun_rows: Sequence[Dict[str, object]],
    band_counts: Dict[str, int],
    primary_band_name: str,
    secondary_band_name: str,
    family_rows: Sequence[Dict[str, object]],
    recurrent_early_neurons: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    base = build_wordclass_summary(
        experiment_id="stage130_multisyntax_noun_context_probe",
        title="Noun 多句法上下文探针",
        status_short="gpt2_multisyntax_noun_context_ready",
        target_lexical_type="noun",
        stage119_summary=stage119_summary,
        target_rows=noun_rows,
        neuron_tables={
            "target_count": len(noun_rows),
            "control_count": 0,
            "layer_count": 12,
            "neurons_per_layer": 3072,
            "target_group_count": len(unique_target_groups(noun_rows)),
            "primary_band_name": primary_band_name,
            "secondary_band_name": secondary_band_name,
            "dominant_general_layer_index": max(family_rows, key=lambda row: row["top20_mean_general_rule_score"])["dominant_general_layer_index"],
            "dominant_general_layer_score": max(row["top20_mean_general_rule_score"] for row in family_rows),
            "wordclass_neuron_basic_probe_score": 0.0,
            "layer_rows": [],
            "top_general_neurons": list(recurrent_early_neurons),
            "top_primary_band_bias_neurons": [],
            "top_secondary_band_bias_neurons": [],
        },
        band_counts=band_counts,
    )
    stage124_summary = load_stage124_summary()
    early_family_count = sum(1 for row in family_rows if int(row["dominant_general_layer_index"]) <= 2)
    l1_family_count = sum(1 for row in family_rows if int(row["dominant_general_layer_index"]) == 1)
    mean_family_score = sum(float(row["top20_mean_general_rule_score"]) for row in family_rows) / max(1, len(family_rows))
    syntax_stability = sum(
        1 for row in family_rows if int(row["dominant_general_layer_index"]) in {0, 1, 2}
    ) / max(1, len(family_rows))
    recurrent_mean_hits = (
        sum(float(row["syntax_hit_count"]) for row in recurrent_early_neurons[:12]) / max(1, min(12, len(recurrent_early_neurons)))
    )
    multisyntax_score = (
        0.40 * syntax_stability
        + 0.30 * clamp01(mean_family_score / 0.55)
        + 0.20 * clamp01(recurrent_mean_hits / 4.0)
        + 0.10 * clamp01(stage124_summary["dominant_general_layer_score"] / 0.50)
    )
    base["family_count"] = len(family_rows)
    base["family_rows"] = list(family_rows)
    base["recurrent_early_neurons"] = list(recurrent_early_neurons)
    base["early_family_count"] = int(early_family_count)
    base["l1_family_count"] = int(l1_family_count)
    base["mean_family_score"] = float(mean_family_score)
    base["syntax_stability_rate"] = float(syntax_stability)
    base["recurrent_early_hit_mean"] = float(recurrent_mean_hits)
    base["multisyntax_noun_context_score"] = float(multisyntax_score)
    return base


def write_multisyntax_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "STAGE130_MULTISYNTAX_NOUN_CONTEXT_PROBE_REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")
    (output_dir / "family_rows.json").write_text(json.dumps(summary["family_rows"], ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "recurrent_early_neurons.json").write_text(
        json.dumps(summary["recurrent_early_neurons"], ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage130: Noun 多句法上下文探针",
        "",
        "## 核心结果",
        f"- 句法簇数量: {summary['family_count']}",
        f"- 早层主导簇数: {summary['early_family_count']}",
        f"- L1 主导簇数: {summary['l1_family_count']}",
        f"- 句法稳定率: {summary['syntax_stability_rate']:.4f}",
        f"- 句法均值得分: {summary['mean_family_score']:.4f}",
        f"- 反复出现的早层神经元平均命中数: {summary['recurrent_early_hit_mean']:.4f}",
        f"- 多句法名词上下文分数: {summary['multisyntax_noun_context_score']:.4f}",
        "",
        "## 各句法簇",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"dominant=L{row['dominant_general_layer_index']}, "
            f"score={row['top20_mean_general_rule_score']:.4f}"
        )
    lines.extend(["", "## 反复出现的早层神经元"])
    for row in summary["recurrent_early_neurons"][:12]:
        lines.append(
            "- "
            f"L{row['layer_index']} N{row['neuron_index']}: "
            f"hits={row['syntax_hit_count']}, "
            f"rule={row['general_rule_score']:.4f}, "
            f"group={row['dominant_group_name']}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果多个句法簇都把名词主导层压到早层，说明名词进入句子后会先做快速定锚，再由更深层做后续聚合。",
            "- 如果早层神经元跨句法反复出现，说明这里更接近稳定编码规则，而不是模板偶然性。",
            "",
        ]
    )
    return "\n".join(lines)


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

    stage119_summary, rows = ensure_stage119_rows(input_dir)
    noun_rows, control_rows = select_rows(rows)
    group_names = unique_target_groups(noun_rows)
    primary_band_name, secondary_band_name, band_counts = top_two_bands(noun_rows)
    model, tokenizer = load_model()
    family_rows, recurrent_early_neurons = build_family_summaries(
        model,
        tokenizer,
        noun_rows,
        control_rows,
        group_names,
        primary_band_name,
        secondary_band_name,
    )
    summary = build_summary_fn(
        stage119_summary=stage119_summary,
        noun_rows=noun_rows,
        band_counts=band_counts,
        primary_band_name=primary_band_name,
        secondary_band_name=secondary_band_name,
        family_rows=family_rows,
        recurrent_early_neurons=recurrent_early_neurons,
    )
    write_multisyntax_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Noun 多句法上下文探针")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage130 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), force=args.force)
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "syntax_stability_rate": summary["syntax_stability_rate"],
                "multisyntax_noun_context_score": summary["multisyntax_noun_context_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
