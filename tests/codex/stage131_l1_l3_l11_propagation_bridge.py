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
from stage130_multisyntax_noun_context_probe import SYNTAX_FAMILIES
from wordclass_neuron_basic_probe_lib import clamp01, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE124_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323"
STAGE128_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage128_noun_static_route_bridge_20260323"
STAGE130_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage131_l1_l3_l11_propagation_bridge_20260323"

BATCH_SIZE = 128
SAMPLE_LIMIT = 2048
EARLY_LAYER = 1
ROUTE_LAYER = 3
LATE_LAYER = 11
NEURON_LIMIT = 12


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cached_summary(output_dir: Path) -> Dict[str, object] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def choose_sample_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    noun_rows = [row for row in rows if row["lexical_type"] == "noun"]
    noun_rows.sort(
        key=lambda row: (
            float(row.get("lexical_type_score", 0.0)),
            float(row.get("effective_encoding_score", 0.0)),
        ),
        reverse=True,
    )
    return noun_rows[:SAMPLE_LIMIT]


def select_early_neurons() -> List[int]:
    summary = load_json(STAGE130_DIR / "summary.json")
    candidates = list(summary["recurrent_early_neurons"])
    preferred = [row for row in candidates if int(row["layer_index"]) == EARLY_LAYER]
    preferred.sort(
        key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])),
        reverse=True,
    )
    indices = [int(row["neuron_index"]) for row in preferred[:NEURON_LIMIT]]
    if len(indices) < NEURON_LIMIT:
        fallback = [row for row in candidates if int(row["layer_index"]) != EARLY_LAYER]
        fallback.sort(
            key=lambda row: (int(row["syntax_hit_count"]), float(row["general_rule_score"])),
            reverse=True,
        )
        for row in fallback:
            idx = int(row["neuron_index"])
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= NEURON_LIMIT:
                break
    return indices[:NEURON_LIMIT]


def select_route_neurons() -> List[int]:
    summary = load_json(STAGE128_DIR / "summary.json")
    return [int(row["neuron_index"]) for row in summary["selected_route_neurons"][:NEURON_LIMIT]]


def select_late_neurons() -> List[int]:
    summary = load_json(STAGE124_DIR / "summary.json")
    rows = [row for row in summary["top_general_neurons"] if int(row["layer_index"]) == LATE_LAYER]
    return [int(row["neuron_index"]) for row in rows[:NEURON_LIMIT]]


def capture_selected_layers(model) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    layer_buffers: Dict[int, torch.Tensor | None] = {
        EARLY_LAYER: None,
        ROUTE_LAYER: None,
        LATE_LAYER: None,
    }
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            layer_buffers[layer_idx] = output.detach().cpu()

        return hook

    for layer_idx in layer_buffers:
        handles.append(model.transformer.h[layer_idx].mlp.act.register_forward_hook(make_hook(layer_idx)))
    return layer_buffers, handles


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


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


def mean_selected_activation(
    layer_tensor: torch.Tensor,
    target_ranges: Sequence[Tuple[int, int]],
    neuron_indices: Sequence[int],
) -> torch.Tensor:
    rows = []
    neuron_index_tensor = torch.tensor(neuron_indices, dtype=torch.long)
    for sample_idx, (start, end) in enumerate(target_ranges):
        token_mean = layer_tensor[sample_idx, start:end, :].mean(dim=0)
        rows.append(token_mean[neuron_index_tensor].mean())
    return torch.stack(rows, dim=0)


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


def run_family_path(
    model,
    tokenizer,
    sample_rows: Sequence[Dict[str, object]],
    family: Dict[str, str],
    early_neurons: Sequence[int],
    route_neurons: Sequence[int],
    late_neurons: Sequence[int],
) -> Dict[str, object]:
    layer_buffers, handles = capture_selected_layers(model)
    early_values: List[float] = []
    route_values: List[float] = []
    late_values: List[float] = []

    try:
        for start in range(0, len(sample_rows), BATCH_SIZE):
            batch_rows = sample_rows[start : start + BATCH_SIZE]
            input_ids, attention_mask, target_ranges = build_batch_inputs(tokenizer, batch_rows, family)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)

            early_batch = mean_selected_activation(layer_buffers[EARLY_LAYER], target_ranges, early_neurons)
            route_batch = mean_selected_activation(layer_buffers[ROUTE_LAYER], target_ranges, route_neurons)
            late_batch = mean_selected_activation(layer_buffers[LATE_LAYER], target_ranges, late_neurons)

            early_values.extend(float(x) for x in early_batch)
            route_values.extend(float(x) for x in route_batch)
            late_values.extend(float(x) for x in late_batch)
    finally:
        remove_hooks(handles)

    l1_l3_corr = correlation(early_values, route_values)
    l3_l11_corr = correlation(route_values, late_values)
    l1_l11_corr = correlation(early_values, late_values)
    path_score = (
        0.35 * clamp01((l1_l3_corr + 1.0) / 2.0)
        + 0.35 * clamp01((l3_l11_corr + 1.0) / 2.0)
        + 0.30 * clamp01((l1_l11_corr + 1.0) / 2.0)
    )
    return {
        "family_name": family["name"],
        "case_count": len(sample_rows),
        "l1_mean": sum(early_values) / len(early_values),
        "l3_mean": sum(route_values) / len(route_values),
        "l11_mean": sum(late_values) / len(late_values),
        "l1_l3_corr": l1_l3_corr,
        "l3_l11_corr": l3_l11_corr,
        "l1_l11_corr": l1_l11_corr,
        "family_path_score": path_score,
    }


def build_summary(
    family_rows: Sequence[Dict[str, object]],
    early_neurons: Sequence[int],
    route_neurons: Sequence[int],
    late_neurons: Sequence[int],
) -> Dict[str, object]:
    mean_l1_l3 = sum(row["l1_l3_corr"] for row in family_rows) / len(family_rows)
    mean_l3_l11 = sum(row["l3_l11_corr"] for row in family_rows) / len(family_rows)
    mean_l1_l11 = sum(row["l1_l11_corr"] for row in family_rows) / len(family_rows)
    mean_path_score = sum(row["family_path_score"] for row in family_rows) / len(family_rows)
    coherent_family_rate = sum(
        1 for row in family_rows if row["l1_l3_corr"] > 0 and row["l3_l11_corr"] > 0
    ) / len(family_rows)
    propagation_score = (
        0.30 * clamp01((mean_l1_l3 + 1.0) / 2.0)
        + 0.30 * clamp01((mean_l3_l11 + 1.0) / 2.0)
        + 0.20 * clamp01((mean_l1_l11 + 1.0) / 2.0)
        + 0.20 * coherent_family_rate
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage131_l1_l3_l11_propagation_bridge",
        "title": "L1/L3/L11 传播桥",
        "status_short": "gpt2_l1_l3_l11_propagation_ready",
        "family_count": len(family_rows),
        "sample_count_per_family": family_rows[0]["case_count"] if family_rows else 0,
        "early_layer_index": EARLY_LAYER,
        "route_layer_index": ROUTE_LAYER,
        "late_layer_index": LATE_LAYER,
        "early_neuron_count": len(early_neurons),
        "route_neuron_count": len(route_neurons),
        "late_neuron_count": len(late_neurons),
        "mean_l1_l3_corr": mean_l1_l3,
        "mean_l3_l11_corr": mean_l3_l11,
        "mean_l1_l11_corr": mean_l1_l11,
        "mean_family_path_score": mean_path_score,
        "coherent_family_rate": coherent_family_rate,
        "l1_l3_l11_propagation_score": propagation_score,
        "family_rows": list(family_rows),
        "selected_early_neurons": list(early_neurons),
        "selected_route_neurons": list(route_neurons),
        "selected_late_neurons": list(late_neurons),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage131: L1/L3/L11 传播桥",
        "",
        "## 核心结果",
        f"- 句法簇数量: {summary['family_count']}",
        f"- 每簇样本数: {summary['sample_count_per_family']}",
        f"- L1-L3 平均相关: {summary['mean_l1_l3_corr']:.4f}",
        f"- L3-L11 平均相关: {summary['mean_l3_l11_corr']:.4f}",
        f"- L1-L11 平均相关: {summary['mean_l1_l11_corr']:.4f}",
        f"- 家族路径平均分: {summary['mean_family_path_score']:.4f}",
        f"- 相干句法簇比率: {summary['coherent_family_rate']:.4f}",
        f"- 传播桥分数: {summary['l1_l3_l11_propagation_score']:.4f}",
        "",
        "## 各句法簇",
    ]
    for row in summary["family_rows"]:
        lines.append(
            "- "
            f"{row['family_name']}: "
            f"l1_l3={row['l1_l3_corr']:.4f}, "
            f"l3_l11={row['l3_l11_corr']:.4f}, "
            f"l1_l11={row['l1_l11_corr']:.4f}, "
            f"path={row['family_path_score']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 理论提示",
            "- 如果 L1 到 L3、L3 到 L11 在多句法上都保持正相关，说明早层定锚、选路链和后层聚合之间存在同向传播。",
            "- 如果 L1-L11 相关弱于分段相关，则更像“经由中间层串联”，而不是直接闭合。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "STAGE131_L1_L3_L11_PROPAGATION_BRIDGE_REPORT.md").write_text(
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
    sample_rows = choose_sample_rows(rows)
    early_neurons = select_early_neurons()
    route_neurons = select_route_neurons()
    late_neurons = select_late_neurons()
    model, tokenizer = load_model()
    family_rows = [
        run_family_path(model, tokenizer, sample_rows, family, early_neurons, route_neurons, late_neurons)
        for family in SYNTAX_FAMILIES
    ]
    summary = build_summary(family_rows, early_neurons, route_neurons, late_neurons)
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="L1/L3/L11 传播桥")
    parser.add_argument("--input-dir", default=str(STAGE119_OUTPUT_DIR), help="Stage119 输出目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage131 输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    summary = run_analysis(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status_short": summary["status_short"],
                "output_dir": str(Path(args.output_dir)),
                "mean_l1_l3_corr": summary["mean_l1_l3_corr"],
                "l1_l3_l11_propagation_score": summary["l1_l3_l11_propagation_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
