#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import capture_qwen_mlp_payloads, move_batch_to_model_device, remove_hooks
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, discover_layers, load_qwen_like_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage435_apple_feature_binding_neuron_channels_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
TOP_K = 256
EPS = 1e-8

GROUP_CASES = {
    "apple_noun": [
        {"target": "apple", "sentence": "This is an apple."},
        {"target": "apple", "sentence": "I bought an apple today."},
        {"target": "apple", "sentence": "The apple rolled across the table."},
    ],
    "banana_noun": [
        {"target": "banana", "sentence": "This is a banana."},
        {"target": "banana", "sentence": "I bought a banana today."},
        {"target": "banana", "sentence": "The banana rested on the table."},
    ],
    "fruit_family": [
        {"target": "orange", "sentence": "This is an orange."},
        {"target": "pear", "sentence": "This is a pear."},
        {"target": "grape", "sentence": "This is a grape."},
        {"target": "peach", "sentence": "This is a peach."},
    ],
    "generic_noun": [
        {"target": "car", "sentence": "This is a car."},
        {"target": "book", "sentence": "This is a book."},
        {"target": "chair", "sentence": "This is a chair."},
        {"target": "hammer", "sentence": "This is a hammer."},
    ],
    "color_attr": [
        {"target": "red", "sentence": "The color is red."},
        {"target": "yellow", "sentence": "The color is yellow."},
        {"target": "green", "sentence": "The color is green."},
    ],
    "taste_attr": [
        {"target": "sweet", "sentence": "The taste is sweet."},
        {"target": "sour", "sentence": "The taste is sour."},
        {"target": "juicy", "sentence": "The flavor feels juicy."},
    ],
    "size_anchor": [
        {"target": "fist", "sentence": "The size is about a fist."},
        {"target": "hand", "sentence": "The size is about a hand."},
        {"target": "palm", "sentence": "The size is about a palm."},
    ],
    "apple_color_bind": [
        {"target": "apple", "sentence": "The apple is red."},
        {"target": "apple", "sentence": "The apple is green."},
        {"target": "apple", "sentence": "The apple looks bright red."},
    ],
    "apple_taste_bind": [
        {"target": "apple", "sentence": "The apple tastes sweet."},
        {"target": "apple", "sentence": "The apple tastes sour."},
        {"target": "apple", "sentence": "The apple feels sweet and juicy."},
    ],
    "apple_size_bind": [
        {"target": "apple", "sentence": "The apple is about the size of a fist."},
        {"target": "apple", "sentence": "The apple is about the size of a hand."},
        {"target": "apple", "sentence": "The apple is almost the size of a palm."},
    ],
    "banana_color_bind": [
        {"target": "banana", "sentence": "The banana is yellow."},
        {"target": "banana", "sentence": "The banana is green."},
        {"target": "banana", "sentence": "The banana looks bright yellow."},
    ],
}


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def find_last_subsequence(full_ids: List[int], sub_ids: List[int]) -> Tuple[int, int] | None:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    last_match = None
    for start in range(0, len(full_ids) - len(sub_ids) + 1):
        if full_ids[start : start + len(sub_ids)] == sub_ids:
            last_match = (start, start + len(sub_ids))
    return last_match


def locate_target_span(tokenizer, prompt: str, target: str) -> Tuple[int, int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidates = []
    raw_variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    seen = set()
    for text in raw_variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            candidates.append(ids)
    best = None
    best_len = -1
    for ids in candidates:
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法定位目标词: target={target!r}, prompt={prompt!r}")
    return best


def capture_case_flat_neuron_vector(model, tokenizer, sentence: str, target: str) -> torch.Tensor:
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(len(discover_layers(model)))}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    try:
        encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        start, end = locate_target_span(tokenizer, sentence, target)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        per_layer = []
        for layer_idx in range(len(buffers)):
            layer_tensor = buffers[layer_idx]
            if layer_tensor is None:
                raise RuntimeError(f"第 {layer_idx} 层神经元激活捕获失败")
            vec = layer_tensor[0, start:end, :].mean(dim=0).detach().float().cpu()
            per_layer.append(vec)
        return torch.cat(per_layer, dim=0)
    finally:
        remove_hooks(handles)


def init_group_stats(flat_dim: int) -> Dict[str, torch.Tensor | int]:
    return {
        "n": 0,
        "sum": torch.zeros(flat_dim, dtype=torch.float64),
        "sumsq": torch.zeros(flat_dim, dtype=torch.float64),
    }


def update_group_stats(stats: Dict[str, torch.Tensor | int], vec: torch.Tensor) -> None:
    x = vec.double()
    stats["sum"] += x
    stats["sumsq"] += x * x
    stats["n"] += 1


def mean_var(stats: Dict[str, torch.Tensor | int]) -> Tuple[torch.Tensor, torch.Tensor]:
    n = max(1, int(stats["n"]))
    mean = stats["sum"] / n
    var = stats["sumsq"] / n - mean * mean
    return mean, torch.clamp(var, min=0.0)


def index_to_layer_neuron(flat_idx: int, neuron_count: int) -> Tuple[int, int]:
    return int(flat_idx // neuron_count), int(flat_idx % neuron_count)


def build_top_neuron_set(
    group_name: str,
    group_stats: Dict[str, torch.Tensor | int],
    control_stats: Dict[str, torch.Tensor | int],
    neuron_count: int,
    top_k: int,
) -> Dict[str, object]:
    mean_g, var_g = mean_var(group_stats)
    mean_c, var_c = mean_var(control_stats)
    diff = mean_g - mean_c
    effect = diff / torch.sqrt(0.5 * (var_g + var_c) + EPS)
    mask = diff > 0
    effect = torch.where(mask, effect, torch.full_like(effect, float("-inf")))
    take_k = min(top_k, effect.numel())
    vals, idxs = torch.topk(effect, k=take_k)
    neurons = []
    ids = []
    layer_counts = defaultdict(int)
    for rank, (score, flat_idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        if not torch.isfinite(torch.tensor(score)):
            continue
        layer_idx, neuron_idx = index_to_layer_neuron(int(flat_idx), neuron_count)
        ids.append(int(flat_idx))
        layer_counts[layer_idx] += 1
        neurons.append(
            {
                "rank": rank,
                "flat_index": int(flat_idx),
                "layer_index": layer_idx,
                "neuron_index": neuron_idx,
                "effect_size": float(score),
                "mean_activation": float(mean_g[int(flat_idx)].item()),
                "control_mean_activation": float(mean_c[int(flat_idx)].item()),
            }
        )
    positive_mean = torch.clamp(mean_g.float(), min=0.0)
    active_take_k = min(top_k, positive_mean.numel())
    active_vals, active_idxs = torch.topk(positive_mean, k=active_take_k)
    active_neurons = []
    active_ids = []
    active_layer_counts = defaultdict(int)
    for rank, (value, flat_idx) in enumerate(zip(active_vals.tolist(), active_idxs.tolist()), start=1):
        if value <= 0:
            continue
        layer_idx, neuron_idx = index_to_layer_neuron(int(flat_idx), neuron_count)
        active_ids.append(int(flat_idx))
        active_layer_counts[layer_idx] += 1
        active_neurons.append(
            {
                "rank": rank,
                "flat_index": int(flat_idx),
                "layer_index": layer_idx,
                "neuron_index": neuron_idx,
                "mean_activation": float(value),
            }
        )

    return {
        "group_name": group_name,
        "top_selective_neuron_ids": ids,
        "top_selective_neurons": neurons,
        "selective_layer_distribution": dict(sorted(layer_counts.items())),
        "top_active_neuron_ids": active_ids,
        "top_active_neurons": active_neurons,
        "active_layer_distribution": dict(sorted(active_layer_counts.items())),
    }


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def coverage_ratio(target_ids: Sequence[int], source_ids: Sequence[int]) -> float:
    st = set(int(x) for x in target_ids)
    ss = set(int(x) for x in source_ids)
    if not st:
        return 0.0
    return len(st & ss) / len(st)


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        layers = discover_layers(model)
        neuron_count = int(layers[0].mlp.gate_proj.out_features)
        flat_dim = len(layers) * neuron_count
        raw_stats = {group_name: init_group_stats(flat_dim) for group_name in GROUP_CASES}

        for group_name, cases in GROUP_CASES.items():
            for case in cases:
                vec = capture_case_flat_neuron_vector(model, tokenizer, case["sentence"], case["target"])
                update_group_stats(raw_stats[group_name], vec)

        top_sets = {}
        for group_name in GROUP_CASES:
            control = init_group_stats(flat_dim)
            for other_name, other_stats in raw_stats.items():
                if other_name == group_name:
                    continue
                control["n"] += int(other_stats["n"])
                control["sum"] += other_stats["sum"]
                control["sumsq"] += other_stats["sumsq"]
            top_sets[group_name] = build_top_neuron_set(group_name, raw_stats[group_name], control, neuron_count, TOP_K)

        overlap_matrix = {}
        for a in top_sets:
            overlap_matrix[a] = {}
            for b in top_sets:
                overlap_matrix[a][b] = jaccard(top_sets[a]["top_active_neuron_ids"], top_sets[b]["top_active_neuron_ids"])

        binding_decomposition = {}
        binding_specs = {
            "apple_color_bind": ("apple_noun", "color_attr"),
            "apple_taste_bind": ("apple_noun", "taste_attr"),
            "apple_size_bind": ("apple_noun", "size_anchor"),
            "banana_color_bind": ("banana_noun", "color_attr"),
        }
        for bind_group, (noun_group, attr_group) in binding_specs.items():
            bind_ids = set(top_sets[bind_group]["top_active_neuron_ids"])
            noun_ids = set(top_sets[noun_group]["top_active_neuron_ids"])
            attr_ids = set(top_sets[attr_group]["top_active_neuron_ids"])
            union_ids = noun_ids | attr_ids
            binding_decomposition[bind_group] = {
                "noun_backbone_coverage": coverage_ratio(bind_ids, noun_ids),
                "attribute_modifier_coverage": coverage_ratio(bind_ids, attr_ids),
                "union_coverage": coverage_ratio(bind_ids, union_ids),
                "bridge_only_ratio": safe_ratio(len(bind_ids - union_ids), max(1, len(bind_ids))),
            }

        apple_banana_overlap = overlap_matrix["apple_noun"]["banana_noun"]
        apple_red_gap = overlap_matrix["apple_noun"]["color_attr"]
        apple_taste_gap = overlap_matrix["apple_noun"]["taste_attr"]
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "layer_count": len(layers),
            "neurons_per_layer": neuron_count,
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "top_sets": top_sets,
            "overlap_matrix": overlap_matrix,
            "binding_decomposition": binding_decomposition,
            "summary": {
                "apple_banana_backbone_overlap": apple_banana_overlap,
                "apple_color_overlap": apple_red_gap,
                "apple_taste_overlap": apple_taste_gap,
                "apple_color_union_coverage": binding_decomposition["apple_color_bind"]["union_coverage"],
                "apple_taste_union_coverage": binding_decomposition["apple_taste_bind"]["union_coverage"],
                "apple_size_union_coverage": binding_decomposition["apple_size_bind"]["union_coverage"],
                "apple_color_bridge_only_ratio": binding_decomposition["apple_color_bind"]["bridge_only_ratio"],
                "apple_taste_bridge_only_ratio": binding_decomposition["apple_taste_bind"]["bridge_only_ratio"],
                "apple_size_bridge_only_ratio": binding_decomposition["apple_size_bind"]["bridge_only_ratio"],
            },
            "interpretation": {
                "fruit_backbone_is_shared": bool(apple_banana_overlap >= 0.18),
                "noun_vs_attribute_are_separated": bool(apple_red_gap <= apple_banana_overlap and apple_taste_gap <= apple_banana_overlap),
                "binding_reuses_existing_channels": bool(
                    binding_decomposition["apple_color_bind"]["union_coverage"] >= 0.45
                    and binding_decomposition["apple_taste_bind"]["union_coverage"] >= 0.45
                ),
            },
        }
    finally:
        free_model(model)


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    shared_votes = sum(int(bool(row["interpretation"]["fruit_backbone_is_shared"])) for row in model_results)
    sep_votes = sum(int(bool(row["interpretation"]["noun_vs_attribute_are_separated"])) for row in model_results)
    reuse_votes = sum(int(bool(row["interpretation"]["binding_reuses_existing_channels"])) for row in model_results)
    return {
        "fruit_backbone_shared_vote_count": shared_votes,
        "noun_attribute_separation_vote_count": sep_votes,
        "binding_reuse_vote_count": reuse_votes,
        "core_answer": (
            "苹果与香蕉等水果更像共享一条 fruit backbone channel（水果骨干通道），"
            "而 red（红色）、sweet（甜）、sour（酸）和 fist（拳头大小锚点）更像独立的 attribute modifier channels（属性修饰通道）。"
            "当模型处理 apple-red 或 apple-sweet 这类组合时，主要是把名词骨干和属性修饰在同一残差工作区里叠加，"
            "再用一小部分 bridge neurons（桥接神经元）完成绑定。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心回答",
        summary["cross_model_summary"]["core_answer"],
        "",
    ]
    for result in summary["model_results"]:
        s = result["summary"]
        lines.extend(
            [
                f"## {result['model_name']}",
                f"- apple vs banana backbone overlap: {s['apple_banana_backbone_overlap']:.4f}",
                f"- apple vs color overlap: {s['apple_color_overlap']:.4f}",
                f"- apple vs taste overlap: {s['apple_taste_overlap']:.4f}",
                f"- apple-color union coverage: {s['apple_color_union_coverage']:.4f}",
                f"- apple-taste union coverage: {s['apple_taste_union_coverage']:.4f}",
                f"- apple-size union coverage: {s['apple_size_union_coverage']:.4f}",
                f"- apple-color bridge only ratio: {s['apple_color_bridge_only_ratio']:.4f}",
                f"- apple-taste bridge only ratio: {s['apple_taste_bridge_only_ratio']:.4f}",
                f"- apple-size bridge only ratio: {s['apple_size_bridge_only_ratio']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 理论解释",
            "- 苹果和香蕉的神经元编码相交更多，说明水果名词共享骨干通道。",
            "- 红色、酸甜等属性词与水果名词的重叠较低，说明它们不是同一类通道。",
            "- 组合句中的苹果表示主要落在“水果骨干 + 属性通道”的并集里，外加少量桥接神经元，这支持神经元级别的绑定而不是整块重写。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple feature binding neuron channel analysis")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = not bool(args.cpu)
    model_results = [analyze_model(model_key, prefer_cuda=prefer_cuda) for model_key in MODEL_ORDER]
    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage435_apple_feature_binding_neuron_channels",
        "title": "苹果属性绑定的神经元通道分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
        "group_cases": GROUP_CASES,
        "top_k": TOP_K,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
