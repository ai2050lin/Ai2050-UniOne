#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage435_apple_feature_binding_neuron_channels import (
    TOP_K,
    build_top_neuron_set,
    capture_case_flat_neuron_vector,
    coverage_ratio,
    init_group_stats,
    jaccard,
    update_group_stats,
)
from qwen3_language_shared import discover_layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage440_attribute_graph_generalization_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]

GROUP_CASES: Dict[str, List[Dict[str, str]]] = {
    "noun_apple": [
        {"target": "apple", "sentence": "This is an apple."},
        {"target": "apple", "sentence": "I picked an apple after lunch."},
        {"target": "apple", "sentence": "The apple rested on the table."},
    ],
    "noun_banana": [
        {"target": "banana", "sentence": "This is a banana."},
        {"target": "banana", "sentence": "I bought a banana this morning."},
        {"target": "banana", "sentence": "The banana stayed near the bowl."},
    ],
    "noun_orange": [
        {"target": "orange", "sentence": "This is an orange."},
        {"target": "orange", "sentence": "The orange rolled across the counter."},
        {"target": "orange", "sentence": "I peeled an orange in the kitchen."},
    ],
    "noun_lemon": [
        {"target": "lemon", "sentence": "This is a lemon."},
        {"target": "lemon", "sentence": "The lemon sat beside the sink."},
        {"target": "lemon", "sentence": "I sliced a lemon for tea."},
    ],
    "noun_pear": [
        {"target": "pear", "sentence": "This is a pear."},
        {"target": "pear", "sentence": "The pear stayed in the basket."},
        {"target": "pear", "sentence": "She carried a pear home."},
    ],
    "noun_grape": [
        {"target": "grape", "sentence": "This is a grape."},
        {"target": "grape", "sentence": "The grape fell from the bunch."},
        {"target": "grape", "sentence": "A grape rolled toward the glass."},
    ],
    "attr_red": [
        {"target": "red", "sentence": "The color is red."},
        {"target": "red", "sentence": "It looks bright red."},
    ],
    "attr_green": [
        {"target": "green", "sentence": "The color is green."},
        {"target": "green", "sentence": "It appears light green."},
    ],
    "attr_yellow": [
        {"target": "yellow", "sentence": "The color is yellow."},
        {"target": "yellow", "sentence": "It shines bright yellow."},
    ],
    "attr_orange_color": [
        {"target": "orange", "sentence": "The color is orange."},
        {"target": "orange", "sentence": "It glows warm orange."},
    ],
    "attr_purple": [
        {"target": "purple", "sentence": "The color is purple."},
        {"target": "purple", "sentence": "It looks deep purple."},
    ],
    "attr_sweet": [
        {"target": "sweet", "sentence": "The taste is sweet."},
        {"target": "sweet", "sentence": "It tastes fresh and sweet."},
    ],
    "attr_sour": [
        {"target": "sour", "sentence": "The taste is sour."},
        {"target": "sour", "sentence": "It tastes sharply sour."},
    ],
    "attr_juicy": [
        {"target": "juicy", "sentence": "The flavor feels juicy."},
        {"target": "juicy", "sentence": "It stays juicy inside."},
    ],
    "attr_tart": [
        {"target": "tart", "sentence": "The taste is tart."},
        {"target": "tart", "sentence": "It feels pleasantly tart."},
    ],
    "attr_fist": [
        {"target": "fist", "sentence": "The size is about a fist."},
        {"target": "fist", "sentence": "It is roughly fist-sized."},
    ],
    "attr_hand": [
        {"target": "hand", "sentence": "The size is about a hand."},
        {"target": "hand", "sentence": "It is almost hand-sized."},
    ],
    "attr_palm": [
        {"target": "palm", "sentence": "The size is about a palm."},
        {"target": "palm", "sentence": "It feels close to palm-sized."},
    ],
    "attr_thumb": [
        {"target": "thumb", "sentence": "The size is about a thumb."},
        {"target": "thumb", "sentence": "It is nearly thumb-sized."},
    ],
    "bind_apple_red": [
        {"target": "apple", "sentence": "The apple is red."},
        {"target": "apple", "sentence": "The apple looked bright red on the tray."},
    ],
    "bind_apple_sweet": [
        {"target": "apple", "sentence": "The apple tastes sweet."},
        {"target": "apple", "sentence": "The apple felt sweet after one bite."},
    ],
    "bind_apple_fist": [
        {"target": "apple", "sentence": "The apple is about the size of a fist."},
        {"target": "apple", "sentence": "The apple was roughly the size of a fist."},
    ],
    "bind_banana_yellow": [
        {"target": "banana", "sentence": "The banana is yellow."},
        {"target": "banana", "sentence": "The banana looked bright yellow in the bowl."},
    ],
    "bind_banana_sweet": [
        {"target": "banana", "sentence": "The banana tastes sweet."},
        {"target": "banana", "sentence": "The banana felt sweet and soft."},
    ],
    "bind_banana_hand": [
        {"target": "banana", "sentence": "The banana is about the size of a hand."},
        {"target": "banana", "sentence": "The banana seemed nearly hand-sized."},
    ],
    "bind_orange_orange_color": [
        {"target": "orange", "sentence": "The orange is orange."},
        {"target": "orange", "sentence": "The orange looked warm orange in the light."},
    ],
    "bind_orange_juicy": [
        {"target": "orange", "sentence": "The orange tastes juicy."},
        {"target": "orange", "sentence": "The orange stayed juicy after peeling."},
    ],
    "bind_orange_palm": [
        {"target": "orange", "sentence": "The orange is about the size of a palm."},
        {"target": "orange", "sentence": "The orange felt close to palm-sized."},
    ],
    "bind_lemon_yellow": [
        {"target": "lemon", "sentence": "The lemon is yellow."},
        {"target": "lemon", "sentence": "The lemon looked pale yellow."},
    ],
    "bind_lemon_sour": [
        {"target": "lemon", "sentence": "The lemon tastes sour."},
        {"target": "lemon", "sentence": "The lemon felt sharply sour."},
    ],
    "bind_lemon_thumb": [
        {"target": "lemon", "sentence": "The lemon is about the size of a thumb."},
        {"target": "lemon", "sentence": "The lemon seemed nearly thumb-sized."},
    ],
    "bind_pear_green": [
        {"target": "pear", "sentence": "The pear is green."},
        {"target": "pear", "sentence": "The pear looked soft green."},
    ],
    "bind_pear_sweet": [
        {"target": "pear", "sentence": "The pear tastes sweet."},
        {"target": "pear", "sentence": "The pear felt sweet and smooth."},
    ],
    "bind_pear_hand": [
        {"target": "pear", "sentence": "The pear is about the size of a hand."},
        {"target": "pear", "sentence": "The pear looked nearly hand-sized."},
    ],
    "bind_grape_purple": [
        {"target": "grape", "sentence": "The grape is purple."},
        {"target": "grape", "sentence": "The grape looked deep purple in the sun."},
    ],
    "bind_grape_sweet": [
        {"target": "grape", "sentence": "The grape tastes sweet."},
        {"target": "grape", "sentence": "The grape felt sweet and juicy."},
    ],
    "bind_grape_thumb": [
        {"target": "grape", "sentence": "The grape is about the size of a thumb."},
        {"target": "grape", "sentence": "The grape seemed nearly thumb-sized."},
    ],
}

BINDING_SPECS: Dict[str, Tuple[str, str, str]] = {
    "bind_apple_red": ("noun_apple", "attr_red", "color"),
    "bind_apple_sweet": ("noun_apple", "attr_sweet", "taste"),
    "bind_apple_fist": ("noun_apple", "attr_fist", "size"),
    "bind_banana_yellow": ("noun_banana", "attr_yellow", "color"),
    "bind_banana_sweet": ("noun_banana", "attr_sweet", "taste"),
    "bind_banana_hand": ("noun_banana", "attr_hand", "size"),
    "bind_orange_orange_color": ("noun_orange", "attr_orange_color", "color"),
    "bind_orange_juicy": ("noun_orange", "attr_juicy", "taste"),
    "bind_orange_palm": ("noun_orange", "attr_palm", "size"),
    "bind_lemon_yellow": ("noun_lemon", "attr_yellow", "color"),
    "bind_lemon_sour": ("noun_lemon", "attr_sour", "taste"),
    "bind_lemon_thumb": ("noun_lemon", "attr_thumb", "size"),
    "bind_pear_green": ("noun_pear", "attr_green", "color"),
    "bind_pear_sweet": ("noun_pear", "attr_sweet", "taste"),
    "bind_pear_hand": ("noun_pear", "attr_hand", "size"),
    "bind_grape_purple": ("noun_grape", "attr_purple", "color"),
    "bind_grape_sweet": ("noun_grape", "attr_sweet", "taste"),
    "bind_grape_thumb": ("noun_grape", "attr_thumb", "size"),
}

NOUN_GROUPS = [key for key in GROUP_CASES if key.startswith("noun_")]
ATTR_GROUPS = [key for key in GROUP_CASES if key.startswith("attr_")]


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


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def build_model_stats(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
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

        binding_rows = []
        support_count = 0
        family_totals: Dict[str, Dict[str, float]] = {
            "color": {"count": 0.0, "union_coverage_sum": 0.0, "bridge_only_sum": 0.0},
            "taste": {"count": 0.0, "union_coverage_sum": 0.0, "bridge_only_sum": 0.0},
            "size": {"count": 0.0, "union_coverage_sum": 0.0, "bridge_only_sum": 0.0},
        }

        for bind_group, (noun_group, attr_group, family_name) in BINDING_SPECS.items():
            bind_ids = set(int(x) for x in top_sets[bind_group]["top_active_neuron_ids"])
            noun_ids = set(int(x) for x in top_sets[noun_group]["top_active_neuron_ids"])
            attr_ids = set(int(x) for x in top_sets[attr_group]["top_active_neuron_ids"])
            union_ids = noun_ids | attr_ids
            row = {
                "bind_group": bind_group,
                "noun_group": noun_group,
                "attr_group": attr_group,
                "family_name": family_name,
                "noun_backbone_coverage": coverage_ratio(bind_ids, noun_ids),
                "attribute_modifier_coverage": coverage_ratio(bind_ids, attr_ids),
                "union_coverage": coverage_ratio(bind_ids, union_ids),
                "bridge_only_ratio": safe_ratio(len(bind_ids - union_ids), max(1, len(bind_ids))),
            }
            row["law_support"] = bool(
                row["union_coverage"] >= 0.60
                and row["bridge_only_ratio"] >= 0.10
                and row["bridge_only_ratio"] <= 0.40
            )
            support_count += int(row["law_support"])
            binding_rows.append(row)
            family_totals[family_name]["count"] += 1.0
            family_totals[family_name]["union_coverage_sum"] += row["union_coverage"]
            family_totals[family_name]["bridge_only_sum"] += row["bridge_only_ratio"]

        noun_pair_overlaps = [
            overlap_matrix[a][b]
            for a, b in combinations(NOUN_GROUPS, 2)
        ]
        noun_attr_overlaps = [
            overlap_matrix[noun_group][attr_group]
            for noun_group in NOUN_GROUPS
            for attr_group in ATTR_GROUPS
        ]

        family_summary = {}
        for family_name, totals in family_totals.items():
            family_summary[family_name] = {
                "count": int(totals["count"]),
                "mean_union_coverage": safe_ratio(totals["union_coverage_sum"], totals["count"]),
                "mean_bridge_only_ratio": safe_ratio(totals["bridge_only_sum"], totals["count"]),
            }

        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "layer_count": len(layers),
            "neurons_per_layer": neuron_count,
            "binding_rows": binding_rows,
            "family_summary": family_summary,
            "summary": {
                "binding_group_count": len(binding_rows),
                "law_support_count": support_count,
                "law_support_rate": safe_ratio(support_count, len(binding_rows)),
                "mean_union_coverage": safe_ratio(sum(row["union_coverage"] for row in binding_rows), len(binding_rows)),
                "mean_bridge_only_ratio": safe_ratio(sum(row["bridge_only_ratio"] for row in binding_rows), len(binding_rows)),
                "mean_fruit_fruit_overlap": safe_ratio(sum(noun_pair_overlaps), len(noun_pair_overlaps)),
                "mean_fruit_attribute_overlap": safe_ratio(sum(noun_attr_overlaps), len(noun_attr_overlaps)),
            },
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = sum(int(row["summary"]["binding_group_count"]) for row in model_results)
    supported = sum(int(row["summary"]["law_support_count"]) for row in model_results)
    return {
        "attribute_graph_total_bindings": total,
        "attribute_graph_support_count": supported,
        "attribute_graph_support_rate": safe_ratio(supported, total),
        "mean_union_coverage": safe_ratio(
            sum(float(row["summary"]["mean_union_coverage"]) for row in model_results),
            len(model_results),
        ),
        "mean_bridge_only_ratio": safe_ratio(
            sum(float(row["summary"]["mean_bridge_only_ratio"]) for row in model_results),
            len(model_results),
        ),
        "mean_fruit_fruit_overlap": safe_ratio(
            sum(float(row["summary"]["mean_fruit_fruit_overlap"]) for row in model_results),
            len(model_results),
        ),
        "mean_fruit_attribute_overlap": safe_ratio(
            sum(float(row["summary"]["mean_fruit_attribute_overlap"]) for row in model_results),
            len(model_results),
        ),
        "core_answer": (
            "如果多数水果-属性组合都能用 noun backbone（名词骨干）与 attribute modifier（属性修饰）并集解释大部分绑定神经元，"
            "同时还保留一块稳定但不占主导的 bridge-only 区域，那么“骨干 + 修饰 + 桥接”就不再只是苹果个案，而是更一般的属性绑定定律。"
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
    for model_row in summary["model_results"]:
        s = model_row["summary"]
        lines.extend(
            [
                f"## {model_row['model_name']}",
                f"- law_support_rate: {s['law_support_rate']:.4f}",
                f"- mean_union_coverage: {s['mean_union_coverage']:.4f}",
                f"- mean_bridge_only_ratio: {s['mean_bridge_only_ratio']:.4f}",
                f"- mean_fruit_fruit_overlap: {s['mean_fruit_fruit_overlap']:.4f}",
                f"- mean_fruit_attribute_overlap: {s['mean_fruit_attribute_overlap']:.4f}",
                "",
            ]
        )
        for family_name, family_row in model_row["family_summary"].items():
            lines.append(
                f"- {family_name}: union={family_row['mean_union_coverage']:.4f}, bridge={family_row['mean_bridge_only_ratio']:.4f}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="更大属性图谱泛化实验")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    model_results = [build_model_stats(model_key, prefer_cuda=prefer_cuda) for model_key in MODEL_ORDER]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage440_attribute_graph_generalization",
        "title": "更大属性图谱泛化实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "top_k": TOP_K,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
        "binding_specs": BINDING_SPECS,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
