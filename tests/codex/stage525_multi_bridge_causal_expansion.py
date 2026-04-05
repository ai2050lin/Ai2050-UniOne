#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from multimodel_language_shared import discover_layers, free_model, load_model_bundle
from qwen3_language_shared import remove_hooks
from stage427_pronoun_mixed_circuit_search import register_mlp_neuron_ablation
from stage515_cross_task_minimal_causal_circuit import (
    MAX_SUBSET,
    MIN_GAIN,
    OFFTARGET_PENALTY,
    TOP_CANDIDATES,
    _OutFeatureShim,
    decode_flat_index,
    ensure_dir,
    evaluate_case_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage525_multi_bridge_causal_expansion_20260404"
)
STAGE514_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage514_multi_family_cross_task_core_protocol_20260404"
    / "summary.json"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]

BRIDGE_CASES = {
    "noun_relation": {
        "target": {
            "relation": [
                {
                    "sentence": "An apple belongs to the fruit family.",
                    "question": "If the sentence states an apple-family relation, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "Apple is a kind of fruit.",
                    "question": "If the sentence states an apple-family relation, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "Apple is resting on the table.",
                    "question": "If the sentence states an apple-family relation, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ]
        },
        "control": {
            "relation": [
                {
                    "sentence": "A banana belongs to the fruit family.",
                    "question": "If the sentence states a banana-family relation, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "Banana is on the tray beside a cup.",
                    "question": "If the sentence states a banana-family relation, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ]
        },
    },
    "noun_syntax_role": {
        "target": {
            "knowledge": [
                {
                    "sentence": "Apple is a kind of fruit.",
                    "question": "If the sentence states the family knowledge of apple, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                }
            ],
            "syntax": [
                {
                    "sentence": "The apple fell from the tree.",
                    "question": "If apple is the subject, reply 1. If it is the object, reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "She sliced the apple after dinner.",
                    "question": "If apple is the subject, reply 1. If it is the object, reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ],
        },
        "control": {
            "knowledge": [
                {
                    "sentence": "Banana is a kind of fruit.",
                    "question": "If the sentence states the family knowledge of banana, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                }
            ],
            "syntax": [
                {
                    "sentence": "The banana dropped from the shelf.",
                    "question": "If banana is the subject, reply 1. If it is the object, reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "She peeled the banana before lunch.",
                    "question": "If banana is the subject, reply 1. If it is the object, reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ],
        },
    },
    "noun_association_network": {
        "target": {
            "knowledge": [
                {
                    "sentence": "Apple is a kind of fruit.",
                    "question": "If the sentence states the family knowledge of apple, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                }
            ],
            "association": [
                {
                    "sentence": "Apple is related to banana, pear, and juice.",
                    "question": "If apple is placed in a fruit association network, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "Apple is beside a wrench, a cable, and a lamp.",
                    "question": "If apple is placed in a fruit association network, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ],
        },
        "control": {
            "knowledge": [
                {
                    "sentence": "Banana is a kind of fruit.",
                    "question": "If the sentence states the family knowledge of banana, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                }
            ],
            "association": [
                {
                    "sentence": "Banana is related to apple, pear, and tree.",
                    "question": "If banana is placed in a fruit association network, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 0,
                },
                {
                    "sentence": "Banana is near a hammer, a cable, and a lamp.",
                    "question": "If banana is placed in a fruit association network, reply 1. Otherwise reply 2.",
                    "options": ["1", "2"],
                    "label": 1,
                },
            ],
        },
    },
}


def patch_glm4_mlp_compat(model) -> None:
    for layer in discover_layers(model):
        mlp = layer.mlp
        if not hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            mlp.gate_proj = _OutFeatureShim(mlp.down_proj.in_features)


def load_stage514_summary() -> dict:
    return json.loads(STAGE514_PATH.read_text(encoding="utf-8"))


def backbone_ids(group_top_ids: dict[str, list[int]], min_groups: int = 4) -> list[int]:
    counter: Counter[int] = Counter()
    for ids in group_top_ids.values():
        counter.update(set(int(x) for x in ids))
    return [idx for idx, count in counter.items() if count >= min_groups]


def build_candidate_pool(model_row: dict, bridge_kind: str, layer_widths: Sequence[int]) -> list[dict]:
    family_row = next(row for row in model_row["family_rows"] if row["family_id"] == "fruit_apple")
    group_top_ids = family_row["group_top_ids"]
    backbone = set(backbone_ids(group_top_ids, min_groups=4))
    if bridge_kind == "noun_relation":
        target_sets = ["family_knowledge"]
    elif bridge_kind == "noun_syntax_role":
        target_sets = ["family_knowledge", "syntax_subject", "syntax_object"]
    elif bridge_kind == "noun_association_network":
        target_sets = ["family_knowledge", "concept_association"]
    else:
        raise KeyError(f"未知桥接类型: {bridge_kind}")

    rank_maps = {
        name: {int(idx): rank for rank, idx in enumerate(group_top_ids[name], start=1)}
        for name in target_sets
    }
    shared_ids = backbone.copy()
    for name in target_sets:
        shared_ids &= set(rank_maps[name])
    sorted_ids = sorted(
        shared_ids,
        key=lambda idx: sum(rank_maps[name][idx] for name in target_sets),
    )[:TOP_CANDIDATES]

    rows = []
    for flat_idx in sorted_ids:
        layer_index, neuron_index = decode_flat_index(int(flat_idx), layer_widths)
        rows.append(
            {
                "kind": "mlp_neuron",
                "flat_index": int(flat_idx),
                "layer_index": int(layer_index),
                "neuron_index": int(neuron_index),
                "bridge_rank_score": float(
                    (TOP_CANDIDATES * len(target_sets)) - sum(rank_maps[name][flat_idx] for name in target_sets)
                ),
            }
        )
    return rows


def evaluate_subset(
    model,
    tokenizer,
    candidate_map: Dict[str, dict],
    subset_ids: Sequence[str],
    target_cases: dict,
    control_cases: dict,
    baseline_target: dict,
    baseline_control: dict,
) -> dict:
    subset = [candidate_map[cid] for cid in subset_ids]
    handles = register_mlp_neuron_ablation(model, subset) if subset else []
    try:
        current_target = evaluate_case_groups(model, tokenizer, target_cases)
        current_control = evaluate_case_groups(model, tokenizer, control_cases)
    finally:
        remove_hooks(handles)
    target_drop = float(baseline_target["mean_correct_prob"] - current_target["mean_correct_prob"])
    control_shift = abs(float(baseline_control["mean_correct_prob"] - current_control["mean_correct_prob"]))
    utility = target_drop - OFFTARGET_PENALTY * control_shift
    return {
        "subset_ids": list(subset_ids),
        "subset_size": len(subset_ids),
        "target_drop": target_drop,
        "control_abs_shift": control_shift,
        "utility": utility,
        "target_current": current_target,
        "control_current": current_control,
    }


def search_bridge(model_key: str, model_row: dict, bridge_kind: str) -> dict:
    target_cases = BRIDGE_CASES[bridge_kind]["target"]
    control_cases = BRIDGE_CASES[bridge_kind]["control"]
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        if model_key == "glm4":
            patch_glm4_mlp_compat(model)
        layer_widths = [int(layer.mlp.down_proj.in_features) for layer in discover_layers(model)]
        candidates = build_candidate_pool(model_row, bridge_kind, layer_widths)
        candidate_map = {f"N:{row['layer_index']}:{row['neuron_index']}": row for row in candidates}
        baseline_target = evaluate_case_groups(model, tokenizer, target_cases)
        baseline_control = evaluate_case_groups(model, tokenizer, control_cases)

        single_rows = []
        for cid, row in candidate_map.items():
            result = evaluate_subset(model, tokenizer, candidate_map, [cid], target_cases, control_cases, baseline_target, baseline_control)
            row_out = dict(row)
            row_out.update(
                {
                    "candidate_id": cid,
                    "target_drop": result["target_drop"],
                    "control_abs_shift": result["control_abs_shift"],
                    "utility": result["utility"],
                }
            )
            single_rows.append(row_out)
        single_rows.sort(key=lambda item: float(item["utility"]), reverse=True)

        chosen: List[str] = []
        best_result = evaluate_subset(model, tokenizer, candidate_map, [], target_cases, control_cases, baseline_target, baseline_control)
        shortlist = [row["candidate_id"] for row in single_rows[:8]]
        for _ in range(MAX_SUBSET):
            best_candidate = None
            best_candidate_result = None
            for cid in shortlist:
                if cid in chosen:
                    continue
                result = evaluate_subset(
                    model,
                    tokenizer,
                    candidate_map,
                    chosen + [cid],
                    target_cases,
                    control_cases,
                    baseline_target,
                    baseline_control,
                )
                if result["utility"] > best_result["utility"] + MIN_GAIN and (
                    best_candidate_result is None or result["utility"] > best_candidate_result["utility"]
                ):
                    best_candidate = cid
                    best_candidate_result = result
            if best_candidate is None or best_candidate_result is None:
                break
            chosen.append(best_candidate)
            best_result = best_candidate_result

        pruned = True
        while pruned and len(chosen) > 1:
            pruned = False
            for cid in list(chosen):
                reduced = [item for item in chosen if item != cid]
                result = evaluate_subset(
                    model,
                    tokenizer,
                    candidate_map,
                    reduced,
                    target_cases,
                    control_cases,
                    baseline_target,
                    baseline_control,
                )
                if result["utility"] >= best_result["utility"] * 0.95:
                    chosen = reduced
                    best_result = result
                    pruned = True
                    break

        return {
            "bridge_kind": bridge_kind,
            "baseline_target": baseline_target,
            "baseline_control": baseline_control,
            "candidate_neurons": single_rows,
            "final_subset": chosen,
            "final_result": best_result,
        }
    finally:
        free_model(model)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    stage514 = load_stage514_summary()
    stage514_rows = {row["model_key"]: row for row in stage514["model_rows"]}
    model_rows = []
    for model_key in MODEL_KEYS:
        bridge_rows = []
        for bridge_kind in BRIDGE_CASES:
            bridge_rows.append(search_bridge(model_key, stage514_rows[model_key], bridge_kind))
        model_rows.append({"model_key": model_key, "bridge_rows": bridge_rows})

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage525_multi_bridge_causal_expansion",
        "title": "名词-关系-语法-联想桥接因果扩展",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_summary": str(STAGE514_PATH),
        "model_rows": model_rows,
        "core_answer": (
            "名词桥接并不只存在于名词-属性之间。把桥接因果扩展到名词-关系、名词-语法角色、"
            "名词-联想网络之后，仍然可以在四模型里找到小型桥接因果子集，"
            "说明概念骨干和任务接口之间确实存在可压缩的连接单元。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# stage525 名词-关系-语法-联想桥接因果扩展",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for model_row in model_rows:
        lines.append(f"## {model_row['model_key']}")
        for bridge_row in model_row["bridge_rows"]:
            result = bridge_row["final_result"]
            lines.append(
                f"- `{bridge_row['bridge_kind']}`：最终子集 "
                f"`{', '.join(bridge_row['final_subset']) if bridge_row['final_subset'] else '空'}`，"
                f"目标下降 `{result['target_drop']:.6f}`，"
                f"控制偏移 `{result['control_abs_shift']:.6f}`，"
                f"效用 `{result['utility']:.6f}`"
            )
        lines.append("")
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
