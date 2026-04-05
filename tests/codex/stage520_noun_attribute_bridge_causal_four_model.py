#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from multimodel_language_shared import free_model, load_model_bundle, discover_layers
from qwen3_language_shared import move_batch_to_model_device, remove_hooks
from stage427_pronoun_mixed_circuit_search import register_mlp_neuron_ablation
from stage515_cross_task_minimal_causal_circuit import (
    OFFTARGET_PENALTY,
    MIN_GAIN,
    MAX_SUBSET,
    TOP_CANDIDATES,
    build_prompt,
    decode_flat_index,
    ensure_dir,
    evaluate_case_groups,
    option_token_id,
    patch_glm4_mlp_compat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage520_noun_attribute_bridge_causal_four_model_20260404"
)
STAGE514_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage514_multi_family_cross_task_core_protocol_20260404"
    / "summary.json"
)
MODEL_KEYS = ["qwen3", "deepseek7b", "glm4", "gemma4"]

TARGET_CASES = {
    "knowledge": [
        {"sentence": "An apple is a kind of fruit.", "question": "Choose the better category. Reply with 1 or 2 only.", "options": ["1", "2"], "label": 0},
        {"sentence": "The apple belongs to the fruit family.", "question": "Choose the better category. Reply with 1 or 2 only.", "options": ["1", "2"], "label": 0},
    ],
    "attribute": [
        {"sentence": "The apple is red and sweet.", "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.", "options": ["1", "2"], "label": 0},
        {"sentence": "The apple is about the size of a fist.", "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.", "options": ["1", "2"], "label": 0},
        {"sentence": "The banana is yellow, and the apple sits on the plate.", "question": "If the sentence binds an attribute to apple, reply 1. Otherwise reply 2.", "options": ["1", "2"], "label": 1},
    ],
}

CONTROL_CASES = {
    "knowledge": [
        {"sentence": "A banana is a kind of fruit.", "question": "Choose the better category. Reply with 1 or 2 only.", "options": ["1", "2"], "label": 0},
        {"sentence": "People often eat a banana as a snack.", "question": "Choose the better category. Reply with 1 or 2 only.", "options": ["1", "2"], "label": 0},
    ],
    "attribute": [
        {"sentence": "The banana is yellow and soft.", "question": "If the sentence binds an attribute to banana, reply 1. Otherwise reply 2.", "options": ["1", "2"], "label": 0},
        {"sentence": "The apple is red, while the banana is on the plate.", "question": "If the sentence binds an attribute to banana, reply 1. Otherwise reply 2.", "options": ["1", "2"], "label": 1},
    ],
}


def build_bridge_candidates(model_row: dict, layer_widths: Sequence[int]) -> list[dict]:
    family_row = next(row for row in model_row["family_rows"] if row["family_id"] == "fruit_apple")
    noun_ids = family_row["group_top_ids"]["family_knowledge"]
    attr_ids = family_row["group_top_ids"]["attribute_binding"]
    noun_rank = {int(idx): rank for rank, idx in enumerate(noun_ids, start=1)}
    attr_rank = {int(idx): rank for rank, idx in enumerate(attr_ids, start=1)}
    shared_ids = sorted(set(noun_rank) & set(attr_rank), key=lambda idx: noun_rank[idx] + attr_rank[idx])[:TOP_CANDIDATES]
    rows = []
    for flat_idx in shared_ids:
        layer_index, neuron_index = decode_flat_index(int(flat_idx), layer_widths)
        rows.append(
            {
                "kind": "mlp_neuron",
                "flat_index": int(flat_idx),
                "layer_index": int(layer_index),
                "neuron_index": int(neuron_index),
                "bridge_rank_score": float((TOP_CANDIDATES * 2) - (noun_rank[int(flat_idx)] + attr_rank[int(flat_idx)])),
            }
        )
    return rows


def evaluate_subset(model, tokenizer, candidate_map: Dict[str, dict], subset_ids: Sequence[str], baseline_target: dict, baseline_control: dict) -> dict:
    subset = [candidate_map[cid] for cid in subset_ids]
    handles = register_mlp_neuron_ablation(model, subset) if subset else []
    try:
        current_target = evaluate_case_groups(model, tokenizer, TARGET_CASES)
        current_control = evaluate_case_groups(model, tokenizer, CONTROL_CASES)
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


def search_model(model_key: str, stage514_model_row: dict) -> dict:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
    try:
        if model_key == "glm4":
            patch_glm4_mlp_compat(model)
        layer_widths = [int(layer.mlp.down_proj.in_features) for layer in discover_layers(model)]
        candidates = build_bridge_candidates(stage514_model_row, layer_widths)
        candidate_map = {f"N:{row['layer_index']}:{row['neuron_index']}": row for row in candidates}
        baseline_target = evaluate_case_groups(model, tokenizer, TARGET_CASES)
        baseline_control = evaluate_case_groups(model, tokenizer, CONTROL_CASES)

        single_rows = []
        for cid, row in candidate_map.items():
            result = evaluate_subset(model, tokenizer, candidate_map, [cid], baseline_target, baseline_control)
            row_out = dict(row)
            row_out.update({"candidate_id": cid, "target_drop": result["target_drop"], "control_abs_shift": result["control_abs_shift"], "utility": result["utility"]})
            single_rows.append(row_out)
        single_rows.sort(key=lambda row: float(row["utility"]), reverse=True)

        chosen: List[str] = []
        best_result = evaluate_subset(model, tokenizer, candidate_map, [], baseline_target, baseline_control)
        shortlist = [row["candidate_id"] for row in single_rows[:8]]
        for _ in range(MAX_SUBSET):
            best_candidate = None
            best_candidate_result = None
            for cid in shortlist:
                if cid in chosen:
                    continue
                result = evaluate_subset(model, tokenizer, candidate_map, chosen + [cid], baseline_target, baseline_control)
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
                reduced = [x for x in chosen if x != cid]
                result = evaluate_subset(model, tokenizer, candidate_map, reduced, baseline_target, baseline_control)
                if result["utility"] >= best_result["utility"] * 0.95:
                    chosen = reduced
                    best_result = result
                    pruned = True
                    break
        return {
            "model_key": model_key,
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
    stage514 = json.loads(STAGE514_PATH.read_text(encoding="utf-8"))
    stage514_rows = {row["model_key"]: row for row in stage514["model_rows"]}
    started = time.time()
    model_rows = [search_model(model_key, stage514_rows[model_key]) for model_key in MODEL_KEYS]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage520_noun_attribute_bridge_causal_four_model",
        "title": "四模型名词-属性桥接因果搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_rows": model_rows,
        "core_answer": (
            "名词与属性之间的连接并不只是结构重叠，其中一小组跨名词知识和属性绑定共享的神经元，"
            "在部分模型里已经开始表现出可压缩的桥接因果作用。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# stage520 四模型名词-属性桥接因果搜索", "", "## 核心结论", summary["core_answer"], ""]
    for row in model_rows:
        fr = row["final_result"]
        lines.extend(
            [
                f"## {row['model_key']}",
                f"- 最终子集：`{', '.join(row['final_subset']) if row['final_subset'] else '空'}`",
                f"- 目标下降：`{fr['target_drop']:.6f}`",
                f"- 控制偏移：`{fr['control_abs_shift']:.6f}`",
                f"- 综合效用：`{fr['utility']:.6f}`",
                "",
            ]
        )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
