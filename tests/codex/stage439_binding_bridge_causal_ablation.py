#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import register_mlp_neuron_ablation
from stage435_apple_feature_binding_neuron_channels import GROUP_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE435_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage435_apple_feature_binding_neuron_channels_20260402"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage439_binding_bridge_causal_ablation_20260402"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
OPTION_DIGITS = ["1", "2", "3", "4"]
CANDIDATE_SIZES = [8, 16, 32, 48]
OFF_TARGET_PENALTY = 0.50
RANDOM_SEED = 20260402

BINDING_FAMILY_SPECS: Dict[str, Dict[str, object]] = {
    "color": {
        "bind_group": "apple_color_bind",
        "noun_group": "apple_noun",
        "attr_group": "color_attr",
        "search_cases": [
            {"case_id": "apple_red_search_1", "task_type": "binding", "sentence": "The apple is red.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
            {"case_id": "apple_green_search_1", "task_type": "binding", "sentence": "The apple is green.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 2},
            {"case_id": "apple_red_search_2", "task_type": "binding", "sentence": "The apple looks bright red on the plate.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
            {"case_id": "noun_search_1", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][0]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "noun_search_2", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][1]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "attr_search_1", "task_type": "attribute_only", "sentence": "The color is red.", "question": "Which color is named in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
            {"case_id": "attr_search_2", "task_type": "attribute_only", "sentence": "The color is green.", "question": "Which color is named in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 2},
        ],
        "heldout_binding_cases": [
            {"case_id": "apple_red_holdout_1", "task_type": "binding", "sentence": "At lunch, the apple stayed red in the bowl.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
            {"case_id": "apple_green_holdout_1", "task_type": "binding", "sentence": "Near the window, the apple remained green all afternoon.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 2},
            {"case_id": "apple_red_holdout_2", "task_type": "binding", "sentence": "The baker noticed that the apple looked deep red.", "question": "Which color is attached to apple in this sentence?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
        ],
    },
    "taste": {
        "bind_group": "apple_taste_bind",
        "noun_group": "apple_noun",
        "attr_group": "taste_attr",
        "search_cases": [
            {"case_id": "apple_sweet_search_1", "task_type": "binding", "sentence": "The apple tastes sweet.", "question": "Which taste is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
            {"case_id": "apple_sour_search_1", "task_type": "binding", "sentence": "The apple tastes sour.", "question": "Which taste is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 1},
            {"case_id": "apple_juicy_search_1", "task_type": "binding", "sentence": "The apple feels juicy after one bite.", "question": "Which taste or flavor property is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 2},
            {"case_id": "noun_search_1", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][0]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "noun_search_2", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][2]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "attr_search_1", "task_type": "attribute_only", "sentence": "The taste is sweet.", "question": "Which taste is named in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
            {"case_id": "attr_search_2", "task_type": "attribute_only", "sentence": "The taste is sour.", "question": "Which taste is named in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 1},
        ],
        "heldout_binding_cases": [
            {"case_id": "apple_sweet_holdout_1", "task_type": "binding", "sentence": "After the first bite, the apple seemed sweet.", "question": "Which taste is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
            {"case_id": "apple_sour_holdout_1", "task_type": "binding", "sentence": "The old apple turned sour by evening.", "question": "Which taste is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 1},
            {"case_id": "apple_juicy_holdout_1", "task_type": "binding", "sentence": "Inside the bag, the apple stayed juicy and fresh.", "question": "Which taste or flavor property is attached to apple in this sentence?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 2},
        ],
    },
    "size": {
        "bind_group": "apple_size_bind",
        "noun_group": "apple_noun",
        "attr_group": "size_anchor",
        "search_cases": [
            {"case_id": "apple_fist_search_1", "task_type": "binding", "sentence": "The apple is about the size of a fist.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 0},
            {"case_id": "apple_hand_search_1", "task_type": "binding", "sentence": "The apple is about the size of a hand.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 1},
            {"case_id": "apple_palm_search_1", "task_type": "binding", "sentence": "The apple is almost the size of a palm.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 2},
            {"case_id": "noun_search_1", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][0]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "noun_search_2", "task_type": "noun_only", "sentence": GROUP_CASES["apple_noun"][1]["sentence"], "question": "Which object is named in this sentence?", "options": ["apple", "banana", "orange", "chair"], "correct_index": 0},
            {"case_id": "attr_search_1", "task_type": "attribute_only", "sentence": "The size is about a fist.", "question": "Which size anchor is named in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 0},
            {"case_id": "attr_search_2", "task_type": "attribute_only", "sentence": "The size is about a hand.", "question": "Which size anchor is named in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 1},
        ],
        "heldout_binding_cases": [
            {"case_id": "apple_fist_holdout_1", "task_type": "binding", "sentence": "On the desk, the apple was roughly the size of a fist.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 0},
            {"case_id": "apple_hand_holdout_1", "task_type": "binding", "sentence": "The farmer said the apple was nearly the size of a hand.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 1},
            {"case_id": "apple_palm_holdout_1", "task_type": "binding", "sentence": "This apple felt close to the size of a palm.", "question": "Which size anchor is attached to apple in this sentence?", "options": ["fist", "hand", "palm", "coin"], "correct_index": 2},
        ],
    },
}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def flush_cuda() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    gc.collect()


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for digit in OPTION_DIGITS:
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"数字 {digit} 不是单 token，当前设计无法继续")
        out[digit] = int(ids[0])
    return out


def prompt_for_case(case: Dict[str, object]) -> str:
    option_text = " ".join(f"{idx + 1} {option}" for idx, option in enumerate(case["options"]))
    return (
        f'Sentence: "{case["sentence"]}"\n'
        f'Question: {case["question"]}\n'
        f"Answer with one digit only: {option_text}\n"
        "Answer:"
    )


def evaluate_case_batch(model, tokenizer, cases: Sequence[Dict[str, object]], digit_token_ids: Dict[str, int]) -> Dict[str, float]:
    prompts = [prompt_for_case(case) for case in cases]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
    candidate_ids = torch.tensor([digit_token_ids[digit] for digit in OPTION_DIGITS], device=logits.device, dtype=torch.long)
    option_logits = logits.index_select(dim=1, index=candidate_ids)
    option_logprobs = option_logits.log_softmax(dim=-1)
    correct_indices = torch.tensor([int(case["correct_index"]) for case in cases], device=logits.device, dtype=torch.long)
    correct_logprobs = option_logprobs.gather(1, correct_indices.unsqueeze(1)).squeeze(1)
    accuracy = (option_logits.argmax(dim=-1) == correct_indices).to(torch.float32)
    masked = option_logits.clone()
    masked.scatter_(1, correct_indices.unsqueeze(1), float("-inf"))
    margins = option_logits.gather(1, correct_indices.unsqueeze(1)).squeeze(1) - masked.max(dim=-1).values
    return {
        "count": float(len(cases)),
        "correct_prob_sum": float(correct_logprobs.exp().sum().item()),
        "correct_logprob_sum": float(correct_logprobs.sum().item()),
        "accuracy_sum": float(accuracy.sum().item()),
        "margin_sum": float(margins.sum().item()),
    }


def merge_totals(total: Dict[str, float], chunk: Dict[str, float]) -> None:
    for key, value in chunk.items():
        total[key] = total.get(key, 0.0) + float(value)


def finalize_totals(total: Dict[str, float]) -> Dict[str, float]:
    count = max(1.0, float(total["count"]))
    return {
        "count": int(total["count"]),
        "mean_correct_prob": float(total["correct_prob_sum"] / count),
        "mean_correct_logprob": float(total["correct_logprob_sum"] / count),
        "accuracy": float(total["accuracy_sum"] / count),
        "mean_margin": float(total["margin_sum"] / count),
    }


def evaluate_case_groups(model, tokenizer, digit_token_ids: Dict[str, int], groups: Dict[str, List[Dict[str, object]]], *, batch_size: int, handles: Sequence[object] | None = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    try:
        by_group: Dict[str, Dict[str, float]] = {}
        for group_name, cases in groups.items():
            total = {"count": 0.0, "correct_prob_sum": 0.0, "correct_logprob_sum": 0.0, "accuracy_sum": 0.0, "margin_sum": 0.0}
            for start in range(0, len(cases), batch_size):
                chunk = evaluate_case_batch(model, tokenizer, cases[start : start + batch_size], digit_token_ids)
                merge_totals(total, chunk)
            by_group[group_name] = finalize_totals(total)
        return {"by_group": by_group}
    finally:
        if handles:
            remove_hooks(handles)


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def build_groups_for_family(family_name: str) -> Dict[str, List[Dict[str, object]]]:
    spec = BINDING_FAMILY_SPECS[family_name]
    search_cases = list(spec["search_cases"])
    return {
        "binding": [case for case in search_cases if case["task_type"] == "binding"],
        "noun_only": [case for case in search_cases if case["task_type"] == "noun_only"],
        "attribute_only": [case for case in search_cases if case["task_type"] == "attribute_only"],
    }


def summarize_condition(baseline_search: Dict[str, object], current_search: Dict[str, object], baseline_heldout: Dict[str, float], current_heldout: Dict[str, float]) -> Dict[str, object]:
    baseline_groups = baseline_search["by_group"]
    current_groups = current_search["by_group"]
    binding_before = float(baseline_groups["binding"]["mean_correct_prob"])
    binding_after = float(current_groups["binding"]["mean_correct_prob"])
    noun_before = float(baseline_groups["noun_only"]["mean_correct_prob"])
    noun_after = float(current_groups["noun_only"]["mean_correct_prob"])
    attr_before = float(baseline_groups["attribute_only"]["mean_correct_prob"])
    attr_after = float(current_groups["attribute_only"]["mean_correct_prob"])
    heldout_before = float(baseline_heldout["mean_correct_prob"])
    heldout_after = float(current_heldout["mean_correct_prob"])
    binding_drop = binding_before - binding_after
    noun_shift = noun_after - noun_before
    attr_shift = attr_after - attr_before
    heldout_drop = heldout_before - heldout_after
    utility = binding_drop - OFF_TARGET_PENALTY * (abs(noun_shift) + abs(attr_shift))
    return {
        "binding_prob_before": binding_before,
        "binding_prob_after": binding_after,
        "binding_drop": float(binding_drop),
        "noun_shift": float(noun_shift),
        "attribute_shift": float(attr_shift),
        "heldout_binding_prob_before": heldout_before,
        "heldout_binding_prob_after": heldout_after,
        "heldout_binding_drop": float(heldout_drop),
        "binding_accuracy_before": float(baseline_groups["binding"]["accuracy"]),
        "binding_accuracy_after": float(current_groups["binding"]["accuracy"]),
        "heldout_binding_accuracy_before": float(baseline_heldout["accuracy"]),
        "heldout_binding_accuracy_after": float(current_heldout["accuracy"]),
        "utility": float(utility),
    }


def flat_id_to_spec(flat_id: int, neuron_count: int) -> Dict[str, int]:
    return {"layer_index": int(flat_id // neuron_count), "neuron_index": int(flat_id % neuron_count)}


def build_candidate_pools(model_row: Dict[str, object], family_name: str) -> Dict[str, List[int]]:
    spec = BINDING_FAMILY_SPECS[family_name]
    top_sets = model_row["top_sets"]
    bind_ids = [int(x) for x in top_sets[spec["bind_group"]]["top_active_neuron_ids"]]
    noun_ids = [int(x) for x in top_sets[spec["noun_group"]]["top_active_neuron_ids"]]
    attr_ids = [int(x) for x in top_sets[spec["attr_group"]]["top_active_neuron_ids"]]
    noun_set = set(noun_ids)
    attr_set = set(attr_ids)
    bind_set = set(bind_ids)
    bridge_pool = [idx for idx in bind_ids if idx not in noun_set and idx not in attr_set]
    noun_pool = [idx for idx in noun_ids if idx not in attr_set]
    attr_pool = [idx for idx in attr_ids if idx not in noun_set]
    other_groups = [key for key in top_sets.keys() if key not in {spec["bind_group"], spec["noun_group"], spec["attr_group"]}]
    other_pool = []
    seen = set()
    for group_name in other_groups:
        for idx in top_sets[group_name]["top_active_neuron_ids"]:
            idx = int(idx)
            if idx in bind_set or idx in noun_set or idx in attr_set or idx in seen:
                continue
            seen.add(idx)
            other_pool.append(idx)
    return {"bridge": bridge_pool, "noun_unique": noun_pool, "attribute_unique": attr_pool, "random_irrelevant": other_pool}


def pick_random_subset(pool: Sequence[int], size: int, *, seed: int) -> List[int]:
    rng = random.Random(seed)
    if len(pool) <= size:
        return list(pool)
    return list(rng.sample(list(pool), size))


def evaluate_subset_condition(model, tokenizer, digit_token_ids: Dict[str, int], baseline_search: Dict[str, object], baseline_heldout: Dict[str, float], search_groups: Dict[str, List[Dict[str, object]]], heldout_cases: Sequence[Dict[str, object]], subset_specs: Sequence[Dict[str, int]], *, batch_size: int) -> Dict[str, object]:
    handles = register_mlp_neuron_ablation(model, subset_specs) if subset_specs else None
    current_search = evaluate_case_groups(model, tokenizer, digit_token_ids, search_groups, batch_size=batch_size, handles=handles)
    flush_cuda()
    handles = register_mlp_neuron_ablation(model, subset_specs) if subset_specs else None
    current_heldout = evaluate_case_groups(model, tokenizer, digit_token_ids, {"binding": list(heldout_cases)}, batch_size=batch_size, handles=handles)["by_group"]["binding"]
    flush_cuda()
    return {"search_metrics": current_search, "heldout_binding_metrics": current_heldout, "effect": summarize_condition(baseline_search, current_search, baseline_heldout, current_heldout)}


def analyze_family(model, tokenizer, digit_token_ids: Dict[str, int], model_row: Dict[str, object], family_name: str, *, batch_size: int) -> Dict[str, object]:
    neuron_count = int(model_row["neurons_per_layer"])
    search_groups = build_groups_for_family(family_name)
    heldout_cases = list(BINDING_FAMILY_SPECS[family_name]["heldout_binding_cases"])
    baseline_search = evaluate_case_groups(model, tokenizer, digit_token_ids, search_groups, batch_size=batch_size, handles=None)
    flush_cuda()
    baseline_heldout = evaluate_case_groups(model, tokenizer, digit_token_ids, {"binding": heldout_cases}, batch_size=batch_size, handles=None)["by_group"]["binding"]
    flush_cuda()
    pools = build_candidate_pools(model_row, family_name)
    candidate_rows: List[Dict[str, object]] = []
    for size in CANDIDATE_SIZES:
        if any(len(pool) < size for key, pool in pools.items() if key != "random_irrelevant"):
            continue
        if len(pools["random_irrelevant"]) < size:
            continue
        subsets = {
            "bridge": list(pools["bridge"][:size]),
            "noun_unique": list(pools["noun_unique"][:size]),
            "attribute_unique": list(pools["attribute_unique"][:size]),
            "random_irrelevant": pick_random_subset(pools["random_irrelevant"], size, seed=RANDOM_SEED + size * 17 + len(family_name)),
        }
        by_condition = {}
        for condition_name, subset_ids in subsets.items():
            subset_specs = [flat_id_to_spec(idx, neuron_count) for idx in subset_ids]
            by_condition[condition_name] = {"subset_size": len(subset_ids), "subset_ids": subset_ids, "evaluation": evaluate_subset_condition(model, tokenizer, digit_token_ids, baseline_search, baseline_heldout, search_groups, heldout_cases, subset_specs, batch_size=batch_size)}
        candidate_rows.append({"candidate_size": size, "conditions": by_condition})
    if not candidate_rows:
        raise RuntimeError(f"{family_name} 没有足够大的桥接池，无法继续")
    best_row = max(candidate_rows, key=lambda row: float(row["conditions"]["bridge"]["evaluation"]["effect"]["utility"]))
    best_size = int(best_row["candidate_size"])
    bridge_effect = best_row["conditions"]["bridge"]["evaluation"]["effect"]
    comparator_summary = {}
    for condition_name in ["noun_unique", "attribute_unique", "random_irrelevant"]:
        other = best_row["conditions"][condition_name]["evaluation"]["effect"]
        comparator_summary[condition_name] = {
            "binding_drop_gap_vs_bridge": float(bridge_effect["binding_drop"] - other["binding_drop"]),
            "heldout_drop_gap_vs_bridge": float(bridge_effect["heldout_binding_drop"] - other["heldout_binding_drop"]),
            "utility_gap_vs_bridge": float(bridge_effect["utility"] - other["utility"]),
        }
    causal_support = bool(
        bridge_effect["binding_drop"] >= 0.08
        and bridge_effect["heldout_binding_drop"] >= 0.05
        and bridge_effect["binding_drop"] > best_row["conditions"]["noun_unique"]["evaluation"]["effect"]["binding_drop"]
        and bridge_effect["binding_drop"] > best_row["conditions"]["attribute_unique"]["evaluation"]["effect"]["binding_drop"]
        and bridge_effect["binding_drop"] > best_row["conditions"]["random_irrelevant"]["evaluation"]["effect"]["binding_drop"]
    )
    return {
        "family_name": family_name,
        "baseline_search": baseline_search,
        "baseline_heldout_binding": baseline_heldout,
        "pool_sizes": {key: len(value) for key, value in pools.items()},
        "candidate_rows": candidate_rows,
        "best_bridge_size": best_size,
        "best_bridge_row": best_row,
        "comparators_at_best_size": comparator_summary,
        "causal_support": causal_support,
    }


def analyze_model(model_key: str, stage435_summary: Dict[str, object], *, batch_size: int, prefer_cuda: bool) -> Dict[str, object]:
    model_row = next(row for row in stage435_summary["model_results"] if row["model_key"] == model_key)
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        family_results = {}
        for family_name in ["color", "taste", "size"]:
            family_results[family_name] = analyze_family(model, tokenizer, digit_token_ids, model_row, family_name, batch_size=batch_size)
        support_count = sum(int(bool(row["causal_support"])) for row in family_results.values())
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "neurons_per_layer": int(model_row["neurons_per_layer"]),
            "family_results": family_results,
            "summary": {"causal_support_count": support_count, "family_count": len(family_results), "causal_support_rate": safe_ratio(support_count, len(family_results))},
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    supported = 0
    total = 0
    for model_row in model_results:
        for family_name, family_row in model_row["family_results"].items():
            total += 1
            supported += int(bool(family_row["causal_support"]))
            bridge_effect = family_row["best_bridge_row"]["conditions"]["bridge"]["evaluation"]["effect"]
            rows.append({"model_key": model_row["model_key"], "family_name": family_name, "best_bridge_size": int(family_row["best_bridge_size"]), "binding_drop": float(bridge_effect["binding_drop"]), "heldout_binding_drop": float(bridge_effect["heldout_binding_drop"]), "utility": float(bridge_effect["utility"]), "causal_support": bool(family_row["causal_support"])})
    return {
        "binding_causal_support_rate": safe_ratio(supported, total),
        "binding_causal_support_count": supported,
        "binding_causal_total": total,
        "rows": rows,
        "core_answer": "如果 bridge neurons（桥接神经元）被打掉后，apple-red、apple-sweet、apple-fist 这类绑定任务明显下降，而 apple 本身的 noun readout（名词读出）和 red 或 sweet 本身的 attribute readout（属性读出）下降更小，就说明桥接神经元不是附带噪声，而是承担了真正的绑定因果角色。",
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心回答", summary["cross_model_summary"]["core_answer"], ""]
    for model_row in summary["model_results"]:
        lines.extend([f"## {model_row['model_name']}", f"- 因果支持率: {model_row['summary']['causal_support_rate']:.4f}", ""])
        for family_name, family_row in model_row["family_results"].items():
            bridge_effect = family_row["best_bridge_row"]["conditions"]["bridge"]["evaluation"]["effect"]
            lines.extend([
                f"### {family_name}",
                f"- best_bridge_size: {family_row['best_bridge_size']}",
                f"- binding_drop: {bridge_effect['binding_drop']:.4f}",
                f"- heldout_binding_drop: {bridge_effect['heldout_binding_drop']:.4f}",
                f"- noun_shift: {bridge_effect['noun_shift']:+.4f}",
                f"- attribute_shift: {bridge_effect['attribute_shift']:+.4f}",
                f"- utility: {bridge_effect['utility']:.4f}",
                f"- causal_support: {family_row['causal_support']}",
                "",
            ])
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="桥接神经元绑定因果消融实验")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    stage435_summary = load_json(STAGE435_SUMMARY_PATH)
    model_results = [analyze_model(model_key, stage435_summary, batch_size=args.batch_size, prefer_cuda=prefer_cuda) for model_key in MODEL_ORDER]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage439_binding_bridge_causal_ablation",
        "title": "桥接神经元绑定因果消融实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "batch_size": int(args.batch_size),
        "candidate_sizes": CANDIDATE_SIZES,
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
