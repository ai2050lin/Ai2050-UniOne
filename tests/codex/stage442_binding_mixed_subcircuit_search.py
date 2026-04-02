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

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import (
    candidate_id,
    register_attention_head_ablation,
    register_mlp_neuron_ablation,
)


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
    / "stage442_binding_mixed_subcircuit_search_20260402"
)

MODEL_ORDER = ["deepseek7b", "qwen3"]
OPTION_DIGITS = ["1", "2", "3", "4"]
HEAD_LAYER_TOP_K = 4
BRIDGE_NEURON_POOL = 48
HEAD_SHORTLIST_SIZE = 8
NEURON_SHORTLIST_SIZE = 8
MAX_SUBSET_SIZE = 6
MIN_GAIN = 1e-4
OFF_TARGET_PENALTY = 0.50

BINDING_TASKS: Dict[str, Dict[str, object]] = {
    "color": {
        "bind_group": "apple_color_bind",
        "noun_group": "apple_noun",
        "attr_group": "color_attr",
        "search_cases": {
            "binding": [
                {"sentence": "The apple is red while the banana is yellow.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
                {"sentence": "The apple is green but the banana is red.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "blue"], "correct_index": 2},
                {"sentence": "The apple stayed red, and the pear stayed green.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
                {"sentence": "The orange looked orange, but the apple looked green.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "orange"], "correct_index": 2},
            ],
            "noun_only": [
                {"sentence": "The apple rested beside the bowl while the banana stayed near the plate.", "question": "Which object is mentioned first?", "options": ["apple", "banana", "orange", "pear"], "correct_index": 0},
                {"sentence": "The apple rolled away before the banana moved.", "question": "Which object moved first in the sentence?", "options": ["apple", "banana", "orange", "pear"], "correct_index": 0},
            ],
            "attribute_only": [
                {"sentence": "Red appears before yellow in this line.", "question": "Which color appears first?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
                {"sentence": "Green is named after red in this line.", "question": "Which color appears second?", "options": ["red", "yellow", "green", "blue"], "correct_index": 2},
            ],
        },
        "heldout_binding_cases": [
            {"sentence": "The apple was red, whereas the lemon was yellow.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "blue"], "correct_index": 0},
            {"sentence": "The apple stayed green while the grape turned purple.", "question": "Which color belongs to apple?", "options": ["green", "purple", "red", "yellow"], "correct_index": 0},
            {"sentence": "The banana was yellow, but the apple looked red.", "question": "Which color belongs to apple?", "options": ["red", "yellow", "green", "orange"], "correct_index": 0},
        ],
    },
    "taste": {
        "bind_group": "apple_taste_bind",
        "noun_group": "apple_noun",
        "attr_group": "taste_attr",
        "search_cases": {
            "binding": [
                {"sentence": "The apple tastes sweet while the lemon tastes sour.", "question": "Which taste belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
                {"sentence": "The apple tastes sour, but the pear tastes sweet.", "question": "Which taste belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 1},
                {"sentence": "The apple felt juicy while the orange felt sour.", "question": "Which taste or flavor property belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 2},
                {"sentence": "The grape felt sweet, but the apple felt juicy.", "question": "Which taste or flavor property belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 2},
            ],
            "noun_only": [
                {"sentence": "The apple stayed on the table while the lemon stayed in the bag.", "question": "Which object is mentioned first?", "options": ["apple", "lemon", "orange", "grape"], "correct_index": 0},
                {"sentence": "The apple rolled away before the pear moved.", "question": "Which object moved first in the sentence?", "options": ["apple", "pear", "orange", "grape"], "correct_index": 0},
            ],
            "attribute_only": [
                {"sentence": "Sweet is written before sour in this line.", "question": "Which taste appears first?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
                {"sentence": "Juicy appears after sour in this line.", "question": "Which taste appears second?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 2},
            ],
        },
        "heldout_binding_cases": [
            {"sentence": "The apple tasted sweet, whereas the lemon tasted sour.", "question": "Which taste belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
            {"sentence": "The pear felt juicy, but the apple felt sour.", "question": "Which taste belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 1},
            {"sentence": "The orange stayed juicy while the apple stayed sweet.", "question": "Which taste belongs to apple?", "options": ["sweet", "sour", "juicy", "bitter"], "correct_index": 0},
        ],
    },
    "size": {
        "bind_group": "apple_size_bind",
        "noun_group": "apple_noun",
        "attr_group": "size_anchor",
        "search_cases": {
            "binding": [
                {"sentence": "The apple is the size of a fist, while the grape is the size of a thumb.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 0},
                {"sentence": "The apple is the size of a hand, but the lemon is the size of a thumb.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 1},
                {"sentence": "The apple is the size of a palm, while the banana is the size of a hand.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 2},
                {"sentence": "The apple is the size of a fist, and the pear is the size of a hand.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 0},
            ],
            "noun_only": [
                {"sentence": "The apple stayed on the tray while the grape stayed in the cup.", "question": "Which object is mentioned first?", "options": ["apple", "grape", "banana", "lemon"], "correct_index": 0},
                {"sentence": "The apple rolled first, and the pear rolled later.", "question": "Which object rolled first?", "options": ["apple", "pear", "banana", "lemon"], "correct_index": 0},
            ],
            "attribute_only": [
                {"sentence": "Fist is listed before hand in this line.", "question": "Which size anchor appears first?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 0},
                {"sentence": "Palm is listed after hand in this line.", "question": "Which size anchor appears second?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 2},
            ],
        },
        "heldout_binding_cases": [
            {"sentence": "The apple was the size of a fist, whereas the orange was the size of a palm.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 0},
            {"sentence": "The apple looked hand-sized, while the grape looked thumb-sized.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 1},
            {"sentence": "The lemon was thumb-sized, but the apple was palm-sized.", "question": "Which size anchor belongs to apple?", "options": ["fist", "hand", "palm", "thumb"], "correct_index": 2},
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


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def register_mixed_ablation(model, candidates: Sequence[Dict[str, object]]) -> List[object]:
    head_specs = [row for row in candidates if row["kind"] == "attention_head"]
    neuron_specs = [row for row in candidates if row["kind"] == "mlp_neuron"]
    handles: List[object] = []
    if head_specs:
        handles.extend(register_attention_head_ablation(model, head_specs))
    if neuron_specs:
        handles.extend(register_mlp_neuron_ablation(model, neuron_specs))
    return handles


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    token_ids = {}
    for digit in OPTION_DIGITS:
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"数字 {digit} 不是单 token")
        token_ids[digit] = int(ids[0])
    return token_ids


def prompt_for_case(case: Dict[str, object]) -> str:
    option_text = " ".join(f"{idx + 1} {option}" for idx, option in enumerate(case["options"]))
    return f'Sentence: "{case["sentence"]}"\nQuestion: {case["question"]}\nAnswer with one digit only: {option_text}\nAnswer:'


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


def evaluate_groups(model, tokenizer, digit_token_ids: Dict[str, int], groups: Dict[str, List[Dict[str, object]]], *, batch_size: int, handles: Sequence[object] | None = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    try:
        by_group = {}
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


def summarize_effect(baseline_search: Dict[str, object], current_search: Dict[str, object], baseline_heldout: Dict[str, float], current_heldout: Dict[str, float]) -> Dict[str, object]:
    before = baseline_search["by_group"]
    after = current_search["by_group"]
    binding_drop = float(before["binding"]["mean_correct_prob"] - after["binding"]["mean_correct_prob"])
    noun_shift = float(after["noun_only"]["mean_correct_prob"] - before["noun_only"]["mean_correct_prob"])
    attr_shift = float(after["attribute_only"]["mean_correct_prob"] - before["attribute_only"]["mean_correct_prob"])
    heldout_drop = float(baseline_heldout["mean_correct_prob"] - current_heldout["mean_correct_prob"])
    utility = binding_drop - OFF_TARGET_PENALTY * (abs(noun_shift) + abs(attr_shift))
    return {
        "binding_drop": binding_drop,
        "noun_shift": noun_shift,
        "attribute_shift": attr_shift,
        "heldout_binding_drop": heldout_drop,
        "binding_accuracy_before": float(before["binding"]["accuracy"]),
        "binding_accuracy_after": float(after["binding"]["accuracy"]),
        "heldout_accuracy_before": float(baseline_heldout["accuracy"]),
        "heldout_accuracy_after": float(current_heldout["accuracy"]),
        "utility": utility,
    }


def build_bridge_pool_and_layers(model_row: Dict[str, object], family_name: str) -> Tuple[List[int], List[int]]:
    task = BINDING_TASKS[family_name]
    bind_ids = [int(x) for x in model_row["top_sets"][task["bind_group"]]["top_active_neuron_ids"]]
    noun_ids = set(int(x) for x in model_row["top_sets"][task["noun_group"]]["top_active_neuron_ids"])
    attr_ids = set(int(x) for x in model_row["top_sets"][task["attr_group"]]["top_active_neuron_ids"])
    neuron_count = int(model_row["neurons_per_layer"])
    bridge_pool = [idx for idx in bind_ids if idx not in noun_ids and idx not in attr_ids]
    layer_counts: Dict[int, int] = defaultdict(int)
    for idx in bridge_pool:
        layer_counts[int(idx // neuron_count)] += 1
    top_layers = [layer for layer, _ in sorted(layer_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:HEAD_LAYER_TOP_K]]
    return bridge_pool[:BRIDGE_NEURON_POOL], top_layers


def build_candidate_rows(model, model_row: Dict[str, object], family_name: str) -> List[Dict[str, object]]:
    bridge_pool, top_layers = build_bridge_pool_and_layers(model_row, family_name)
    n_heads = int(getattr(model.config, "num_attention_heads"))
    rows: List[Dict[str, object]] = []
    for layer_idx in top_layers:
        for head_idx in range(n_heads):
            rows.append({"kind": "attention_head", "layer_index": int(layer_idx), "head_index": int(head_idx)})
    neuron_count = int(model_row["neurons_per_layer"])
    for flat_idx in bridge_pool:
        rows.append({"kind": "mlp_neuron", "layer_index": int(flat_idx // neuron_count), "neuron_index": int(flat_idx % neuron_count)})
    return rows


def evaluate_subset(model, tokenizer, digit_token_ids: Dict[str, int], search_groups: Dict[str, List[Dict[str, object]]], heldout_cases: Sequence[Dict[str, object]], baseline_search: Dict[str, object], baseline_heldout: Dict[str, float], subset: Sequence[Dict[str, object]], *, batch_size: int) -> Dict[str, object]:
    handles = register_mixed_ablation(model, subset) if subset else None
    current_search = evaluate_groups(model, tokenizer, digit_token_ids, search_groups, batch_size=batch_size, handles=handles)
    flush_cuda()
    handles = register_mixed_ablation(model, subset) if subset else None
    current_heldout = evaluate_groups(model, tokenizer, digit_token_ids, {"binding": list(heldout_cases)}, batch_size=batch_size, handles=handles)["by_group"]["binding"]
    flush_cuda()
    return {"search_metrics": current_search, "heldout_metrics": current_heldout, "effect": summarize_effect(baseline_search, current_search, baseline_heldout, current_heldout)}


def score_single_candidates(model, tokenizer, digit_token_ids: Dict[str, int], search_groups: Dict[str, List[Dict[str, object]]], heldout_cases: Sequence[Dict[str, object]], baseline_search: Dict[str, object], baseline_heldout: Dict[str, float], candidates: Sequence[Dict[str, object]], *, batch_size: int) -> List[Dict[str, object]]:
    rows = []
    for candidate in candidates:
        try:
            eval_row = evaluate_subset(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, [candidate], batch_size=batch_size)
        except RuntimeError:
            flush_cuda()
            continue
        row = dict(candidate)
        row["candidate_id"] = candidate_id(candidate)
        row["effect"] = eval_row["effect"]
        rows.append(row)
    rows.sort(key=lambda row: (float(row["effect"]["utility"]), float(row["effect"]["binding_drop"]), float(row["effect"]["heldout_binding_drop"])), reverse=True)
    return rows


def shortlist_candidates(scored_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    head_rows = [row for row in scored_rows if row["kind"] == "attention_head"][:HEAD_SHORTLIST_SIZE]
    neuron_rows = [row for row in scored_rows if row["kind"] == "mlp_neuron"][:NEURON_SHORTLIST_SIZE]
    merged = {row["candidate_id"]: row for row in head_rows + neuron_rows}
    out = list(merged.values())
    out.sort(key=lambda row: (float(row["effect"]["utility"]), float(row["effect"]["binding_drop"])), reverse=True)
    return out


def evaluate_subset_cached(model, tokenizer, digit_token_ids: Dict[str, int], search_groups: Dict[str, List[Dict[str, object]]], heldout_cases: Sequence[Dict[str, object]], baseline_search: Dict[str, object], baseline_heldout: Dict[str, float], candidate_map: Dict[str, Dict[str, object]], subset_ids: Sequence[str], cache: Dict[Tuple[str, ...], Dict[str, object]], *, batch_size: int) -> Dict[str, object]:
    key = tuple(sorted(subset_ids))
    if key in cache:
        return cache[key]
    subset = [candidate_map[cid] for cid in key]
    result = evaluate_subset(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, subset, batch_size=batch_size)
    result["subset_ids"] = list(key)
    cache[key] = result
    return result


def greedy_mixed_search(model, tokenizer, digit_token_ids: Dict[str, int], search_groups: Dict[str, List[Dict[str, object]]], heldout_cases: Sequence[Dict[str, object]], baseline_search: Dict[str, object], baseline_heldout: Dict[str, float], shortlist: Sequence[Dict[str, object]], *, batch_size: int) -> Dict[str, object]:
    candidate_map = {row["candidate_id"]: row for row in shortlist}
    cache: Dict[Tuple[str, ...], Dict[str, object]] = {}
    chosen_ids: List[str] = []
    current_effect = {"utility": 0.0, "binding_drop": 0.0, "heldout_binding_drop": 0.0}
    trace = []
    for step in range(MAX_SUBSET_SIZE):
        best_candidate = None
        best_result = None
        best_gain = None
        for candidate in shortlist:
            cid = candidate["candidate_id"]
            if cid in chosen_ids:
                continue
            try:
                trial = evaluate_subset_cached(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, candidate_map, chosen_ids + [cid], cache, batch_size=batch_size)
            except RuntimeError:
                flush_cuda()
                continue
            gain = float(trial["effect"]["utility"]) - float(current_effect["utility"])
            if best_result is None or gain > best_gain + 1e-12 or (abs(gain - best_gain) <= 1e-12 and float(trial["effect"]["binding_drop"]) > float(best_result["effect"]["binding_drop"])):
                best_candidate = candidate
                best_result = trial
                best_gain = gain
        if best_result is None or best_gain is None or best_gain <= MIN_GAIN:
            break
        chosen_ids.append(best_candidate["candidate_id"])
        current_effect = best_result["effect"]
        trace.append({"step": step + 1, "added_candidate": best_candidate["candidate_id"], "kind": best_candidate["kind"], "utility": float(current_effect["utility"]), "binding_drop": float(current_effect["binding_drop"]), "heldout_binding_drop": float(current_effect["heldout_binding_drop"]), "gain": float(best_gain), "subset_ids": list(chosen_ids)})

    pruned_ids = list(chosen_ids)
    pruned_result = evaluate_subset_cached(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, candidate_map, pruned_ids, cache, batch_size=batch_size)
    changed = True
    prune_trace = []
    while changed and len(pruned_ids) > 1:
        changed = False
        ref_utility = float(pruned_result["effect"]["utility"])
        ref_binding = float(pruned_result["effect"]["binding_drop"])
        for cid in list(pruned_ids):
            trial_ids = [x for x in pruned_ids if x != cid]
            try:
                trial = evaluate_subset_cached(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, candidate_map, trial_ids, cache, batch_size=batch_size)
            except RuntimeError:
                flush_cuda()
                continue
            utility_keep = float(trial["effect"]["utility"]) >= ref_utility * 0.95
            binding_keep = float(trial["effect"]["binding_drop"]) >= ref_binding * 0.95
            if utility_keep and binding_keep:
                prune_trace.append({"removed_candidate": cid, "before_subset_ids": list(pruned_ids), "after_subset_ids": list(trial_ids)})
                pruned_ids = trial_ids
                pruned_result = trial
                changed = True
                break
    final_greedy_result = None
    if chosen_ids:
        try:
            final_greedy_result = evaluate_subset_cached(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, candidate_map, chosen_ids, cache, batch_size=batch_size)
        except RuntimeError:
            flush_cuda()
    if final_greedy_result is None:
        final_greedy_result = {
            "search_metrics": baseline_search,
            "heldout_metrics": baseline_heldout,
            "effect": summarize_effect(baseline_search, baseline_search, baseline_heldout, baseline_heldout),
            "subset_ids": list(chosen_ids),
        }
    return {
        "shortlist": shortlist,
        "greedy_subset_ids": chosen_ids,
        "greedy_result": final_greedy_result,
        "greedy_trace": trace,
        "pruned_subset_ids": pruned_ids,
        "pruned_result": pruned_result,
        "prune_trace": prune_trace,
    }


def analyze_family(model, tokenizer, digit_token_ids: Dict[str, int], model_row: Dict[str, object], family_name: str, *, batch_size: int) -> Dict[str, object]:
    task = BINDING_TASKS[family_name]
    search_groups = {key: list(value) for key, value in task["search_cases"].items()}
    heldout_cases = list(task["heldout_binding_cases"])
    baseline_search = evaluate_groups(model, tokenizer, digit_token_ids, search_groups, batch_size=batch_size)
    flush_cuda()
    baseline_heldout = evaluate_groups(model, tokenizer, digit_token_ids, {"binding": heldout_cases}, batch_size=batch_size)["by_group"]["binding"]
    flush_cuda()
    raw_candidate_rows = build_candidate_rows(model, model_row, family_name)
    candidate_probe = {
        "attention_head_ok": False,
        "mlp_neuron_ok": False,
        "probe_errors": {},
    }
    candidate_rows = []
    first_head = next((row for row in raw_candidate_rows if row["kind"] == "attention_head"), None)
    if first_head is not None:
        try:
            evaluate_subset(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, [first_head], batch_size=batch_size)
            flush_cuda()
            candidate_probe["attention_head_ok"] = True
            candidate_rows.extend(row for row in raw_candidate_rows if row["kind"] == "attention_head")
        except RuntimeError as exc:
            flush_cuda()
            candidate_probe["probe_errors"]["attention_head"] = str(exc)
    first_neuron = next((row for row in raw_candidate_rows if row["kind"] == "mlp_neuron"), None)
    if first_neuron is not None:
        try:
            evaluate_subset(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, [first_neuron], batch_size=batch_size)
            flush_cuda()
            candidate_probe["mlp_neuron_ok"] = True
            candidate_rows.extend(row for row in raw_candidate_rows if row["kind"] == "mlp_neuron")
        except RuntimeError as exc:
            flush_cuda()
            candidate_probe["probe_errors"]["mlp_neuron"] = str(exc)
    scored_rows = score_single_candidates(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, candidate_rows, batch_size=batch_size)
    shortlist = shortlist_candidates(scored_rows)
    search_state = greedy_mixed_search(model, tokenizer, digit_token_ids, search_groups, heldout_cases, baseline_search, baseline_heldout, shortlist, batch_size=batch_size)
    best_single = scored_rows[0] if scored_rows else None
    best_single_head = next((row for row in scored_rows if row["kind"] == "attention_head"), None)
    best_single_neuron = next((row for row in scored_rows if row["kind"] == "mlp_neuron"), None)
    final_effect = search_state["pruned_result"]["effect"]
    mixed_support = bool(float(final_effect["binding_drop"]) >= 0.05 and float(final_effect["heldout_binding_drop"]) >= 0.03 and float(final_effect["utility"]) > 0.0)
    return {
        "family_name": family_name,
        "baseline_search": baseline_search,
        "baseline_heldout_binding": baseline_heldout,
        "candidate_probe": candidate_probe,
        "raw_candidate_count": len(raw_candidate_rows),
        "candidate_count": len(candidate_rows),
        "shortlist_size": len(shortlist),
        "best_single_candidate": best_single,
        "best_single_head": best_single_head,
        "best_single_neuron": best_single_neuron,
        "search_state": search_state,
        "mixed_support": mixed_support,
    }


def analyze_model(model_key: str, stage435_summary: Dict[str, object], *, batch_size: int, prefer_cuda: bool) -> Dict[str, object]:
    model_row = next(row for row in stage435_summary["model_results"] if row["model_key"] == model_key)
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        family_results = {}
        for family_name in ["color", "taste", "size"]:
            family_results[family_name] = analyze_family(model, tokenizer, digit_token_ids, model_row, family_name, batch_size=batch_size)
        support_count = sum(int(bool(row["mixed_support"])) for row in family_results.values())
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "family_results": family_results,
            "summary": {"mixed_support_count": support_count, "family_count": len(family_results), "mixed_support_rate": float(support_count / max(1, len(family_results)))},
        }
    finally:
        free_model(model)


def build_failed_model_result(model_key: str, error_text: str, *, prefer_cuda: bool) -> Dict[str, object]:
    return {
        "model_key": model_key,
        "model_name": MODEL_SPECS[model_key]["model_name"],
        "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
        "error": error_text,
        "family_results": {},
        "summary": {
            "mixed_support_count": 0,
            "family_count": 0,
            "mixed_support_rate": 0.0,
        },
    }


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    support_count = 0
    total = 0
    failed_models = []
    for model_row in model_results:
        if model_row.get("error"):
            failed_models.append(
                {
                    "model_key": model_row["model_key"],
                    "model_name": model_row["model_name"],
                    "error": model_row["error"],
                }
            )
        if not model_row["family_results"]:
            continue
        for family_name, family_row in model_row["family_results"].items():
            total += 1
            support_count += int(bool(family_row["mixed_support"]))
            final_effect = family_row["search_state"]["pruned_result"]["effect"]
            rows.append({"model_key": model_row["model_key"], "family_name": family_name, "subset_ids": family_row["search_state"]["pruned_subset_ids"], "binding_drop": float(final_effect["binding_drop"]), "heldout_binding_drop": float(final_effect["heldout_binding_drop"]), "utility": float(final_effect["utility"]), "mixed_support": bool(family_row["mixed_support"])})
    return {
        "mixed_support_count": support_count,
        "mixed_total": total,
        "mixed_support_rate": float(support_count / max(1, total)),
        "rows": rows,
        "failed_models": failed_models,
        "failed_model_count": len(failed_models),
        "core_answer": "如果加入 attention head（注意力头）之后，绑定消融效果明显强于只消融桥接神经元，就说明桥接项不是单独的稀疏神经元集合，而更像“头负责跨对象配对，神经元负责局部绑定写入”的混合子回路。",
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", "", "## 核心回答", summary["cross_model_summary"]["core_answer"], ""]
    failed_models = summary["cross_model_summary"].get("failed_models", [])
    if failed_models:
        lines.extend(["## 失败模型", ""])
        for row in failed_models:
            lines.extend([f"- {row['model_name']} ({row['model_key']}): {row['error']}", ""])
    for model_row in summary["model_results"]:
        lines.extend([f"## {model_row['model_name']}", f"- mixed_support_rate: {model_row['summary']['mixed_support_rate']:.4f}", ""])
        if model_row.get("error"):
            lines.extend([f"- error: {model_row['error']}", ""])
            continue
        for family_name, family_row in model_row["family_results"].items():
            eff = family_row["search_state"]["pruned_result"]["effect"]
            probe = family_row["candidate_probe"]
            lines.extend([
                f"### {family_name}",
                f"- candidate_probe: head_ok={probe['attention_head_ok']}, neuron_ok={probe['mlp_neuron_ok']}",
                f"- candidate_count: {family_row['candidate_count']} / raw_candidate_count: {family_row['raw_candidate_count']}",
                f"- subset: {family_row['search_state']['pruned_subset_ids']}",
                f"- binding_drop: {eff['binding_drop']:.4f}",
                f"- heldout_binding_drop: {eff['heldout_binding_drop']:.4f}",
                f"- utility: {eff['utility']:.4f}",
                f"- mixed_support: {family_row['mixed_support']}",
                "",
            ])
            if probe["probe_errors"]:
                lines.extend([f"- probe_errors: {probe['probe_errors']}", ""])
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绑定最小混合子回路搜索")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--models", default=",".join(MODEL_ORDER), help="逗号分隔的模型键，例如 deepseek7b,qwen3")
    return parser.parse_args()


def build_summary(model_results: Sequence[Dict[str, object]], *, start_time: float, prefer_cuda: bool, batch_size: int) -> Dict[str, object]:
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage442_binding_mixed_subcircuit_search",
        "title": "绑定最小混合子回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "batch_size": int(batch_size),
        "model_results": list(model_results),
        "cross_model_summary": build_cross_model_summary(model_results),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    stage435_summary = load_json(STAGE435_SUMMARY_PATH)
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    model_results = []
    output_dir = Path(args.output_dir)
    for model_key in selected_models:
        try:
            model_result = analyze_model(model_key, stage435_summary, batch_size=args.batch_size, prefer_cuda=prefer_cuda)
        except Exception as exc:
            flush_cuda()
            model_result = build_failed_model_result(model_key, str(exc), prefer_cuda=prefer_cuda)
        model_results.append(model_result)
        write_outputs(build_summary(model_results, start_time=start_time, prefer_cuda=prefer_cuda, batch_size=args.batch_size), output_dir)
    summary = build_summary(model_results, start_time=start_time, prefer_cuda=prefer_cuda, batch_size=args.batch_size)
    write_outputs(summary, output_dir)


if __name__ == "__main__":
    main()
