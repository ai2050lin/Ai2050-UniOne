#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import capture_qwen_mlp_payloads, discover_layers, move_batch_to_model_device, remove_hooks
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import candidate_id, register_mlp_neuron_ablation
from stage434_apple_polysemy_factorized_switch import APPLE_BRAND_CASES, APPLE_FRUIT_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE448_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage478_apple_switch_minimal_subcircuit_{time.strftime('%Y%m%d')}"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
RAW_TOP_K_PER_SIGN = 8
SHORTLIST_TOP_K_PER_SIGN = 4
MAX_SUBSET_SIZE = 6
MIN_GAIN = 1e-4
PRUNE_RATIO = 0.95
OFF_TARGET_PENALTY = 0.50

BANANA_CONTROL_CASES = [
    {"target": "banana", "label": "fruit_neutral", "sentence": "I bought a banana this morning.", "sense_label": 0},
    {"target": "banana", "label": "fruit_color", "sentence": "The banana is yellow and ripe.", "sense_label": 0},
    {"target": "banana", "label": "fruit_taste", "sentence": "The banana tastes sweet today.", "sense_label": 0},
    {"target": "banana", "label": "fruit_size", "sentence": "The banana is about the size of a hand.", "sense_label": 0},
]


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


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for digit in ["1", "2"]:
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"数字 {digit} 不是单 token")
        out[digit] = int(ids[0])
    return out


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
    variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    best = None
    best_len = -1
    seen = set()
    for text in variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法定位目标词: target={target!r} prompt={prompt!r}")
    return best


def build_sense_prompt(case: Dict[str, object]) -> str:
    word = str(case["target"])
    return (
        f'Sentence: "{case["sentence"]}"\n'
        f'Question: In this sentence, does the word {word} refer to 1 fruit or 2 company?\n'
        "Answer with one digit only: 1 fruit 2 company\n"
        "Answer:"
    )


def capture_case_layer_neuron_vector(model, tokenizer, layer_idx: int, sentence: str, target: str) -> torch.Tensor:
    buffers, handles = capture_qwen_mlp_payloads(model, {layer_idx: "neuron_in"})
    try:
        encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        start, end = locate_target_span(tokenizer, sentence, target)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        layer_tensor = buffers[layer_idx]
        if layer_tensor is None:
            raise RuntimeError(f"第 {layer_idx} 层神经元输入捕获失败")
        return layer_tensor[0, start:end, :].mean(dim=0).detach().float().cpu()
    finally:
        remove_hooks(handles)


def build_case_rows(model, tokenizer, layer_idx: int, cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for case in cases:
        rows.append(
            {
                "sentence": case["sentence"],
                "target": case["target"],
                "label": case["label"],
                "sense_label": int(case["sense_label"]),
                "layer_neuron_vector": capture_case_layer_neuron_vector(
                    model,
                    tokenizer,
                    layer_idx,
                    str(case["sentence"]),
                    str(case["target"]),
                ),
            }
        )
    return rows


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def top_ids_by_signed_diff(diff_vec: torch.Tensor, top_k: int, *, positive: bool) -> List[int]:
    work = diff_vec.float()
    masked = torch.where(
        work > 0 if positive else work < 0,
        work if positive else -work,
        torch.full_like(work, float("-inf")),
    )
    take_k = min(top_k, masked.numel())
    vals, idxs = torch.topk(masked, k=take_k)
    rows: List[int] = []
    for value, idx in zip(vals.tolist(), idxs.tolist()):
        if not torch.isfinite(torch.tensor(value)):
            continue
        rows.append(int(idx))
    return rows


def evaluate_cases(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    cases: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    total_correct_prob = 0.0
    total_accuracy = 0.0
    total = 0
    per_case = []
    candidate_ids = None
    for start in range(0, len(cases), batch_size):
        batch_cases = list(cases[start : start + batch_size])
        prompts = [build_sense_prompt(case) for case in batch_cases]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
        if candidate_ids is None:
            candidate_ids = torch.tensor(
                [digit_token_ids["1"], digit_token_ids["2"]],
                device=logits.device,
                dtype=torch.long,
            )
        option_logits = logits.index_select(dim=1, index=candidate_ids)
        log_probs = option_logits.log_softmax(dim=-1)
        labels = torch.tensor([int(case["sense_label"]) for case in batch_cases], device=logits.device, dtype=torch.long)
        correct_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).exp()
        predictions = option_logits.argmax(dim=-1)
        for idx, case in enumerate(batch_cases):
            is_correct = bool(int(predictions[idx].item()) == int(case["sense_label"]))
            correct_prob = float(correct_probs[idx].item())
            total += 1
            total_correct_prob += correct_prob
            total_accuracy += float(is_correct)
            per_case.append(
                {
                    "sentence": case["sentence"],
                    "target": case["target"],
                    "label": case["label"],
                    "expected_label": int(case["sense_label"]),
                    "predicted_label": int(predictions[idx].item()),
                    "correct_prob": correct_prob,
                    "is_correct": is_correct,
                }
            )
    return {
        "count": total,
        "mean_correct_prob": safe_ratio(total_correct_prob, total),
        "accuracy": safe_ratio(total_accuracy, total),
        "per_case": per_case,
    }


def evaluate_groups(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    *,
    batch_size: int,
    handles: Sequence[object] | None = None,
) -> Dict[str, object]:
    try:
        by_group = {}
        for group_name, cases in groups.items():
            by_group[group_name] = evaluate_cases(model, tokenizer, digit_token_ids, list(cases), batch_size=batch_size)
        return {"by_group": by_group}
    finally:
        if handles:
            remove_hooks(handles)


def summarize_effect(baseline: Dict[str, object], current: Dict[str, object]) -> Dict[str, float]:
    before = baseline["by_group"]
    after = current["by_group"]
    search_drop = float(before["search"]["mean_correct_prob"] - after["search"]["mean_correct_prob"])
    heldout_drop = float(before["heldout"]["mean_correct_prob"] - after["heldout"]["mean_correct_prob"])
    control_shift = float(after["control"]["mean_correct_prob"] - before["control"]["mean_correct_prob"])
    control_abs_shift = abs(control_shift)
    utility = 0.5 * (search_drop + heldout_drop) - OFF_TARGET_PENALTY * control_abs_shift
    return {
        "search_drop": search_drop,
        "heldout_drop": heldout_drop,
        "control_shift": control_shift,
        "control_abs_shift": control_abs_shift,
        "search_accuracy_before": float(before["search"]["accuracy"]),
        "search_accuracy_after": float(after["search"]["accuracy"]),
        "heldout_accuracy_before": float(before["heldout"]["accuracy"]),
        "heldout_accuracy_after": float(after["heldout"]["accuracy"]),
        "control_accuracy_before": float(before["control"]["accuracy"]),
        "control_accuracy_after": float(after["control"]["accuracy"]),
        "utility": utility,
    }


def evaluate_subset(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    baseline: Dict[str, object],
    subset: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    handles = register_mlp_neuron_ablation(model, subset) if subset else None
    current = evaluate_groups(model, tokenizer, digit_token_ids, groups, batch_size=batch_size, handles=handles)
    return {"group_metrics": current, "effect": summarize_effect(baseline, current)}


def score_single_candidates(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    baseline: Dict[str, object],
    candidates: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> List[Dict[str, object]]:
    rows = []
    for candidate in candidates:
        eval_row = evaluate_subset(model, tokenizer, digit_token_ids, groups, baseline, [candidate], batch_size=batch_size)
        row = dict(candidate)
        row["candidate_id"] = candidate_id(candidate)
        row["effect"] = eval_row["effect"]
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["effect"]["utility"]),
            float(row["effect"]["heldout_drop"]),
            float(row["effect"]["search_drop"]),
        ),
        reverse=True,
    )
    return rows


def shortlist_candidates(scored_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    pos_rows = [row for row in scored_rows if row["sense_direction"] == "brand"][:SHORTLIST_TOP_K_PER_SIGN]
    neg_rows = [row for row in scored_rows if row["sense_direction"] == "fruit"][:SHORTLIST_TOP_K_PER_SIGN]
    merged = {row["candidate_id"]: row for row in pos_rows + neg_rows}
    out = list(merged.values())
    out.sort(key=lambda row: float(row["effect"]["utility"]), reverse=True)
    return out


def evaluate_subset_cached(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    baseline: Dict[str, object],
    candidate_map: Dict[str, Dict[str, object]],
    subset_ids: Sequence[str],
    cache: Dict[Tuple[str, ...], Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    key = tuple(sorted(subset_ids))
    if key in cache:
        return cache[key]
    subset = [candidate_map[cid] for cid in key]
    result = evaluate_subset(model, tokenizer, digit_token_ids, groups, baseline, subset, batch_size=batch_size)
    result["subset_ids"] = list(key)
    cache[key] = result
    return result


def greedy_search(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    baseline: Dict[str, object],
    shortlist: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    candidate_map = {row["candidate_id"]: row for row in shortlist}
    cache: Dict[Tuple[str, ...], Dict[str, object]] = {}
    chosen_ids: List[str] = []
    current_utility = 0.0
    greedy_trace = []

    for step in range(MAX_SUBSET_SIZE):
        best_candidate = None
        best_result = None
        best_gain = None
        for candidate in shortlist:
            cid = candidate["candidate_id"]
            if cid in chosen_ids:
                continue
            trial = evaluate_subset_cached(
                model,
                tokenizer,
                digit_token_ids,
                groups,
                baseline,
                candidate_map,
                chosen_ids + [cid],
                cache,
                batch_size=batch_size,
            )
            gain = float(trial["effect"]["utility"]) - float(current_utility)
            if best_result is None or gain > float(best_gain):
                best_candidate = candidate
                best_result = trial
                best_gain = gain
        if best_result is None or float(best_gain) < MIN_GAIN:
            break
        chosen_ids.append(str(best_candidate["candidate_id"]))
        current_utility = float(best_result["effect"]["utility"])
        greedy_trace.append(
            {
                "step": step + 1,
                "added_candidate": best_candidate["candidate_id"],
                "gain": float(best_gain),
                "utility": float(best_result["effect"]["utility"]),
                "search_drop": float(best_result["effect"]["search_drop"]),
                "heldout_drop": float(best_result["effect"]["heldout_drop"]),
                "control_abs_shift": float(best_result["effect"]["control_abs_shift"]),
                "subset_ids": list(chosen_ids),
            }
        )

    greedy_result = evaluate_subset_cached(
        model,
        tokenizer,
        digit_token_ids,
        groups,
        baseline,
        candidate_map,
        chosen_ids,
        cache,
        batch_size=batch_size,
    ) if chosen_ids else {"effect": {"utility": 0.0, "search_drop": 0.0, "heldout_drop": 0.0, "control_abs_shift": 0.0}, "subset_ids": []}

    pruned_ids = list(chosen_ids)
    prune_trace = []
    changed = True
    while changed and len(pruned_ids) > 1:
        changed = False
        ref = evaluate_subset_cached(
            model,
            tokenizer,
            digit_token_ids,
            groups,
            baseline,
            candidate_map,
            pruned_ids,
            cache,
            batch_size=batch_size,
        )
        ref_utility = float(ref["effect"]["utility"])
        for cid in list(pruned_ids):
            trial_ids = [x for x in pruned_ids if x != cid]
            trial = evaluate_subset_cached(
                model,
                tokenizer,
                digit_token_ids,
                groups,
                baseline,
                candidate_map,
                trial_ids,
                cache,
                batch_size=batch_size,
            )
            if float(trial["effect"]["utility"]) >= ref_utility * PRUNE_RATIO:
                prune_trace.append(
                    {
                        "removed_candidate": cid,
                        "before_subset_ids": list(pruned_ids),
                        "after_subset_ids": list(trial_ids),
                        "before_utility": ref_utility,
                        "after_utility": float(trial["effect"]["utility"]),
                    }
                )
                pruned_ids = trial_ids
                changed = True
                break

    pruned_result = evaluate_subset_cached(
        model,
        tokenizer,
        digit_token_ids,
        groups,
        baseline,
        candidate_map,
        pruned_ids,
        cache,
        batch_size=batch_size,
    ) if pruned_ids else {"effect": {"utility": 0.0, "search_drop": 0.0, "heldout_drop": 0.0, "control_abs_shift": 0.0}, "subset_ids": []}

    return {
        "greedy_subset_ids": chosen_ids,
        "greedy_result": greedy_result,
        "greedy_trace": greedy_trace,
        "pruned_subset_ids": pruned_ids,
        "pruned_result": pruned_result,
        "prune_trace": prune_trace,
    }


def split_cases() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    search_cases: List[Dict[str, object]] = []
    heldout_cases: List[Dict[str, object]] = []

    fruit_search = [APPLE_FRUIT_CASES[0], APPLE_FRUIT_CASES[1], APPLE_FRUIT_CASES[4]]
    fruit_heldout = [APPLE_FRUIT_CASES[2], APPLE_FRUIT_CASES[3]]
    brand_search = [APPLE_BRAND_CASES[0], APPLE_BRAND_CASES[1], APPLE_BRAND_CASES[4]]
    brand_heldout = [APPLE_BRAND_CASES[2], APPLE_BRAND_CASES[3]]

    for case in fruit_search:
        row = dict(case)
        row["sense_label"] = 0
        search_cases.append(row)
    for case in brand_search:
        row = dict(case)
        row["sense_label"] = 1
        search_cases.append(row)
    for case in fruit_heldout:
        row = dict(case)
        row["sense_label"] = 0
        heldout_cases.append(row)
    for case in brand_heldout:
        row = dict(case)
        row["sense_label"] = 1
        heldout_cases.append(row)
    return search_cases, heldout_cases


def build_candidate_rows(diff_vec: torch.Tensor, layer_idx: int) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for neuron_idx in top_ids_by_signed_diff(diff_vec, RAW_TOP_K_PER_SIGN, positive=True):
        candidates.append(
            {
                "kind": "mlp_neuron",
                "layer_index": int(layer_idx),
                "neuron_index": int(neuron_idx),
                "sense_direction": "brand",
                "activation_delta": float(diff_vec[int(neuron_idx)].item()),
            }
        )
    for neuron_idx in top_ids_by_signed_diff(diff_vec, RAW_TOP_K_PER_SIGN, positive=False):
        candidates.append(
            {
                "kind": "mlp_neuron",
                "layer_index": int(layer_idx),
                "neuron_index": int(neuron_idx),
                "sense_direction": "fruit",
                "activation_delta": float(diff_vec[int(neuron_idx)].item()),
            }
        )
    return candidates


def build_stage448_reference(stage448_summary: Dict[str, object], model_key: str) -> Dict[str, object]:
    row = next(item for item in stage448_summary["model_results"] if item["model_key"] == model_key)
    return {
        "best_sensitive_layer": int(row["best_sensitive_layer"]["layer_index"]),
        "best_sensitive_excess_switch_drop": float(row["best_sensitive_layer"]["excess_switch_drop"]),
        "best_sensitive_switch_prob_drop": float(row["best_sensitive_layer"]["switch_prob_drop"]),
        "best_sensitive_control_prob_drop": float(row["best_sensitive_layer"]["control_prob_drop"]),
    }


def analyze_model(model_key: str, stage448_summary: Dict[str, object], *, use_cuda: bool, batch_size: int) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        reference = build_stage448_reference(stage448_summary, model_key)
        layer_idx = int(reference["best_sensitive_layer"])
        search_cases, heldout_cases = split_cases()
        search_rows = build_case_rows(model, tokenizer, layer_idx, search_cases)
        heldout_rows = build_case_rows(model, tokenizer, layer_idx, heldout_cases)
        fruit_mean = mean_tensors([row["layer_neuron_vector"] for row in search_rows if int(row["sense_label"]) == 0])
        brand_mean = mean_tensors([row["layer_neuron_vector"] for row in search_rows if int(row["sense_label"]) == 1])
        diff_vec = brand_mean - fruit_mean

        groups = {
            "search": search_cases,
            "heldout": heldout_cases,
            "control": BANANA_CONTROL_CASES,
        }
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_groups(model, tokenizer, digit_token_ids, groups, batch_size=batch_size)
        raw_candidates = build_candidate_rows(diff_vec, layer_idx)
        single_scores = score_single_candidates(model, tokenizer, digit_token_ids, groups, baseline, raw_candidates, batch_size=batch_size)
        shortlist = shortlist_candidates(single_scores)
        search_state = greedy_search(model, tokenizer, digit_token_ids, groups, baseline, shortlist, batch_size=batch_size)

        final_subset_ids = list(search_state["pruned_subset_ids"])
        candidate_map = {row["candidate_id"]: row for row in shortlist}
        final_subset = [candidate_map[cid] for cid in final_subset_ids]
        final_effect = search_state["pruned_result"]["effect"]
        recovered_fraction = safe_ratio(
            float(final_effect["search_drop"]),
            float(reference["best_sensitive_excess_switch_drop"]),
        )

        return {
            "model_key": model_key,
            "model_name": str(MODEL_SPECS[model_key]["model_name"]),
            "used_cuda": bool(use_cuda),
            "batch_size": int(batch_size),
            "search_case_count": len(search_cases),
            "heldout_case_count": len(heldout_cases),
            "control_case_count": len(BANANA_CONTROL_CASES),
            "best_sensitive_layer": layer_idx,
            "stage448_reference": reference,
            "baseline_metrics": baseline,
            "raw_candidate_count": len(raw_candidates),
            "shortlist_count": len(shortlist),
            "raw_candidates": [
                {
                    "candidate_id": candidate_id(row),
                    "sense_direction": row["sense_direction"],
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "activation_delta": float(row["activation_delta"]),
                }
                for row in raw_candidates
            ],
            "single_scores": [
                {
                    "candidate_id": row["candidate_id"],
                    "sense_direction": row["sense_direction"],
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "activation_delta": float(row["activation_delta"]),
                    "effect": row["effect"],
                }
                for row in single_scores
            ],
            "shortlist": [
                {
                    "candidate_id": row["candidate_id"],
                    "sense_direction": row["sense_direction"],
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "activation_delta": float(row["activation_delta"]),
                    "effect": row["effect"],
                }
                for row in shortlist
            ],
            "search_state": search_state,
            "final_subset": [
                {
                    "candidate_id": candidate_id(row),
                    "sense_direction": row["sense_direction"],
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "activation_delta": float(row["activation_delta"]),
                }
                for row in final_subset
            ],
            "final_effect": final_effect,
            "recovered_fraction_vs_stage448": recovered_fraction,
            "search_mean_diff_abs": float(torch.abs(diff_vec).mean().item()),
            "search_top_abs_concentration_ratio": safe_ratio(
                sum(abs(float(row["activation_delta"])) for row in raw_candidates),
                float(torch.abs(diff_vec).sum().item()),
            ),
            "search_rows": [
                {
                    "label": row["label"],
                    "sentence": row["sentence"],
                    "sense_label": int(row["sense_label"]),
                }
                for row in search_rows
            ],
            "heldout_rows": [
                {
                    "label": row["label"],
                    "sentence": row["sentence"],
                    "sense_label": int(row["sense_label"]),
                }
                for row in heldout_rows
            ],
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    return {
        model_key: {
            "best_sensitive_layer": int(row["best_sensitive_layer"]),
            "raw_candidate_count": int(row["raw_candidate_count"]),
            "shortlist_count": int(row["shortlist_count"]),
            "final_subset_size": len(row["final_subset"]),
            "final_search_drop": float(row["final_effect"]["search_drop"]),
            "final_heldout_drop": float(row["final_effect"]["heldout_drop"]),
            "final_control_abs_shift": float(row["final_effect"]["control_abs_shift"]),
            "final_utility": float(row["final_effect"]["utility"]),
            "recovered_fraction_vs_stage448": float(row["recovered_fraction_vs_stage448"]),
        }
        for model_key, row in model_rows.items()
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 目标: 在苹果多义切换敏感层搜索最小神经元子回路",
        "- 候选池: 敏感层中品牌偏置与水果偏置神经元",
        "- 算法: 单神经元快筛 + 贪心组合搜索 + 反向剪枝",
        "",
    ]
    for model_key in MODEL_ORDER:
        row = summary["models"][model_key]
        eff = row["final_effect"]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 模型名: {row['model_name']}",
                f"- 敏感层: L{row['best_sensitive_layer']}",
                f"- 原始候选数: {row['raw_candidate_count']}",
                f"- shortlist 数: {row['shortlist_count']}",
                f"- 最终子集大小: {len(row['final_subset'])}",
                f"- 最终子集: {', '.join(item['candidate_id'] for item in row['final_subset']) or '空'}",
                f"- search_drop: {eff['search_drop']:+.4f}",
                f"- heldout_drop: {eff['heldout_drop']:+.4f}",
                f"- control_abs_shift: {eff['control_abs_shift']:+.4f}",
                f"- utility: {eff['utility']:+.4f}",
                f"- 相对 stage448 最强层消融恢复比例: {row['recovered_fraction_vs_stage448']:.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果多义切换最小神经元子回路搜索")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=2, help="推理批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    stage448_summary = {
        "model_results": []
    }
    for model_key in MODEL_ORDER:
        suffix = "qwen3_cpu" if model_key == "qwen3" else "deepseek7b_cpu"
        path = STAGE448_SUMMARY_PATH / suffix / "summary.json"
        stage448_summary["model_results"].append(load_json(path)["model_results"][0])

    start_time = time.time()
    model_rows = {}
    for model_key in MODEL_ORDER:
        model_rows[model_key] = analyze_model(model_key, stage448_summary, use_cuda=use_cuda, batch_size=int(args.batch_size))
    elapsed = time.time() - start_time

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage478_apple_switch_minimal_subcircuit",
        "title": "苹果多义切换最小神经元子回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "search_config": {
            "raw_top_k_per_sign": RAW_TOP_K_PER_SIGN,
            "shortlist_top_k_per_sign": SHORTLIST_TOP_K_PER_SIGN,
            "max_subset_size": MAX_SUBSET_SIZE,
            "min_gain": MIN_GAIN,
            "prune_ratio": PRUNE_RATIO,
            "off_target_penalty": OFF_TARGET_PENALTY,
        },
        "models": model_rows,
        "cross_model_summary": build_cross_model_summary(model_rows),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage478_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_final_subset_size": len(model_rows["qwen3"]["final_subset"]),
                "deepseek7b_final_subset_size": len(model_rows["deepseek7b"]["final_subset"]),
                "qwen3_final_utility": model_rows["qwen3"]["final_effect"]["utility"],
                "deepseek7b_final_utility": model_rows["deepseek7b"]["final_effect"]["utility"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
