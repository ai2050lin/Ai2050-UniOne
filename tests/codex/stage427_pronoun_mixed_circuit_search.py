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

from qwen3_language_shared import PROJECT_ROOT, discover_layers
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage425_sentence_context_wordclass_causal import (
    SENTENCE_CASES,
    WORD_CLASSES,
    evaluate_case_batch,
    finalize_metric_totals,
    merge_metric_totals,
    resolve_digit_token_ids,
)


STAGE423_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
STAGE426_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage426_pronoun_minimal_causal_mechanism_20260330"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage427_pronoun_mixed_circuit_search_20260330"
)

TARGET_CLASS = "pronoun"
SEARCH_CONTROL_CLASSES = ["noun", "adjective", "verb", "adverb", "preposition"]
SEARCH_CONTROL_CASES_PER_CLASS = 2
HEAD_LAYER_LIMIT = 3
HEAD_SHORTLIST_SIZE = 8
NEURON_POOL = 20
NEURON_SHORTLIST_SIZE = 8
MAX_SUBSET_SIZE = 6
OFF_TARGET_PENALTY = 0.50
PRUNE_RATIO = 0.95
MIN_GAIN = 1e-4


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


def build_search_case_groups() -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {TARGET_CLASS: list(SENTENCE_CASES[TARGET_CLASS])}
    for class_name in SEARCH_CONTROL_CLASSES:
        groups[class_name] = list(SENTENCE_CASES[class_name][:SEARCH_CONTROL_CASES_PER_CLASS])
    return groups


def evaluate_case_groups(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, List[Dict[str, str]]],
    *,
    batch_size: int,
    handles: Sequence[object] | None = None,
) -> Dict[str, object]:
    try:
        by_class = {}
        aggregate = {
            "count": 0.0,
            "correct_prob_sum": 0.0,
            "correct_logprob_sum": 0.0,
            "accuracy_sum": 0.0,
            "margin_sum": 0.0,
        }
        for class_name, cases in groups.items():
            totals = {
                "count": 0.0,
                "correct_prob_sum": 0.0,
                "correct_logprob_sum": 0.0,
                "accuracy_sum": 0.0,
                "margin_sum": 0.0,
            }
            for start in range(0, len(cases), batch_size):
                batch_cases = cases[start : start + batch_size]
                labels = [class_name for _ in batch_cases]
                chunk = evaluate_case_batch(
                    model,
                    tokenizer,
                    batch_cases,
                    labels,
                    digit_token_ids,
                )
                merge_metric_totals(totals, chunk)
                merge_metric_totals(aggregate, chunk)
            by_class[class_name] = finalize_metric_totals(totals)
        return {
            "by_class": by_class,
            "aggregate": finalize_metric_totals(aggregate),
        }
    finally:
        if handles:
            remove_hooks(handles)


def summarize_search_effect(
    baseline: Dict[str, object],
    current: Dict[str, object],
    *,
    control_classes: Sequence[str],
    off_target_penalty: float,
) -> Dict[str, object]:
    baseline_by_class = baseline["by_class"]
    current_by_class = current["by_class"]
    target_prob_before = float(baseline_by_class[TARGET_CLASS]["mean_correct_prob"])
    target_prob_after = float(current_by_class[TARGET_CLASS]["mean_correct_prob"])
    target_acc_before = float(baseline_by_class[TARGET_CLASS]["accuracy"])
    target_acc_after = float(current_by_class[TARGET_CLASS]["accuracy"])
    target_drop = target_prob_before - target_prob_after
    target_acc_drop = target_acc_before - target_acc_after

    control_abs_shifts = []
    control_signed_shifts = []
    by_class_delta = {}
    for class_name in control_classes:
        before = float(baseline_by_class[class_name]["mean_correct_prob"])
        after = float(current_by_class[class_name]["mean_correct_prob"])
        delta = after - before
        by_class_delta[class_name] = float(delta)
        control_abs_shifts.append(abs(delta))
        control_signed_shifts.append(delta)

    off_target_abs_shift = sum(control_abs_shifts) / max(1, len(control_abs_shifts))
    off_target_signed_mean = sum(control_signed_shifts) / max(1, len(control_signed_shifts))
    utility = target_drop - off_target_penalty * off_target_abs_shift
    return {
        "target_prob_before": float(target_prob_before),
        "target_prob_after": float(target_prob_after),
        "target_drop": float(target_drop),
        "target_accuracy_before": float(target_acc_before),
        "target_accuracy_after": float(target_acc_after),
        "target_accuracy_drop": float(target_acc_drop),
        "off_target_abs_shift": float(off_target_abs_shift),
        "off_target_signed_mean": float(off_target_signed_mean),
        "utility": float(utility),
        "control_prob_deltas": by_class_delta,
    }


def summarize_full_delta(
    baseline: Dict[str, object],
    current: Dict[str, object],
) -> Dict[str, object]:
    baseline_by_class = baseline["by_class"]
    current_by_class = current["by_class"]
    out = {}
    for class_name in WORD_CLASSES:
        out[class_name] = {
            "correct_prob_delta": float(current_by_class[class_name]["mean_correct_prob"])
            - float(baseline_by_class[class_name]["mean_correct_prob"]),
            "accuracy_delta": float(current_by_class[class_name]["accuracy"])
            - float(baseline_by_class[class_name]["accuracy"]),
        }
    return {
        "delta_by_class": out,
        "target_prob_delta": float(out[TARGET_CLASS]["correct_prob_delta"]),
        "target_accuracy_delta": float(out[TARGET_CLASS]["accuracy_delta"]),
    }


def get_head_dim(model) -> int:
    hidden_size = int(getattr(model.config, "hidden_size"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    return hidden_size // n_heads


def register_attention_head_ablation(
    model,
    head_specs: Sequence[Dict[str, object]],
) -> List[object]:
    handles = []
    layers = discover_layers(model)
    head_dim = get_head_dim(model)
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for row in head_specs:
        by_layer[int(row["layer_index"])].append(int(row["head_index"]))

    for layer_idx, head_indices in by_layer.items():
        module = layers[layer_idx].self_attn.o_proj
        unique_heads = sorted(set(head_indices))

        def make_pre_hook(heads: List[int]):
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0].clone()
                for head_idx in heads:
                    start = head_idx * head_dim
                    end = start + head_dim
                    hidden[..., start:end] = 0
                if len(inputs) == 1:
                    return (hidden,)
                return (hidden, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(unique_heads)))
    return handles


def register_mlp_neuron_ablation(
    model,
    neuron_specs: Sequence[Dict[str, object]],
) -> List[object]:
    handles = []
    layers = discover_layers(model)
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for row in neuron_specs:
        by_layer[int(row["layer_index"])].append(int(row["neuron_index"]))

    for layer_idx, neuron_indices in by_layer.items():
        module = layers[layer_idx].mlp.down_proj
        index_tensor = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long)

        def make_pre_hook(indices: torch.Tensor):
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0].clone()
                hidden[..., indices.to(hidden.device)] = 0
                if len(inputs) == 1:
                    return (hidden,)
                return (hidden, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(index_tensor)))
    return handles


def register_mixed_ablation(
    model,
    candidates: Sequence[Dict[str, object]],
) -> List[object]:
    head_specs = [row for row in candidates if row["kind"] == "attention_head"]
    neuron_specs = [row for row in candidates if row["kind"] == "mlp_neuron"]
    handles: List[object] = []
    if head_specs:
        handles.extend(register_attention_head_ablation(model, head_specs))
    if neuron_specs:
        handles.extend(register_mlp_neuron_ablation(model, neuron_specs))
    return handles


def candidate_id(candidate: Dict[str, object]) -> str:
    if candidate["kind"] == "attention_head":
        return f"H:{candidate['layer_index']}:{candidate['head_index']}"
    return f"N:{candidate['layer_index']}:{candidate['neuron_index']}"


def evaluate_subset_cached(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    search_groups: Dict[str, List[Dict[str, str]]],
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
    handles = register_mixed_ablation(model, subset) if subset else None
    current = evaluate_case_groups(
        model,
        tokenizer,
        digit_token_ids,
        search_groups,
        batch_size=batch_size,
        handles=handles,
    )
    result = summarize_search_effect(
        baseline_search,
        current,
        control_classes=SEARCH_CONTROL_CLASSES,
        off_target_penalty=OFF_TARGET_PENALTY,
    )
    result["subset_ids"] = list(key)
    result["subset_size"] = len(key)
    cache[key] = result
    return result


def score_single_candidates(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    search_groups: Dict[str, List[Dict[str, str]]],
    candidates: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> List[Dict[str, object]]:
    rows = []
    for candidate in candidates:
        handles = register_mixed_ablation(model, [candidate])
        current = evaluate_case_groups(
            model,
            tokenizer,
            digit_token_ids,
            search_groups,
            batch_size=batch_size,
            handles=handles,
        )
        effect = summarize_search_effect(
            baseline_search,
            current,
            control_classes=SEARCH_CONTROL_CLASSES,
            off_target_penalty=OFF_TARGET_PENALTY,
        )
        row = dict(candidate)
        row["candidate_id"] = candidate_id(candidate)
        row["search_effect"] = effect
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["search_effect"]["utility"]),
            float(row["search_effect"]["target_drop"]),
        ),
        reverse=True,
    )
    return rows


def build_head_candidates(model, top_layers: Sequence[int]) -> List[Dict[str, object]]:
    n_heads = int(getattr(model.config, "num_attention_heads"))
    rows = []
    for layer_idx in top_layers:
        for head_idx in range(n_heads):
            rows.append(
                {
                    "kind": "attention_head",
                    "layer_index": int(layer_idx),
                    "head_index": int(head_idx),
                }
            )
    return rows


def build_neuron_candidates(
    class_summary: Dict[str, object],
    *,
    neuron_pool: int,
) -> List[Dict[str, object]]:
    rows = []
    for row in class_summary["top_neurons"][:neuron_pool]:
        rows.append(
            {
                "kind": "mlp_neuron",
                "layer_index": int(row["layer_index"]),
                "neuron_index": int(row["neuron_index"]),
                "score": float(row["score"]),
                "effect_size": float(row["effect_size"]),
            }
        )
    return rows


def shortlist_candidates(
    scored_heads: Sequence[Dict[str, object]],
    scored_neurons: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    out = []
    out.extend(list(scored_heads[:HEAD_SHORTLIST_SIZE]))
    out.extend(list(scored_neurons[:NEURON_SHORTLIST_SIZE]))
    uniq = {}
    for row in out:
        uniq[row["candidate_id"]] = row
    merged = list(uniq.values())
    merged.sort(
        key=lambda row: (
            float(row["search_effect"]["utility"]),
            float(row["search_effect"]["target_drop"]),
        ),
        reverse=True,
    )
    return merged


def greedy_forward_search(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    search_groups: Dict[str, List[Dict[str, str]]],
    shortlist: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, object]:
    candidate_map = {row["candidate_id"]: row for row in shortlist}
    chosen_ids: List[str] = []
    cache: Dict[Tuple[str, ...], Dict[str, object]] = {}
    current = {
        "subset_ids": [],
        "subset_size": 0,
        "target_drop": 0.0,
        "target_accuracy_drop": 0.0,
        "off_target_abs_shift": 0.0,
        "off_target_signed_mean": 0.0,
        "utility": 0.0,
        "control_prob_deltas": {},
    }
    trace = []

    for step in range(MAX_SUBSET_SIZE):
        best_row = None
        best_result = None
        best_gain = None
        for candidate in shortlist:
            cid = candidate["candidate_id"]
            if cid in chosen_ids:
                continue
            trial_ids = chosen_ids + [cid]
            result = evaluate_subset_cached(
                model,
                tokenizer,
                digit_token_ids,
                baseline_search,
                search_groups,
                candidate_map,
                trial_ids,
                cache,
                batch_size=batch_size,
            )
            gain = float(result["utility"]) - float(current["utility"])
            if best_result is None or gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12
                and float(result["target_drop"]) > float(best_result["target_drop"])
            ):
                best_row = candidate
                best_result = result
                best_gain = gain
        if best_result is None or best_gain is None or best_gain <= MIN_GAIN:
            break
        chosen_ids.append(best_row["candidate_id"])
        current = best_result
        trace.append(
            {
                "step": step + 1,
                "added_candidate": best_row["candidate_id"],
                "added_kind": best_row["kind"],
                "target_drop": float(current["target_drop"]),
                "off_target_abs_shift": float(current["off_target_abs_shift"]),
                "utility": float(current["utility"]),
                "gain": float(best_gain),
                "subset_ids": list(chosen_ids),
            }
        )

    return {
        "candidate_map": candidate_map,
        "cache": cache,
        "greedy_subset_ids": chosen_ids,
        "greedy_result": current,
        "greedy_trace": trace,
    }


def backward_prune(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    search_groups: Dict[str, List[Dict[str, str]]],
    search_state: Dict[str, object],
    *,
    batch_size: int,
) -> Dict[str, object]:
    subset_ids = list(search_state["greedy_subset_ids"])
    candidate_map = search_state["candidate_map"]
    cache = search_state["cache"]
    reference = dict(search_state["greedy_result"])
    prune_trace = []

    if not subset_ids:
        return {
            "final_subset_ids": [],
            "final_result": reference,
            "prune_trace": prune_trace,
        }

    changed = True
    while changed:
        changed = False
        for cid in list(subset_ids):
            trial_ids = [x for x in subset_ids if x != cid]
            result = evaluate_subset_cached(
                model,
                tokenizer,
                digit_token_ids,
                baseline_search,
                search_groups,
                candidate_map,
                trial_ids,
                cache,
                batch_size=batch_size,
            )
            keep_target = float(result["target_drop"]) >= PRUNE_RATIO * float(reference["target_drop"])
            keep_utility = float(result["utility"]) >= PRUNE_RATIO * float(reference["utility"])
            if keep_target and keep_utility:
                subset_ids = trial_ids
                prune_trace.append(
                    {
                        "removed_candidate": cid,
                        "remaining_subset_ids": list(subset_ids),
                        "target_drop": float(result["target_drop"]),
                        "utility": float(result["utility"]),
                    }
                )
                changed = True
                break

    final_result = evaluate_subset_cached(
        model,
        tokenizer,
        digit_token_ids,
        baseline_search,
        search_groups,
        candidate_map,
        subset_ids,
        cache,
        batch_size=batch_size,
    )
    return {
        "final_subset_ids": subset_ids,
        "final_result": final_result,
        "prune_trace": prune_trace,
    }


def describe_subset(
    subset_ids: Sequence[str],
    candidate_map: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows = []
    for cid in subset_ids:
        row = dict(candidate_map[cid])
        rows.append(row)
    return rows


def evaluate_final_subset_full(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    subset: Sequence[Dict[str, object]],
    full_baseline: Dict[str, object],
    *,
    batch_size: int,
) -> Dict[str, object]:
    handles = register_mixed_ablation(model, subset) if subset else None
    current = evaluate_case_groups(
        model,
        tokenizer,
        digit_token_ids,
        SENTENCE_CASES,
        batch_size=batch_size,
        handles=handles,
    )
    return summarize_full_delta(full_baseline, current)


def build_model_summary(
    model_key: str,
    stage423_summary: Dict[str, object],
    stage426_summary: Dict[str, object],
    *,
    batch_size: int,
    use_cuda: bool,
) -> Dict[str, object]:
    pronoun_stage423 = stage423_summary["models"][model_key]["classes"][TARGET_CLASS]
    top_layers = [int(row["layer_index"]) for row in pronoun_stage423["top_layers_by_mass"][:HEAD_LAYER_LIMIT]]

    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        search_groups = build_search_case_groups()
        baseline_search = evaluate_case_groups(
            model,
            tokenizer,
            digit_token_ids,
            search_groups,
            batch_size=batch_size,
        )
        full_baseline = evaluate_case_groups(
            model,
            tokenizer,
            digit_token_ids,
            SENTENCE_CASES,
            batch_size=batch_size,
        )

        head_candidates = build_head_candidates(model, top_layers)
        neuron_candidates = build_neuron_candidates(pronoun_stage423, neuron_pool=NEURON_POOL)
        scored_heads = score_single_candidates(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            search_groups,
            head_candidates,
            batch_size=batch_size,
        )
        scored_neurons = score_single_candidates(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            search_groups,
            neuron_candidates,
            batch_size=batch_size,
        )
        shortlist = shortlist_candidates(scored_heads, scored_neurons)
        search_state = greedy_forward_search(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            search_groups,
            shortlist,
            batch_size=batch_size,
        )
        pruned = backward_prune(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            search_groups,
            search_state,
            batch_size=batch_size,
        )
        subset = describe_subset(pruned["final_subset_ids"], search_state["candidate_map"])
        final_full = evaluate_final_subset_full(
            model,
            tokenizer,
            digit_token_ids,
            subset,
            full_baseline,
            batch_size=batch_size,
        )

        stage426_model = stage426_summary["models"][model_key]
        ref_mlp = float(stage426_model["mlp_top_layer_ablation"]["target_prob_delta"])
        ref_attn = float(stage426_model["attention_top_layer_ablation"]["target_prob_delta"])
        ref_best = max(abs(ref_mlp), abs(ref_attn))
        recovered_fraction = (
            abs(float(final_full["target_prob_delta"])) / ref_best if ref_best > 1e-8 else None
        )

        subset_kind_counts = defaultdict(int)
        for row in subset:
            subset_kind_counts[row["kind"]] += 1

        return {
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "model_path": str(MODEL_SPECS[model_key]["model_path"]),
            "top_layers_for_head_search": top_layers,
            "digit_token_ids": digit_token_ids,
            "search_baseline": baseline_search,
            "full_baseline": full_baseline,
            "single_unit_screening": {
                "head_candidate_count": len(head_candidates),
                "neuron_candidate_count": len(neuron_candidates),
                "top_heads": scored_heads[:12],
                "top_neurons": scored_neurons[:12],
            },
            "shortlist": shortlist,
            "greedy_trace": search_state["greedy_trace"],
            "prune_trace": pruned["prune_trace"],
            "final_subset": subset,
            "final_subset_kind_counts": dict(sorted(subset_kind_counts.items())),
            "final_search_effect": pruned["final_result"],
            "final_full_effect": final_full,
            "stage426_reference": {
                "mlp_top_layer_target_prob_delta": ref_mlp,
                "attention_top_layer_target_prob_delta": ref_attn,
                "best_module_abs_target_prob_delta": ref_best,
                "recovered_fraction_vs_best_module": recovered_fraction,
            },
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out = {}
    for model_key, payload in model_payloads.items():
        out[model_key] = {
            "final_subset_size": len(payload["final_subset"]),
            "final_head_count": int(payload["final_subset_kind_counts"].get("attention_head", 0)),
            "final_neuron_count": int(payload["final_subset_kind_counts"].get("mlp_neuron", 0)),
            "final_search_target_drop": float(payload["final_search_effect"]["target_drop"]),
            "final_search_utility": float(payload["final_search_effect"]["utility"]),
            "final_full_target_prob_delta": float(payload["final_full_effect"]["target_prob_delta"]),
            "recovered_fraction_vs_best_module": payload["stage426_reference"]["recovered_fraction_vs_best_module"],
        }
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 算法: 单元快筛 + 混合贪心搜索 + 反向剪枝",
        "- 候选池: pronoun 顶部层 attention heads + pronoun 高分 MLP neurons",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        payload = summary["models"][model_key]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 模型名: {payload['model_name']}",
                f"- head 搜索层: {payload['top_layers_for_head_search']}",
                f"- 最终子集大小: {len(payload['final_subset'])}",
                f"- 最终子集组成: {payload['final_subset_kind_counts']}",
                f"- 搜索阶段 target_drop: {payload['final_search_effect']['target_drop']:+.4f}",
                f"- 搜索阶段 utility: {payload['final_search_effect']['utility']:+.4f}",
                f"- 全量复核 target_prob_delta: {payload['final_full_effect']['target_prob_delta']:+.4f}",
                f"- 相对 stage426 最强模块恢复比例: "
                f"{payload['stage426_reference']['recovered_fraction_vs_best_module']}",
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
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 与 DeepSeek7B 代词混合回路搜索")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    stage423_summary = load_json(STAGE423_SUMMARY_PATH)
    stage426_summary = load_json(STAGE426_SUMMARY_PATH)

    start_time = time.time()
    model_payloads = {}
    for model_key in ["qwen3", "deepseek7b"]:
        model_payloads[model_key] = build_model_summary(
            model_key,
            stage423_summary,
            stage426_summary,
            batch_size=int(args.batch_size),
            use_cuda=use_cuda,
        )

    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage427_pronoun_mixed_circuit_search",
        "title": "Qwen3 与 DeepSeek7B 代词混合回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "search_config": {
            "target_class": TARGET_CLASS,
            "search_control_classes": SEARCH_CONTROL_CLASSES,
            "search_control_cases_per_class": SEARCH_CONTROL_CASES_PER_CLASS,
            "head_layer_limit": HEAD_LAYER_LIMIT,
            "head_shortlist_size": HEAD_SHORTLIST_SIZE,
            "neuron_pool": NEURON_POOL,
            "neuron_shortlist_size": NEURON_SHORTLIST_SIZE,
            "max_subset_size": MAX_SUBSET_SIZE,
            "off_target_penalty": OFF_TARGET_PENALTY,
            "prune_ratio": PRUNE_RATIO,
        },
        "models": model_payloads,
        "cross_model_summary": build_cross_model_summary(model_payloads),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage427_pronoun_mixed_circuit_search_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_final_subset_size": len(model_payloads["qwen3"]["final_subset"]),
                "deepseek7b_final_subset_size": len(model_payloads["deepseek7b"]["final_subset"]),
                "qwen3_final_full_target_prob_delta": model_payloads["qwen3"]["final_full_effect"][
                    "target_prob_delta"
                ],
                "deepseek7b_final_full_target_prob_delta": model_payloads["deepseek7b"]["final_full_effect"][
                    "target_prob_delta"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
