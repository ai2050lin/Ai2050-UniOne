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

from qwen3_language_shared import discover_layers
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import (
    candidate_id,
    register_attention_head_ablation,
    register_mlp_neuron_ablation,
)
from stage478_apple_switch_minimal_subcircuit import (
    BANANA_CONTROL_CASES,
    OUTPUT_DIR as STAGE478_OUTPUT_DIR,
    build_sense_prompt,
    evaluate_groups,
    load_json,
    resolve_digit_token_ids,
    safe_ratio,
    split_cases,
    summarize_effect,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE446_ROOT = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage446_polysemy_neuron_overlap_and_switch_axis_ablation_20260403"
)
STAGE478_SUMMARY_PATH = STAGE478_OUTPUT_DIR / "summary.json"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage479_apple_switch_mixed_circuit_search_{time.strftime('%Y%m%d')}"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
HEAD_SHORTLIST_SIZE = 6
NEURON_SHORTLIST_SIZE = 8
MAX_SUBSET_SIZE = 6
MIN_GAIN = 1e-4
PRUNE_RATIO = 0.95
OFF_TARGET_PENALTY = 0.50


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


def register_mixed_ablation(model, candidates: Sequence[Dict[str, object]]) -> List[object]:
    head_specs = [row for row in candidates if row["kind"] == "attention_head"]
    neuron_specs = [row for row in candidates if row["kind"] == "mlp_neuron"]
    handles: List[object] = []
    if head_specs:
        handles.extend(register_attention_head_ablation(model, head_specs))
    if neuron_specs:
        handles.extend(register_mlp_neuron_ablation(model, neuron_specs))
    return handles


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
    handles = register_mixed_ablation(model, subset) if subset else None
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
    head_rows = [row for row in scored_rows if row["kind"] == "attention_head"][:HEAD_SHORTLIST_SIZE]
    neuron_rows = [row for row in scored_rows if row["kind"] == "mlp_neuron"][:NEURON_SHORTLIST_SIZE]
    merged = {row["candidate_id"]: row for row in head_rows + neuron_rows}
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


def greedy_mixed_search(
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
    trace = []

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
        trace.append(
            {
                "step": step + 1,
                "added_candidate": best_candidate["candidate_id"],
                "kind": best_candidate["kind"],
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
        "greedy_trace": trace,
        "pruned_subset_ids": pruned_ids,
        "pruned_result": pruned_result,
        "prune_trace": prune_trace,
    }


def load_stage446_neuron_rows(model_key: str) -> List[Dict[str, object]]:
    path = STAGE446_ROOT / model_key / "summary.json"
    summary = load_json(path)
    model_row = summary["model_results"][0]
    rows: List[Dict[str, object]] = []
    for item in model_row["switch_neuron_summary"]["brand_biased_neurons"][:4]:
        rows.append(
            {
                "kind": "mlp_neuron",
                "layer_index": int(item["layer_index"]),
                "neuron_index": int(item["neuron_index"]),
                "sense_direction": "brand",
                "activation_delta": float(item["activation_delta"]),
                "source": "stage446_global_brand",
            }
        )
    for item in model_row["switch_neuron_summary"]["fruit_biased_neurons"][:4]:
        rows.append(
            {
                "kind": "mlp_neuron",
                "layer_index": int(item["layer_index"]),
                "neuron_index": int(item["neuron_index"]),
                "sense_direction": "fruit",
                "activation_delta": float(item["activation_delta"]),
                "source": "stage446_global_fruit",
            }
        )
    return rows


def load_stage478_neuron_rows(model_key: str, stage478_summary: Dict[str, object]) -> Tuple[int, List[Dict[str, object]], Dict[str, object]]:
    row = stage478_summary["models"][model_key]
    return int(row["best_sensitive_layer"]), [
        {
            "kind": "mlp_neuron",
            "layer_index": int(item["layer_index"]),
            "neuron_index": int(item["neuron_index"]),
            "sense_direction": str(item["sense_direction"]),
            "activation_delta": float(item["activation_delta"]),
            "source": "stage478_sensitive_diff",
        }
        for item in row["raw_candidates"]
    ], row


def build_head_candidates(model, sensitive_layer: int) -> List[Dict[str, object]]:
    n_heads = int(getattr(model.config, "num_attention_heads"))
    head_rows = []
    for head_idx in range(n_heads):
        head_rows.append(
            {
                "kind": "attention_head",
                "layer_index": int(sensitive_layer),
                "head_index": int(head_idx),
                "source": "sensitive_layer_heads",
            }
        )
    return head_rows


def merge_neuron_candidates(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for row in rows:
        cid = candidate_id(row)
        if cid not in merged:
            merged[cid] = dict(row)
        else:
            old = merged[cid]
            if abs(float(row["activation_delta"])) > abs(float(old["activation_delta"])):
                merged[cid] = dict(row)
    out = list(merged.values())
    out.sort(key=lambda row: abs(float(row["activation_delta"])), reverse=True)
    return out


def build_groups() -> Dict[str, Sequence[Dict[str, object]]]:
    search_cases, heldout_cases = split_cases()
    return {
        "search": search_cases,
        "heldout": heldout_cases,
        "control": BANANA_CONTROL_CASES,
    }


def analyze_model(model_key: str, stage478_summary: Dict[str, object], *, use_cuda: bool, batch_size: int) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        sensitive_layer, sensitive_neurons, stage478_row = load_stage478_neuron_rows(model_key, stage478_summary)
        global_neurons = load_stage446_neuron_rows(model_key)
        neuron_candidates = merge_neuron_candidates(list(sensitive_neurons) + list(global_neurons))
        head_candidates = build_head_candidates(model, sensitive_layer)
        raw_candidates = list(head_candidates) + list(neuron_candidates)

        groups = build_groups()
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_groups(model, tokenizer, digit_token_ids, groups, batch_size=batch_size)
        single_scores = score_single_candidates(model, tokenizer, digit_token_ids, groups, baseline, raw_candidates, batch_size=batch_size)
        shortlist = shortlist_candidates(single_scores)
        search_state = greedy_mixed_search(model, tokenizer, digit_token_ids, groups, baseline, shortlist, batch_size=batch_size)

        final_subset_ids = list(search_state["pruned_subset_ids"])
        candidate_map = {row["candidate_id"]: row for row in shortlist}
        final_subset = [candidate_map[cid] for cid in final_subset_ids]
        final_effect = search_state["pruned_result"]["effect"]
        recovered_fraction_vs_stage478 = safe_ratio(
            float(final_effect["utility"]),
            float(stage478_row["final_effect"]["utility"]),
        )

        kind_counts: Dict[str, int] = {}
        for row in final_subset:
            kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1

        return {
            "model_key": model_key,
            "model_name": str(MODEL_SPECS[model_key]["model_name"]),
            "used_cuda": bool(use_cuda),
            "batch_size": int(batch_size),
            "sensitive_layer": int(sensitive_layer),
            "baseline_metrics": baseline,
            "stage478_reference": {
                "final_subset": stage478_row["final_subset"],
                "final_effect": stage478_row["final_effect"],
                "final_subset_size": len(stage478_row["final_subset"]),
            },
            "raw_head_candidate_count": len(head_candidates),
            "raw_neuron_candidate_count": len(neuron_candidates),
            "raw_candidate_count": len(raw_candidates),
            "shortlist_count": len(shortlist),
            "single_scores": [
                {
                    "candidate_id": row["candidate_id"],
                    "kind": row["kind"],
                    "layer_index": int(row["layer_index"]),
                    "source": row.get("source"),
                    "head_index": int(row["head_index"]) if row["kind"] == "attention_head" else None,
                    "neuron_index": int(row["neuron_index"]) if row["kind"] == "mlp_neuron" else None,
                    "sense_direction": row.get("sense_direction"),
                    "activation_delta": float(row.get("activation_delta", 0.0)),
                    "effect": row["effect"],
                }
                for row in single_scores
            ],
            "shortlist": [
                {
                    "candidate_id": row["candidate_id"],
                    "kind": row["kind"],
                    "layer_index": int(row["layer_index"]),
                    "source": row.get("source"),
                    "head_index": int(row["head_index"]) if row["kind"] == "attention_head" else None,
                    "neuron_index": int(row["neuron_index"]) if row["kind"] == "mlp_neuron" else None,
                    "sense_direction": row.get("sense_direction"),
                    "activation_delta": float(row.get("activation_delta", 0.0)),
                    "effect": row["effect"],
                }
                for row in shortlist
            ],
            "search_state": search_state,
            "final_subset": [
                {
                    "candidate_id": candidate_id(row),
                    "kind": row["kind"],
                    "layer_index": int(row["layer_index"]),
                    "source": row.get("source"),
                    "head_index": int(row["head_index"]) if row["kind"] == "attention_head" else None,
                    "neuron_index": int(row["neuron_index"]) if row["kind"] == "mlp_neuron" else None,
                    "sense_direction": row.get("sense_direction"),
                    "activation_delta": float(row.get("activation_delta", 0.0)),
                }
                for row in final_subset
            ],
            "final_subset_kind_counts": kind_counts,
            "final_effect": final_effect,
            "recovered_fraction_vs_stage478_utility": recovered_fraction_vs_stage478,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    return {
        model_key: {
            "sensitive_layer": int(row["sensitive_layer"]),
            "final_subset_size": len(row["final_subset"]),
            "final_head_count": int(row["final_subset_kind_counts"].get("attention_head", 0)),
            "final_neuron_count": int(row["final_subset_kind_counts"].get("mlp_neuron", 0)),
            "final_search_drop": float(row["final_effect"]["search_drop"]),
            "final_heldout_drop": float(row["final_effect"]["heldout_drop"]),
            "final_control_abs_shift": float(row["final_effect"]["control_abs_shift"]),
            "final_utility": float(row["final_effect"]["utility"]),
            "utility_gain_vs_stage478": float(row["recovered_fraction_vs_stage478_utility"]),
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
        "- 目标: 在苹果切换任务中搜索 attention head + 神经元的混合回路",
        "- 头候选: 敏感层全部注意力头",
        "- 神经元候选: stage478 敏感层差分神经元 + stage446 全局切换偏置神经元",
        "- 算法: 单元快筛 + 混合贪心搜索 + 反向剪枝",
        "",
    ]
    for model_key in MODEL_ORDER:
        row = summary["models"][model_key]
        eff = row["final_effect"]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 模型名: {row['model_name']}",
                f"- 敏感层: L{row['sensitive_layer']}",
                f"- 原始候选数: {row['raw_candidate_count']}（头 {row['raw_head_candidate_count']}，神经元 {row['raw_neuron_candidate_count']}）",
                f"- shortlist 数: {row['shortlist_count']}",
                f"- 最终子集大小: {len(row['final_subset'])}",
                f"- 最终子集组成: {row['final_subset_kind_counts']}",
                f"- 最终子集: {', '.join(item['candidate_id'] for item in row['final_subset']) or '空'}",
                f"- search_drop: {eff['search_drop']:+.4f}",
                f"- heldout_drop: {eff['heldout_drop']:+.4f}",
                f"- control_abs_shift: {eff['control_abs_shift']:+.4f}",
                f"- utility: {eff['utility']:+.4f}",
                f"- 相对 stage478 utility 倍数: {row['recovered_fraction_vs_stage478_utility']:.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果多义切换混合回路搜索")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=2, help="推理批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    stage478_summary = load_json(STAGE478_SUMMARY_PATH)

    start_time = time.time()
    model_rows = {}
    for model_key in MODEL_ORDER:
        model_rows[model_key] = analyze_model(model_key, stage478_summary, use_cuda=use_cuda, batch_size=int(args.batch_size))
    elapsed = time.time() - start_time

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage479_apple_switch_mixed_circuit_search",
        "title": "苹果多义切换混合回路搜索",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "search_config": {
            "head_shortlist_size": HEAD_SHORTLIST_SIZE,
            "neuron_shortlist_size": NEURON_SHORTLIST_SIZE,
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
                "status_short": "stage479_ready",
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
