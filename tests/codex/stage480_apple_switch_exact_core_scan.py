#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage427_pronoun_mixed_circuit_search import candidate_id
from stage478_apple_switch_minimal_subcircuit import BANANA_CONTROL_CASES, resolve_digit_token_ids, split_cases
from stage479_apple_switch_mixed_circuit_search import (
    evaluate_groups,
    load_json,
    register_mixed_ablation,
    remove_hooks,
    safe_ratio,
    summarize_effect,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE479_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage479_apple_switch_mixed_circuit_search_20260403"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage480_apple_switch_exact_core_scan_{time.strftime('%Y%m%d')}"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    import gc

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def build_groups() -> Dict[str, Sequence[Dict[str, object]]]:
    search_cases, heldout_cases = split_cases()
    return {
        "search": search_cases,
        "heldout": heldout_cases,
        "control": BANANA_CONTROL_CASES,
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
    handles = register_mixed_ablation(model, subset) if subset else None
    try:
        current = evaluate_groups(model, tokenizer, digit_token_ids, groups, batch_size=batch_size, handles=None)
    finally:
        if handles:
            remove_hooks(handles)
    return {"group_metrics": current, "effect": summarize_effect(baseline, current)}


def exact_factorial(n: int) -> int:
    return math.factorial(n)


def shapley_values(candidates: Sequence[Dict[str, object]], subset_effect_map: Dict[Tuple[str, ...], Dict[str, float]], *, metric_key: str) -> Dict[str, float]:
    ids = [candidate_id(row) for row in candidates]
    n = len(ids)
    fact = [exact_factorial(i) for i in range(n + 1)]
    total_fact = fact[n]
    out = {cid: 0.0 for cid in ids}
    for cid in ids:
        others = [x for x in ids if x != cid]
        for r in range(0, len(others) + 1):
            weight = (fact[r] * fact[n - r - 1]) / total_fact
            for subset in itertools.combinations(others, r):
                subset_key = tuple(sorted(subset))
                with_key = tuple(sorted(subset + (cid,)))
                before = float(subset_effect_map[subset_key][metric_key]) if subset_key in subset_effect_map else 0.0
                after = float(subset_effect_map[with_key][metric_key])
                out[cid] += weight * (after - before)
    return out


def leave_one_out(candidates: Sequence[Dict[str, object]], full_effect: Dict[str, float], subset_effect_map: Dict[Tuple[str, ...], Dict[str, float]], *, metric_key: str) -> Dict[str, float]:
    ids = [candidate_id(row) for row in candidates]
    full_key = tuple(sorted(ids))
    ref = float(subset_effect_map[full_key][metric_key]) if full_key in subset_effect_map else float(full_effect[metric_key])
    out = {}
    for cid in ids:
        remain = tuple(sorted(x for x in ids if x != cid))
        remain_effect = float(subset_effect_map[remain][metric_key]) if remain in subset_effect_map else 0.0
        out[cid] = ref - remain_effect
    return out


def find_min_subset_for_threshold(subset_rows: Sequence[Dict[str, object]], *, metric_key: str, threshold: float) -> Dict[str, object] | None:
    eligible = [row for row in subset_rows if float(row["effect"][metric_key]) >= threshold]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            int(row["subset_size"]),
            -float(row["effect"][metric_key]),
            -float(row["effect"]["utility"]),
        )
    )
    return eligible[0]


def build_core_candidates(model_key: str, stage479_summary: Dict[str, object]) -> List[Dict[str, object]]:
    row = stage479_summary["models"][model_key]
    final_ids = [item["candidate_id"] for item in row["final_subset"]]
    if model_key == "qwen3":
        extra_ids = ["H:5:8", "H:5:0"]
        wanted = final_ids + [cid for cid in extra_ids if cid not in final_ids]
    else:
        wanted = list(final_ids)

    shortlist_map = {item["candidate_id"]: item for item in row["shortlist"]}
    out = []
    for cid in wanted:
        item = shortlist_map[cid]
        row_obj = {
            "kind": item["kind"],
            "layer_index": int(item["layer_index"]),
            "source": item.get("source"),
        }
        if item["kind"] == "attention_head":
            row_obj["head_index"] = int(item["head_index"])
        else:
            row_obj["neuron_index"] = int(item["neuron_index"])
            row_obj["sense_direction"] = item.get("sense_direction")
            row_obj["activation_delta"] = float(item.get("activation_delta", 0.0))
        out.append(row_obj)
    return out


def evaluate_all_subsets(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    groups: Dict[str, Sequence[Dict[str, object]]],
    baseline: Dict[str, object],
    candidates: Sequence[Dict[str, object]],
    *,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, ...], Dict[str, float]]]:
    rows = []
    effect_map: Dict[Tuple[str, ...], Dict[str, float]] = {tuple(): {"search_drop": 0.0, "heldout_drop": 0.0, "control_abs_shift": 0.0, "utility": 0.0}}
    for r in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            subset_ids = sorted(candidate_id(item) for item in combo)
            result = evaluate_subset(model, tokenizer, digit_token_ids, groups, baseline, list(combo), batch_size=batch_size)
            row = {
                "subset_ids": subset_ids,
                "subset_size": len(combo),
                "subset_kind_counts": {
                    "attention_head": sum(1 for item in combo if item["kind"] == "attention_head"),
                    "mlp_neuron": sum(1 for item in combo if item["kind"] == "mlp_neuron"),
                },
                "effect": result["effect"],
            }
            rows.append(row)
            effect_map[tuple(subset_ids)] = result["effect"]
    rows.sort(key=lambda row: (float(row["effect"]["utility"]), float(row["effect"]["heldout_drop"]), -int(row["subset_size"])), reverse=True)
    return rows, effect_map


def analyze_model(model_key: str, stage479_summary: Dict[str, object], *, use_cuda: bool, batch_size: int) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        candidate_rows = build_core_candidates(model_key, stage479_summary)
        groups = build_groups()
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_groups(model, tokenizer, digit_token_ids, groups, batch_size=batch_size)
        subset_rows, effect_map = evaluate_all_subsets(model, tokenizer, digit_token_ids, groups, baseline, candidate_rows, batch_size=batch_size)

        full_key = tuple(sorted(candidate_id(row) for row in candidate_rows))
        full_effect = effect_map[full_key]
        best_by_utility = max(subset_rows, key=lambda row: float(row["effect"]["utility"]))
        best_by_heldout = max(subset_rows, key=lambda row: float(row["effect"]["heldout_drop"]))
        shapley_utility = shapley_values(candidate_rows, effect_map, metric_key="utility")
        shapley_heldout = shapley_values(candidate_rows, effect_map, metric_key="heldout_drop")
        loo_utility = leave_one_out(candidate_rows, full_effect, effect_map, metric_key="utility")
        loo_heldout = leave_one_out(candidate_rows, full_effect, effect_map, metric_key="heldout_drop")

        utility_thresholds = {
            "50pct": 0.5 * float(best_by_utility["effect"]["utility"]),
            "70pct": 0.7 * float(best_by_utility["effect"]["utility"]),
            "90pct": 0.9 * float(best_by_utility["effect"]["utility"]),
        }
        heldout_thresholds = {
            "50pct": 0.5 * float(best_by_heldout["effect"]["heldout_drop"]),
            "70pct": 0.7 * float(best_by_heldout["effect"]["heldout_drop"]),
            "90pct": 0.9 * float(best_by_heldout["effect"]["heldout_drop"]),
        }
        min_subsets = {
            "utility": {
                key: find_min_subset_for_threshold(subset_rows, metric_key="utility", threshold=value)
                for key, value in utility_thresholds.items()
            },
            "heldout_drop": {
                key: find_min_subset_for_threshold(subset_rows, metric_key="heldout_drop", threshold=value)
                for key, value in heldout_thresholds.items()
            },
        }

        candidate_stats = []
        for row in candidate_rows:
            cid = candidate_id(row)
            candidate_stats.append(
                {
                    "candidate_id": cid,
                    "kind": row["kind"],
                    "layer_index": int(row["layer_index"]),
                    "head_index": int(row["head_index"]) if row["kind"] == "attention_head" else None,
                    "neuron_index": int(row["neuron_index"]) if row["kind"] == "mlp_neuron" else None,
                    "source": row.get("source"),
                    "sense_direction": row.get("sense_direction"),
                    "activation_delta": float(row.get("activation_delta", 0.0)),
                    "shapley_utility": float(shapley_utility[cid]),
                    "shapley_heldout_drop": float(shapley_heldout[cid]),
                    "leave_one_out_utility": float(loo_utility[cid]),
                    "leave_one_out_heldout_drop": float(loo_heldout[cid]),
                }
            )
        candidate_stats.sort(key=lambda row: row["shapley_utility"], reverse=True)

        return {
            "model_key": model_key,
            "model_name": str(MODEL_SPECS[model_key]["model_name"]),
            "used_cuda": bool(use_cuda),
            "batch_size": int(batch_size),
            "candidate_count": len(candidate_rows),
            "candidates": candidate_stats,
            "baseline_metrics": baseline,
            "full_set_effect": full_effect,
            "best_subset_by_utility": best_by_utility,
            "best_subset_by_heldout_drop": best_by_heldout,
            "min_subsets": min_subsets,
            "subset_rows": subset_rows,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out = {}
    for key, row in model_rows.items():
        out[key] = {
            "candidate_count": int(row["candidate_count"]),
            "best_utility_subset_size": int(row["best_subset_by_utility"]["subset_size"]),
            "best_utility": float(row["best_subset_by_utility"]["effect"]["utility"]),
            "best_heldout_subset_size": int(row["best_subset_by_heldout_drop"]["subset_size"]),
            "best_heldout_drop": float(row["best_subset_by_heldout_drop"]["effect"]["heldout_drop"]),
        }
    return out


def compact_subset(row: Dict[str, object] | None) -> Dict[str, object] | None:
    if row is None:
        return None
    return {
        "subset_ids": row["subset_ids"],
        "subset_size": int(row["subset_size"]),
        "subset_kind_counts": row["subset_kind_counts"],
        "effect": row["effect"],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 目标: 对苹果切换核心做精确子集穷举，区分骨架与增强器",
        "",
    ]
    for model_key in MODEL_ORDER:
        row = summary["models"][model_key]
        best_u = row["best_subset_by_utility"]
        best_h = row["best_subset_by_heldout_drop"]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 候选数: {row['candidate_count']}",
                f"- utility 最优子集: {', '.join(best_u['subset_ids'])}",
                f"- utility 最优值: {best_u['effect']['utility']:+.4f}",
                f"- heldout_drop 最优子集: {', '.join(best_h['subset_ids'])}",
                f"- heldout_drop 最优值: {best_h['effect']['heldout_drop']:+.4f}",
                f"- 50% utility 最小子集: {compact_subset(row['min_subsets']['utility']['50pct'])}",
                f"- 70% utility 最小子集: {compact_subset(row['min_subsets']['utility']['70pct'])}",
                f"- 90% utility 最小子集: {compact_subset(row['min_subsets']['utility']['90pct'])}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果切换核心精确子集扫描")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=2, help="推理批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    stage479_summary = load_json(STAGE479_SUMMARY_PATH)
    start_time = time.time()
    model_rows = {}
    for model_key in MODEL_ORDER:
        model_rows[model_key] = analyze_model(model_key, stage479_summary, use_cuda=use_cuda, batch_size=int(args.batch_size))
    elapsed = time.time() - start_time

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage480_apple_switch_exact_core_scan",
        "title": "苹果切换核心精确子集扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "models": model_rows,
        "cross_model_summary": build_cross_model_summary(model_rows),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage480_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_best_utility_subset": model_rows["qwen3"]["best_subset_by_utility"]["subset_ids"],
                "deepseek7b_best_utility_subset": model_rows["deepseek7b"]["best_subset_by_utility"]["subset_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
