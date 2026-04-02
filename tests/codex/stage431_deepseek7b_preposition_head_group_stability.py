#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage425_sentence_context_wordclass_causal import SENTENCE_CASES, resolve_digit_token_ids
from stage430_deepseek7b_preposition_mixed_circuit_search import (
    HELDOUT_PREPOSITION_CASES,
    OFF_TARGET_PENALTY,
    SEARCH_CONTROL_CLASSES,
    TARGET_CLASS,
    build_search_case_groups,
    evaluate_case_groups,
    evaluate_target_only,
    register_attention_head_ablation,
    summarize_full_delta,
    summarize_search_effect,
    summarize_target_only_delta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE430_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage430_deepseek7b_preposition_mixed_circuit_search_20260402"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage431_deepseek7b_preposition_head_group_stability_20260402"
)
MODEL_KEY = "deepseek7b"
THRESHOLD_RATIOS = [0.50, 0.70, 0.80, 0.90]
PARTIAL_SUBSET_TABLE_NAME = "subset_table_partial.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


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


def subset_key(ids: Sequence[str]) -> str:
    if not ids:
        return "EMPTY"
    return "|".join(sorted(ids))


def powerset_nonempty(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    for size in range(1, len(items) + 1):
        for combo in itertools.combinations(items, size):
            yield combo


def evaluate_subset(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    baseline_full: Dict[str, object],
    baseline_heldout: Dict[str, float],
    search_groups: Dict[str, List[Dict[str, str]]],
    candidate_map: Dict[str, Dict[str, object]],
    subset_ids: Sequence[str],
    *,
    batch_size: int,
) -> Dict[str, object]:
    subset = [candidate_map[cid] for cid in sorted(subset_ids)]

    search_handles = register_attention_head_ablation(model, subset) if subset else None
    current_search = evaluate_case_groups(
        model,
        tokenizer,
        digit_token_ids,
        search_groups,
        batch_size=batch_size,
        handles=search_handles,
    )
    flush_cuda()
    search_effect = summarize_search_effect(
        baseline_search,
        current_search,
        control_classes=SEARCH_CONTROL_CLASSES,
        off_target_penalty=OFF_TARGET_PENALTY,
    )

    full_handles = register_attention_head_ablation(model, subset) if subset else None
    current_full = evaluate_case_groups(
        model,
        tokenizer,
        digit_token_ids,
        SENTENCE_CASES,
        batch_size=batch_size,
        handles=full_handles,
    )
    flush_cuda()
    full_effect = summarize_full_delta(baseline_full, current_full)

    heldout_handles = register_attention_head_ablation(model, subset) if subset else None
    current_heldout = evaluate_target_only(
        model,
        tokenizer,
        digit_token_ids,
        HELDOUT_PREPOSITION_CASES,
        batch_size=batch_size,
        handles=heldout_handles,
    )
    flush_cuda()
    heldout_effect = summarize_target_only_delta(baseline_heldout, current_heldout)

    return {
        "subset_ids": list(sorted(subset_ids)),
        "subset_size": len(subset_ids),
        "search_effect": search_effect,
        "full_effect": full_effect,
        "heldout_effect": heldout_effect,
    }


def initial_subset_table(
    baseline_search: Dict[str, object],
    baseline_heldout: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    return {
        "EMPTY": {
            "subset_ids": [],
            "subset_size": 0,
            "search_effect": {
                "target_prob_before": float(baseline_search["by_class"][TARGET_CLASS]["mean_correct_prob"]),
                "target_prob_after": float(baseline_search["by_class"][TARGET_CLASS]["mean_correct_prob"]),
                "target_drop": 0.0,
                "target_accuracy_before": float(baseline_search["by_class"][TARGET_CLASS]["accuracy"]),
                "target_accuracy_after": float(baseline_search["by_class"][TARGET_CLASS]["accuracy"]),
                "target_accuracy_drop": 0.0,
                "off_target_abs_shift": 0.0,
                "off_target_signed_mean": 0.0,
                "utility": 0.0,
                "control_prob_deltas": {name: 0.0 for name in SEARCH_CONTROL_CLASSES},
            },
            "full_effect": {
                "delta_by_class": {
                    class_name: {"correct_prob_delta": 0.0, "accuracy_delta": 0.0}
                    for class_name in SENTENCE_CASES
                },
                "target_prob_delta": 0.0,
                "target_accuracy_delta": 0.0,
            },
            "heldout_effect": {
                "target_prob_before": float(baseline_heldout["mean_correct_prob"]),
                "target_prob_after": float(baseline_heldout["mean_correct_prob"]),
                "target_prob_delta": 0.0,
                "target_drop": 0.0,
                "target_accuracy_before": float(baseline_heldout["accuracy"]),
                "target_accuracy_after": float(baseline_heldout["accuracy"]),
                "target_accuracy_delta": 0.0,
                "target_accuracy_drop": 0.0,
            },
        }
    }


def load_partial_subset_table(path: Path) -> Dict[str, Dict[str, object]] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    subset_table = payload.get("subset_table")
    if not isinstance(subset_table, dict):
        return None
    return subset_table


def save_partial_subset_table(
    path: Path,
    subset_table: Dict[str, Dict[str, object]],
    *,
    completed_count: int,
    total_count: int,
) -> None:
    save_json(
        path,
        {
            "subset_table": subset_table,
            "completed_count": int(completed_count),
            "total_count": int(total_count),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def build_subset_table(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    baseline_full: Dict[str, object],
    baseline_heldout: Dict[str, float],
    search_groups: Dict[str, List[Dict[str, str]]],
    candidate_map: Dict[str, Dict[str, object]],
    *,
    batch_size: int,
    initial_table: Dict[str, Dict[str, object]] | None,
    partial_path: Path | None,
    max_new_subsets: int | None,
    pause_seconds: float,
) -> Dict[str, Dict[str, object]]:
    table = initial_table or initial_subset_table(baseline_search, baseline_heldout)
    all_combos = list(powerset_nonempty(list(candidate_map.keys())))
    completed = {key for key in table.keys() if key != "EMPTY"}
    new_processed = 0
    for combo in all_combos:
        combo_key = subset_key(combo)
        if combo_key in completed:
            continue
        row = evaluate_subset(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            baseline_full,
            baseline_heldout,
            search_groups,
            candidate_map,
            combo,
            batch_size=batch_size,
        )
        table[combo_key] = row
        completed.add(combo_key)
        new_processed += 1
        if partial_path is not None:
            save_partial_subset_table(
                partial_path,
                table,
                completed_count=len(completed),
                total_count=len(all_combos),
            )
        flush_cuda()
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        if max_new_subsets is not None and new_processed >= max_new_subsets:
            break
    return table


def sort_subset_rows(rows: Iterable[Dict[str, object]], *, metric_path: Tuple[str, ...]) -> List[Dict[str, object]]:
    def metric(row: Dict[str, object]) -> float:
        cur = row
        for key in metric_path:
            cur = cur[key]
        return float(cur)

    return sorted(rows, key=lambda row: (metric(row), -int(row["subset_size"])), reverse=True)


def find_minimal_threshold_subsets(
    rows: Sequence[Dict[str, object]],
    *,
    best_full_target_drop: float,
) -> List[Dict[str, object]]:
    out = []
    scored = [row for row in rows if int(row["subset_size"]) > 0]
    for ratio in THRESHOLD_RATIOS:
        threshold = ratio * best_full_target_drop
        candidates = [row for row in scored if -float(row["full_effect"]["target_prob_delta"]) >= threshold]
        if not candidates:
            out.append({"ratio": ratio, "threshold_target_drop": threshold, "subset": None})
            continue
        candidates.sort(
            key=lambda row: (
                int(row["subset_size"]),
                -float(row["search_effect"]["utility"]),
                -(-float(row["full_effect"]["target_prob_delta"])),
            )
        )
        out.append(
            {
                "ratio": ratio,
                "threshold_target_drop": threshold,
                "subset": candidates[0],
            }
        )
    return out


def compute_leave_one_out(
    subset_table: Dict[str, Dict[str, object]],
    full_subset_ids: Sequence[str],
) -> List[Dict[str, object]]:
    full_key = subset_key(full_subset_ids)
    full_row = subset_table[full_key]
    full_search_drop = float(full_row["search_effect"]["target_drop"])
    full_search_utility = float(full_row["search_effect"]["utility"])
    full_full_drop = -float(full_row["full_effect"]["target_prob_delta"])
    full_heldout_drop = float(full_row["heldout_effect"]["target_drop"])
    rows = []
    for cid in full_subset_ids:
        reduced_ids = [x for x in full_subset_ids if x != cid]
        reduced_row = subset_table[subset_key(reduced_ids)]
        rows.append(
            {
                "candidate_id": cid,
                "search_target_drop_loss": full_search_drop - float(reduced_row["search_effect"]["target_drop"]),
                "search_utility_loss": full_search_utility - float(reduced_row["search_effect"]["utility"]),
                "full_target_drop_loss": full_full_drop - (-float(reduced_row["full_effect"]["target_prob_delta"])),
                "heldout_target_drop_loss": full_heldout_drop - float(reduced_row["heldout_effect"]["target_drop"]),
                "remaining_subset_ids": reduced_ids,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["full_target_drop_loss"]),
            float(row["heldout_target_drop_loss"]),
            float(row["search_utility_loss"]),
        ),
        reverse=True,
    )
    return rows


def compute_pair_synergy(
    subset_table: Dict[str, Dict[str, object]],
    candidate_ids: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    for cid_a, cid_b in itertools.combinations(candidate_ids, 2):
        pair_row = subset_table[subset_key([cid_a, cid_b])]
        row_a = subset_table[subset_key([cid_a])]
        row_b = subset_table[subset_key([cid_b])]
        rows.append(
            {
                "pair_ids": [cid_a, cid_b],
                "pair_search_utility": float(pair_row["search_effect"]["utility"]),
                "pair_full_target_drop": -float(pair_row["full_effect"]["target_prob_delta"]),
                "pair_heldout_target_drop": float(pair_row["heldout_effect"]["target_drop"]),
                "utility_synergy": float(pair_row["search_effect"]["utility"])
                - float(row_a["search_effect"]["utility"])
                - float(row_b["search_effect"]["utility"]),
                "full_target_drop_synergy": -float(pair_row["full_effect"]["target_prob_delta"])
                + float(row_a["full_effect"]["target_prob_delta"])
                + float(row_b["full_effect"]["target_prob_delta"]),
                "heldout_target_drop_synergy": float(pair_row["heldout_effect"]["target_drop"])
                - float(row_a["heldout_effect"]["target_drop"])
                - float(row_b["heldout_effect"]["target_drop"]),
            }
        )
    rows.sort(
        key=lambda row: (float(row["full_target_drop_synergy"]), float(row["heldout_target_drop_synergy"])),
        reverse=True,
    )
    return rows


def compute_shapley(
    subset_table: Dict[str, Dict[str, object]],
    candidate_ids: Sequence[str],
) -> List[Dict[str, object]]:
    n = len(candidate_ids)
    factorial_n = math.factorial(n)
    rows = []
    for cid in candidate_ids:
        utility_value = 0.0
        full_drop_value = 0.0
        heldout_drop_value = 0.0
        others = [x for x in candidate_ids if x != cid]
        for r in range(0, len(others) + 1):
            for combo in itertools.combinations(others, r):
                base_row = subset_table[subset_key(combo)]
                with_row = subset_table[subset_key(list(combo) + [cid])]
                weight = math.factorial(r) * math.factorial(n - r - 1) / factorial_n
                utility_value += weight * (
                    float(with_row["search_effect"]["utility"]) - float(base_row["search_effect"]["utility"])
                )
                full_drop_value += weight * (
                    -float(with_row["full_effect"]["target_prob_delta"])
                    + float(base_row["full_effect"]["target_prob_delta"])
                )
                heldout_drop_value += weight * (
                    float(with_row["heldout_effect"]["target_drop"]) - float(base_row["heldout_effect"]["target_drop"])
                )
        rows.append(
            {
                "candidate_id": cid,
                "shapley_search_utility": utility_value,
                "shapley_full_target_drop": full_drop_value,
                "shapley_heldout_target_drop": heldout_drop_value,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["shapley_full_target_drop"]),
            float(row["shapley_heldout_target_drop"]),
            float(row["shapley_search_utility"]),
        ),
        reverse=True,
    )
    return rows


def build_candidate_descriptions(final_subset: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out = {}
    for row in final_subset:
        out[str(row["candidate_id"])] = {
            "layer_index": int(row["layer_index"]),
            "head_index": int(row["head_index"]),
            "kind": str(row["kind"]),
        }
    return out


def build_stable_core_ranking(
    leave_one_out: Sequence[Dict[str, object]],
    shapley_rows: Sequence[Dict[str, object]],
    candidate_meta: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    loo_map = {row["candidate_id"]: row for row in leave_one_out}
    out = []
    for row in shapley_rows:
        cid = row["candidate_id"]
        meta = candidate_meta[cid]
        loo_row = loo_map.get(cid)
        out.append(
            {
                "candidate_id": cid,
                "layer_index": meta["layer_index"],
                "head_index": meta["head_index"],
                "shapley_full_target_drop": float(row["shapley_full_target_drop"]),
                "shapley_heldout_target_drop": float(row["shapley_heldout_target_drop"]),
                "leave_one_out_full_target_drop_loss": (
                    float(loo_row["full_target_drop_loss"]) if loo_row is not None else 0.0
                ),
                "leave_one_out_heldout_target_drop_loss": (
                    float(loo_row["heldout_target_drop_loss"]) if loo_row is not None else 0.0
                ),
            }
        )
    out.sort(
        key=lambda row: (
            float(row["shapley_full_target_drop"]),
            float(row["leave_one_out_full_target_drop_loss"]),
            float(row["shapley_heldout_target_drop"]),
        ),
        reverse=True,
    )
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## Setup",
        f"- Timestamp UTC: {summary['timestamp_utc']}",
        f"- Use CUDA: {summary['used_cuda']}",
        f"- Batch size: {summary['batch_size']}",
        f"- Candidate heads: {list(summary['candidate_heads'].keys())}",
        f"- Best full subset: {summary['best_full_subset']['subset_ids']}",
        f"- Best full target drop: {-float(summary['best_full_subset']['full_effect']['target_prob_delta']):.4f}",
        f"- Best heldout subset: {summary['best_heldout_subset']['subset_ids']}",
        f"- Best heldout target drop: {float(summary['best_heldout_subset']['heldout_effect']['target_drop']):.4f}",
        "",
        "## Stable Core",
    ]
    for row in summary["stable_core_ranking"]:
        lines.append(
            f"- {row['candidate_id']}: full_shapley={row['shapley_full_target_drop']:.4f}, "
            f"heldout_shapley={row['shapley_heldout_target_drop']:.4f}, "
            f"loo_full={row['leave_one_out_full_target_drop_loss']:.4f}"
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
    parser = argparse.ArgumentParser(description="DeepSeek7B preposition head-group exact scan")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Forward batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--max-new-subsets",
        type=int,
        default=None,
        help="Process at most this many new subsets in the current run",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.75,
        help="Pause between new subset evaluations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stage430_summary = load_json(STAGE430_SUMMARY_PATH)
    final_subset = stage430_summary["model"]["final_subset"]
    candidate_meta = build_candidate_descriptions(final_subset)
    candidate_ids = list(candidate_meta.keys())
    candidate_map = {row["candidate_id"]: row for row in final_subset}
    output_dir = Path(args.output_dir)
    partial_path = output_dir / PARTIAL_SUBSET_TABLE_NAME

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[MODEL_KEY]["model_path"], prefer_cuda=use_cuda)
    start_time = time.time()
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        search_groups = build_search_case_groups()
        baseline_search = evaluate_case_groups(
            model,
            tokenizer,
            digit_token_ids,
            search_groups,
            batch_size=int(args.batch_size),
        )
        baseline_full = evaluate_case_groups(
            model,
            tokenizer,
            digit_token_ids,
            SENTENCE_CASES,
            batch_size=int(args.batch_size),
        )
        baseline_heldout = evaluate_target_only(
            model,
            tokenizer,
            digit_token_ids,
            HELDOUT_PREPOSITION_CASES,
            batch_size=int(args.batch_size),
        )
        initial_table = load_partial_subset_table(partial_path)
        subset_table = build_subset_table(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            baseline_full,
            baseline_heldout,
            search_groups,
            candidate_map,
            batch_size=int(args.batch_size),
            initial_table=initial_table,
            partial_path=partial_path,
            max_new_subsets=args.max_new_subsets,
            pause_seconds=float(args.pause_seconds),
        )
    finally:
        free_model(model)

    elapsed = time.time() - start_time
    nonempty_rows = [row for key, row in subset_table.items() if key != "EMPTY"]
    total_subset_count = (2 ** len(candidate_ids)) - 1
    is_complete_scan = len(nonempty_rows) == total_subset_count
    best_search_subset = sort_subset_rows(nonempty_rows, metric_path=("search_effect", "utility"))[0]
    best_full_subset = sort_subset_rows(nonempty_rows, metric_path=("full_effect", "target_prob_delta"))[0]
    best_heldout_subset = sort_subset_rows(nonempty_rows, metric_path=("heldout_effect", "target_drop"))[0]
    best_full_target_drop = -float(best_full_subset["full_effect"]["target_prob_delta"])
    leave_one_out = compute_leave_one_out(subset_table, best_full_subset["subset_ids"]) if is_complete_scan else []
    pair_synergy = compute_pair_synergy(subset_table, candidate_ids) if is_complete_scan else []
    shapley_rows = compute_shapley(subset_table, candidate_ids) if is_complete_scan else []
    stable_core_ranking = (
        build_stable_core_ranking(leave_one_out, shapley_rows, candidate_meta) if is_complete_scan else []
    )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage431_deepseek7b_preposition_head_group_stability",
        "title": "DeepSeek7B preposition head-group exact scan",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": bool(use_cuda),
        "batch_size": int(args.batch_size),
        "source_stage430_summary": str(STAGE430_SUMMARY_PATH),
        "candidate_heads": candidate_meta,
        "baseline_search": baseline_search,
        "baseline_full": baseline_full,
        "baseline_heldout": baseline_heldout,
        "heldout_preposition_cases": HELDOUT_PREPOSITION_CASES,
        "subset_count_scanned": len(nonempty_rows),
        "subset_count_total": total_subset_count,
        "is_complete_scan": is_complete_scan,
        "subset_table": subset_table,
        "best_search_subset": best_search_subset,
        "best_full_subset": best_full_subset,
        "best_heldout_subset": best_heldout_subset,
        "minimal_threshold_subsets": (
            find_minimal_threshold_subsets(
                nonempty_rows,
                best_full_target_drop=best_full_target_drop,
            )
            if is_complete_scan
            else []
        ),
        "leave_one_out": leave_one_out,
        "pair_synergy": pair_synergy,
        "shapley": shapley_rows,
        "stable_core_ranking": stable_core_ranking,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
