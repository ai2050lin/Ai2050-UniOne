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
from stage427_pronoun_mixed_circuit_search import (
    OFF_TARGET_PENALTY,
    SEARCH_CONTROL_CLASSES,
    TARGET_CLASS,
    build_search_case_groups,
    evaluate_case_groups,
    register_attention_head_ablation,
    summarize_full_delta,
    summarize_search_effect,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE427_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage427_pronoun_mixed_circuit_search_20260330"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage428_deepseek7b_pronoun_head_group_stability_20260402"
)
MODEL_KEY = "deepseek7b"
THRESHOLD_RATIOS = [0.50, 0.70, 0.80, 0.90]


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

    return {
        "subset_ids": list(sorted(subset_ids)),
        "subset_size": len(subset_ids),
        "search_effect": search_effect,
        "full_effect": full_effect,
    }


def build_subset_table(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_search: Dict[str, object],
    baseline_full: Dict[str, object],
    search_groups: Dict[str, List[Dict[str, str]]],
    candidate_map: Dict[str, Dict[str, object]],
    *,
    batch_size: int,
) -> Dict[str, Dict[str, object]]:
    table: Dict[str, Dict[str, object]] = {
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
        }
    }
    ids = list(candidate_map.keys())
    for combo in powerset_nonempty(ids):
        row = evaluate_subset(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            baseline_full,
            search_groups,
            candidate_map,
            combo,
            batch_size=batch_size,
        )
        table[subset_key(combo)] = row
    return table


def sort_subset_rows(rows: Iterable[Dict[str, object]], *, metric_path: Tuple[str, ...]) -> List[Dict[str, object]]:
    def metric(row: Dict[str, object]) -> float:
        cur = row
        for key in metric_path:
            cur = cur[key]
        return float(cur)

    return sorted(
        rows,
        key=lambda row: (metric(row), -int(row["subset_size"])),
        reverse=True,
    )


def extract_best_by_size(
    rows: Sequence[Dict[str, object]],
    *,
    metric_path: Tuple[str, ...],
) -> Dict[str, Dict[str, object]]:
    best: Dict[int, Dict[str, object]] = {}

    def metric(row: Dict[str, object]) -> float:
        cur = row
        for key in metric_path:
            cur = cur[key]
        return float(cur)

    for row in rows:
        size = int(row["subset_size"])
        prev = best.get(size)
        if prev is None or metric(row) > metric(prev):
            best[size] = row
    return {str(size): best[size] for size in sorted(best)}


def find_minimal_threshold_subsets(
    rows: Sequence[Dict[str, object]],
    *,
    best_full_target_drop: float,
) -> List[Dict[str, object]]:
    out = []
    scored = [row for row in rows if int(row["subset_size"]) > 0]
    for ratio in THRESHOLD_RATIOS:
        threshold = ratio * best_full_target_drop
        candidates = [
            row
            for row in scored
            if -float(row["full_effect"]["target_prob_delta"]) >= threshold
        ]
        if not candidates:
            out.append(
                {
                    "ratio": ratio,
                    "threshold_target_drop": threshold,
                    "subset": None,
                }
            )
            continue
        candidates.sort(
            key=lambda row: (
                int(row["subset_size"]),
                -float(row["search_effect"]["utility"]),
                -(-float(row["full_effect"]["target_prob_delta"])),
            )
        )
        chosen = candidates[0]
        out.append(
            {
                "ratio": ratio,
                "threshold_target_drop": threshold,
                "subset": chosen,
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
    rows = []
    for cid in full_subset_ids:
        reduced_ids = [x for x in full_subset_ids if x != cid]
        reduced_row = subset_table[subset_key(reduced_ids)]
        reduced_search_drop = float(reduced_row["search_effect"]["target_drop"])
        reduced_search_utility = float(reduced_row["search_effect"]["utility"])
        reduced_full_drop = -float(reduced_row["full_effect"]["target_prob_delta"])
        rows.append(
            {
                "candidate_id": cid,
                "search_target_drop_loss": full_search_drop - reduced_search_drop,
                "search_utility_loss": full_search_utility - reduced_search_utility,
                "full_target_drop_loss": full_full_drop - reduced_full_drop,
                "remaining_subset_ids": reduced_ids,
            }
        )
    rows.sort(key=lambda row: (float(row["full_target_drop_loss"]), float(row["search_utility_loss"])), reverse=True)
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
        pair_search_utility = float(pair_row["search_effect"]["utility"])
        pair_search_drop = float(pair_row["search_effect"]["target_drop"])
        single_search_utility = float(row_a["search_effect"]["utility"]) + float(row_b["search_effect"]["utility"])
        single_search_drop = float(row_a["search_effect"]["target_drop"]) + float(row_b["search_effect"]["target_drop"])
        pair_full_drop = -float(pair_row["full_effect"]["target_prob_delta"])
        single_full_drop = -float(row_a["full_effect"]["target_prob_delta"]) - float(row_b["full_effect"]["target_prob_delta"])
        rows.append(
            {
                "pair_ids": [cid_a, cid_b],
                "pair_search_utility": pair_search_utility,
                "pair_search_target_drop": pair_search_drop,
                "pair_full_target_drop": pair_full_drop,
                "utility_synergy": pair_search_utility - single_search_utility,
                "search_target_drop_synergy": pair_search_drop - single_search_drop,
                "full_target_drop_synergy": pair_full_drop - single_full_drop,
            }
        )
    rows.sort(key=lambda row: (float(row["utility_synergy"]), float(row["full_target_drop_synergy"])), reverse=True)
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
        search_drop_value = 0.0
        full_drop_value = 0.0
        others = [x for x in candidate_ids if x != cid]
        for r in range(0, len(others) + 1):
            for combo in itertools.combinations(others, r):
                combo_key = subset_key(combo)
                with_key = subset_key(list(combo) + [cid])
                base_row = subset_table[combo_key]
                with_row = subset_table[with_key]
                weight = math.factorial(r) * math.factorial(n - r - 1) / factorial_n
                utility_value += weight * (
                    float(with_row["search_effect"]["utility"]) - float(base_row["search_effect"]["utility"])
                )
                search_drop_value += weight * (
                    float(with_row["search_effect"]["target_drop"]) - float(base_row["search_effect"]["target_drop"])
                )
                full_drop_value += weight * (
                    -float(with_row["full_effect"]["target_prob_delta"])
                    + float(base_row["full_effect"]["target_prob_delta"])
                )
        rows.append(
            {
                "candidate_id": cid,
                "shapley_search_utility": utility_value,
                "shapley_search_target_drop": search_drop_value,
                "shapley_full_target_drop": full_drop_value,
            }
        )
    rows.sort(
        key=lambda row: (float(row["shapley_search_utility"]), float(row["shapley_full_target_drop"])),
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


def build_core_summary(
    leave_one_out: Sequence[Dict[str, object]],
    shapley_rows: Sequence[Dict[str, object]],
    candidate_meta: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    loo_map = {row["candidate_id"]: row for row in leave_one_out}
    out = []
    for row in shapley_rows:
        cid = row["candidate_id"]
        meta = candidate_meta[cid]
        out.append(
            {
                "candidate_id": cid,
                "layer_index": meta["layer_index"],
                "head_index": meta["head_index"],
                "shapley_search_utility": float(row["shapley_search_utility"]),
                "shapley_full_target_drop": float(row["shapley_full_target_drop"]),
                "leave_one_out_full_target_drop_loss": float(loo_map[cid]["full_target_drop_loss"]),
                "leave_one_out_search_utility_loss": float(loo_map[cid]["search_utility_loss"]),
            }
        )
    out.sort(
        key=lambda row: (
            float(row["shapley_search_utility"]),
            float(row["leave_one_out_full_target_drop_loss"]),
        ),
        reverse=True,
    )
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 对 stage427 找到的 DeepSeek7B 六头代词回路做精确穷举扫描",
        "- 统计对象: 所有非空头组子集、Shapley 贡献、留一剔除必要性、两两协同",
        "",
        "## 核心结果",
        f"- 最强搜索子集: {summary['best_search_subset']['subset_ids']}",
        f"- 最强搜索 utility: {summary['best_search_subset']['search_effect']['utility']:+.4f}",
        f"- 最强全量复核子集: {summary['best_full_subset']['subset_ids']}",
        f"- 最强全量复核 target_prob_delta: {summary['best_full_subset']['full_effect']['target_prob_delta']:+.4f}",
        "",
        "## 稳定核心排序",
    ]
    for row in summary["stable_core_ranking"][:6]:
        lines.append(
            f"- {row['candidate_id']}: layer={row['layer_index']}, head={row['head_index']}, "
            f"shapley_utility={row['shapley_search_utility']:.4f}, "
            f"loo_full_drop_loss={row['leave_one_out_full_target_drop_loss']:.4f}"
        )
    lines.append("")
    lines.append("## 最优阈值子集")
    for row in summary["minimal_threshold_subsets"]:
        if row["subset"] is None:
            lines.append(f"- ratio={row['ratio']:.2f}: 未找到满足阈值的子集")
        else:
            lines.append(
                f"- ratio={row['ratio']:.2f}: subset={row['subset']['subset_ids']}, "
                f"size={row['subset']['subset_size']}, "
                f"full_target_prob_delta={row['subset']['full_effect']['target_prob_delta']:+.4f}"
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
    parser = argparse.ArgumentParser(description="DeepSeek7B 代词头组稳定性扫描")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stage427_summary = load_json(STAGE427_SUMMARY_PATH)
    final_subset = stage427_summary["models"][MODEL_KEY]["final_subset"]
    candidate_meta = build_candidate_descriptions(final_subset)
    candidate_ids = list(candidate_meta.keys())
    candidate_map = {row["candidate_id"]: row for row in final_subset}

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
        subset_table = build_subset_table(
            model,
            tokenizer,
            digit_token_ids,
            baseline_search,
            baseline_full,
            search_groups,
            candidate_map,
            batch_size=int(args.batch_size),
        )
    finally:
        free_model(model)

    elapsed = time.time() - start_time
    nonempty_rows = [row for key, row in subset_table.items() if key != "EMPTY"]
    best_search_subset = sort_subset_rows(nonempty_rows, metric_path=("search_effect", "utility"))[0]
    best_full_subset = sort_subset_rows(
        nonempty_rows,
        metric_path=("full_effect", "target_prob_delta"),
    )[-1]
    # previous line is not right for most negative values; compute explicitly
    best_full_subset = min(nonempty_rows, key=lambda row: float(row["full_effect"]["target_prob_delta"]))
    best_by_size_search = extract_best_by_size(nonempty_rows, metric_path=("search_effect", "utility"))
    best_by_size_full = {
        size: min(
            [row for row in nonempty_rows if str(row["subset_size"]) == size],
            key=lambda row: float(row["full_effect"]["target_prob_delta"]),
        )
        for size in sorted({str(row["subset_size"]) for row in nonempty_rows}, key=int)
    }
    minimal_threshold_subsets = find_minimal_threshold_subsets(
        nonempty_rows,
        best_full_target_drop=-float(best_full_subset["full_effect"]["target_prob_delta"]),
    )
    leave_one_out = compute_leave_one_out(subset_table, candidate_ids)
    pair_synergy = compute_pair_synergy(subset_table, candidate_ids)
    shapley_rows = compute_shapley(subset_table, candidate_ids)
    stable_core_ranking = build_core_summary(leave_one_out, shapley_rows, candidate_meta)

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage428_deepseek7b_pronoun_head_group_stability",
        "title": "DeepSeek7B 代词头组稳定性扫描",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "source_stage427_summary": str(STAGE427_SUMMARY_PATH),
        "candidate_heads": candidate_meta,
        "baseline_search": baseline_search,
        "baseline_full": baseline_full,
        "candidate_count": len(candidate_ids),
        "subset_count_scanned": len(nonempty_rows),
        "subset_table": {
            subset_key_: row for subset_key_, row in sorted(subset_table.items(), key=lambda item: (item[1]["subset_size"], item[0]))
        },
        "best_search_subset": best_search_subset,
        "best_full_subset": best_full_subset,
        "best_by_size_search": best_by_size_search,
        "best_by_size_full": best_by_size_full,
        "minimal_threshold_subsets": minimal_threshold_subsets,
        "leave_one_out": leave_one_out,
        "pair_synergy": pair_synergy,
        "shapley": shapley_rows,
        "stable_core_ranking": stable_core_ranking,
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage428_deepseek7b_pronoun_head_group_stability_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "best_search_subset_size": best_search_subset["subset_size"],
                "best_search_utility": best_search_subset["search_effect"]["utility"],
                "best_full_subset_size": best_full_subset["subset_size"],
                "best_full_target_prob_delta": best_full_subset["full_effect"]["target_prob_delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
