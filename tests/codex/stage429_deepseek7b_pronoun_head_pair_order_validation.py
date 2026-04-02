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
from stage428_deepseek7b_pronoun_head_group_stability import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE428_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage428_deepseek7b_pronoun_head_group_stability_20260402"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage429_deepseek7b_pronoun_head_pair_order_validation_20260402"
)
MODEL_KEY = "deepseek7b"

HELDOUT_PRONOUN_CASES: List[Dict[str, str]] = [
    {"word": "they", "sentence": "By sunrise, [they] had already packed the instruments."},
    {"word": "them", "sentence": "The guide pointed toward [them] near the far gate."},
    {"word": "us", "sentence": "The letter reached [us] after the holiday ended."},
    {"word": "him", "sentence": "The noise startled [him] in the middle of the speech."},
    {"word": "her", "sentence": "The manager thanked [her] for the careful repair."},
    {"word": "ours", "sentence": "The blue tent is [ours] for the weekend."},
    {"word": "hers", "sentence": "The silver notebook is [hers] now."},
    {"word": "this", "sentence": "[This] sounds different in the quiet room."},
    {"word": "those", "sentence": "[Those] belong in the bottom drawer."},
    {"word": "who", "sentence": "[Who] answered the radio before dawn?"},
    {"word": "which", "sentence": "[Which] should stay near the west wall?"},
    {"word": "someone", "sentence": "[Someone] left a sealed envelope on the desk."},
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


def evaluate_target_only(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    cases: Sequence[Dict[str, str]],
    *,
    batch_size: int,
    handles: Sequence[object] | None = None,
) -> Dict[str, float]:
    groups = {TARGET_CLASS: list(cases)}
    result = evaluate_case_groups(
        model,
        tokenizer,
        digit_token_ids,
        groups,
        batch_size=batch_size,
        handles=handles,
    )
    return result["by_class"][TARGET_CLASS]


def summarize_target_only_delta(
    baseline: Dict[str, float],
    current: Dict[str, float],
) -> Dict[str, float]:
    prob_before = float(baseline["mean_correct_prob"])
    prob_after = float(current["mean_correct_prob"])
    acc_before = float(baseline["accuracy"])
    acc_after = float(current["accuracy"])
    return {
        "target_prob_before": prob_before,
        "target_prob_after": prob_after,
        "target_prob_delta": prob_after - prob_before,
        "target_drop": prob_before - prob_after,
        "target_accuracy_before": acc_before,
        "target_accuracy_after": acc_after,
        "target_accuracy_delta": acc_after - acc_before,
        "target_accuracy_drop": acc_before - acc_after,
    }


def normalize_candidate_map(summary: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for cid, meta in summary["candidate_heads"].items():
        out[str(cid)] = {
            "candidate_id": str(cid),
            "layer_index": int(meta["layer_index"]),
            "head_index": int(meta["head_index"]),
            "kind": str(meta["kind"]),
        }
    return out


def select_core_and_boosters(summary: Dict[str, object]) -> Tuple[List[str], List[str], str, List[str]]:
    stable = summary["stable_core_ranking"]
    core_ids = [str(row["candidate_id"]) for row in stable[:3]]
    booster_ids = [str(row["candidate_id"]) for row in stable[3:6]]
    candidate_meta = normalize_candidate_map(summary)
    route_pair = [cid for cid in core_ids if int(candidate_meta[cid]["layer_index"]) == 2]
    if len(route_pair) != 2:
        raise RuntimeError(f"Cannot identify two layer-2 route heads from stable core: {route_pair}")
    integrator_ids = [cid for cid in core_ids if cid not in route_pair]
    if len(integrator_ids) != 1:
        raise RuntimeError(f"Cannot identify one layer-3 integrator head from stable core: {integrator_ids}")
    return core_ids, booster_ids, integrator_ids[0], sorted(route_pair)


def build_subset_plan(
    booster_ids: Sequence[str],
    integrator_id: str,
    route_pair: Sequence[str],
) -> List[Tuple[str, List[str]]]:
    assert len(route_pair) == 2
    route_a, route_b = route_pair
    return [
        ("empty", []),
        ("single_integrator", [integrator_id]),
        ("single_route_a", [route_a]),
        ("single_route_b", [route_b]),
        ("single_booster_1", [booster_ids[0]]),
        ("single_booster_2", [booster_ids[1]]),
        ("single_booster_3", [booster_ids[2]]),
        ("pair_route", [route_a, route_b]),
        ("pair_integrator_route_a", [integrator_id, route_a]),
        ("pair_integrator_route_b", [integrator_id, route_b]),
        ("triple_core", [integrator_id, route_a, route_b]),
        ("quad_core_plus_booster_1", [integrator_id, route_a, route_b, booster_ids[0]]),
        ("quad_core_plus_booster_2", [integrator_id, route_a, route_b, booster_ids[1]]),
        ("quad_core_plus_booster_3", [integrator_id, route_a, route_b, booster_ids[2]]),
    ]


def evaluate_subset(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_heldout: Dict[str, float],
    candidate_map: Dict[str, Dict[str, object]],
    stage428_subset_table: Dict[str, Dict[str, object]],
    subset_ids: Sequence[str],
    *,
    batch_size: int,
) -> Dict[str, object]:
    subset = [candidate_map[cid] for cid in sorted(subset_ids)]
    table_key = "EMPTY" if not subset_ids else "|".join(sorted(subset_ids))
    if table_key not in stage428_subset_table:
        raise KeyError(f"Missing subset in stage428 table: {table_key}")
    cached_row = stage428_subset_table[table_key]
    search_effect = cached_row["search_effect"]
    full_effect = cached_row["full_effect"]

    heldout_handles = register_attention_head_ablation(model, subset) if subset else None
    current_heldout = evaluate_target_only(
        model,
        tokenizer,
        digit_token_ids,
        HELDOUT_PRONOUN_CASES,
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
        "heldout_pronoun_effect": heldout_effect,
    }


def build_subset_rows(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline_heldout: Dict[str, float],
    candidate_map: Dict[str, Dict[str, object]],
    stage428_subset_table: Dict[str, Dict[str, object]],
    subset_plan: Sequence[Tuple[str, List[str]]],
    *,
    batch_size: int,
) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for label, subset_ids in subset_plan:
        rows[label] = evaluate_subset(
            model,
            tokenizer,
            digit_token_ids,
            baseline_heldout,
            candidate_map,
            stage428_subset_table,
            subset_ids,
            batch_size=batch_size,
        )
    return rows


def target_drop_full(row: Dict[str, object]) -> float:
    return -float(row["full_effect"]["target_prob_delta"])


def target_drop_heldout(row: Dict[str, object]) -> float:
    return float(row["heldout_pronoun_effect"]["target_drop"])


def utility(row: Dict[str, object]) -> float:
    return float(row["search_effect"]["utility"])


def pair_synergy(
    pair_row: Dict[str, object],
    row_a: Dict[str, object],
    row_b: Dict[str, object],
) -> Dict[str, float]:
    return {
        "utility_synergy": utility(pair_row) - utility(row_a) - utility(row_b),
        "full_target_drop_synergy": target_drop_full(pair_row) - target_drop_full(row_a) - target_drop_full(row_b),
        "heldout_target_drop_synergy": target_drop_heldout(pair_row)
        - target_drop_heldout(row_a)
        - target_drop_heldout(row_b),
    }


def conditional_gain(
    base_row: Dict[str, object],
    added_row: Dict[str, object],
    candidate_id: str,
    context_ids: Sequence[str],
) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "context_ids": list(sorted(context_ids)),
        "added_subset_ids": list(added_row["subset_ids"]),
        "utility_gain": utility(added_row) - utility(base_row),
        "full_target_drop_gain": target_drop_full(added_row) - target_drop_full(base_row),
        "heldout_target_drop_gain": target_drop_heldout(added_row) - target_drop_heldout(base_row),
    }


def build_order_sequences(
    rows: Dict[str, Dict[str, object]],
    best_booster_label: str,
    best_booster_id: str,
) -> List[Dict[str, object]]:
    route_first = [
        ("empty", rows["empty"]),
        ("single_route_a", rows["single_route_a"]),
        ("pair_route", rows["pair_route"]),
        ("triple_core", rows["triple_core"]),
        (f"core_plus_{best_booster_id}", rows[best_booster_label]),
    ]
    integrator_first_a = [
        ("empty", rows["empty"]),
        ("single_integrator", rows["single_integrator"]),
        ("pair_integrator_route_a", rows["pair_integrator_route_a"]),
        ("triple_core", rows["triple_core"]),
        route_first[-1],
    ]
    integrator_first_b = [
        ("empty", rows["empty"]),
        ("single_integrator", rows["single_integrator"]),
        ("pair_integrator_route_b", rows["pair_integrator_route_b"]),
        ("triple_core", rows["triple_core"]),
        route_first[-1],
    ]

    sequences = []
    for label, steps in [
        ("route_first", route_first),
        ("integrator_first_via_route_a", integrator_first_a),
        ("integrator_first_via_route_b", integrator_first_b),
    ]:
        step_rows = []
        prev = steps[0][1]
        for step_label, current in steps[1:]:
            step_rows.append(
                {
                    "step_label": step_label,
                    "subset_ids": list(current["subset_ids"]),
                    "incremental_utility_gain": utility(current) - utility(prev),
                    "incremental_full_target_drop_gain": target_drop_full(current) - target_drop_full(prev),
                    "incremental_heldout_target_drop_gain": target_drop_heldout(current)
                    - target_drop_heldout(prev),
                    "cumulative_utility": utility(current),
                    "cumulative_full_target_drop": target_drop_full(current),
                    "cumulative_heldout_target_drop": target_drop_heldout(current),
                }
            )
            prev = current
        sequences.append({"sequence_label": label, "steps": step_rows})
    return sequences


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## Setup",
        f"- Timestamp UTC: {summary['timestamp_utc']}",
        f"- Use CUDA: {summary['used_cuda']}",
        f"- Batch size: {summary['batch_size']}",
        f"- Core heads: {summary['core_heads']}",
        f"- Route pair: {summary['route_pair']}",
        f"- Integrator head: {summary['integrator_head']}",
        f"- Booster candidates: {summary['booster_heads']}",
        f"- Best booster: {summary['best_booster']['candidate_id']}",
        "",
        "## Findings",
        f"- Best pair: {summary['best_core_pair']['pair_ids']}",
        f"- Best pair full pronoun drop: {summary['best_core_pair']['full_target_drop']:.4f}",
        f"- Integrator gain on route pair: {summary['mechanism_inference']['integrator_after_route_pair']['full_target_drop_gain']:.4f}",
        f"- Best reverse gain on integrator: {summary['mechanism_inference']['best_route_after_integrator']['full_target_drop_gain']:.4f}",
        f"- Full order support margin: {summary['mechanism_inference']['route_then_integrate_margin_full']:.4f}",
        f"- Heldout order support margin: {summary['mechanism_inference']['route_then_integrate_margin_heldout']:.4f}",
        f"- Heldout best booster effect: {summary['best_booster']['candidate_id']} -> {summary['best_booster']['heldout_target_drop_gain']:.4f}",
        "",
        "## Reading",
        "- If the layer-3 head gains more on top of the layer-2 route pair than the reverse direction gains on top of the integrator, the circuit looks more like route-first then integrate.",
        "- If a booster head matters mainly after the core triple exists, it behaves more like an amplifier than a skeleton head.",
    ]
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek7B pronoun core head pair and order validation")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Forward batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stage428_summary = load_json(STAGE428_SUMMARY_PATH)
    candidate_map = normalize_candidate_map(stage428_summary)
    core_ids, booster_ids, integrator_id, route_pair = select_core_and_boosters(stage428_summary)
    subset_plan = build_subset_plan(booster_ids, integrator_id, route_pair)
    stage428_subset_table = stage428_summary["subset_table"]
    baseline_search = stage428_summary["baseline_search"]
    baseline_full = stage428_summary["baseline_full"]

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[MODEL_KEY]["model_path"], prefer_cuda=use_cuda)
    start_time = time.time()
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline_heldout = evaluate_target_only(
            model,
            tokenizer,
            digit_token_ids,
            HELDOUT_PRONOUN_CASES,
            batch_size=int(args.batch_size),
        )
        rows = build_subset_rows(
            model,
            tokenizer,
            digit_token_ids,
            baseline_heldout,
            candidate_map,
            stage428_subset_table,
            subset_plan,
            batch_size=int(args.batch_size),
        )
    finally:
        free_model(model)

    elapsed = time.time() - start_time

    single_booster_labels = ["single_booster_1", "single_booster_2", "single_booster_3"]
    quad_booster_labels = ["quad_core_plus_booster_1", "quad_core_plus_booster_2", "quad_core_plus_booster_3"]
    booster_gains = []
    for single_label, quad_label in zip(single_booster_labels, quad_booster_labels):
        booster_gains.append(
            {
                "quad_label": quad_label,
                **conditional_gain(
                    rows["triple_core"],
                    rows[quad_label],
                    rows[single_label]["subset_ids"][0],
                    rows["triple_core"]["subset_ids"],
                ),
            }
        )
    booster_gains.sort(
        key=lambda row: (float(row["heldout_target_drop_gain"]), float(row["full_target_drop_gain"])),
        reverse=True,
    )
    best_booster = booster_gains[0]

    pair_route_synergy = pair_synergy(rows["pair_route"], rows["single_route_a"], rows["single_route_b"])
    pair_integrator_route_a_synergy = pair_synergy(
        rows["pair_integrator_route_a"],
        rows["single_integrator"],
        rows["single_route_a"],
    )
    pair_integrator_route_b_synergy = pair_synergy(
        rows["pair_integrator_route_b"],
        rows["single_integrator"],
        rows["single_route_b"],
    )

    core_pairs = [
        {
            "pair_ids": list(rows["pair_route"]["subset_ids"]),
            "utility": utility(rows["pair_route"]),
            "full_target_drop": target_drop_full(rows["pair_route"]),
            "heldout_target_drop": target_drop_heldout(rows["pair_route"]),
            **pair_route_synergy,
        },
        {
            "pair_ids": list(rows["pair_integrator_route_a"]["subset_ids"]),
            "utility": utility(rows["pair_integrator_route_a"]),
            "full_target_drop": target_drop_full(rows["pair_integrator_route_a"]),
            "heldout_target_drop": target_drop_heldout(rows["pair_integrator_route_a"]),
            **pair_integrator_route_a_synergy,
        },
        {
            "pair_ids": list(rows["pair_integrator_route_b"]["subset_ids"]),
            "utility": utility(rows["pair_integrator_route_b"]),
            "full_target_drop": target_drop_full(rows["pair_integrator_route_b"]),
            "heldout_target_drop": target_drop_heldout(rows["pair_integrator_route_b"]),
            **pair_integrator_route_b_synergy,
        },
    ]
    core_pairs.sort(
        key=lambda row: (float(row["full_target_drop"]), float(row["heldout_target_drop"])),
        reverse=True,
    )
    best_core_pair = core_pairs[0]

    integrator_after_route_pair = conditional_gain(
        rows["pair_route"],
        rows["triple_core"],
        integrator_id,
        rows["pair_route"]["subset_ids"],
    )
    route_a_after_integrator = conditional_gain(
        rows["single_integrator"],
        rows["pair_integrator_route_a"],
        route_pair[0],
        rows["single_integrator"]["subset_ids"],
    )
    route_b_after_integrator = conditional_gain(
        rows["single_integrator"],
        rows["pair_integrator_route_b"],
        route_pair[1],
        rows["single_integrator"]["subset_ids"],
    )
    best_route_after_integrator = max(
        [route_a_after_integrator, route_b_after_integrator],
        key=lambda row: (float(row["full_target_drop_gain"]), float(row["heldout_target_drop_gain"])),
    )

    order_sequences = build_order_sequences(
        rows,
        str(best_booster["quad_label"]),
        str(best_booster["candidate_id"]),
    )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage429_deepseek7b_pronoun_head_pair_order_validation",
        "title": "DeepSeek7B pronoun core head pair and order validation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": bool(use_cuda),
        "batch_size": int(args.batch_size),
        "source_stage428_summary": str(STAGE428_SUMMARY_PATH),
        "core_heads": core_ids,
        "route_pair": list(route_pair),
        "integrator_head": integrator_id,
        "booster_heads": booster_ids,
        "candidate_heads": candidate_map,
        "baseline_search": baseline_search,
        "baseline_full": baseline_full,
        "baseline_heldout_pronoun": baseline_heldout,
        "heldout_pronoun_cases": HELDOUT_PRONOUN_CASES,
        "targeted_subsets": rows,
        "core_pair_scores": core_pairs,
        "best_core_pair": best_core_pair,
        "conditional_gains": {
            "integrator_after_route_pair": integrator_after_route_pair,
            "route_a_after_integrator": route_a_after_integrator,
            "route_b_after_integrator": route_b_after_integrator,
            "booster_after_core": booster_gains,
        },
        "best_booster": best_booster,
        "order_sequences": order_sequences,
        "mechanism_inference": {
            "integrator_after_route_pair": integrator_after_route_pair,
            "best_route_after_integrator": best_route_after_integrator,
            "route_then_integrate_margin_full": float(integrator_after_route_pair["full_target_drop_gain"])
            - float(best_route_after_integrator["full_target_drop_gain"]),
            "route_then_integrate_margin_heldout": float(integrator_after_route_pair["heldout_target_drop_gain"])
            - float(best_route_after_integrator["heldout_target_drop_gain"]),
            "booster_is_post_core": {
                "candidate_id": best_booster["candidate_id"],
                "full_target_drop_gain_after_core": float(best_booster["full_target_drop_gain"]),
                "heldout_target_drop_gain_after_core": float(best_booster["heldout_target_drop_gain"]),
            },
            "working_hypothesis": (
                "If the layer-2 route pair has the strongest synergy, and the layer-3 head gains more on top of that pair "
                "than the reverse direction gains on top of the integrator, the pronoun circuit is better explained as "
                "route first, integrate second, and booster last."
            ),
        },
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
