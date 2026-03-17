from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage3_causal_closure import (  # noqa: E402
    analyze_family_effect,
    category_readout,
    compute_signature,
    read_json,
    read_jsonl,
    register_ablation,
    remove_handles,
    sample_random_like,
)
from deepseek7b_stage4_minimal_circuit_search import (  # noqa: E402
    baseline_importance,
)
from deepseek7b_three_pool_structure_scan import (  # noqa: E402
    GateCollector,
    LexemeItem,
    layer_distribution,
    load_model,
)


def stage4_candidate_key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return (
        str(row["item"]["term"]),
        str(row["item"]["category"]),
        str(row["source_kind"]),
    )


def sort_stage4_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            1 if bool(row["pair_metrics"]["joint_binding_hit"]) else 0,
            float(row["pair_metrics"]["joint_adv_score"]),
            float(row["pair_metrics"]["margin_adv_vs_random"]),
            float(row["pair_metrics"]["category_adv_vs_random"]),
            -int(row["subset_size"]),
        ),
        reverse=True,
    )


def normalize_lexeme(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalpha())


def is_category_word(item: Dict[str, object]) -> bool:
    term = normalize_lexeme(str(item.get("term", "")))
    category = normalize_lexeme(str(item.get("category", "")))
    if not term or not category:
        return False
    return term == category or term.rstrip("s") == category


def is_prototype_proxy_row(row: Dict[str, object]) -> bool:
    return str(row.get("source_kind", "")) == "family_prototype"


def lane_matches(
    item: Dict[str, object],
    lane_mode: str,
    source_kind: str = "",
) -> bool:
    is_proto = is_category_word(item) or source_kind == "family_prototype"
    if lane_mode == "prototype":
        return is_proto
    if lane_mode == "instance":
        return not is_proto
    return True


def candidate_allowed_in_lane(
    item: Dict[str, object],
    lane_mode: str,
    source_kind: str = "",
    prototype_term_mode: str = "any",
    allow_prototype_proxy: bool = True,
) -> bool:
    if not lane_matches(item, lane_mode, source_kind=source_kind):
        return False
    if lane_mode != "prototype":
        return True
    if source_kind == "family_prototype" and not allow_prototype_proxy:
        return False
    if prototype_term_mode == "category_only" and not is_category_word(item):
        return False
    return True


def subset_overlap_ratio(left: Dict[str, object], right: Dict[str, object]) -> float:
    left_ids = set(int(x) for x in left.get("subset_flat_indices", []))
    right_ids = set(int(x) for x in right.get("subset_flat_indices", []))
    if not left_ids or not right_ids:
        return 0.0
    inter = len(left_ids & right_ids)
    union = len(left_ids | right_ids)
    return float(inter / union) if union else 0.0


def selection_score(
    row: Dict[str, object],
    chosen: Sequence[Dict[str, object]],
    overlap_penalty: float,
    category_word_penalty: float = 0.0,
    margin_adv_threshold: float = 0.0,
    margin_adv_penalty: float = 0.0,
) -> float:
    pair = row["pair_metrics"]
    base = (
        float(pair["joint_adv_score"])
        + 0.5 * float(pair["margin_adv_vs_random"])
        + (0.25 if bool(pair["joint_binding_hit"]) else 0.0)
    )
    if category_word_penalty > 0.0 and is_category_word(row.get("item", {})):
        base -= float(category_word_penalty)
    if margin_adv_penalty > 0.0 and float(pair["margin_adv_vs_random"]) <= margin_adv_threshold:
        base -= float(margin_adv_penalty)
    if not chosen:
        return base
    max_overlap = max(subset_overlap_ratio(row, prev) for prev in chosen)
    return float(base - overlap_penalty * max_overlap)


def select_stage4_candidates(
    rows: Sequence[Dict[str, object]],
    max_candidates: int,
    per_category_limit: int,
    overlap_penalty: float = 0.15,
    max_overlap: float = 0.80,
    require_category_coverage: bool = False,
    category_word_penalty: float = 0.0,
    margin_adv_threshold: float = 0.0,
    margin_adv_penalty: float = 0.0,
) -> List[Dict[str, object]]:
    ordered = sort_stage4_rows(rows)
    chosen: List[Dict[str, object]] = []
    seen = set()
    category_counts: Dict[str, int] = {}
    remaining = list(ordered)

    def pick_next(preferred_categories: set[str] | None = None) -> int | None:
        best_idx = None
        best_score = None
        for idx, row in enumerate(remaining):
            key = stage4_candidate_key(row)
            category = str(row["item"]["category"])
            if key in seen:
                continue
            if category_counts.get(category, 0) >= per_category_limit:
                continue
            if preferred_categories is not None and category not in preferred_categories:
                continue
            if chosen and max(subset_overlap_ratio(row, prev) for prev in chosen) > max_overlap:
                continue
            score = selection_score(
                row,
                chosen,
                overlap_penalty=overlap_penalty,
                category_word_penalty=category_word_penalty,
                margin_adv_threshold=margin_adv_threshold,
                margin_adv_penalty=margin_adv_penalty,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    if require_category_coverage and max_candidates > 0:
        categories = {str(row["item"]["category"]) for row in ordered}
        while remaining and len(chosen) < max_candidates and categories:
            missing = {cat for cat in categories if category_counts.get(cat, 0) <= 0}
            if not missing:
                break
            best_idx = pick_next(preferred_categories=missing)
            if best_idx is None:
                break
            row = remaining.pop(best_idx)
            key = stage4_candidate_key(row)
            category = str(row["item"]["category"])
            seen.add(key)
            category_counts[category] = category_counts.get(category, 0) + 1
            chosen.append(row)

    while remaining and len(chosen) < max_candidates:
        best_idx = pick_next()
        if best_idx is None:
            break
        row = remaining.pop(best_idx)
        key = stage4_candidate_key(row)
        category = str(row["item"]["category"])
        seen.add(key)
        category_counts[category] = category_counts.get(category, 0) + 1
        chosen.append(row)
    return chosen


def limit_candidate_indices(
    row: Dict[str, object],
    baseline_signature: Dict[str, object],
    max_neurons: int,
    d_ff: int | None = None,
    max_per_layer: int = 0,
) -> List[int]:
    raw = [int(x) for x in row["subset_flat_indices"]]
    importance = baseline_importance(baseline_signature)
    ranked = sorted(
        sorted(set(raw)),
        key=lambda idx: (
            1 if idx in importance else 0,
            importance.get(idx, 0.0),
            -idx,
        ),
        reverse=True,
    )
    if d_ff is None or max_per_layer <= 0:
        return ranked[: max(1, max_neurons)]
    chosen: List[int] = []
    layer_counts: Dict[int, int] = {}
    leftovers: List[int] = []
    for idx in ranked:
        layer = idx // d_ff
        if layer_counts.get(layer, 0) >= max_per_layer:
            leftovers.append(idx)
            continue
        chosen.append(idx)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(chosen) >= max(1, max_neurons):
            return chosen
    for idx in leftovers:
        chosen.append(idx)
        if len(chosen) >= max(1, max_neurons):
            break
    return chosen


def effect_score(margin_value: float, category_value: float, alpha: float) -> float:
    return float(margin_value + alpha * category_value)


def strict_effect_score(
    margin_value: float,
    category_value: float,
    alpha: float,
    margin_adv_threshold: float = 0.0,
    margin_adv_penalty: float = 0.0,
) -> float:
    score = effect_score(margin_value, category_value, alpha)
    if margin_adv_penalty > 0.0 and float(margin_value) <= margin_adv_threshold:
        score -= float(margin_adv_penalty)
    return float(score)


def choose_representative_baselines(
    baselines: Sequence[Dict[str, object]],
    selected_categories: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    wanted = {str(category) for category in selected_categories}
    for row in baselines:
        category = str(row["item"]["category"])
        if category not in wanted:
            continue
        grouped.setdefault(category, []).append(row)
    return {
        category: max(
            rows,
            key=lambda row: (
                float(row["baseline_readout"].get("category_margin", 0.0)),
                float(row["baseline_readout"].get("correct_prob", 0.0)),
                str(row["item"]["term"]),
            ),
        )
        for category, rows in grouped.items()
    }


def build_prototype_proxy_rows(
    families: Sequence[Dict[str, object]],
    baselines: Sequence[Dict[str, object]],
    selected_categories: Sequence[str],
) -> List[Dict[str, object]]:
    representative_map = choose_representative_baselines(baselines, selected_categories)
    rows: List[Dict[str, object]] = []
    for family in families:
        if str(family.get("record_type")) != "family_prototype":
            continue
        if str(family.get("pool")) != "closure":
            continue
        category = str(family.get("category"))
        if category not in representative_map:
            continue
        prototype_indices = [int(x) for x in family.get("prototype_top_indices", [])]
        if not prototype_indices:
            continue
        stability = float(family.get("mean_prompt_stability", 0.0))
        representative = representative_map[category]
        rows.append(
            {
                "record_type": "stage5_prototype_proxy",
                "item": dict(representative["item"]),
                "source_kind": "family_prototype",
                "variant": "subset",
                "subset_size": len(prototype_indices),
                "subset_flat_indices": prototype_indices,
                "pair_metrics": {
                    "joint_binding_hit": False,
                    "joint_adv_score": stability * 0.01,
                    "margin_adv_vs_random": 0.0,
                    "category_adv_vs_random": 0.0,
                },
                "prototype_category": category,
                "prototype_mean_prompt_stability": stability,
            }
        )
    return rows


def evaluate_ablation(
    model,
    tok,
    collector: GateCollector,
    item: LexemeItem,
    baseline_sig: Dict[str, object],
    baseline_readout: Dict[str, object],
    category_proto: Dict[str, object],
    proto_map: Dict[str, Dict[str, object]],
    selected_categories: Sequence[str],
    flat_indices: Sequence[int],
    signature_top_k: int,
) -> Dict[str, object]:
    handles = register_ablation(model, flat_indices, collector.d_ff) if flat_indices else []
    try:
        ablated_sig = compute_signature(model, tok, collector, item, top_k=signature_top_k)
        ablated_readout = category_readout(
            model,
            tok,
            term=item.term,
            correct_category=item.category,
            all_categories=selected_categories,
        )
    finally:
        remove_handles(handles)

    effects = analyze_family_effect(
        item=type("FocusLike", (), {"term": item.term, "category": item.category, "role": "candidate"})(),
        category_proto=category_proto,
        all_protos_by_category=proto_map,
        baseline_sig=baseline_sig,
        ablated_sig=ablated_sig,
        baseline_readout=baseline_readout,
        ablated_readout=ablated_readout,
    )
    return {
        "effects": effects,
        "ablated_signature": ablated_sig,
        "ablated_readout": ablated_readout,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], circuits: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# DeepSeek Stage5 Readout-Coupled Search Report",
        "",
        f"- Candidate count: {summary['candidate_count']}",
        f"- Neuron rows: {summary['neuron_row_count']}",
        f"- Circuit rows: {summary['circuit_row_count']}",
        f"- Positive micro-circuits: {summary['positive_micro_circuit_count']}",
        "",
        "## Top Micro-Circuits",
    ]
    top_rows = sorted(
        circuits,
        key=lambda row: (
            float(row["adv_metrics"].get("strict_joint_adv_score", row["adv_metrics"]["joint_adv_score"])),
            float(row["adv_metrics"]["joint_adv_score"]),
        ),
        reverse=True,
    )[:20]
    for row in top_rows:
        lines.append(
            "- "
            f"{row['item']['category']} / {row['item']['term']} / {row['source_kind']} / {row['circuit_kind']} "
            f"/ size={row['neuron_count']} / joint_adv={row['adv_metrics']['joint_adv_score']:.6f} "
            f"/ strict_joint_adv={row['adv_metrics'].get('strict_joint_adv_score', row['adv_metrics']['joint_adv_score']):.6f} "
            f"/ margin_adv={row['adv_metrics']['margin_adv_vs_random']:.6f} "
            f"/ category_adv={row['adv_metrics']['category_adv_vs_random']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek stage5 readout-coupled micro-circuit search")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--stage2-families", default="tempdata/deepseek7b_three_pool_stage2_focus_520_bf16_20260316/families.jsonl")
    ap.add_argument("--stage3-summary", default="tempdata/deepseek7b_stage3_causal_closure_520_20260316/summary.json")
    ap.add_argument("--stage3-baselines", default="tempdata/deepseek7b_stage3_causal_closure_520_20260316/baselines.jsonl")
    ap.add_argument("--stage4-results", default="tempdata/deepseek7b_stage4_minimal_circuit_520_20260316/results.jsonl")
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--per-category-limit", type=int, default=2)
    ap.add_argument("--max-neurons-per-candidate", type=int, default=12)
    ap.add_argument("--max-neurons-per-layer", type=int, default=4)
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--candidate-overlap-penalty", type=float, default=0.15)
    ap.add_argument("--max-candidate-overlap", type=float, default=0.80)
    ap.add_argument("--require-category-coverage", action="store_true")
    ap.add_argument("--category-word-penalty", type=float, default=0.0)
    ap.add_argument("--margin-adv-threshold", type=float, default=0.0)
    ap.add_argument("--margin-adv-penalty", type=float, default=0.0)
    ap.add_argument("--lane-mode", choices=["mixed", "prototype", "instance"], default="mixed")
    ap.add_argument("--prototype-term-mode", choices=["any", "category_only"], default="any")
    ap.add_argument("--disable-prototype-proxy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage5_readout_coupled_520_20260316")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    stage3_summary = read_json(args.stage3_summary)
    baselines = read_jsonl(args.stage3_baselines)
    stage4_rows = read_jsonl(args.stage4_results)

    selected_categories = [str(x) for x in stage3_summary["selected_categories"]]
    proto_map = {
        str(row["category"]): row
        for row in families
        if str(row.get("pool")) == "closure"
        if str(row["category"]) in selected_categories
    }
    baseline_map = {
        (str(row["item"]["term"]), str(row["item"]["category"])): row
        for row in baselines
    }
    stage4_subset_rows = [
        row
        for row in stage4_rows
        if str(row.get("variant")) == "subset"
    ]
    prototype_proxy_rows = build_prototype_proxy_rows(
        families=families,
        baselines=baselines,
        selected_categories=selected_categories,
    )
    candidate_source_rows = list(stage4_subset_rows)
    if args.lane_mode in {"mixed", "prototype"} and not bool(args.disable_prototype_proxy):
        candidate_source_rows.extend(prototype_proxy_rows)
    subset_rows = [
        row
        for row in candidate_source_rows
        if candidate_allowed_in_lane(
            row.get("item", {}),
            args.lane_mode,
            source_kind=str(row.get("source_kind", "")),
            prototype_term_mode=args.prototype_term_mode,
            allow_prototype_proxy=not bool(args.disable_prototype_proxy),
        )
    ]
    candidates = select_stage4_candidates(
        rows=subset_rows,
        max_candidates=args.max_candidates,
        per_category_limit=args.per_category_limit,
        overlap_penalty=args.candidate_overlap_penalty,
        max_overlap=args.max_candidate_overlap,
        require_category_coverage=args.require_category_coverage,
        category_word_penalty=args.category_word_penalty,
        margin_adv_threshold=args.margin_adv_threshold,
        margin_adv_penalty=args.margin_adv_penalty,
    )

    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)
    candidate_rows: List[Dict[str, object]] = []
    neuron_rows: List[Dict[str, object]] = []
    circuit_rows: List[Dict[str, object]] = []

    try:
        for cand_idx, cand in enumerate(candidates):
            term = str(cand["item"]["term"])
            category = str(cand["item"]["category"])
            baseline_row = baseline_map[(term, category)]
            baseline_sig = baseline_row["baseline_signature"]
            baseline_readout = baseline_row["baseline_readout"]
            category_proto = proto_map[category]
            item = LexemeItem(term=term, category=category, language="ascii")
            candidate_indices = limit_candidate_indices(
                row=cand,
                baseline_signature=baseline_sig,
                max_neurons=args.max_neurons_per_candidate,
                d_ff=collector.d_ff,
                max_per_layer=args.max_neurons_per_layer,
            )
            full_eval = evaluate_ablation(
                model,
                tok,
                collector,
                item,
                baseline_sig,
                baseline_readout,
                category_proto,
                proto_map,
                selected_categories,
                candidate_indices,
                args.signature_top_k,
            )
            full_random = sample_random_like(
                candidate_indices,
                collector.total_neurons,
                seed=args.seed + cand_idx * 1000 + 1,
            )
            full_random_eval = evaluate_ablation(
                model,
                tok,
                collector,
                item,
                baseline_sig,
                baseline_readout,
                category_proto,
                proto_map,
                selected_categories,
                full_random,
                args.signature_top_k,
            )
            full_joint_adv = effect_score(
                float(full_eval["effects"]["margin_drop"] - full_random_eval["effects"]["margin_drop"]),
                float(full_eval["effects"]["category_margin_drop"] - full_random_eval["effects"]["category_margin_drop"]),
                args.score_alpha,
            )
            full_margin_adv = float(full_eval["effects"]["margin_drop"] - full_random_eval["effects"]["margin_drop"])
            full_category_adv = float(
                full_eval["effects"]["category_margin_drop"] - full_random_eval["effects"]["category_margin_drop"]
            )
            full_strict_joint_adv = strict_effect_score(
                full_margin_adv,
                full_category_adv,
                args.score_alpha,
                margin_adv_threshold=args.margin_adv_threshold,
                margin_adv_penalty=args.margin_adv_penalty,
            )
            candidate_rows.append(
                {
                    "record_type": "stage5_candidate",
                    "item": cand["item"],
                    "source_kind": cand["source_kind"],
                    "is_category_word": is_category_word(cand["item"]),
                    "is_prototype_proxy": is_prototype_proxy_row(cand),
                    "original_subset_size": int(cand["subset_size"]),
                    "candidate_neuron_count": len(candidate_indices),
                    "candidate_indices": [int(x) for x in candidate_indices],
                    "candidate_layer_distribution": layer_distribution(candidate_indices, collector.d_ff),
                    "full_subset_effects": full_eval["effects"],
                    "full_random_effects": full_random_eval["effects"],
                    "full_margin_adv_vs_random": full_margin_adv,
                    "full_category_adv_vs_random": full_category_adv,
                    "full_joint_adv_score": full_joint_adv,
                    "full_strict_joint_adv_score": full_strict_joint_adv,
                }
            )

            rescue_rows = []
            for neuron_idx, neuron in enumerate(candidate_indices):
                solo_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    [neuron],
                    args.signature_top_k,
                )
                solo_random = sample_random_like(
                    [neuron],
                    collector.total_neurons,
                    seed=args.seed + cand_idx * 1000 + neuron_idx * 17 + 11,
                )
                solo_random_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    solo_random,
                    args.signature_top_k,
                )
                loo_indices = [idx for idx in candidate_indices if idx != neuron]
                loo_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    loo_indices,
                    args.signature_top_k,
                ) if loo_indices else full_eval

                margin_rescue = float(full_eval["effects"]["margin_drop"] - loo_eval["effects"]["margin_drop"])
                category_rescue = float(
                    full_eval["effects"]["category_margin_drop"] - loo_eval["effects"]["category_margin_drop"]
                )
                solo_margin_adv = float(
                    solo_eval["effects"]["margin_drop"] - solo_random_eval["effects"]["margin_drop"]
                )
                solo_category_adv = float(
                    solo_eval["effects"]["category_margin_drop"] - solo_random_eval["effects"]["category_margin_drop"]
                )
                rescue_score = effect_score(margin_rescue, category_rescue, args.score_alpha)
                solo_score = effect_score(solo_margin_adv, solo_category_adv, args.score_alpha)
                row = {
                    "record_type": "stage5_neuron",
                    "item": cand["item"],
                    "source_kind": cand["source_kind"],
                    "candidate_neuron_count": len(candidate_indices),
                    "neuron_index": int(neuron),
                    "layer_distribution": layer_distribution([neuron], collector.d_ff),
                    "leave_one_out_effects": loo_eval["effects"],
                    "solo_effects": solo_eval["effects"],
                    "solo_random_effects": solo_random_eval["effects"],
                    "margin_rescue": margin_rescue,
                    "category_rescue": category_rescue,
                    "solo_margin_adv": solo_margin_adv,
                    "solo_category_adv": solo_category_adv,
                    "rescue_joint_score": rescue_score,
                    "solo_joint_score": solo_score,
                }
                neuron_rows.append(row)
                rescue_rows.append(row)

            top_neurons = sorted(
                rescue_rows,
                key=lambda row: (float(row["rescue_joint_score"]), float(row["solo_joint_score"])),
                reverse=True,
            )
            for circuit_kind, count in (("top1", 1), ("top2", 2), ("top3", 3)):
                chosen = [int(row["neuron_index"]) for row in top_neurons[:count] if count <= len(top_neurons)]
                if len(chosen) != count:
                    continue
                circuit_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    chosen,
                    args.signature_top_k,
                )
                circuit_random = sample_random_like(
                    chosen,
                    collector.total_neurons,
                    seed=args.seed + cand_idx * 2000 + count * 101 + 23,
                )
                circuit_random_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    circuit_random,
                    args.signature_top_k,
                )
                margin_adv = float(
                    circuit_eval["effects"]["margin_drop"] - circuit_random_eval["effects"]["margin_drop"]
                )
                category_adv = float(
                    circuit_eval["effects"]["category_margin_drop"]
                    - circuit_random_eval["effects"]["category_margin_drop"]
                )
                joint_adv = effect_score(margin_adv, category_adv, args.score_alpha)
                strict_joint_adv = strict_effect_score(
                    margin_adv,
                    category_adv,
                    args.score_alpha,
                    margin_adv_threshold=args.margin_adv_threshold,
                    margin_adv_penalty=args.margin_adv_penalty,
                )
                circuit_rows.append(
                    {
                        "record_type": "stage5_micro_circuit",
                        "item": cand["item"],
                        "source_kind": cand["source_kind"],
                        "circuit_kind": circuit_kind,
                        "neuron_count": count,
                        "neuron_indices": chosen,
                        "layer_distribution": layer_distribution(chosen, collector.d_ff),
                        "effects": circuit_eval["effects"],
                        "random_effects": circuit_random_eval["effects"],
                        "adv_metrics": {
                            "margin_adv_vs_random": margin_adv,
                            "category_adv_vs_random": category_adv,
                            "joint_adv_score": joint_adv,
                            "strict_joint_adv_score": strict_joint_adv,
                        },
                    }
                )
    finally:
        collector.close()

    positive_circuits = [
        row
        for row in circuit_rows
        if float(row["adv_metrics"]["margin_adv_vs_random"]) > 0.0
        and float(row["adv_metrics"]["category_adv_vs_random"]) > 0.0
    ]
    strict_positive_circuits = [
        row
        for row in circuit_rows
        if float(row["adv_metrics"]["margin_adv_vs_random"]) > args.margin_adv_threshold
        and float(row["adv_metrics"]["category_adv_vs_random"]) > 0.0
    ]
    summary = {
        "record_type": "stage5_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "candidate_count": len(candidate_rows),
        "neuron_row_count": len(neuron_rows),
        "circuit_row_count": len(circuit_rows),
        "positive_micro_circuit_count": len(positive_circuits),
        "selected_categories": selected_categories,
        "max_neurons_per_candidate": args.max_neurons_per_candidate,
        "max_neurons_per_layer": args.max_neurons_per_layer,
        "candidate_overlap_penalty": args.candidate_overlap_penalty,
        "max_candidate_overlap": args.max_candidate_overlap,
        "require_category_coverage": bool(args.require_category_coverage),
        "category_word_penalty": args.category_word_penalty,
        "margin_adv_threshold": args.margin_adv_threshold,
        "margin_adv_penalty": args.margin_adv_penalty,
        "lane_mode": args.lane_mode,
        "prototype_term_mode": args.prototype_term_mode,
        "disable_prototype_proxy": bool(args.disable_prototype_proxy),
        "lane_pool_row_count": len(subset_rows),
        "category_word_candidate_count": int(sum(1 for row in candidate_rows if bool(row["is_category_word"]))),
        "prototype_proxy_candidate_count": int(sum(1 for row in candidate_rows if bool(row.get("is_prototype_proxy")))),
        "strict_real_category_candidate_count": int(
            sum(
                1
                for row in candidate_rows
                if bool(row["is_category_word"]) and not bool(row.get("is_prototype_proxy"))
            )
        ),
        "mean_candidate_full_joint_adv": float(
            np.mean([row["full_joint_adv_score"] for row in candidate_rows]) if candidate_rows else 0.0
        ),
        "mean_candidate_full_strict_joint_adv": float(
            np.mean([row["full_strict_joint_adv_score"] for row in candidate_rows]) if candidate_rows else 0.0
        ),
        "mean_neuron_rescue_joint_score": float(
            np.mean([row["rescue_joint_score"] for row in neuron_rows]) if neuron_rows else 0.0
        ),
        "mean_neuron_solo_joint_score": float(
            np.mean([row["solo_joint_score"] for row in neuron_rows]) if neuron_rows else 0.0
        ),
        "mean_micro_circuit_joint_adv": float(
            np.mean([row["adv_metrics"]["joint_adv_score"] for row in circuit_rows]) if circuit_rows else 0.0
        ),
        "mean_micro_circuit_strict_joint_adv": float(
            np.mean([row["adv_metrics"]["strict_joint_adv_score"] for row in circuit_rows]) if circuit_rows else 0.0
        ),
        "strict_positive_micro_circuit_count": len(strict_positive_circuits),
        "top_micro_circuits": [
            {
                "term": row["item"]["term"],
                "category": row["item"]["category"],
                "source_kind": row["source_kind"],
                "circuit_kind": row["circuit_kind"],
                "neuron_count": row["neuron_count"],
                "margin_adv_vs_random": row["adv_metrics"]["margin_adv_vs_random"],
                "category_adv_vs_random": row["adv_metrics"]["category_adv_vs_random"],
                "joint_adv_score": row["adv_metrics"]["joint_adv_score"],
                "strict_joint_adv_score": row["adv_metrics"]["strict_joint_adv_score"],
            }
            for row in sorted(
                circuit_rows,
                key=lambda x: (
                    float(x["adv_metrics"]["strict_joint_adv_score"]),
                    float(x["adv_metrics"]["joint_adv_score"]),
                ),
                reverse=True,
            )[:20]
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "candidates.jsonl", candidate_rows)
    write_jsonl(out_dir / "neurons.jsonl", neuron_rows)
    write_jsonl(out_dir / "micro_circuits.jsonl", circuit_rows)
    write_report(out_dir / "REPORT.md", summary, circuit_rows)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "candidate_count": len(candidate_rows),
                "neuron_row_count": len(neuron_rows),
                "circuit_row_count": len(circuit_rows),
                "positive_micro_circuit_count": len(positive_circuits),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
