from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage3_causal_closure import (  # noqa: E402
    FocusTerm,
    analyze_family_effect,
    category_readout,
    compute_signature,
    load_focus_terms,
    read_json,
    read_jsonl,
    register_ablation,
    remove_handles,
    sample_random_like,
)
from deepseek7b_three_pool_structure_scan import (  # noqa: E402
    GateCollector,
    LexemeItem,
    layer_distribution,
    load_model,
)


def parse_subset_sizes(text: str) -> List[int]:
    out = []
    seen = set()
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        value = int(s)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def pick_subset_sizes(total: int, requested: Sequence[int]) -> List[int]:
    out = []
    seen = set()
    for value in requested:
        if value <= 0 or value >= total:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if total > 1 and (total - 1) not in seen:
        out.insert(0, total - 1)
    return out


def baseline_importance(signature: Dict[str, object]) -> Dict[int, float]:
    return {
        int(idx): float(val)
        for idx, val in zip(signature["signature_top_indices"], signature["signature_top_values"])
    }


def index_frequency(rows: Sequence[Dict[str, object]]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for row in rows:
        for idx in row.get("intervention", {}).get("flat_indices", []):
            counts[int(idx)] += 1
    return counts


def common_index_set(
    counts: Dict[int, int],
    total_rows: int,
    max_fraction: float,
) -> set[int]:
    if total_rows <= 0 or max_fraction <= 0.0:
        return set()
    threshold = float(total_rows) * float(max_fraction)
    return {int(idx) for idx, count in counts.items() if float(count) >= threshold}


def rank_neurons_by_baseline(
    indices: Sequence[int],
    signature: Dict[str, object],
    d_ff: int,
    common_indices: set[int] | None = None,
) -> List[int]:
    importance = baseline_importance(signature)
    common_indices = common_indices or set()
    uniq = sorted({int(x) for x in indices})
    return sorted(
        uniq,
        key=lambda idx: (
            0 if idx in common_indices else 1,
            1 if idx in importance else 0,
            importance.get(idx, 0.0),
            idx // d_ff,
            -idx,
        ),
        reverse=True,
    )


def index_rows(
    rows: Sequence[Dict[str, object]],
    key_fields: Sequence[str],
) -> Dict[Tuple[str, ...], Dict[str, object]]:
    out: Dict[Tuple[str, ...], Dict[str, object]] = {}
    for row in rows:
        key_parts: List[str] = []
        for field in key_fields:
            if field == "term":
                key_parts.append(str(row["item"]["term"]))
            elif field == "category":
                key_parts.append(str(row["item"]["category"]))
            elif field == "kind":
                key_parts.append(str(row["intervention"]["kind"]))
            else:
                raise KeyError(field)
        out[tuple(key_parts)] = row
    return out


def nested_mean(rows: Sequence[Dict[str, object]], value_key: str) -> float:
    return float(np.mean([float(row[value_key]) for row in rows])) if rows else 0.0


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# DeepSeek Stage4 Minimal Circuit Search Report",
        "",
        f"- Family count: {summary['family_count']}",
        f"- Term count: {summary['term_count']}",
        f"- Pair count: {summary['evaluation_pair_count']}",
        f"- Result rows: {summary['result_row_count']}",
        f"- Margin-preserving hits: {summary['margin_preserving_hit_count']}",
        f"- Joint-binding hits: {summary['joint_binding_hit_count']}",
        "",
        "## Top Joint Candidates",
    ]
    top_rows = sorted(rows, key=lambda row: float(row["pair_metrics"]["joint_adv_score"]), reverse=True)[:20]
    for row in top_rows:
        lines.append(
            "- "
            f"{row['item']['category']} / {row['item']['term']} / {row['source_kind']} / size={row['subset_size']} "
            f"/ joint_adv={row['pair_metrics']['joint_adv_score']:.6f} "
            f"/ margin_adv={row['pair_metrics']['margin_adv_vs_random']:.6f} "
            f"/ category_adv={row['pair_metrics']['category_adv_vs_random']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek stage4 minimal circuit search")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--focus-manifest", default="tempdata/deepseek7b_stage2_focus_520_20260316/focus_manifest.json")
    ap.add_argument("--stage2-families", default="tempdata/deepseek7b_three_pool_stage2_focus_520_bf16_20260316/families.jsonl")
    ap.add_argument("--stage3-summary", default="tempdata/deepseek7b_stage3_causal_closure_520_20260316/summary.json")
    ap.add_argument("--stage3-baselines", default="tempdata/deepseek7b_stage3_causal_closure_520_20260316/baselines.jsonl")
    ap.add_argument("--stage3-interventions", default="tempdata/deepseek7b_stage3_causal_closure_520_20260316/interventions.jsonl")
    ap.add_argument("--source-kinds", default="family_shared,combined")
    ap.add_argument("--subset-sizes", default="48,32,24,16,12,8,6,4")
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--margin-preserve-threshold", type=float, default=0.8)
    ap.add_argument("--global-common-max-fraction", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage4_minimal_circuit_520_20260316")
    args = ap.parse_args()

    t0 = time.time()
    focus_terms = load_focus_terms(args.focus_manifest)
    stage3_summary = read_json(args.stage3_summary)
    families = read_jsonl(args.stage2_families)
    baselines = read_jsonl(args.stage3_baselines)
    interventions = read_jsonl(args.stage3_interventions)

    selected_categories = [str(x) for x in stage3_summary["selected_categories"]]
    proto_map = {
        str(row["category"]): row
        for row in families
        if str(row.get("pool")) == "closure"
        if str(row["category"]) in selected_categories
    }
    baseline_map = index_rows(baselines, ("term", "category"))
    intervention_map = index_rows(interventions, ("term", "category", "kind"))
    source_kinds = [x.strip() for x in args.source_kinds.split(",") if x.strip()]
    subset_sizes = parse_subset_sizes(args.subset_sizes)
    intervention_counts = index_frequency(interventions)
    common_indices = common_index_set(
        intervention_counts,
        total_rows=len(interventions),
        max_fraction=args.global_common_max_fraction,
    )

    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)
    result_rows: List[Dict[str, object]] = []

    try:
        for family_idx, category in enumerate(selected_categories):
            if category not in proto_map or category not in focus_terms:
                continue
            category_proto = proto_map[category]
            term_rows = list(focus_terms[category])
            for term_idx, focus_term in enumerate(term_rows):
                base_key = (focus_term.term, focus_term.category)
                baseline_row = baseline_map.get(base_key)
                if not baseline_row:
                    continue
                baseline_sig = baseline_row["baseline_signature"]
                baseline_readout = baseline_row["baseline_readout"]
                item = LexemeItem(term=focus_term.term, category=focus_term.category, language="ascii")
                for source_kind in source_kinds:
                    full_row = intervention_map.get((focus_term.term, focus_term.category, source_kind))
                    if not full_row:
                        continue
                    full_indices = [int(x) for x in full_row["intervention"]["flat_indices"]]
                    ranked_indices = rank_neurons_by_baseline(
                        full_indices,
                        baseline_sig,
                        collector.d_ff,
                        common_indices=common_indices,
                    )
                    actual_sizes = pick_subset_sizes(len(ranked_indices), subset_sizes)
                    full_margin_drop = float(full_row["effects"]["margin_drop"])
                    full_category_drop = float(full_row["effects"]["category_margin_drop"])
                    for size_idx, subset_size in enumerate(actual_sizes):
                        subset = ranked_indices[:subset_size]
                        random_subset = sample_random_like(
                            subset,
                            collector.total_neurons,
                            seed=args.seed + family_idx * 1009 + term_idx * 97 + size_idx * 17 + len(source_kind),
                        )

                        pair_rows = []
                        for variant, indices in (("subset", subset), ("random", random_subset)):
                            handles = register_ablation(model, indices, collector.d_ff) if indices else []
                            try:
                                ablated_sig = compute_signature(model, tok, collector, item, top_k=args.signature_top_k)
                                ablated_readout = category_readout(
                                    model,
                                    tok,
                                    term=focus_term.term,
                                    correct_category=focus_term.category,
                                    all_categories=selected_categories,
                                )
                            finally:
                                remove_handles(handles)

                            effects = analyze_family_effect(
                                item=focus_term,
                                category_proto=category_proto,
                                all_protos_by_category=proto_map,
                                baseline_sig=baseline_sig,
                                ablated_sig=ablated_sig,
                                baseline_readout=baseline_readout,
                                ablated_readout=ablated_readout,
                            )
                            pair_rows.append(
                                {
                                    "record_type": "stage4_minimal_circuit_result",
                                    "item": {
                                        "term": focus_term.term,
                                        "category": focus_term.category,
                                        "role": focus_term.role,
                                    },
                                    "source_kind": source_kind,
                                    "variant": variant,
                                    "subset_size": subset_size,
                                    "full_neuron_count": len(ranked_indices),
                                    "subset_layer_distribution": layer_distribution(indices, collector.d_ff),
                                    "subset_flat_indices": [int(x) for x in indices],
                                    "effects": effects,
                                    "full_reference": {
                                        "margin_drop": full_margin_drop,
                                        "category_margin_drop": full_category_drop,
                                    },
                                    "margin_preserve_ratio": float(
                                        effects["margin_drop"] / full_margin_drop
                                    )
                                    if full_margin_drop > 1e-12
                                    else 0.0,
                                    "category_preserve_ratio": float(
                                        effects["category_margin_drop"] / full_category_drop
                                    )
                                    if abs(full_category_drop) > 1e-12
                                    else None,
                                }
                            )

                        subset_row = pair_rows[0]
                        random_row = pair_rows[1]
                        margin_adv = float(subset_row["effects"]["margin_drop"] - random_row["effects"]["margin_drop"])
                        category_adv = float(
                            subset_row["effects"]["category_margin_drop"] - random_row["effects"]["category_margin_drop"]
                        )
                        joint_adv = float(margin_adv + 256.0 * category_adv)
                        joint_binding = bool(
                            subset_row["margin_preserve_ratio"] >= args.margin_preserve_threshold
                            and margin_adv > 0.0
                            and category_adv > 0.0
                        )
                        for row in pair_rows:
                            row["pair_metrics"] = {
                                "margin_adv_vs_random": margin_adv,
                                "category_adv_vs_random": category_adv,
                                "joint_adv_score": joint_adv,
                                "joint_binding_hit": joint_binding,
                            }
                            result_rows.append(row)
    finally:
        collector.close()

    subset_only = [row for row in result_rows if row["variant"] == "subset"]
    margin_hits = [
        row
        for row in subset_only
        if float(row["margin_preserve_ratio"]) >= args.margin_preserve_threshold
    ]
    joint_hits = [row for row in subset_only if bool(row["pair_metrics"]["joint_binding_hit"])]

    summary = {
        "record_type": "stage4_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "family_count": len(selected_categories),
        "term_count": len({(row['item']['term'], row['item']['category']) for row in subset_only}),
        "evaluation_pair_count": len(subset_only),
        "result_row_count": len(result_rows),
        "selected_categories": selected_categories,
        "source_kinds": source_kinds,
        "subset_sizes": subset_sizes,
        "global_common_max_fraction": args.global_common_max_fraction,
        "global_common_index_count": len(common_indices),
        "margin_preserving_hit_count": len(margin_hits),
        "joint_binding_hit_count": len(joint_hits),
        "mean_subset_margin_drop": nested_mean(
            [{"value": row["effects"]["margin_drop"]} for row in subset_only],
            "value",
        ),
        "mean_subset_category_margin_drop": nested_mean(
            [{"value": row["effects"]["category_margin_drop"]} for row in subset_only],
            "value",
        ),
        "mean_margin_adv_vs_random": nested_mean(
            [{"value": row["pair_metrics"]["margin_adv_vs_random"]} for row in subset_only],
            "value",
        ),
        "mean_category_adv_vs_random": nested_mean(
            [{"value": row["pair_metrics"]["category_adv_vs_random"]} for row in subset_only],
            "value",
        ),
        "top_joint_candidates": [
            {
                "term": row["item"]["term"],
                "category": row["item"]["category"],
                "source_kind": row["source_kind"],
                "subset_size": row["subset_size"],
                "margin_preserve_ratio": row["margin_preserve_ratio"],
                "margin_adv_vs_random": row["pair_metrics"]["margin_adv_vs_random"],
                "category_adv_vs_random": row["pair_metrics"]["category_adv_vs_random"],
                "joint_adv_score": row["pair_metrics"]["joint_adv_score"],
            }
            for row in sorted(subset_only, key=lambda x: float(x["pair_metrics"]["joint_adv_score"]), reverse=True)[:20]
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "results.jsonl", result_rows)
    write_report(out_dir / "REPORT.md", summary, subset_only)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "result_row_count": len(result_rows),
                "evaluation_pair_count": len(subset_only),
                "joint_binding_hit_count": len(joint_hits),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
