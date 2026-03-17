from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage3_causal_closure import (  # noqa: E402
    read_json,
    read_jsonl,
    sample_random_like,
)
from deepseek7b_stage5_readout_coupled_search import (  # noqa: E402
    effect_score,
    evaluate_ablation,
)
from deepseek7b_three_pool_structure_scan import (  # noqa: E402
    GateCollector,
    LexemeItem,
    load_model,
)


def top_rows_by_category(
    rows: Sequence[Dict[str, object]],
    per_category_limit: int,
) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    ordered = sorted(rows, key=lambda row: float(row.get("full_joint_adv_score", 0.0)), reverse=True)
    for row in ordered:
        category = str(row["item"]["category"])
        bucket = out.setdefault(category, [])
        if len(bucket) >= per_category_limit:
            continue
        bucket.append(row)
    return out


def unique_union(left: Sequence[int], right: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for idx in list(left) + list(right):
        value = int(idx)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def paired_categories(
    prototype_by_category: Dict[str, List[Dict[str, object]]],
    instance_by_category: Dict[str, List[Dict[str, object]]],
) -> List[str]:
    return sorted(set(prototype_by_category) & set(instance_by_category))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# DeepSeek Stage6 Prototype-Instance Decomposition Report",
        "",
        f"- Pair count: {summary['pair_count']}",
        f"- Category count: {summary['category_count']}",
        f"- Mean prototype joint adv: {summary['mean_proto_joint_adv']:.6f}",
        f"- Mean instance joint adv: {summary['mean_instance_joint_adv']:.6f}",
        f"- Mean union joint adv: {summary['mean_union_joint_adv']:.6f}",
        f"- Mean union synergy joint: {summary['mean_union_synergy_joint']:.6f}",
        "",
        "## Top Pair Rows",
    ]
    top_rows = sorted(rows, key=lambda row: float(row["union_joint_adv"]), reverse=True)[:20]
    for row in top_rows:
        lines.append(
            "- "
            f"{row['category']} / proto={row['prototype_term']} / inst={row['instance_term']} "
            f"/ proto_joint={row['proto_joint_adv']:.6f}"
            f" / inst_joint={row['instance_joint_adv']:.6f}"
            f" / union_joint={row['union_joint_adv']:.6f}"
            f" / union_synergy={row['union_synergy_joint']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek stage6 prototype-instance decomposition")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--stage2-families", default="tempdata/deepseek7b_three_pool_stage2_focus_cleanup_1504_bf16_20260317/families.jsonl")
    ap.add_argument("--stage3-summary", default="tempdata/deepseek7b_stage3_causal_closure_cleanup_1504_20260317/summary.json")
    ap.add_argument("--stage3-baselines", default="tempdata/deepseek7b_stage3_causal_closure_cleanup_1504_20260317/baselines.jsonl")
    ap.add_argument("--prototype-candidates", default="tempdata/deepseek7b_stage5_readout_coupled_cleanup_debiased_prototype_1504_20260317/candidates.jsonl")
    ap.add_argument("--instance-candidates", default="tempdata/deepseek7b_stage5_readout_coupled_cleanup_debiased_instance_lane_1504_20260317/candidates.jsonl")
    ap.add_argument("--max-instance-terms-per-category", type=int, default=2)
    ap.add_argument("--signature-top-k", type=int, default=256)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tempdata/deepseek7b_stage6_prototype_instance_decomposition_1504_20260317")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    stage3_summary = read_json(args.stage3_summary)
    baselines = read_jsonl(args.stage3_baselines)
    prototype_rows = read_jsonl(args.prototype_candidates)
    instance_rows = read_jsonl(args.instance_candidates)

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
    prototype_by_category = top_rows_by_category(prototype_rows, per_category_limit=1)
    instance_by_category = top_rows_by_category(
        instance_rows,
        per_category_limit=max(1, args.max_instance_terms_per_category),
    )
    categories = paired_categories(prototype_by_category, instance_by_category)

    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)
    result_rows: List[Dict[str, object]] = []

    try:
        for cat_idx, category in enumerate(categories):
            category_proto = proto_map[category]
            prototype_row = prototype_by_category[category][0]
            prototype_indices = [int(x) for x in prototype_row["candidate_indices"]]
            for inst_idx, instance_row in enumerate(instance_by_category[category]):
                term = str(instance_row["item"]["term"])
                baseline_row = baseline_map[(term, category)]
                baseline_sig = baseline_row["baseline_signature"]
                baseline_readout = baseline_row["baseline_readout"]
                item = LexemeItem(term=term, category=category, language="ascii")

                instance_indices = [int(x) for x in instance_row["candidate_indices"]]
                union_indices = unique_union(prototype_indices, instance_indices)

                proto_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    prototype_indices,
                    args.signature_top_k,
                )
                proto_random_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    sample_random_like(
                        prototype_indices,
                        collector.total_neurons,
                        seed=args.seed + cat_idx * 100 + inst_idx * 11 + 1,
                    ),
                    args.signature_top_k,
                )
                instance_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    instance_indices,
                    args.signature_top_k,
                )
                instance_random_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    sample_random_like(
                        instance_indices,
                        collector.total_neurons,
                        seed=args.seed + cat_idx * 100 + inst_idx * 11 + 2,
                    ),
                    args.signature_top_k,
                )
                union_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    union_indices,
                    args.signature_top_k,
                )
                union_random_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    sample_random_like(
                        union_indices,
                        collector.total_neurons,
                        seed=args.seed + cat_idx * 100 + inst_idx * 11 + 3,
                    ),
                    args.signature_top_k,
                )

                proto_margin_adv = float(proto_eval["effects"]["margin_drop"] - proto_random_eval["effects"]["margin_drop"])
                proto_category_adv = float(
                    proto_eval["effects"]["category_margin_drop"] - proto_random_eval["effects"]["category_margin_drop"]
                )
                instance_margin_adv = float(
                    instance_eval["effects"]["margin_drop"] - instance_random_eval["effects"]["margin_drop"]
                )
                instance_category_adv = float(
                    instance_eval["effects"]["category_margin_drop"]
                    - instance_random_eval["effects"]["category_margin_drop"]
                )
                union_margin_adv = float(
                    union_eval["effects"]["margin_drop"] - union_random_eval["effects"]["margin_drop"]
                )
                union_category_adv = float(
                    union_eval["effects"]["category_margin_drop"] - union_random_eval["effects"]["category_margin_drop"]
                )
                proto_joint_adv = effect_score(proto_margin_adv, proto_category_adv, args.score_alpha)
                instance_joint_adv = effect_score(instance_margin_adv, instance_category_adv, args.score_alpha)
                union_joint_adv = effect_score(union_margin_adv, union_category_adv, args.score_alpha)

                synergy_margin = float(
                    union_eval["effects"]["margin_drop"]
                    - max(float(proto_eval["effects"]["margin_drop"]), float(instance_eval["effects"]["margin_drop"]))
                )
                synergy_category = float(
                    union_eval["effects"]["category_margin_drop"]
                    - max(
                        float(proto_eval["effects"]["category_margin_drop"]),
                        float(instance_eval["effects"]["category_margin_drop"]),
                    )
                )
                union_synergy_joint = effect_score(synergy_margin, synergy_category, args.score_alpha)

                result_rows.append(
                    {
                        "record_type": "stage6_decomposition_result",
                        "category": category,
                        "prototype_term": str(prototype_row["item"]["term"]),
                        "instance_term": term,
                        "prototype_neuron_count": len(prototype_indices),
                        "instance_neuron_count": len(instance_indices),
                        "union_neuron_count": len(union_indices),
                        "overlap_neuron_count": len(set(prototype_indices) & set(instance_indices)),
                        "proto_joint_adv": proto_joint_adv,
                        "instance_joint_adv": instance_joint_adv,
                        "union_joint_adv": union_joint_adv,
                        "union_synergy_joint": union_synergy_joint,
                        "proto_margin_adv": proto_margin_adv,
                        "instance_margin_adv": instance_margin_adv,
                        "union_margin_adv": union_margin_adv,
                        "proto_category_adv": proto_category_adv,
                        "instance_category_adv": instance_category_adv,
                        "union_category_adv": union_category_adv,
                        "proto_effects": proto_eval["effects"],
                        "instance_effects": instance_eval["effects"],
                        "union_effects": union_eval["effects"],
                    }
                )
    finally:
        collector.close()

    summary = {
        "record_type": "stage6_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "category_count": len(categories),
        "pair_count": len(result_rows),
        "categories": categories,
        "mean_proto_joint_adv": float(np.mean([row["proto_joint_adv"] for row in result_rows]) if result_rows else 0.0),
        "mean_instance_joint_adv": float(np.mean([row["instance_joint_adv"] for row in result_rows]) if result_rows else 0.0),
        "mean_union_joint_adv": float(np.mean([row["union_joint_adv"] for row in result_rows]) if result_rows else 0.0),
        "mean_union_synergy_joint": float(np.mean([row["union_synergy_joint"] for row in result_rows]) if result_rows else 0.0),
        "top_pairs": [
            {
                "category": row["category"],
                "prototype_term": row["prototype_term"],
                "instance_term": row["instance_term"],
                "proto_joint_adv": row["proto_joint_adv"],
                "instance_joint_adv": row["instance_joint_adv"],
                "union_joint_adv": row["union_joint_adv"],
                "union_synergy_joint": row["union_synergy_joint"],
            }
            for row in sorted(result_rows, key=lambda row: row["union_joint_adv"], reverse=True)[:20]
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "results.jsonl", result_rows)
    write_report(out_dir / "REPORT.md", summary, result_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "pair_count": len(result_rows),
                "category_count": len(categories),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
