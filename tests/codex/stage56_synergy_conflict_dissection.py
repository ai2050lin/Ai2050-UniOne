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
    layer_distribution,
    load_model,
)


def select_target_pair(
    rows: Sequence[Dict[str, object]],
    category: str = "",
    instance_term: str = "",
    prefer_conflict: bool = True,
) -> Dict[str, object]:
    filtered = list(rows)
    if category:
        filtered = [row for row in filtered if str(row.get("category")) == category]
    if instance_term:
        filtered = [row for row in filtered if str(row.get("instance_term")) == instance_term]
    if not filtered:
        raise ValueError("no matching stage6 target row found")
    if prefer_conflict:
        return min(
            filtered,
            key=lambda row: (
                float(row.get("union_synergy_joint", 0.0)),
                float(row.get("union_joint_adv", 0.0)),
            ),
        )
    return max(
        filtered,
        key=lambda row: (
            float(row.get("union_joint_adv", 0.0)),
            float(row.get("union_synergy_joint", 0.0)),
        ),
    )


def find_candidate_row(rows: Sequence[Dict[str, object]], category: str, term: str) -> Dict[str, object]:
    for row in rows:
        item = row.get("item", {})
        if str(item.get("category")) == category and str(item.get("term")) == term:
            return row
    raise ValueError(f"candidate row not found: {category} / {term}")


def partition_union_indices(prototype_indices: Sequence[int], instance_indices: Sequence[int]) -> Dict[str, List[int]]:
    proto = [int(x) for x in prototype_indices]
    inst = [int(x) for x in instance_indices]
    proto_set = set(proto)
    inst_set = set(inst)
    return {
        "prototype_only": [idx for idx in proto if idx not in inst_set],
        "instance_only": [idx for idx in inst if idx not in proto_set],
        "overlap": [idx for idx in proto if idx in inst_set],
        "union": list(dict.fromkeys(proto + inst)),
    }


def joint_from_eval(eval_result: Dict[str, object], random_eval: Dict[str, object], alpha: float) -> Dict[str, float]:
    margin_adv = float(eval_result["effects"]["margin_drop"] - random_eval["effects"]["margin_drop"])
    category_adv = float(
        eval_result["effects"]["category_margin_drop"] - random_eval["effects"]["category_margin_drop"]
    )
    return {
        "margin_adv": margin_adv,
        "category_adv": category_adv,
        "joint_adv": float(effect_score(margin_adv, category_adv, alpha)),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(summary: Dict[str, object], neuron_rows: Sequence[Dict[str, object]], cross_rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Stage56 Synergy Conflict Dissection Report",
        "",
        f"- Category: {summary['category']}",
        f"- Prototype term: {summary['prototype_term']}",
        f"- Instance term: {summary['instance_term']}",
        f"- Prototype joint adv: {summary['proto_joint_adv']:.6f}",
        f"- Instance joint adv: {summary['instance_joint_adv']:.6f}",
        f"- Union joint adv: {summary['union_joint_adv']:.6f}",
        f"- Union synergy joint: {summary['union_synergy_joint']:.6f}",
        "",
        "## Top Rescue Neurons",
    ]
    for row in sorted(neuron_rows, key=lambda item: float(item["union_rescue_joint"]), reverse=True)[:10]:
        lines.append(
            "- "
            f"group={row['group']} / neuron={row['neuron_index']} / rescue_joint={row['union_rescue_joint']:.6f} "
            f"/ loo_joint={row['loo_joint_adv']:.6f} / solo_joint={row['solo_joint_adv']:.6f}"
        )
    lines.extend(["", "## Top Cross Pairs"])
    for row in sorted(cross_rows, key=lambda item: float(item["pair_joint_adv"]), reverse=True)[:10]:
        lines.append(
            "- "
            f"proto={row['prototype_neuron']} / inst={row['instance_neuron']} / pair_joint={row['pair_joint_adv']:.6f} "
            f"/ pair_category_adv={row['pair_category_adv']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Dissect a negative-synergy or conflict pair at single-neuron level")
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--stage2-families", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage1_three_pool/families.jsonl")
    ap.add_argument("--stage3-summary", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage3_causal_closure/summary.json")
    ap.add_argument("--stage3-baselines", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage3_causal_closure/baselines.jsonl")
    ap.add_argument("--prototype-candidates", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage5_prototype/candidates.jsonl")
    ap.add_argument("--instance-candidates", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage5_instance/candidates.jsonl")
    ap.add_argument("--stage6-results", default="tempdata/stage56_fruit_family_closure_block_real_20260317/qwen3_4b/stage6_prototype_instance_decomposition/results.jsonl")
    ap.add_argument("--category", default="fruit")
    ap.add_argument("--instance-term", default="apple")
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tests/codex_temp/stage56_synergy_conflict_dissection_qwen_fruit_apple_20260317")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    baselines = read_jsonl(args.stage3_baselines)
    stage3_summary = read_json(args.stage3_summary)
    prototype_rows = read_jsonl(args.prototype_candidates)
    instance_rows = read_jsonl(args.instance_candidates)
    stage6_rows = read_jsonl(args.stage6_results)

    target_row = select_target_pair(stage6_rows, category=args.category, instance_term=args.instance_term, prefer_conflict=True)
    category = str(target_row["category"])
    prototype_term = str(target_row["prototype_term"])
    instance_term = str(target_row["instance_term"])

    prototype_row = find_candidate_row(prototype_rows, category, prototype_term)
    instance_row = find_candidate_row(instance_rows, category, instance_term)
    partitions = partition_union_indices(prototype_row["candidate_indices"], instance_row["candidate_indices"])

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
    baseline_row = baseline_map[(instance_term, category)]
    baseline_sig = baseline_row["baseline_signature"]
    baseline_readout = baseline_row["baseline_readout"]
    category_proto = proto_map[category]
    item = LexemeItem(term=instance_term, category=category, language="ascii")

    model, tok, model_ref = load_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    collector = GateCollector(model)
    neuron_rows: List[Dict[str, object]] = []
    cross_rows: List[Dict[str, object]] = []

    try:
        union_indices = [int(x) for x in partitions["union"]]
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
            sample_random_like(union_indices, collector.total_neurons, seed=args.seed + 1),
            args.signature_top_k,
        )
        union_joint = joint_from_eval(union_eval, union_random_eval, args.score_alpha)

        group_by_neuron: Dict[int, str] = {}
        for group_name in ("prototype_only", "instance_only", "overlap"):
            for neuron in partitions[group_name]:
                group_by_neuron[int(neuron)] = group_name

        for offset, neuron in enumerate(union_indices):
            loo_indices = [idx for idx in union_indices if idx != neuron]
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
            )
            loo_random_eval = evaluate_ablation(
                model,
                tok,
                collector,
                item,
                baseline_sig,
                baseline_readout,
                category_proto,
                proto_map,
                selected_categories,
                sample_random_like(loo_indices, collector.total_neurons, seed=args.seed + 100 + offset),
                args.signature_top_k,
            ) if loo_indices else union_random_eval
            loo_joint = joint_from_eval(loo_eval, loo_random_eval, args.score_alpha)

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
                sample_random_like([neuron], collector.total_neurons, seed=args.seed + 200 + offset),
                args.signature_top_k,
            )
            solo_joint = joint_from_eval(solo_eval, solo_random_eval, args.score_alpha)
            neuron_rows.append(
                {
                    "neuron_index": int(neuron),
                    "group": group_by_neuron.get(int(neuron), "union_only"),
                    "layer_distribution": layer_distribution([neuron], collector.d_ff),
                    "union_rescue_joint": float(loo_joint["joint_adv"] - union_joint["joint_adv"]),
                    "union_rescue_category_adv": float(loo_joint["category_adv"] - union_joint["category_adv"]),
                    "loo_joint_adv": float(loo_joint["joint_adv"]),
                    "solo_joint_adv": float(solo_joint["joint_adv"]),
                    "solo_category_adv": float(solo_joint["category_adv"]),
                }
            )

        for proto_offset, proto_neuron in enumerate(partitions["prototype_only"]):
            for inst_offset, inst_neuron in enumerate(partitions["instance_only"]):
                pair_indices = [int(proto_neuron), int(inst_neuron)]
                pair_eval = evaluate_ablation(
                    model,
                    tok,
                    collector,
                    item,
                    baseline_sig,
                    baseline_readout,
                    category_proto,
                    proto_map,
                    selected_categories,
                    pair_indices,
                    args.signature_top_k,
                )
                pair_random_eval = evaluate_ablation(
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
                        pair_indices,
                        collector.total_neurons,
                        seed=args.seed + 500 + proto_offset * 31 + inst_offset,
                    ),
                    args.signature_top_k,
                )
                pair_joint = joint_from_eval(pair_eval, pair_random_eval, args.score_alpha)
                cross_rows.append(
                    {
                        "prototype_neuron": int(proto_neuron),
                        "instance_neuron": int(inst_neuron),
                        "pair_joint_adv": float(pair_joint["joint_adv"]),
                        "pair_margin_adv": float(pair_joint["margin_adv"]),
                        "pair_category_adv": float(pair_joint["category_adv"]),
                    }
                )
    finally:
        collector.close()

    summary = {
        "record_type": "stage56_synergy_conflict_dissection_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "category": category,
        "prototype_term": prototype_term,
        "instance_term": instance_term,
        "proto_joint_adv": float(target_row["proto_joint_adv"]),
        "instance_joint_adv": float(target_row["instance_joint_adv"]),
        "union_joint_adv": float(target_row["union_joint_adv"]),
        "union_synergy_joint": float(target_row["union_synergy_joint"]),
        "prototype_neuron_count": len(prototype_row["candidate_indices"]),
        "instance_neuron_count": len(instance_row["candidate_indices"]),
        "prototype_only_count": len(partitions["prototype_only"]),
        "instance_only_count": len(partitions["instance_only"]),
        "overlap_count": len(partitions["overlap"]),
        "top_rescue_neurons": sorted(neuron_rows, key=lambda row: float(row["union_rescue_joint"]), reverse=True)[:10],
        "top_cross_pairs": sorted(cross_rows, key=lambda row: float(row["pair_joint_adv"]), reverse=True)[:10],
        "mean_union_rescue_joint": float(np.mean([row["union_rescue_joint"] for row in neuron_rows]) if neuron_rows else 0.0),
        "positive_rescue_neuron_count": int(sum(1 for row in neuron_rows if float(row["union_rescue_joint"]) > 0.0)),
        "mean_cross_pair_joint_adv": float(np.mean([row["pair_joint_adv"] for row in cross_rows]) if cross_rows else 0.0),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "rescue_neurons.jsonl", neuron_rows)
    write_jsonl(out_dir / "cross_pairs.jsonl", cross_rows)
    (out_dir / "REPORT.md").write_text(build_report(summary, neuron_rows, cross_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "rescue_neuron_count": len(neuron_rows),
                "cross_pair_count": len(cross_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
