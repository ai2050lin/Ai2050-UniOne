from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
from stage56_synergy_conflict_dissection import (  # noqa: E402
    find_candidate_row,
    partition_union_indices,
    select_target_pair,
)


def generate_remove_sets(
    indices: Sequence[int],
    max_remove: int,
    min_remove: int = 1,
) -> List[Tuple[int, ...]]:
    ordered = [int(x) for x in indices]
    out: List[Tuple[int, ...]] = []
    start_size = max(1, int(min_remove))
    end_size = max(start_size, int(max_remove))
    for size in range(start_size, end_size + 1):
        out.extend(tuple(combo) for combo in itertools.combinations(ordered, size))
    return out


def remove_indices(full: Sequence[int], removed: Sequence[int]) -> List[int]:
    removed_set = {int(x) for x in removed}
    return [int(x) for x in full if int(x) not in removed_set]


def main() -> None:
    ap = argparse.ArgumentParser(description="Search for pruned conflict subsets that flip a negative union back to positive")
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
    ap.add_argument("--conflict-summary", default="tests/codex_temp/stage56_synergy_conflict_dissection_qwen_fruit_apple_20260317/summary.json")
    ap.add_argument("--category", default="fruit")
    ap.add_argument("--instance-term", default="apple")
    ap.add_argument("--top-conflict-neurons", type=int, default=5)
    ap.add_argument("--min-remove", type=int, default=1)
    ap.add_argument("--max-remove", type=int, default=3)
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tests/codex_temp/stage56_conflict_pruned_flip_search_qwen_fruit_apple_20260317")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    baselines = read_jsonl(args.stage3_baselines)
    stage3_summary = read_json(args.stage3_summary)
    prototype_rows = read_jsonl(args.prototype_candidates)
    instance_rows = read_jsonl(args.instance_candidates)
    stage6_rows = read_jsonl(args.stage6_results)
    conflict_summary = read_json(Path(args.conflict_summary))

    target_row = select_target_pair(stage6_rows, category=args.category, instance_term=args.instance_term, prefer_conflict=True)
    category = str(target_row["category"])
    prototype_term = str(target_row["prototype_term"])
    instance_term = str(target_row["instance_term"])

    prototype_row = find_candidate_row(prototype_rows, category, prototype_term)
    instance_row = find_candidate_row(instance_rows, category, instance_term)
    partitions = partition_union_indices(prototype_row["candidate_indices"], instance_row["candidate_indices"])
    union_indices = [int(x) for x in partitions["union"]]

    top_conflict_neurons = [
        int(row["neuron_index"])
        for row in list(conflict_summary.get("top_rescue_neurons", []))[: max(1, args.top_conflict_neurons)]
    ]
    remove_sets = generate_remove_sets(
        top_conflict_neurons,
        max_remove=args.max_remove,
        min_remove=args.min_remove,
    )

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
    result_rows: List[Dict[str, object]] = []

    try:
        proto_joint_adv = float(target_row["proto_joint_adv"])
        instance_joint_adv = float(target_row["instance_joint_adv"])
        proto_effects = dict(target_row["proto_effects"])
        instance_effects = dict(target_row["instance_effects"])

        for combo_idx, removed in enumerate(remove_sets):
            pruned_indices = remove_indices(union_indices, removed)
            if not pruned_indices:
                continue

            pruned_eval = evaluate_ablation(
                model,
                tok,
                collector,
                item,
                baseline_sig,
                baseline_readout,
                category_proto,
                proto_map,
                selected_categories,
                pruned_indices,
                args.signature_top_k,
            )
            pruned_random_eval = evaluate_ablation(
                model,
                tok,
                collector,
                item,
                baseline_sig,
                baseline_readout,
                category_proto,
                proto_map,
                selected_categories,
                sample_random_like(pruned_indices, collector.total_neurons, seed=args.seed + 1000 + combo_idx),
                args.signature_top_k,
            )
            pruned_margin_adv = float(
                pruned_eval["effects"]["margin_drop"] - pruned_random_eval["effects"]["margin_drop"]
            )
            pruned_category_adv = float(
                pruned_eval["effects"]["category_margin_drop"] - pruned_random_eval["effects"]["category_margin_drop"]
            )
            pruned_joint_adv = float(effect_score(pruned_margin_adv, pruned_category_adv, args.score_alpha))

            synergy_margin = float(
                pruned_eval["effects"]["margin_drop"]
                - max(float(proto_effects["margin_drop"]), float(instance_effects["margin_drop"]))
            )
            synergy_category = float(
                pruned_eval["effects"]["category_margin_drop"]
                - max(
                    float(proto_effects["category_margin_drop"]),
                    float(instance_effects["category_margin_drop"]),
                )
            )
            pruned_union_synergy_joint = float(effect_score(synergy_margin, synergy_category, args.score_alpha))
            strict_positive_after_prune = (
                pruned_joint_adv > proto_joint_adv
                and pruned_joint_adv > instance_joint_adv
                and pruned_union_synergy_joint > 0.0
            )

            result_rows.append(
                {
                    "removed_neurons": [int(x) for x in removed],
                    "removed_count": len(removed),
                    "remaining_union_neuron_count": len(pruned_indices),
                    "pruned_joint_adv": pruned_joint_adv,
                    "pruned_category_adv": pruned_category_adv,
                    "pruned_union_synergy_joint": pruned_union_synergy_joint,
                    "joint_gain_vs_original_union": pruned_joint_adv - float(target_row["union_joint_adv"]),
                    "strict_positive_after_prune": strict_positive_after_prune,
                }
            )
    finally:
        collector.close()

    ordered_rows = sorted(
        result_rows,
        key=lambda row: (
            1 if bool(row["strict_positive_after_prune"]) else 0,
            float(row["pruned_joint_adv"]),
            float(row["pruned_union_synergy_joint"]),
        ),
        reverse=True,
    )

    summary = {
        "record_type": "stage56_conflict_pruned_flip_search_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "category": category,
        "prototype_term": prototype_term,
        "instance_term": instance_term,
        "original_proto_joint_adv": float(target_row["proto_joint_adv"]),
        "original_instance_joint_adv": float(target_row["instance_joint_adv"]),
        "original_union_joint_adv": float(target_row["union_joint_adv"]),
        "original_union_synergy_joint": float(target_row["union_synergy_joint"]),
        "top_conflict_neuron_count": len(top_conflict_neurons),
        "min_remove": int(args.min_remove),
        "max_remove": int(args.max_remove),
        "searched_combo_count": len(ordered_rows),
        "strict_positive_flip_count": int(sum(1 for row in ordered_rows if bool(row["strict_positive_after_prune"]))),
        "best_rows": ordered_rows[:10],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_lines = [
        "# Stage56 Conflict Pruned Flip Search Report",
        "",
        f"- Category: {category}",
        f"- Prototype term: {prototype_term}",
        f"- Instance term: {instance_term}",
        f"- Top conflict neurons: {len(top_conflict_neurons)}",
        f"- Remove size range: {int(args.min_remove)}..{int(args.max_remove)}",
        f"- Original union joint adv: {float(target_row['union_joint_adv']):.6f}",
        f"- Original union synergy joint: {float(target_row['union_synergy_joint']):.6f}",
        f"- Strict positive flips: {summary['strict_positive_flip_count']}",
        "",
        "## Best Rows",
    ]
    for row in ordered_rows[:10]:
        report_lines.append(
            "- "
            f"removed={row['removed_neurons']} / pruned_joint={row['pruned_joint_adv']:.6f} "
            f"/ pruned_synergy={row['pruned_union_synergy_joint']:.6f} "
            f"/ gain_vs_union={row['joint_gain_vs_original_union']:.6f} "
            f"/ strict_positive={row['strict_positive_after_prune']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "searched_combo_count": len(ordered_rows),
                "strict_positive_flip_count": summary["strict_positive_flip_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
