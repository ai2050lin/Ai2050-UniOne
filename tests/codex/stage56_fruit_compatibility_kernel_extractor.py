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
    layer_distribution,
    load_model,
)
from stage56_synergy_conflict_dissection import (  # noqa: E402
    find_candidate_row,
    partition_union_indices,
)


def read_success_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        rows.extend(read_jsonl(path))
    return [row for row in rows if bool(row.get("strict_positive_after_prune"))]


def remaining_kernel(union_indices: Sequence[int], removed_neurons: Sequence[int]) -> Tuple[int, ...]:
    removed = {int(x) for x in removed_neurons}
    return tuple(int(idx) for idx in union_indices if int(idx) not in removed)


def unique_kernels(success_rows: Sequence[Dict[str, object]], union_indices: Sequence[int]) -> List[Tuple[int, ...]]:
    seen = set()
    out: List[Tuple[int, ...]] = []
    for row in success_rows:
        kernel = remaining_kernel(union_indices, row.get("removed_neurons", []))
        if kernel and kernel not in seen:
            seen.add(kernel)
            out.append(kernel)
    return out


def iter_proper_subsets(kernel: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    ordered = [int(x) for x in kernel]
    for size in range(1, len(ordered)):
        for combo in itertools.combinations(ordered, size):
            yield tuple(combo)


def neuron_support_counts(kernels: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    counts: Dict[int, int] = {}
    for kernel in kernels:
        for neuron in kernel:
            counts[int(neuron)] = counts.get(int(neuron), 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [
        {
            "neuron_index": int(neuron),
            "support_count": int(count),
        }
        for neuron, count in ordered
    ]


def kernel_signature(kernel: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in kernel)


def evaluate_kernel(
    *,
    model,
    tok,
    collector: GateCollector,
    item: LexemeItem,
    baseline_sig,
    baseline_readout,
    category_proto: Dict[str, object],
    proto_map: Dict[str, object],
    selected_categories: Sequence[str],
    proto_joint_adv: float,
    instance_joint_adv: float,
    proto_effects: Dict[str, object],
    instance_effects: Dict[str, object],
    kernel: Sequence[int],
    signature_top_k: int,
    score_alpha: float,
    seed: int,
    random_trials: int,
) -> Dict[str, object]:
    kernel_indices = [int(x) for x in kernel]
    eval_result = evaluate_ablation(
        model,
        tok,
        collector,
        item,
        baseline_sig,
        baseline_readout,
        category_proto,
        proto_map,
        selected_categories,
        kernel_indices,
        signature_top_k,
    )
    joint_adv_trials: List[float] = []
    category_adv_trials: List[float] = []
    for trial_idx in range(max(1, int(random_trials))):
        random_eval = evaluate_ablation(
            model,
            tok,
            collector,
            item,
            baseline_sig,
            baseline_readout,
            category_proto,
            proto_map,
            selected_categories,
            sample_random_like(kernel_indices, collector.total_neurons, seed=seed + trial_idx),
            signature_top_k,
        )
        margin_adv = float(eval_result["effects"]["margin_drop"] - random_eval["effects"]["margin_drop"])
        category_adv = float(
            eval_result["effects"]["category_margin_drop"] - random_eval["effects"]["category_margin_drop"]
        )
        joint_adv_trials.append(float(effect_score(margin_adv, category_adv, score_alpha)))
        category_adv_trials.append(category_adv)

    joint_adv = float(np.mean(joint_adv_trials))
    category_adv = float(np.mean(category_adv_trials))
    synergy_margin = float(
        eval_result["effects"]["margin_drop"]
        - max(float(proto_effects["margin_drop"]), float(instance_effects["margin_drop"]))
    )
    synergy_category = float(
        eval_result["effects"]["category_margin_drop"]
        - max(float(proto_effects["category_margin_drop"]), float(instance_effects["category_margin_drop"]))
    )
    synergy_joint = float(effect_score(synergy_margin, synergy_category, score_alpha))
    strict_positive_mean = joint_adv > proto_joint_adv and joint_adv > instance_joint_adv and synergy_joint > 0.0
    strict_positive_all_trials = (
        synergy_joint > 0.0
        and all(value > proto_joint_adv for value in joint_adv_trials)
        and all(value > instance_joint_adv for value in joint_adv_trials)
    )
    return {
        "kernel": kernel_indices,
        "kernel_size": len(kernel_indices),
        "kernel_signature": kernel_signature(kernel_indices),
        "joint_adv": joint_adv,
        "joint_adv_min": float(np.min(joint_adv_trials)),
        "joint_adv_max": float(np.max(joint_adv_trials)),
        "joint_adv_std": float(np.std(joint_adv_trials)),
        "category_adv": category_adv,
        "synergy_joint": synergy_joint,
        "strict_positive": strict_positive_mean,
        "strict_positive_all_trials": strict_positive_all_trials,
        "random_trial_count": int(max(1, int(random_trials))),
        "layer_distribution": layer_distribution(kernel_indices, collector.d_ff),
    }


def build_report(
    summary: Dict[str, object],
    kernel_rows: Sequence[Dict[str, object]],
    subset_rows: Sequence[Dict[str, object]],
    support_rows: Sequence[Dict[str, object]],
) -> str:
    lines = [
        "# Stage56 Fruit Compatibility Kernel Extractor Report",
        "",
        f"- Category: {summary['category']}",
        f"- Prototype term: {summary['prototype_term']}",
        f"- Instance term: {summary['instance_term']}",
        f"- Original union joint adv: {summary['original_union_joint_adv']:.6f}",
        f"- Original union synergy joint: {summary['original_union_synergy_joint']:.6f}",
        f"- Successful prune rows: {summary['successful_prune_row_count']}",
        f"- Unique candidate kernels: {summary['unique_kernel_count']}",
        f"- Minimal robust kernels: {summary['minimal_strict_positive_kernel_count']}",
        "",
        "## Candidate Kernels",
    ]
    for row in kernel_rows:
        lines.append(
            "- "
            f"kernel={row['kernel']} / joint_mean={row['joint_adv']:.6f} / joint_min={row['joint_adv_min']:.6f} "
            f"/ synergy={row['synergy_joint']:.6f} / strict_all={row['strict_positive_all_trials']} "
            f"/ minimal={row['is_minimal_strict_positive']}"
        )
    lines.extend(["", "## Strict-Positive Subsets"])
    positive_subset_rows = [row for row in subset_rows if bool(row["strict_positive_all_trials"])]
    if positive_subset_rows:
        for row in positive_subset_rows:
            lines.append(
                "- "
                f"parent={row['parent_kernel']} / subset={row['kernel']} / joint_mean={row['joint_adv']:.6f} "
                f"/ joint_min={row['joint_adv_min']:.6f} / synergy={row['synergy_joint']:.6f}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Neuron Support"])
    for row in support_rows:
        lines.append(f"- neuron={row['neuron_index']} / support_count={row['support_count']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract and verify fruit compatibility kernels from successful flip rows")
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
    ap.add_argument(
        "--flip-result-paths",
        nargs="+",
        default=[
            "tests/codex_temp/stage56_conflict_pruned_flip_search_qwen_fruit_apple_r1_3_20260317/results.jsonl",
            "tests/codex_temp/stage56_conflict_pruned_flip_search_qwen_fruit_apple_r4_5_20260317/results.jsonl",
            "tests/codex_temp/stage56_conflict_pruned_flip_search_qwen_fruit_apple_r6_7_20260317/results.jsonl",
        ],
    )
    ap.add_argument("--category", default="fruit")
    ap.add_argument("--instance-term", default="apple")
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--random-trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tests/codex_temp/stage56_fruit_compatibility_kernel_extractor_qwen_20260317")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    baselines = read_jsonl(args.stage3_baselines)
    stage3_summary = read_json(args.stage3_summary)
    prototype_rows = read_jsonl(args.prototype_candidates)
    instance_rows = read_jsonl(args.instance_candidates)
    stage6_rows = read_jsonl(args.stage6_results)
    success_rows = read_success_rows(args.flip_result_paths)

    target_row = None
    for row in stage6_rows:
        if str(row.get("category")) == args.category and str(row.get("instance_term")) == args.instance_term:
            target_row = row
            break
    if target_row is None:
        raise ValueError("stage6 target row not found")

    category = str(target_row["category"])
    prototype_term = str(target_row["prototype_term"])
    instance_term = str(target_row["instance_term"])
    prototype_row = find_candidate_row(prototype_rows, category, prototype_term)
    instance_row = find_candidate_row(instance_rows, category, instance_term)
    partitions = partition_union_indices(prototype_row["candidate_indices"], instance_row["candidate_indices"])
    union_indices = [int(x) for x in partitions["union"]]
    kernels = unique_kernels(success_rows, union_indices)
    support_rows = neuron_support_counts(kernels)

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
    kernel_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []

    try:
        proto_joint_adv = float(target_row["proto_joint_adv"])
        instance_joint_adv = float(target_row["instance_joint_adv"])
        proto_effects = dict(target_row["proto_effects"])
        instance_effects = dict(target_row["instance_effects"])

        for kernel_idx, kernel in enumerate(kernels):
            kernel_row = evaluate_kernel(
                model=model,
                tok=tok,
                collector=collector,
                item=item,
                baseline_sig=baseline_sig,
                baseline_readout=baseline_readout,
                category_proto=category_proto,
                proto_map=proto_map,
                selected_categories=selected_categories,
                proto_joint_adv=proto_joint_adv,
                instance_joint_adv=instance_joint_adv,
                proto_effects=proto_effects,
                instance_effects=instance_effects,
                kernel=kernel,
                signature_top_k=args.signature_top_k,
                score_alpha=args.score_alpha,
                seed=args.seed + 1000 + kernel_idx,
                random_trials=args.random_trials,
            )

            has_positive_subset = False
            for subset_idx, subset in enumerate(iter_proper_subsets(kernel)):
                subset_row = evaluate_kernel(
                    model=model,
                    tok=tok,
                    collector=collector,
                    item=item,
                    baseline_sig=baseline_sig,
                    baseline_readout=baseline_readout,
                    category_proto=category_proto,
                    proto_map=proto_map,
                    selected_categories=selected_categories,
                    proto_joint_adv=proto_joint_adv,
                    instance_joint_adv=instance_joint_adv,
                    proto_effects=proto_effects,
                    instance_effects=instance_effects,
                    kernel=subset,
                    signature_top_k=args.signature_top_k,
                    score_alpha=args.score_alpha,
                    seed=args.seed + 5000 + kernel_idx * 100 + subset_idx,
                    random_trials=args.random_trials,
                )
                subset_row["parent_kernel"] = list(kernel)
                subset_rows.append(subset_row)
                if bool(subset_row["strict_positive_all_trials"]):
                    has_positive_subset = True

            kernel_row["has_strict_positive_subset"] = has_positive_subset
            kernel_row["is_minimal_strict_positive"] = bool(kernel_row["strict_positive_all_trials"]) and not has_positive_subset
            kernel_rows.append(kernel_row)
    finally:
        collector.close()

    minimal_rows = [row for row in kernel_rows if bool(row["is_minimal_strict_positive"])]
    intersection_kernel = []
    if kernels:
        intersection = set(kernels[0])
        for kernel in kernels[1:]:
            intersection &= set(kernel)
        intersection_kernel = sorted(int(x) for x in intersection)

    summary = {
        "record_type": "stage56_fruit_compatibility_kernel_extractor_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "category": category,
        "prototype_term": prototype_term,
        "instance_term": instance_term,
        "original_union_joint_adv": float(target_row["union_joint_adv"]),
        "original_union_synergy_joint": float(target_row["union_synergy_joint"]),
        "successful_prune_row_count": len(success_rows),
        "unique_kernel_count": len(kernels),
        "random_trial_count": int(args.random_trials),
        "minimal_strict_positive_kernel_count": len(minimal_rows),
        "successful_kernel_intersection": intersection_kernel,
        "successful_kernel_union": sorted({int(x) for kernel in kernels for x in kernel}),
        "neuron_support_counts": support_rows,
        "kernel_rows": kernel_rows,
        "minimal_kernels": minimal_rows,
        "positive_subset_count": int(sum(1 for row in subset_rows if bool(row["strict_positive_all_trials"]))),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "kernel_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in kernel_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "subset_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in subset_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "REPORT.md").write_text(build_report(summary, kernel_rows, subset_rows, support_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "unique_kernel_count": len(kernels),
                "minimal_strict_positive_kernel_count": len(minimal_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
