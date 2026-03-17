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
from stage56_synergy_conflict_dissection import (  # noqa: E402
    find_candidate_row,
    partition_union_indices,
)


def build_probe_sets(
    union_indices: Sequence[int],
    robust_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    rows = [
        {
            "probe_name": "original_union",
            "indices": [int(x) for x in union_indices],
        }
    ]
    for kernel_row in robust_summary.get("kernel_rows", []):
        rows.append(
            {
                "probe_name": f"kernel_{','.join(str(int(x)) for x in kernel_row.get('kernel', []))}",
                "indices": [int(x) for x in kernel_row.get("kernel", [])],
            }
        )
    return rows


def summarize_trials(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe baseline variance for union and compatibility kernels")
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
    ap.add_argument("--robust-kernel-summary", default="tests/codex_temp/stage56_fruit_compatibility_kernel_extractor_qwen_robust_20260317/summary.json")
    ap.add_argument("--category", default="fruit")
    ap.add_argument("--instance-term", default="apple")
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--random-trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="tests/codex_temp/stage56_baseline_variance_probe_qwen_20260317")
    args = ap.parse_args()

    t0 = time.time()
    families = read_jsonl(args.stage2_families)
    baselines = read_jsonl(args.stage3_baselines)
    stage3_summary = read_json(args.stage3_summary)
    prototype_rows = read_jsonl(args.prototype_candidates)
    instance_rows = read_jsonl(args.instance_candidates)
    stage6_rows = read_jsonl(args.stage6_results)
    robust_summary = read_json(args.robust_kernel_summary)

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
    probe_sets = build_probe_sets(union_indices, robust_summary)

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
    probe_rows: List[Dict[str, object]] = []

    try:
        proto_joint_adv = float(target_row["proto_joint_adv"])
        instance_joint_adv = float(target_row["instance_joint_adv"])
        proto_effects = dict(target_row["proto_effects"])
        instance_effects = dict(target_row["instance_effects"])

        for probe_idx, probe in enumerate(probe_sets):
            indices = [int(x) for x in probe["indices"]]
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
                indices,
                args.signature_top_k,
            )
            joint_trials: List[float] = []
            category_trials: List[float] = []
            for trial_idx in range(max(1, int(args.random_trials))):
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
                    sample_random_like(indices, collector.total_neurons, seed=args.seed + probe_idx * 1000 + trial_idx),
                    args.signature_top_k,
                )
                margin_adv = float(eval_result["effects"]["margin_drop"] - random_eval["effects"]["margin_drop"])
                category_adv = float(
                    eval_result["effects"]["category_margin_drop"] - random_eval["effects"]["category_margin_drop"]
                )
                joint_trials.append(float(effect_score(margin_adv, category_adv, args.score_alpha)))
                category_trials.append(category_adv)

            synergy_margin = float(
                eval_result["effects"]["margin_drop"]
                - max(float(proto_effects["margin_drop"]), float(instance_effects["margin_drop"]))
            )
            synergy_category = float(
                eval_result["effects"]["category_margin_drop"]
                - max(
                    float(proto_effects["category_margin_drop"]),
                    float(instance_effects["category_margin_drop"]),
                )
            )
            synergy_joint = float(effect_score(synergy_margin, synergy_category, args.score_alpha))
            joint_stats = summarize_trials(joint_trials)
            category_stats = summarize_trials(category_trials)
            probe_rows.append(
                {
                    "probe_name": str(probe["probe_name"]),
                    "indices": indices,
                    "probe_size": len(indices),
                    "layer_distribution": layer_distribution(indices, collector.d_ff),
                    "random_trial_count": int(args.random_trials),
                    "joint_adv_mean": joint_stats["mean"],
                    "joint_adv_std": joint_stats["std"],
                    "joint_adv_min": joint_stats["min"],
                    "joint_adv_max": joint_stats["max"],
                    "category_adv_mean": category_stats["mean"],
                    "category_adv_std": category_stats["std"],
                    "synergy_joint": synergy_joint,
                    "beats_proto_count": int(sum(1 for value in joint_trials if value > proto_joint_adv)),
                    "beats_instance_count": int(sum(1 for value in joint_trials if value > instance_joint_adv)),
                    "strict_all_trials": bool(
                        synergy_joint > 0.0
                        and all(value > proto_joint_adv for value in joint_trials)
                        and all(value > instance_joint_adv for value in joint_trials)
                    ),
                }
            )
    finally:
        collector.close()

    summary = {
        "record_type": "stage56_baseline_variance_probe_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "model_id": args.model_id,
        "model_ref": model_ref,
        "category": category,
        "prototype_term": prototype_term,
        "instance_term": instance_term,
        "original_union_joint_adv": float(target_row["union_joint_adv"]),
        "original_union_synergy_joint": float(target_row["union_synergy_joint"]),
        "random_trial_count": int(args.random_trials),
        "probe_count": len(probe_rows),
        "probe_rows": probe_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "probe_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in probe_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_lines = [
        "# Stage56 Baseline Variance Probe Report",
        "",
        f"- Category: {category}",
        f"- Prototype term: {prototype_term}",
        f"- Instance term: {instance_term}",
        f"- Random trials: {int(args.random_trials)}",
        "",
        "## Probe Rows",
    ]
    for row in probe_rows:
        report_lines.append(
            "- "
            f"{row['probe_name']} / size={row['probe_size']} / joint_mean={row['joint_adv_mean']:.6f} "
            f"/ joint_std={row['joint_adv_std']:.6f} / joint_min={row['joint_adv_min']:.6f} "
            f"/ synergy={row['synergy_joint']:.6f} / strict_all={row['strict_all_trials']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "probe_count": len(probe_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
