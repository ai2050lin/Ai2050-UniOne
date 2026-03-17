from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deepseek7b_stage3_causal_closure import read_json, read_jsonl  # noqa: E402
from deepseek7b_three_pool_structure_scan import GateCollector, LexemeItem, load_model  # noqa: E402
from stage56_strong_weak_combo_probe import (  # noqa: E402
    build_probe_subsets,
    build_random_eval_metrics,
    build_stage_paths,
    build_neuron_strength_table,
    choose_best,
    classify_case,
    filter_neuron_rows,
    match_candidate_row,
    split_strong_weak,
    unique_union,
)


CASE_GROUPS = [
    {
        "label": "qwen_real",
        "model_id": "Qwen/Qwen3-4B",
        "model_root": ROOT / "tempdata" / "stage56_real_category_closure_block_strict_20260317_1922" / "qwen3_4b",
    },
    {
        "label": "deepseek_real",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_root": ROOT / "tempdata" / "stage56_real_category_closure_block_strict_20260317_1922" / "deepseek_7b",
    },
    {
        "label": "qwen_fruit",
        "model_id": "Qwen/Qwen3-4B",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_real_20260317" / "qwen3_4b",
    },
    {
        "label": "deepseek_fruit",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_real_20260317" / "deepseek_7b",
    },
    {
        "label": "glm_fruit",
        "model_id": "zai-org/GLM-4-9B-Chat-HF",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_glm_real_20260317" / "zai_org_glm_4_9b_chat_hf",
    },
]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dominant_structure(case_role: str, best_strong: Dict[str, object], best_weak: Dict[str, object], best_mixed: Dict[str, object]) -> str:
    strong_mean = float(best_strong["metrics"]["joint_adv_mean"])
    weak_mean = float(best_weak["metrics"]["joint_adv_mean"])
    mixed_mean = float(best_mixed["metrics"]["joint_adv_mean"])
    if case_role == "weak_bridge_positive":
        return "bridge_dominant"
    if weak_mean > strong_mean and weak_mean > 0.0:
        return "weak_dominant"
    if strong_mean > 0.0 and mixed_mean < strong_mean:
        return "strong_core_dominant"
    if mixed_mean <= 0.0 and strong_mean <= 0.0 and weak_mean <= 0.0:
        return "global_failure"
    return "mixed_or_unresolved"


def aggregate_counts(rows: Sequence[Dict[str, object]], key_name: str) -> Dict[str, int]:
    ctr = Counter(str(row[key_name]) for row in rows)
    return {key: int(value) for key, value in sorted(ctr.items())}


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Multicategory Strong Weak Taxonomy Report",
        "",
        f"- Case count: {summary['case_count']}",
        f"- Model count: {summary['model_count']}",
        f"- Category count: {summary['category_count']}",
        f"- Weak bridge positive count: {summary['weak_bridge_positive_count']}",
        f"- Strong core dominant count: {summary['dominant_structure_counts'].get('strong_core_dominant', 0)}",
        "",
        "## By Model",
    ]
    for model_id, block in summary["per_model"].items():
        lines.append(
            f"- {model_id}: cases={block['case_count']} / weak_bridge_positive={block['weak_bridge_positive_count']} "
            f"/ mean_best_strong={block['mean_best_strong_joint_adv']:.6f} "
            f"/ mean_best_weak={block['mean_best_weak_joint_adv']:.6f} "
            f"/ mean_best_mixed={block['mean_best_mixed_joint_adv']:.6f}"
        )
    lines.extend(["", "## Top Cases"])
    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row["best_mixed"]["metrics"]["joint_adv_mean"]),
            float(row["best_strong"]["metrics"]["joint_adv_mean"]),
        ),
        reverse=True,
    )[:20]
    for row in top_rows:
        lines.append(
            "- "
            f"{row['group_label']} / {row['category']} / proto={row['prototype_term']} / inst={row['instance_term']} "
            f"/ dominant={row['dominant_structure']} / role={row['case_role']} "
            f"/ best_strong={row['best_strong']['label']}:{row['best_strong']['metrics']['joint_adv_mean']:.6f} "
            f"/ best_weak={row['best_weak']['label']}:{row['best_weak']['metrics']['joint_adv_mean']:.6f} "
            f"/ best_mixed={row['best_mixed']['label']}:{row['best_mixed']['metrics']['joint_adv_mean']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run multicategory strong/weak taxonomy aggregation")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--random-repeats", type=int, default=2)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_multicategory_strong_weak_taxonomy_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups_by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for group in CASE_GROUPS:
        groups_by_model[str(group["model_id"])].append(group)

    case_rows: List[Dict[str, object]] = []
    for model_idx, (model_id, groups) in enumerate(groups_by_model.items()):
        model, tok, model_ref = load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        collector = GateCollector(model)
        try:
            for group_idx, group in enumerate(groups):
                group_label = str(group["label"])
                model_root = Path(group["model_root"])
                paths = build_stage_paths(model_root)
                families = read_jsonl(paths["stage1_families"])
                stage3_summary = read_json(paths["stage3_summary"])
                baselines = read_jsonl(paths["stage3_baselines"])
                proto_candidates = read_jsonl(paths["stage5_proto_candidates"])
                proto_neurons = read_jsonl(paths["stage5_proto_neurons"])
                inst_candidates = read_jsonl(paths["stage5_inst_candidates"])
                inst_neurons = read_jsonl(paths["stage5_inst_neurons"])
                stage6_rows = read_jsonl(paths["stage6_results"])

                selected_categories = [str(x) for x in stage3_summary["selected_categories"]]
                proto_map = {
                    str(row["category"]): row
                    for row in families
                    if str(row.get("record_type")) == "family_prototype"
                    and str(row.get("pool")) == "closure"
                    and str(row["category"]) in selected_categories
                }

                for case_idx, stage6_row in enumerate(stage6_rows):
                    category = str(stage6_row["category"])
                    prototype_term = str(stage6_row["prototype_term"])
                    instance_term = str(stage6_row["instance_term"])
                    item = LexemeItem(term=instance_term, category=category, language="ascii")
                    baseline_row = next(
                        row
                        for row in baselines
                        if str(row["item"]["term"]) == instance_term and str(row["item"]["category"]) == category
                    )
                    baseline_sig = baseline_row["baseline_signature"]
                    baseline_readout = baseline_row["baseline_readout"]
                    category_proto = proto_map[category]

                    proto_candidate = match_candidate_row(proto_candidates, prototype_term, category)
                    inst_candidate = match_candidate_row(inst_candidates, instance_term, category)
                    proto_indices = [int(x) for x in proto_candidate["candidate_indices"]]
                    inst_indices = [int(x) for x in inst_candidate["candidate_indices"]]
                    union_indices = unique_union(proto_indices, inst_indices)

                    strength_rows = build_neuron_strength_table(
                        union_indices=union_indices,
                        proto_rows=filter_neuron_rows(proto_neurons, prototype_term, category),
                        inst_rows=filter_neuron_rows(inst_neurons, instance_term, category),
                    )
                    strong_indices, weak_indices = split_strong_weak(strength_rows)
                    probe_subsets = build_probe_subsets(strong_indices, weak_indices, union_indices)

                    evaluated_subsets: List[Dict[str, object]] = []
                    for subset_idx, subset in enumerate(probe_subsets):
                        metrics = build_random_eval_metrics(
                            model=model,
                            tok=tok,
                            collector=collector,
                            item=item,
                            baseline_sig=baseline_sig,
                            baseline_readout=baseline_readout,
                            category_proto=category_proto,
                            proto_map=proto_map,
                            selected_categories=selected_categories,
                            indices=subset["indices"],
                            signature_top_k=args.signature_top_k,
                            random_repeats=args.random_repeats,
                            seed_base=args.seed + model_idx * 100000 + group_idx * 10000 + case_idx * 1000 + subset_idx * 100,
                            alpha=args.score_alpha,
                        )
                        evaluated_subsets.append(
                            {
                                "label": subset["label"],
                                "subset_type": (
                                    "strong_only"
                                    if set(subset["indices"]).issubset(set(strong_indices))
                                    else "weak_only"
                                    if set(subset["indices"]).issubset(set(weak_indices))
                                    else "mixed"
                                ),
                                "metrics": metrics,
                            }
                        )

                    strong_rows = [row for row in evaluated_subsets if row["subset_type"] == "strong_only"]
                    weak_rows = [row for row in evaluated_subsets if row["subset_type"] == "weak_only"]
                    mixed_rows = [row for row in evaluated_subsets if row["subset_type"] == "mixed"]
                    best_strong = choose_best(strong_rows)
                    best_weak = choose_best(weak_rows) if weak_rows else {"label": "none", "metrics": {"joint_adv_mean": 0.0}}
                    best_mixed = choose_best(mixed_rows) if mixed_rows else best_strong
                    case_role = classify_case(best_strong, best_weak, best_mixed)
                    case_rows.append(
                        {
                            "record_type": "stage56_multicategory_strong_weak_case",
                            "group_label": group_label,
                            "model_id": model_id,
                            "model_ref": model_ref,
                            "model_root": str(model_root),
                            "category": category,
                            "prototype_term": prototype_term,
                            "instance_term": instance_term,
                            "strong_indices": strong_indices,
                            "weak_indices": weak_indices,
                            "strength_rows": strength_rows,
                            "best_strong": best_strong,
                            "best_weak": best_weak,
                            "best_mixed": best_mixed,
                            "case_role": case_role,
                            "dominant_structure": dominant_structure(case_role, best_strong, best_weak, best_mixed),
                            "stage6_reference": {
                                "proto_joint_adv": float(stage6_row["proto_joint_adv"]),
                                "instance_joint_adv": float(stage6_row["instance_joint_adv"]),
                                "union_joint_adv": float(stage6_row["union_joint_adv"]),
                                "union_synergy_joint": float(stage6_row["union_synergy_joint"]),
                            },
                        }
                    )
        finally:
            collector.close()
            del collector
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_model_summary: Dict[str, Dict[str, object]] = {}
    for model_id in sorted({str(row["model_id"]) for row in case_rows}):
        rows = [row for row in case_rows if str(row["model_id"]) == model_id]
        per_model_summary[model_id] = {
            "case_count": len(rows),
            "weak_bridge_positive_count": int(sum(1 for row in rows if row["case_role"] == "weak_bridge_positive")),
            "weak_drag_or_conflict_count": int(sum(1 for row in rows if row["case_role"] == "weak_drag_or_conflict")),
            "mean_best_strong_joint_adv": float(np.mean([row["best_strong"]["metrics"]["joint_adv_mean"] for row in rows])) if rows else 0.0,
            "mean_best_weak_joint_adv": float(np.mean([row["best_weak"]["metrics"]["joint_adv_mean"] for row in rows])) if rows else 0.0,
            "mean_best_mixed_joint_adv": float(np.mean([row["best_mixed"]["metrics"]["joint_adv_mean"] for row in rows])) if rows else 0.0,
        }

    summary = {
        "record_type": "stage56_multicategory_strong_weak_taxonomy_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "case_count": len(case_rows),
        "model_count": len(per_model_summary),
        "category_count": len({str(row["category"]) for row in case_rows}),
        "group_count": len({str(row["group_label"]) for row in case_rows}),
        "weak_bridge_positive_count": int(sum(1 for row in case_rows if row["case_role"] == "weak_bridge_positive")),
        "weak_partial_bridge_count": int(sum(1 for row in case_rows if row["case_role"] == "weak_partial_bridge")),
        "weak_dominant_positive_count": int(sum(1 for row in case_rows if row["case_role"] == "weak_dominant_positive")),
        "weak_drag_or_conflict_count": int(sum(1 for row in case_rows if row["case_role"] == "weak_drag_or_conflict")),
        "case_role_counts": aggregate_counts(case_rows, "case_role"),
        "dominant_structure_counts": aggregate_counts(case_rows, "dominant_structure"),
        "per_model": per_model_summary,
    }

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "cases.jsonl", case_rows)
    write_report(output_dir / "REPORT.md", summary, case_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "case_count": len(case_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
