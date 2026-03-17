from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

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


DEFAULT_CASES = [
    {
        "model_id": "Qwen/Qwen3-4B",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_real_20260317" / "qwen3_4b",
        "category": "fruit",
    },
    {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_real_20260317" / "deepseek_7b",
        "category": "fruit",
    },
    {
        "model_id": "zai-org/GLM-4-9B-Chat-HF",
        "model_root": ROOT / "tempdata" / "stage56_fruit_family_closure_block_glm_real_20260317" / "zai_org_glm_4_9b_chat_hf",
        "category": "fruit",
    },
]


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


def stable_key(indices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in indices))


def build_stage_paths(model_root: Path) -> Dict[str, Path]:
    return {
        "stage1_families": model_root / "stage1_three_pool" / "families.jsonl",
        "stage3_summary": model_root / "stage3_causal_closure" / "summary.json",
        "stage3_baselines": model_root / "stage3_causal_closure" / "baselines.jsonl",
        "stage5_proto_candidates": model_root / "stage5_prototype" / "candidates.jsonl",
        "stage5_proto_neurons": model_root / "stage5_prototype" / "neurons.jsonl",
        "stage5_inst_candidates": model_root / "stage5_instance" / "candidates.jsonl",
        "stage5_inst_neurons": model_root / "stage5_instance" / "neurons.jsonl",
        "stage6_results": model_root / "stage6_prototype_instance_decomposition" / "results.jsonl",
    }


def match_candidate_row(rows: Sequence[Dict[str, object]], term: str, category: str) -> Dict[str, object]:
    for row in rows:
        item = row.get("item", {})
        if str(item.get("term")) == term and str(item.get("category")) == category:
            return row
    raise KeyError(f"missing candidate row: term={term}, category={category}")


def filter_neuron_rows(
    rows: Sequence[Dict[str, object]],
    term: str,
    category: str,
) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("item", {}).get("term")) == term
        and str(row.get("item", {}).get("category")) == category
    ]


def build_neuron_strength_table(
    union_indices: Sequence[int],
    proto_rows: Sequence[Dict[str, object]],
    inst_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    proto_map = {int(row["neuron_index"]): row for row in proto_rows}
    inst_map = {int(row["neuron_index"]): row for row in inst_rows}
    rows: List[Dict[str, object]] = []
    for idx in union_indices:
        proto = proto_map.get(int(idx))
        inst = inst_map.get(int(idx))
        proto_strength = 0.0
        inst_strength = 0.0
        if proto is not None:
            proto_strength = max(
                abs(float(proto.get("rescue_joint_score", 0.0))),
                abs(float(proto.get("solo_joint_score", 0.0))),
            )
        if inst is not None:
            inst_strength = max(
                abs(float(inst.get("rescue_joint_score", 0.0))),
                abs(float(inst.get("solo_joint_score", 0.0))),
            )
        rows.append(
            {
                "neuron_index": int(idx),
                "proto_strength": float(proto_strength),
                "inst_strength": float(inst_strength),
                "strength_score": float(max(proto_strength, inst_strength)),
                "proto_rescue_joint_score": float(proto.get("rescue_joint_score", 0.0)) if proto else 0.0,
                "proto_solo_joint_score": float(proto.get("solo_joint_score", 0.0)) if proto else 0.0,
                "inst_rescue_joint_score": float(inst.get("rescue_joint_score", 0.0)) if inst else 0.0,
                "inst_solo_joint_score": float(inst.get("solo_joint_score", 0.0)) if inst else 0.0,
                "lane_presence": "".join(
                    part
                    for part, present in (
                        ("P", proto is not None),
                        ("I", inst is not None),
                    )
                    if present
                ) or "-",
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["strength_score"]),
            float(row["proto_strength"]),
            float(row["inst_strength"]),
            -int(row["neuron_index"]),
        ),
        reverse=True,
    )
    return rows


def split_strong_weak(rows: Sequence[Dict[str, object]]) -> Tuple[List[int], List[int]]:
    if not rows:
        return [], []
    strong_count = max(1, int(math.ceil(len(rows) / 2.0)))
    strong = [int(row["neuron_index"]) for row in rows[:strong_count]]
    weak = [int(row["neuron_index"]) for row in rows[strong_count:]]
    return strong, weak


def build_probe_subsets(strong: Sequence[int], weak: Sequence[int], union_indices: Sequence[int]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[Tuple[int, ...]] = set()

    def add(label: str, indices: Sequence[int]) -> None:
        cleaned = [int(x) for x in indices]
        if not cleaned:
            return
        key = stable_key(cleaned)
        if key in seen:
            return
        seen.add(key)
        out.append({"label": label, "indices": list(key)})

    add("strong_top1", strong[:1])
    add("strong_top2", strong[:2])
    add("strong_top3", strong[:3])
    add("strong_full", strong)
    add("weak_top1", weak[:1])
    add("weak_top2", weak[:2])
    add("weak_full", weak)
    add("union_full", union_indices)

    strong_top1 = strong[:1]
    strong_top2 = strong[:2]
    for weak_idx in weak:
        add(f"mix_top1_plus_{weak_idx}", list(strong_top1) + [int(weak_idx)])
        add(f"mix_top2_plus_{weak_idx}", list(strong_top2) + [int(weak_idx)])
    for strong_idx in strong[: min(2, len(strong))]:
        for weak_idx in weak[: min(3, len(weak))]:
            add(f"mix_pair_{strong_idx}_{weak_idx}", [int(strong_idx), int(weak_idx)])
    return out


def build_random_eval_metrics(
    model,
    tok,
    collector: GateCollector,
    item: LexemeItem,
    baseline_sig: Dict[str, object],
    baseline_readout: Dict[str, object],
    category_proto: Dict[str, object],
    proto_map: Dict[str, Dict[str, object]],
    selected_categories: Sequence[str],
    indices: Sequence[int],
    signature_top_k: int,
    random_repeats: int,
    seed_base: int,
    alpha: float,
) -> Dict[str, object]:
    eval_row = evaluate_ablation(
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
        signature_top_k,
    )
    joint_values: List[float] = []
    margin_values: List[float] = []
    category_values: List[float] = []
    random_effects = []
    for rep in range(random_repeats):
        random_indices = sample_random_like(
            indices,
            collector.total_neurons,
            seed=seed_base + rep,
        )
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
            random_indices,
            signature_top_k,
        )
        margin_adv = float(eval_row["effects"]["margin_drop"] - random_eval["effects"]["margin_drop"])
        category_adv = float(
            eval_row["effects"]["category_margin_drop"] - random_eval["effects"]["category_margin_drop"]
        )
        joint_adv = effect_score(margin_adv, category_adv, alpha)
        margin_values.append(margin_adv)
        category_values.append(category_adv)
        joint_values.append(joint_adv)
        random_effects.append(
            {
                "random_indices": [int(x) for x in random_indices],
                "margin_adv": float(margin_adv),
                "category_adv": float(category_adv),
                "joint_adv": float(joint_adv),
            }
        )
    return {
        "subset_size": len(indices),
        "indices": [int(x) for x in indices],
        "effects": eval_row["effects"],
        "joint_adv_mean": float(np.mean(joint_values) if joint_values else 0.0),
        "joint_adv_min": float(np.min(joint_values) if joint_values else 0.0),
        "joint_adv_max": float(np.max(joint_values) if joint_values else 0.0),
        "joint_adv_std": float(np.std(joint_values) if joint_values else 0.0),
        "margin_adv_mean": float(np.mean(margin_values) if margin_values else 0.0),
        "category_adv_mean": float(np.mean(category_values) if category_values else 0.0),
        "random_effects": random_effects,
    }


def classify_case(best_strong: Dict[str, object], best_weak: Dict[str, object], best_mixed: Dict[str, object]) -> str:
    strong_mean = float(best_strong["metrics"]["joint_adv_mean"])
    weak_mean = float(best_weak["metrics"]["joint_adv_mean"])
    mixed_mean = float(best_mixed["metrics"]["joint_adv_mean"])
    if mixed_mean > strong_mean and mixed_mean > 0.0:
        return "weak_bridge_positive"
    if mixed_mean > strong_mean:
        return "weak_partial_bridge"
    if weak_mean > strong_mean and weak_mean > 0.0:
        return "weak_dominant_positive"
    if mixed_mean < strong_mean:
        return "weak_drag_or_conflict"
    return "neutral_or_unresolved"


def choose_best(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row["metrics"]["joint_adv_mean"]),
            float(row["metrics"]["joint_adv_min"]),
            -int(row["metrics"]["subset_size"]),
        ),
    )


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Strong Weak Combo Probe Report",
        "",
        f"- Case count: {summary['case_count']}",
        f"- Weak bridge positive count: {summary['weak_bridge_positive_count']}",
        f"- Weak drag/conflict count: {summary['weak_drag_or_conflict_count']}",
        f"- Mean best mixed joint adv: {summary['mean_best_mixed_joint_adv']:.6f}",
        "",
        "## Cases",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['model_id']} / {row['category']} / proto={row['prototype_term']} / inst={row['instance_term']} "
            f"/ role={row['case_role']} / best_strong={row['best_strong']['label']}:{row['best_strong']['metrics']['joint_adv_mean']:.6f} "
            f"/ best_weak={row['best_weak']['label']}:{row['best_weak']['metrics']['joint_adv_mean']:.6f} "
            f"/ best_mixed={row['best_mixed']['label']}:{row['best_mixed']['metrics']['joint_adv_mean']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe strong/weak neuron mixture behavior on fruit cases")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--signature-top-k", type=int, default=64)
    ap.add_argument("--random-repeats", type=int, default=4)
    ap.add_argument("--score-alpha", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strong_weak_combo_probe_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: List[Dict[str, object]] = []
    for case_idx, case in enumerate(DEFAULT_CASES):
        model_id = str(case["model_id"])
        model_root = Path(case["model_root"])
        category = str(case["category"])
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
        stage6_row = next(row for row in stage6_rows if str(row["category"]) == category)
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

        model, tok, model_ref = load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        collector = GateCollector(model)
        try:
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
                    seed_base=args.seed + case_idx * 10000 + subset_idx * 100,
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
        finally:
            collector.close()
            del collector
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        strong_rows = [row for row in evaluated_subsets if row["subset_type"] == "strong_only"]
        weak_rows = [row for row in evaluated_subsets if row["subset_type"] == "weak_only"]
        mixed_rows = [row for row in evaluated_subsets if row["subset_type"] == "mixed"]
        best_strong = choose_best(strong_rows)
        best_weak = choose_best(weak_rows) if weak_rows else {"label": "none", "metrics": {"joint_adv_mean": 0.0}}
        best_mixed = choose_best(mixed_rows) if mixed_rows else best_strong

        result_rows.append(
            {
                "record_type": "stage56_strong_weak_combo_case",
                "model_id": model_id,
                "model_ref": model_ref,
                "model_root": str(model_root),
                "category": category,
                "prototype_term": prototype_term,
                "instance_term": instance_term,
                "proto_indices": proto_indices,
                "instance_indices": inst_indices,
                "union_indices": union_indices,
                "strength_rows": strength_rows,
                "strong_indices": strong_indices,
                "weak_indices": weak_indices,
                "best_strong": best_strong,
                "best_weak": best_weak,
                "best_mixed": best_mixed,
                "case_role": classify_case(best_strong, best_weak, best_mixed),
                "stage6_reference": {
                    "proto_joint_adv": float(stage6_row["proto_joint_adv"]),
                    "instance_joint_adv": float(stage6_row["instance_joint_adv"]),
                    "union_joint_adv": float(stage6_row["union_joint_adv"]),
                    "union_synergy_joint": float(stage6_row["union_synergy_joint"]),
                },
                "all_subset_rows": evaluated_subsets,
            }
        )

    summary = {
        "record_type": "stage56_strong_weak_combo_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "case_count": len(result_rows),
        "weak_bridge_positive_count": int(sum(1 for row in result_rows if row["case_role"] == "weak_bridge_positive")),
        "weak_partial_bridge_count": int(sum(1 for row in result_rows if row["case_role"] == "weak_partial_bridge")),
        "weak_dominant_positive_count": int(sum(1 for row in result_rows if row["case_role"] == "weak_dominant_positive")),
        "weak_drag_or_conflict_count": int(sum(1 for row in result_rows if row["case_role"] == "weak_drag_or_conflict")),
        "mean_best_strong_joint_adv": float(
            np.mean([row["best_strong"]["metrics"]["joint_adv_mean"] for row in result_rows]) if result_rows else 0.0
        ),
        "mean_best_weak_joint_adv": float(
            np.mean([row["best_weak"]["metrics"]["joint_adv_mean"] for row in result_rows]) if result_rows else 0.0
        ),
        "mean_best_mixed_joint_adv": float(
            np.mean([row["best_mixed"]["metrics"]["joint_adv_mean"] for row in result_rows]) if result_rows else 0.0
        ),
    }

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "results.jsonl", result_rows)
    write_report(output_dir / "REPORT.md", summary, result_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "case_count": len(result_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
