from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "tests" / "codex"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

DEFAULT_TAXONOMY_PATH = (
    ROOT / "tests" / "codex_temp" / "stage56_multicategory_strong_weak_taxonomy_20260318" / "cases.jsonl"
)
AXES = ("style", "logic", "syntax")
FIELD_PROXY_NAMES = (
    "prototype_field_proxy",
    "instance_field_proxy",
    "bridge_field_proxy",
    "conflict_field_proxy",
    "mismatch_field_proxy",
)
FIELD_SHORT_NAMES = {
    "prototype_field_proxy": "P",
    "instance_field_proxy": "I",
    "bridge_field_proxy": "B",
    "conflict_field_proxy": "X",
    "mismatch_field_proxy": "M",
}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def first_token_id(tok, text: str) -> int | None:
    ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0]) if ids else None


def prompt_variants(term: str) -> List[Dict[str, str]]:
    return [
        {"axis": "control", "variant": "plain", "prompt": f"The concept {term} belongs to"},
        {"axis": "style", "variant": "chat", "prompt": f"In a short chat, the category of {term} is"},
        {"axis": "style", "variant": "formal", "prompt": f"In formal classification, {term} belongs to the category of"},
        {
            "axis": "logic",
            "variant": "causal",
            "prompt": f"Because {term} is one member of a broader class, {term} belongs to",
        },
        {
            "axis": "logic",
            "variant": "contrast",
            "prompt": f"{term} is not an animal or a vehicle; {term} belongs to",
        },
        {"axis": "syntax", "variant": "simple", "prompt": f"{term} is a"},
        {"axis": "syntax", "variant": "embedded", "prompt": f"The category to which {term} belongs is"},
    ]


def category_margin_for_prompt(model, tok, prompt: str, correct_category: str, all_categories: Sequence[str]) -> Dict[str, float]:
    from deepseek7b_three_pool_structure_scan import run_prompt

    out = run_prompt(model, tok, prompt)
    logits = out.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=0)
    masses = []
    for category in all_categories:
        token_id = first_token_id(tok, " " + category)
        mass = float(probs[token_id].item()) if token_id is not None else 0.0
        masses.append((category, mass))
    correct_mass = next((mass for cat, mass in masses if cat == correct_category), 0.0)
    best_other = max((mass for cat, mass in masses if cat != correct_category), default=0.0)
    return {
        "correct_prob": float(correct_mass),
        "best_other_prob": float(best_other),
        "category_margin": float(correct_mass - best_other),
    }


def effect_drop(baseline: Dict[str, float], ablated: Dict[str, float]) -> float:
    return float(baseline["category_margin"] - ablated["category_margin"])


def compute_gate_proxies(
    prototype_drop: float,
    instance_drop: float,
    strong_drop: float,
    mixed_drop: float,
) -> Dict[str, float]:
    return {
        "prototype_field_proxy": float(prototype_drop),
        "instance_field_proxy": float(instance_drop),
        "bridge_field_proxy": float(max(mixed_drop - strong_drop, 0.0)),
        "conflict_field_proxy": float(max(strong_drop - mixed_drop, 0.0)),
        "mismatch_field_proxy": float(max(max(prototype_drop, instance_drop) - mixed_drop, 0.0)),
    }


def direction_label(delta: float, threshold: float = 1e-6) -> str:
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "neutral"


def mean_gate_proxies(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    return {
        field_name: safe_mean([float(row["gate_proxies"][field_name]) for row in rows])
        for field_name in FIELD_PROXY_NAMES
    }


def compute_axis_gate_summary(variant_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    control_row = next(row for row in variant_rows if row["axis"] == "control")
    control_fields = {field_name: float(control_row["gate_proxies"][field_name]) for field_name in FIELD_PROXY_NAMES}
    axes: Dict[str, Dict[str, object]] = {}
    for axis in AXES:
        axis_rows = [row for row in variant_rows if row["axis"] == axis]
        axis_means = mean_gate_proxies(axis_rows)
        deltas = {
            field_name: float(axis_means[field_name] - control_fields[field_name])
            for field_name in FIELD_PROXY_NAMES
        }
        axes[axis] = {
            "variant_count": len(axis_rows),
            "mean_fields": axis_means,
            "deltas": deltas,
            "directions": {
                field_name: direction_label(deltas[field_name])
                for field_name in FIELD_PROXY_NAMES
            },
            "mean_strong_drop": safe_mean([float(row["strong_drop"]) for row in axis_rows]),
            "mean_mixed_drop": safe_mean([float(row["mixed_drop"]) for row in axis_rows]),
            "mean_bridge_gain": safe_mean([float(row["bridge_gain"]) for row in axis_rows]),
        }
    return {
        "control_fields": control_fields,
        "control_strong_drop": float(control_row["strong_drop"]),
        "control_mixed_drop": float(control_row["mixed_drop"]),
        "control_bridge_gain": float(control_row["bridge_gain"]),
        "axes": axes,
    }


def filter_taxonomy_rows(
    rows: Sequence[Dict[str, object]],
    group_labels: Sequence[str],
    categories: Sequence[str],
    case_roles: Sequence[str],
    max_cases_per_model: int,
) -> List[Dict[str, object]]:
    selected = []
    allowed_groups = {value for value in group_labels if value}
    allowed_categories = {value for value in categories if value}
    allowed_roles = {value for value in case_roles if value}
    per_model_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        group_label = str(row["group_label"])
        category = str(row["category"])
        case_role = str(row["case_role"])
        model_id = str(row["model_id"])
        if allowed_groups and group_label not in allowed_groups:
            continue
        if allowed_categories and category not in allowed_categories:
            continue
        if allowed_roles and case_role not in allowed_roles:
            continue
        if max_cases_per_model > 0 and per_model_counts[model_id] >= max_cases_per_model:
            continue
        selected.append(row)
        per_model_counts[model_id] += 1
    return selected


def build_summary(case_rows: Sequence[Dict[str, object]], runtime_sec: float) -> Dict[str, object]:
    per_model_case_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        per_model_case_rows[str(row["model_id"])].append(row)

    per_axis = {}
    for axis in AXES:
        per_axis[axis] = {
            "mean_deltas": {
                field_name: safe_mean(
                    [float(row["axis_gate_summary"]["axes"][axis]["deltas"][field_name]) for row in case_rows]
                )
                for field_name in FIELD_PROXY_NAMES
            },
            "direction_counts": {
                field_name: {
                    direction: int(
                        sum(
                            1
                            for row in case_rows
                            if str(row["axis_gate_summary"]["axes"][axis]["directions"][field_name]) == direction
                        )
                    )
                    for direction in ("positive", "negative", "neutral")
                }
                for field_name in FIELD_PROXY_NAMES
            },
        }

    return {
        "record_type": "stage56_generation_gate_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(runtime_sec),
        "case_count": len(case_rows),
        "per_axis": per_axis,
        "per_model": {
            model_id: {
                "case_count": len(model_rows),
                "per_axis": {
                    axis: {
                        "mean_deltas": {
                            field_name: safe_mean(
                                [
                                    float(model_row["axis_gate_summary"]["axes"][axis]["deltas"][field_name])
                                    for model_row in model_rows
                                ]
                            )
                            for field_name in FIELD_PROXY_NAMES
                        }
                    }
                    for axis in AXES
                },
            }
            for model_id, model_rows in sorted(per_model_case_rows.items())
        },
    }


def write_report(path: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Stage56 Generation Gate Coupling Report",
        "",
        f"- Case count: {summary['case_count']}",
    ]
    for axis in AXES:
        axis_block = summary["per_axis"][axis]
        deltas = axis_block["mean_deltas"]
        lines.append(
            f"- {axis}: "
            f"P={deltas['prototype_field_proxy']:.6f}, "
            f"I={deltas['instance_field_proxy']:.6f}, "
            f"B={deltas['bridge_field_proxy']:.6f}, "
            f"X={deltas['conflict_field_proxy']:.6f}, "
            f"M={deltas['mismatch_field_proxy']:.6f}"
        )
    lines.extend(["", "## Representative Cases"])
    top_rows = sorted(
        rows,
        key=lambda row: (
            abs(float(row["axis_gate_summary"]["axes"]["logic"]["deltas"]["bridge_field_proxy"])),
            abs(float(row["axis_gate_summary"]["axes"]["style"]["deltas"]["mismatch_field_proxy"])),
            abs(float(row["axis_gate_summary"]["axes"]["syntax"]["deltas"]["prototype_field_proxy"])),
        ),
        reverse=True,
    )[:20]
    for row in top_rows:
        parts = [
            f"{row['group_label']} / {row['category']} / proto={row['prototype_term']} / inst={row['instance_term']}"
        ]
        for axis in AXES:
            deltas = row["axis_gate_summary"]["axes"][axis]["deltas"]
            parts.append(
                f"{axis}: "
                f"P={deltas['prototype_field_proxy']:.6f}, "
                f"I={deltas['instance_field_proxy']:.6f}, "
                f"B={deltas['bridge_field_proxy']:.6f}, "
                f"X={deltas['conflict_field_proxy']:.6f}, "
                f"M={deltas['mismatch_field_proxy']:.6f}"
            )
        lines.append("- " + " | ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_arg(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Couple style/logic/syntax generation gates with P/I/B/X/M proxies")
    ap.add_argument("--taxonomy-cases", default=str(DEFAULT_TAXONOMY_PATH))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--group-labels", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument("--case-roles", default="")
    ap.add_argument("--max-cases-per-model", type=int, default=0)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_coupling_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    from deepseek7b_stage3_causal_closure import register_ablation, remove_handles
    from deepseek7b_three_pool_structure_scan import gate_spec_for_layer, load_model
    from stage56_strong_weak_combo_probe import build_stage_paths, match_candidate_row

    args = parse_args()
    t0 = time.time()
    taxonomy_rows = read_jsonl(Path(args.taxonomy_cases))
    taxonomy_rows = filter_taxonomy_rows(
        taxonomy_rows,
        group_labels=parse_csv_arg(args.group_labels),
        categories=parse_csv_arg(args.categories),
        case_roles=parse_csv_arg(args.case_roles),
        max_cases_per_model=int(args.max_cases_per_model),
    )

    rows_by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in taxonomy_rows:
        rows_by_model[str(row["model_id"])].append(row)

    stage_cache: Dict[str, Dict[str, object]] = {}
    case_rows: List[Dict[str, object]] = []
    for model_id, model_cases in rows_by_model.items():
        model, tok, model_ref = load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        d_ff = int(gate_spec_for_layer(model.model.layers[0]).d_ff)
        try:
            for case in model_cases:
                model_root = Path(case["model_root"])
                cache_key = str(model_root)
                if cache_key not in stage_cache:
                    paths = build_stage_paths(model_root)
                    stage3_summary = read_json(paths["stage3_summary"])
                    stage_cache[cache_key] = {
                        "selected_categories": [str(x) for x in stage3_summary["selected_categories"]],
                        "proto_candidates": read_jsonl(paths["stage5_proto_candidates"]),
                        "inst_candidates": read_jsonl(paths["stage5_inst_candidates"]),
                    }
                cached = stage_cache[cache_key]
                selected_categories = [str(x) for x in cached["selected_categories"]]
                category = str(case["category"])
                prototype_term = str(case["prototype_term"])
                instance_term = str(case["instance_term"])
                proto_candidate = match_candidate_row(cached["proto_candidates"], prototype_term, category)
                inst_candidate = match_candidate_row(cached["inst_candidates"], instance_term, category)
                prototype_indices = [int(x) for x in proto_candidate["candidate_indices"]]
                instance_indices = [int(x) for x in inst_candidate["candidate_indices"]]
                best_strong_indices = [int(x) for x in case["best_strong"]["metrics"]["indices"]]
                best_mixed_indices = [int(x) for x in case["best_mixed"]["metrics"]["indices"]]

                variant_rows = []
                for spec in prompt_variants(instance_term):
                    baseline = category_margin_for_prompt(
                        model,
                        tok,
                        spec["prompt"],
                        correct_category=category,
                        all_categories=selected_categories,
                    )

                    proto_handles = register_ablation(model, prototype_indices, d_ff) if prototype_indices else []
                    try:
                        prototype_readout = category_margin_for_prompt(
                            model,
                            tok,
                            spec["prompt"],
                            correct_category=category,
                            all_categories=selected_categories,
                        )
                    finally:
                        remove_handles(proto_handles)

                    inst_handles = register_ablation(model, instance_indices, d_ff) if instance_indices else []
                    try:
                        instance_readout = category_margin_for_prompt(
                            model,
                            tok,
                            spec["prompt"],
                            correct_category=category,
                            all_categories=selected_categories,
                        )
                    finally:
                        remove_handles(inst_handles)

                    strong_handles = register_ablation(model, best_strong_indices, d_ff) if best_strong_indices else []
                    try:
                        strong_readout = category_margin_for_prompt(
                            model,
                            tok,
                            spec["prompt"],
                            correct_category=category,
                            all_categories=selected_categories,
                        )
                    finally:
                        remove_handles(strong_handles)

                    mixed_handles = register_ablation(model, best_mixed_indices, d_ff) if best_mixed_indices else []
                    try:
                        mixed_readout = category_margin_for_prompt(
                            model,
                            tok,
                            spec["prompt"],
                            correct_category=category,
                            all_categories=selected_categories,
                        )
                    finally:
                        remove_handles(mixed_handles)

                    prototype_drop = effect_drop(baseline, prototype_readout)
                    instance_drop = effect_drop(baseline, instance_readout)
                    strong_drop = effect_drop(baseline, strong_readout)
                    mixed_drop = effect_drop(baseline, mixed_readout)
                    bridge_gain = float(mixed_drop - strong_drop)
                    gate_proxies = compute_gate_proxies(
                        prototype_drop=prototype_drop,
                        instance_drop=instance_drop,
                        strong_drop=strong_drop,
                        mixed_drop=mixed_drop,
                    )
                    variant_rows.append(
                        {
                            "axis": spec["axis"],
                            "variant": spec["variant"],
                            "prompt": spec["prompt"],
                            "baseline": baseline,
                            "prototype_ablated": prototype_readout,
                            "instance_ablated": instance_readout,
                            "strong_ablated": strong_readout,
                            "mixed_ablated": mixed_readout,
                            "prototype_drop": float(prototype_drop),
                            "instance_drop": float(instance_drop),
                            "strong_drop": float(strong_drop),
                            "mixed_drop": float(mixed_drop),
                            "bridge_gain": float(bridge_gain),
                            "gate_proxies": gate_proxies,
                        }
                    )

                axis_gate_summary = compute_axis_gate_summary(variant_rows)
                case_rows.append(
                    {
                        "record_type": "stage56_generation_gate_case",
                        "group_label": str(case["group_label"]),
                        "model_id": model_id,
                        "model_ref": model_ref,
                        "model_root": str(model_root),
                        "category": category,
                        "prototype_term": prototype_term,
                        "instance_term": instance_term,
                        "case_role": str(case["case_role"]),
                        "dominant_structure": str(case["dominant_structure"]),
                        "prototype_indices": prototype_indices,
                        "instance_indices": instance_indices,
                        "best_strong_indices": best_strong_indices,
                        "best_mixed_indices": best_mixed_indices,
                        "axis_gate_summary": axis_gate_summary,
                        "variant_rows": variant_rows,
                    }
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = build_summary(case_rows, runtime_sec=float(time.time() - t0))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "cases.jsonl", case_rows)
    write_report(out_dir / "REPORT.md", summary, case_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "case_count": len(case_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
