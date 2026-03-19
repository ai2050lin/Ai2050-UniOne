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


def parse_csv_arg(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


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


def build_layer_profile(gate_by_layer: np.ndarray, indices: Sequence[int], d_ff: int) -> List[float]:
    gate_by_layer = np.asarray(gate_by_layer, dtype=np.float32)
    n_layers = int(gate_by_layer.shape[0])
    values = np.zeros(n_layers, dtype=np.float32)
    counts = np.zeros(n_layers, dtype=np.float32)
    for raw_idx in indices:
        idx = int(raw_idx)
        layer_idx = idx // d_ff
        local_idx = idx % d_ff
        if 0 <= layer_idx < n_layers:
            values[layer_idx] += abs(float(gate_by_layer[layer_idx, local_idx]))
            counts[layer_idx] += 1.0
    out = []
    for layer_idx in range(n_layers):
        if counts[layer_idx] > 0:
            out.append(float(values[layer_idx] / counts[layer_idx]))
        else:
            out.append(0.0)
    return out


def subset_global_mean(gate_flat: np.ndarray, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    values = [abs(float(gate_flat[int(idx)])) for idx in indices]
    return safe_mean(values)


def dominant_layer_label(values: Sequence[float]) -> str:
    if not values:
        return "layer_0"
    layer_idx = int(max(range(len(values)), key=lambda idx: (abs(float(values[idx])), -idx)))
    return f"layer_{layer_idx}"


def dominant_head_label(values: Sequence[Sequence[float]]) -> str:
    if not values:
        return "layer_0_head_0"
    best = (0.0, 0, 0)
    for layer_idx, row in enumerate(values):
        for head_idx, value in enumerate(row):
            candidate = (abs(float(value)), -layer_idx, -head_idx)
            if candidate > best:
                best = candidate
                best_value = float(value)
                best_layer = layer_idx
                best_head = head_idx
    if best[0] == 0.0:
        best_value = 0.0
        best_layer = 0
        best_head = 0
    _ = best_value
    return f"layer_{best_layer}_head_{best_head}"


def run_prompt_internal(model, tok, collector, prompt: str) -> Dict[str, object]:
    collector.reset()
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(
            **enc,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
    gate_by_layer = np.stack([buf.numpy() for buf in collector.buffers if buf is not None], axis=0).astype(np.float32)
    gate_flat = collector.get_flat()
    hidden_last = np.stack(
        [state[0, -1, :].detach().float().cpu().numpy() for state in out.hidden_states],
        axis=0,
    ).astype(np.float32)
    attention_last = np.stack(
        [att[0, :, -1, :].detach().float().cpu().numpy().mean(axis=-1) for att in out.attentions],
        axis=0,
    ).astype(np.float32)
    return {
        "gate_by_layer": gate_by_layer,
        "gate_flat": gate_flat,
        "hidden_last": hidden_last,
        "attention_last": attention_last,
    }


def build_variant_internal_row(
    stats: Dict[str, object],
    control: Dict[str, object],
    d_ff: int,
    prototype_indices: Sequence[int],
    instance_indices: Sequence[int],
    strong_indices: Sequence[int],
    mixed_indices: Sequence[int],
) -> Dict[str, object]:
    gate_flat = np.asarray(stats["gate_flat"], dtype=np.float32)
    gate_by_layer = np.asarray(stats["gate_by_layer"], dtype=np.float32)
    control_gate_by_layer = np.asarray(control["gate_by_layer"], dtype=np.float32)
    hidden_last = np.asarray(stats["hidden_last"], dtype=np.float32)
    control_hidden_last = np.asarray(control["hidden_last"], dtype=np.float32)
    attention_last = np.asarray(stats["attention_last"], dtype=np.float32)
    control_attention_last = np.asarray(control["attention_last"], dtype=np.float32)

    hidden_shift = np.linalg.norm(hidden_last - control_hidden_last, axis=1).astype(np.float32)
    attention_delta = (attention_last - control_attention_last).astype(np.float32)
    mlp_layer_delta = np.mean(np.abs(gate_by_layer - control_gate_by_layer), axis=1).astype(np.float32)
    prototype_profile = build_layer_profile(gate_by_layer, prototype_indices, d_ff)
    instance_profile = build_layer_profile(gate_by_layer, instance_indices, d_ff)
    strong_profile = build_layer_profile(gate_by_layer, strong_indices, d_ff)
    mixed_profile = build_layer_profile(gate_by_layer, mixed_indices, d_ff)
    return {
        "prototype_gate_mean": float(subset_global_mean(gate_flat, prototype_indices)),
        "instance_gate_mean": float(subset_global_mean(gate_flat, instance_indices)),
        "strong_gate_mean": float(subset_global_mean(gate_flat, strong_indices)),
        "mixed_gate_mean": float(subset_global_mean(gate_flat, mixed_indices)),
        "prototype_layer_profile": prototype_profile,
        "instance_layer_profile": instance_profile,
        "strong_layer_profile": strong_profile,
        "mixed_layer_profile": mixed_profile,
        "hidden_shift_profile": [float(x) for x in hidden_shift.tolist()],
        "mlp_layer_delta_profile": [float(x) for x in mlp_layer_delta.tolist()],
        "attention_head_delta_profile": [[float(v) for v in row] for row in attention_delta.tolist()],
    }


def mean_list(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    return [float(x) for x in np.mean(np.asarray(vectors, dtype=np.float32), axis=0).tolist()]


def mean_matrix(matrices: Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
    if not matrices:
        return []
    return [
        [float(v) for v in row]
        for row in np.mean(np.asarray(matrices, dtype=np.float32), axis=0).tolist()
    ]


def aggregate_axis_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "variant_count": 0,
            "mean_prototype_gate_delta": 0.0,
            "mean_instance_gate_delta": 0.0,
            "mean_strong_gate_delta": 0.0,
            "mean_mixed_gate_delta": 0.0,
            "mean_bridge_gate_delta": 0.0,
            "mean_hidden_shift_profile": [],
            "mean_mlp_layer_delta_profile": [],
            "mean_attention_head_delta_profile": [],
            "dominant_hidden_layer": "layer_0",
            "dominant_mlp_layer": "layer_0",
            "dominant_attention_head": "layer_0_head_0",
        }
    hidden_profile = mean_list([row["hidden_shift_profile"] for row in rows])
    mlp_profile = mean_list([row["mlp_layer_delta_profile"] for row in rows])
    attention_profile = mean_matrix([row["attention_head_delta_profile"] for row in rows])
    return {
        "variant_count": len(rows),
        "mean_prototype_gate_delta": safe_mean([float(row["prototype_gate_delta"]) for row in rows]),
        "mean_instance_gate_delta": safe_mean([float(row["instance_gate_delta"]) for row in rows]),
        "mean_strong_gate_delta": safe_mean([float(row["strong_gate_delta"]) for row in rows]),
        "mean_mixed_gate_delta": safe_mean([float(row["mixed_gate_delta"]) for row in rows]),
        "mean_bridge_gate_delta": safe_mean([float(row["bridge_gate_delta"]) for row in rows]),
        "mean_hidden_shift_profile": hidden_profile,
        "mean_mlp_layer_delta_profile": mlp_profile,
        "mean_attention_head_delta_profile": attention_profile,
        "dominant_hidden_layer": dominant_layer_label(hidden_profile),
        "dominant_mlp_layer": dominant_layer_label(mlp_profile),
        "dominant_attention_head": dominant_head_label(attention_profile),
    }


def aggregate_axis_blocks(blocks: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not blocks:
        return aggregate_axis_rows([])
    hidden_profile = mean_list([block["mean_hidden_shift_profile"] for block in blocks])
    mlp_profile = mean_list([block["mean_mlp_layer_delta_profile"] for block in blocks])
    attention_profile = mean_matrix([block["mean_attention_head_delta_profile"] for block in blocks])
    return {
        "variant_count": int(sum(int(block["variant_count"]) for block in blocks)),
        "mean_prototype_gate_delta": safe_mean([float(block["mean_prototype_gate_delta"]) for block in blocks]),
        "mean_instance_gate_delta": safe_mean([float(block["mean_instance_gate_delta"]) for block in blocks]),
        "mean_strong_gate_delta": safe_mean([float(block["mean_strong_gate_delta"]) for block in blocks]),
        "mean_mixed_gate_delta": safe_mean([float(block["mean_mixed_gate_delta"]) for block in blocks]),
        "mean_bridge_gate_delta": safe_mean([float(block["mean_bridge_gate_delta"]) for block in blocks]),
        "mean_hidden_shift_profile": hidden_profile,
        "mean_mlp_layer_delta_profile": mlp_profile,
        "mean_attention_head_delta_profile": attention_profile,
        "dominant_hidden_layer": dominant_layer_label(hidden_profile),
        "dominant_mlp_layer": dominant_layer_label(mlp_profile),
        "dominant_attention_head": dominant_head_label(attention_profile),
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 Generation Gate Internal Map Report",
        "",
        f"- Case count: {summary['case_count']}",
        f"- Model count: {summary['model_count']}",
        "",
        "## Per Model",
    ]
    for model_id, model_block in summary["per_model"].items():
        lines.append(f"- {model_id}: cases={model_block['case_count']}")
        for axis in AXES:
            axis_block = model_block["per_axis"][axis]
            lines.append(
                "  - "
                f"{axis}: "
                f"proto={axis_block['mean_prototype_gate_delta']:.6f}, "
                f"inst={axis_block['mean_instance_gate_delta']:.6f}, "
                f"strong={axis_block['mean_strong_gate_delta']:.6f}, "
                f"mixed={axis_block['mean_mixed_gate_delta']:.6f}, "
                f"bridge={axis_block['mean_bridge_gate_delta']:.6f}, "
                f"hidden={axis_block['dominant_hidden_layer']}, "
                f"mlp={axis_block['dominant_mlp_layer']}, "
                f"head={axis_block['dominant_attention_head']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map generation gate prompt axes to internal layer/head/MLP coordinates")
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
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_internal_map_20260318"),
    )
    return ap.parse_args()


def main() -> None:
    from deepseek7b_three_pool_structure_scan import GateCollector, gate_spec_for_layer, load_model
    from stage56_generation_gate_coupling import read_json as coupling_read_json
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
        collector = GateCollector(model)
        d_ff = int(gate_spec_for_layer(model.model.layers[0]).d_ff)
        try:
            for case in model_cases:
                model_root = Path(case["model_root"])
                cache_key = str(model_root)
                if cache_key not in stage_cache:
                    paths = build_stage_paths(model_root)
                    stage3_summary = coupling_read_json(paths["stage3_summary"])
                    stage_cache[cache_key] = {
                        "selected_categories": [str(x) for x in stage3_summary["selected_categories"]],
                        "proto_candidates": read_jsonl(paths["stage5_proto_candidates"]),
                        "inst_candidates": read_jsonl(paths["stage5_inst_candidates"]),
                    }
                cached = stage_cache[cache_key]
                category = str(case["category"])
                prototype_term = str(case["prototype_term"])
                instance_term = str(case["instance_term"])
                proto_candidate = match_candidate_row(cached["proto_candidates"], prototype_term, category)
                inst_candidate = match_candidate_row(cached["inst_candidates"], instance_term, category)
                prototype_indices = [int(x) for x in proto_candidate["candidate_indices"]]
                instance_indices = [int(x) for x in inst_candidate["candidate_indices"]]
                best_strong_indices = [int(x) for x in case["best_strong"]["metrics"]["indices"]]
                best_mixed_indices = [int(x) for x in case["best_mixed"]["metrics"]["indices"]]

                prompt_rows = []
                internal_rows = []
                for spec in prompt_variants(instance_term):
                    prompt_row = {
                        "axis": spec["axis"],
                        "variant": spec["variant"],
                        "prompt": spec["prompt"],
                        "stats": run_prompt_internal(model, tok, collector, spec["prompt"]),
                    }
                    prompt_rows.append(prompt_row)

                control_row = next(row for row in prompt_rows if row["axis"] == "control")
                control_internal = build_variant_internal_row(
                    control_row["stats"],
                    control_row["stats"],
                    d_ff=d_ff,
                    prototype_indices=prototype_indices,
                    instance_indices=instance_indices,
                    strong_indices=best_strong_indices,
                    mixed_indices=best_mixed_indices,
                )
                for prompt_row in prompt_rows:
                    variant_internal = build_variant_internal_row(
                        prompt_row["stats"],
                        control_row["stats"],
                        d_ff=d_ff,
                        prototype_indices=prototype_indices,
                        instance_indices=instance_indices,
                        strong_indices=best_strong_indices,
                        mixed_indices=best_mixed_indices,
                    )
                    internal_rows.append(
                        {
                            "axis": prompt_row["axis"],
                            "variant": prompt_row["variant"],
                            "prompt": prompt_row["prompt"],
                            "prototype_gate_delta": float(
                                variant_internal["prototype_gate_mean"] - control_internal["prototype_gate_mean"]
                            ),
                            "instance_gate_delta": float(
                                variant_internal["instance_gate_mean"] - control_internal["instance_gate_mean"]
                            ),
                            "strong_gate_delta": float(
                                variant_internal["strong_gate_mean"] - control_internal["strong_gate_mean"]
                            ),
                            "mixed_gate_delta": float(
                                variant_internal["mixed_gate_mean"] - control_internal["mixed_gate_mean"]
                            ),
                            "bridge_gate_delta": float(
                                (variant_internal["mixed_gate_mean"] - control_internal["mixed_gate_mean"])
                                - (variant_internal["strong_gate_mean"] - control_internal["strong_gate_mean"])
                            ),
                            "prototype_layer_profile": variant_internal["prototype_layer_profile"],
                            "instance_layer_profile": variant_internal["instance_layer_profile"],
                            "strong_layer_profile": variant_internal["strong_layer_profile"],
                            "mixed_layer_profile": variant_internal["mixed_layer_profile"],
                            "hidden_shift_profile": variant_internal["hidden_shift_profile"],
                            "mlp_layer_delta_profile": variant_internal["mlp_layer_delta_profile"],
                            "attention_head_delta_profile": variant_internal["attention_head_delta_profile"],
                        }
                    )

                axis_summary = {
                    axis: aggregate_axis_rows([row for row in internal_rows if row["axis"] == axis])
                    for axis in AXES
                }
                case_rows.append(
                    {
                        "record_type": "stage56_generation_gate_internal_case",
                        "group_label": str(case["group_label"]),
                        "model_id": model_id,
                        "model_ref": model_ref,
                        "category": category,
                        "prototype_term": prototype_term,
                        "instance_term": instance_term,
                        "case_role": str(case["case_role"]),
                        "dominant_structure": str(case["dominant_structure"]),
                        "axis_internal_summary": axis_summary,
                        "internal_rows": internal_rows,
                    }
                )
        finally:
            collector.close()
            del collector
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "record_type": "stage56_generation_gate_internal_map_summary",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": float(time.time() - t0),
        "case_count": len(case_rows),
        "model_count": len(rows_by_model),
        "per_model": {
            model_id: {
                "case_count": len(model_rows),
                "per_axis": {
                    axis: aggregate_axis_blocks(
                        [row["axis_internal_summary"][axis] for row in model_rows]
                    )
                    for axis in AXES
                },
            }
            for model_id, model_rows in sorted(
                ((model_id, [row for row in case_rows if row["model_id"] == model_id]) for model_id in rows_by_model),
                key=lambda kv: kv[0],
            )
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "cases.jsonl", case_rows)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "summary": str(out_dir / "summary.json"),
                "case_count": len(case_rows),
                "model_count": len(rows_by_model),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
