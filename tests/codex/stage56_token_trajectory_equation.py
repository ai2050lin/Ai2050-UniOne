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

from deepseek7b_three_pool_structure_scan import gate_spec_for_layer, load_model  # noqa: E402
from stage56_generation_gate_internal_map import prompt_variants  # noqa: E402

AXES = ("style", "logic", "syntax")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mean_list(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    array = np.nan_to_num(np.asarray(vectors, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(x) for x in np.mean(array, axis=0).tolist()]


def mean_matrix(matrices: Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
    if not matrices:
        return []
    array = np.nan_to_num(np.asarray(matrices, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return [
        [float(v) for v in row]
        for row in np.mean(array, axis=0).tolist()
    ]


def profile_values(row: Dict[str, object], raw_key: str, summary_key: str) -> List[float]:
    values = row.get(raw_key)
    if values is None:
        values = row.get(summary_key, [])
    return list(values) if isinstance(values, list) else []


def tail_align(array: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    current = int(arr.shape[0])
    if current >= length:
        return arr[-length:]
    pad_shape = (length - current,) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=np.float32)
    return np.concatenate([pad, arr], axis=0)


def tail_position_labels(length: int) -> List[str]:
    return [f"tail_pos_-{idx}" for idx in range(length, 0, -1)]


def dominant_tail_position(profile: Sequence[float]) -> str:
    if not profile:
        return "tail_pos_-1"
    labels = tail_position_labels(len(profile))
    best_idx = max(range(len(profile)), key=lambda idx: (abs(float(profile[idx])), idx))
    return labels[best_idx]


def dominant_attention_head(values: Sequence[Sequence[float]]) -> str:
    best_value = -1.0
    best_layer = 0
    best_head = 0
    for layer_idx, row in enumerate(values):
        for head_idx, value in enumerate(row):
            current = abs(float(value))
            if current > best_value:
                best_value = current
                best_layer = layer_idx
                best_head = head_idx
    return f"layer_{best_layer}_head_{best_head}"


class FullTokenGateCollector:
    def __init__(self, model) -> None:
        self.layers = list(model.model.layers)
        self.specs = [gate_spec_for_layer(layer) for layer in self.layers]
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for layer_idx, spec in enumerate(self.specs):
            self.handles.append(spec.module.register_forward_hook(self._make_hook(layer_idx)))

    def _make_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            gate = tensor[..., self.specs[layer_idx].gate_start : self.specs[layer_idx].gate_end]
            self.buffers[layer_idx] = gate[0].detach().float().cpu()
            return output

        return _hook

    def reset(self) -> None:
        for idx in range(len(self.buffers)):
            self.buffers[idx] = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_prompt_trace(model, tok, collector: FullTokenGateCollector, prompt: str) -> Dict[str, np.ndarray]:
    collector.reset()
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt")
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.inference_mode():
        out = model(
            **enc,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
    gate_seq = np.stack([buf.numpy() for buf in collector.buffers if buf is not None], axis=0).astype(np.float32)
    hidden_seq = np.stack(
        [state[0].detach().float().cpu().numpy() for state in out.hidden_states],
        axis=0,
    ).astype(np.float32)
    attention_seq = np.stack(
        [att[0].detach().float().cpu().numpy() for att in out.attentions],
        axis=0,
    ).astype(np.float32)
    return {
        "input_ids": enc["input_ids"][0].detach().cpu().numpy(),
        "gate_seq": gate_seq,
        "hidden_seq": hidden_seq,
        "attention_seq": attention_seq,
    }


def build_trace_delta(control: Dict[str, object], variant: Dict[str, object], tail_tokens: int) -> Dict[str, object]:
    control_ids = np.asarray(control["input_ids"], dtype=np.int64)
    variant_ids = np.asarray(variant["input_ids"], dtype=np.int64)
    control_hidden = np.asarray(control["hidden_seq"], dtype=np.float32)
    variant_hidden = np.asarray(variant["hidden_seq"], dtype=np.float32)
    control_gate = np.asarray(control["gate_seq"], dtype=np.float32)
    variant_gate = np.asarray(variant["gate_seq"], dtype=np.float32)
    control_attention = np.asarray(control["attention_seq"], dtype=np.float32)
    variant_attention = np.asarray(variant["attention_seq"], dtype=np.float32)

    hidden_len = min(control_hidden.shape[1], variant_hidden.shape[1])
    gate_len = min(control_gate.shape[1], variant_gate.shape[1])
    attention_len = min(control_attention.shape[2], variant_attention.shape[2])

    hidden_delta = np.linalg.norm(
        variant_hidden[:, -hidden_len:, :] - control_hidden[:, -hidden_len:, :],
        axis=-1,
    )
    gate_delta = np.mean(
        np.abs(variant_gate[:, -gate_len:, :] - control_gate[:, -gate_len:, :]),
        axis=-1,
    )
    attention_delta = np.mean(
        np.abs(
            variant_attention[:, :, -attention_len:, -attention_len:]
            - control_attention[:, :, -attention_len:, -attention_len:]
        ),
        axis=(2, 3),
    )

    hidden_delta = np.nan_to_num(hidden_delta, nan=0.0, posinf=0.0, neginf=0.0)
    gate_delta = np.nan_to_num(gate_delta, nan=0.0, posinf=0.0, neginf=0.0)
    attention_delta = np.nan_to_num(attention_delta, nan=0.0, posinf=0.0, neginf=0.0)

    hidden_token_profile = tail_align(np.mean(hidden_delta, axis=0), tail_tokens)
    mlp_token_profile = tail_align(np.mean(gate_delta, axis=0), tail_tokens)
    hidden_layer_profile = np.mean(hidden_delta, axis=1).astype(np.float32)
    mlp_layer_profile = np.mean(gate_delta, axis=1).astype(np.float32)

    return {
        "control_token_count": int(control_ids.shape[0]),
        "variant_token_count": int(variant_ids.shape[0]),
        "hidden_token_profile": [float(x) for x in hidden_token_profile.tolist()],
        "mlp_token_profile": [float(x) for x in mlp_token_profile.tolist()],
        "hidden_layer_profile": [float(x) for x in hidden_layer_profile.tolist()],
        "mlp_layer_profile": [float(x) for x in mlp_layer_profile.tolist()],
        "attention_head_profile": [[float(v) for v in row] for row in attention_delta.tolist()],
        "dominant_hidden_tail_position": dominant_tail_position(hidden_token_profile.tolist()),
        "dominant_mlp_tail_position": dominant_tail_position(mlp_token_profile.tolist()),
        "dominant_hidden_layer": f"layer_{int(np.argmax(hidden_layer_profile))}",
        "dominant_mlp_layer": f"layer_{int(np.argmax(mlp_layer_profile))}",
        "dominant_attention_head": dominant_attention_head(attention_delta.tolist()),
    }


def select_representative_cases(rows: Sequence[Dict[str, object]], max_cases_per_model: int) -> List[Dict[str, object]]:
    preferred_roles = [
        "strict_positive_pair",
        "weak_bridge_positive",
        "strong_bridge_positive",
        "bridge_dominant",
    ]
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model_id", ""))].append(row)

    selected: List[Dict[str, object]] = []
    for model_id, candidates in sorted(grouped.items()):
        ordered = sorted(
            candidates,
            key=lambda row: (
                preferred_roles.index(str(row.get("case_role", "")))
                if str(row.get("case_role", "")) in preferred_roles
                else len(preferred_roles),
                str(row.get("category", "")),
                str(row.get("instance_term", "")),
            ),
        )
        if max_cases_per_model <= 0:
            selected.extend(ordered)
        else:
            selected.extend(ordered[:max_cases_per_model])
    return selected


def aggregate_axis_rows(rows: Sequence[Dict[str, object]], tail_tokens: int) -> Dict[str, object]:
    mean_hidden_token_profile = mean_list(
        [profile_values(row, "hidden_token_profile", "mean_hidden_token_profile") for row in rows]
    )
    mean_mlp_token_profile = mean_list(
        [profile_values(row, "mlp_token_profile", "mean_mlp_token_profile") for row in rows]
    )
    mean_hidden_layer_profile = mean_list(
        [profile_values(row, "hidden_layer_profile", "mean_hidden_layer_profile") for row in rows]
    )
    mean_mlp_layer_profile = mean_list(
        [profile_values(row, "mlp_layer_profile", "mean_mlp_layer_profile") for row in rows]
    )
    mean_attention_head_profile = mean_matrix(
        [
            row.get("attention_head_profile", row.get("mean_attention_head_profile", []))
            for row in rows
            if row.get("attention_head_profile", row.get("mean_attention_head_profile", []))
        ]
    )

    hidden_layer_array = np.nan_to_num(np.asarray(mean_hidden_layer_profile or [0.0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mlp_layer_array = np.nan_to_num(np.asarray(mean_mlp_layer_profile or [0.0], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "variant_count": len(rows),
        "tail_tokens": tail_tokens,
        "tail_position_labels": tail_position_labels(tail_tokens),
        "mean_hidden_token_profile": mean_hidden_token_profile,
        "mean_mlp_token_profile": mean_mlp_token_profile,
        "mean_hidden_layer_profile": mean_hidden_layer_profile,
        "mean_mlp_layer_profile": mean_mlp_layer_profile,
        "mean_attention_head_profile": mean_attention_head_profile,
        "dominant_hidden_tail_position": dominant_tail_position(mean_hidden_token_profile),
        "dominant_mlp_tail_position": dominant_tail_position(mean_mlp_token_profile),
        "dominant_hidden_layer": f"layer_{int(np.argmax(hidden_layer_array))}",
        "dominant_mlp_layer": f"layer_{int(np.argmax(mlp_layer_array))}",
        "dominant_attention_head": dominant_attention_head(mean_attention_head_profile),
    }


def summarize_case_axis(case: Dict[str, object], axis: str, axis_rows: Sequence[Dict[str, object]], tail_tokens: int) -> Dict[str, object]:
    block = aggregate_axis_rows(axis_rows, tail_tokens=tail_tokens)
    hidden_profile = list(block.get("mean_hidden_token_profile", []))
    mlp_profile = list(block.get("mean_mlp_token_profile", []))
    hidden_total = float(sum(abs(float(v)) for v in hidden_profile))
    mlp_total = float(sum(abs(float(v)) for v in mlp_profile))
    late_window = 4
    hidden_late_focus = float(sum(abs(float(v)) for v in hidden_profile[-late_window:])) / max(hidden_total, 1e-12)
    mlp_late_focus = float(sum(abs(float(v)) for v in mlp_profile[-late_window:])) / max(mlp_total, 1e-12)
    stage6_reference = dict(case.get("stage6_reference", {}))
    return {
        "record_type": "stage56_token_trajectory_case",
        "model_id": str(case.get("model_id", "")),
        "model_ref": str(case.get("model_ref", "")),
        "case_key": (
            f"{case.get('model_id', '')}|{case.get('category', '')}|"
            f"{case.get('prototype_term', '')}|{case.get('instance_term', '')}"
        ),
        "category": str(case.get("category", "")),
        "prototype_term": str(case.get("prototype_term", "")),
        "instance_term": str(case.get("instance_term", "")),
        "case_role": str(case.get("case_role", "")),
        "axis": axis,
        "hidden_total": hidden_total,
        "mlp_total": mlp_total,
        "hidden_late_focus": hidden_late_focus,
        "mlp_late_focus": mlp_late_focus,
        "proto_joint_adv": safe_float(stage6_reference.get("proto_joint_adv")),
        "instance_joint_adv": safe_float(stage6_reference.get("instance_joint_adv")),
        "union_joint_adv": safe_float(stage6_reference.get("union_joint_adv")),
        "union_synergy_joint": safe_float(stage6_reference.get("union_synergy_joint")),
        **block,
    }


def build_axis_stage6_link(case_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    axis_blocks: Dict[str, object] = {}
    for axis in AXES:
        axis_rows = [row for row in case_rows if str(row.get("axis", "")) == axis]
        hidden_totals = [safe_float(row.get("hidden_total")) for row in axis_rows]
        mlp_totals = [safe_float(row.get("mlp_total")) for row in axis_rows]
        hidden_late_focus = [safe_float(row.get("hidden_late_focus")) for row in axis_rows]
        mlp_late_focus = [safe_float(row.get("mlp_late_focus")) for row in axis_rows]
        proto_joint_adv = [safe_float(row.get("proto_joint_adv")) for row in axis_rows]
        instance_joint_adv = [safe_float(row.get("instance_joint_adv")) for row in axis_rows]
        union_joint_adv = [safe_float(row.get("union_joint_adv")) for row in axis_rows]
        union_synergy_joint = [safe_float(row.get("union_synergy_joint")) for row in axis_rows]
        axis_blocks[axis] = {
            "case_count": len(axis_rows),
            "mean_hidden_total": average(hidden_totals),
            "mean_mlp_total": average(mlp_totals),
            "mean_hidden_late_focus": average(hidden_late_focus),
            "mean_mlp_late_focus": average(mlp_late_focus),
            "corr_hidden_total_to_proto_joint_adv": pearson_corr(hidden_totals, proto_joint_adv),
            "corr_hidden_total_to_instance_joint_adv": pearson_corr(hidden_totals, instance_joint_adv),
            "corr_hidden_total_to_union_joint_adv": pearson_corr(hidden_totals, union_joint_adv),
            "corr_hidden_total_to_union_synergy_joint": pearson_corr(hidden_totals, union_synergy_joint),
            "corr_mlp_total_to_proto_joint_adv": pearson_corr(mlp_totals, proto_joint_adv),
            "corr_mlp_total_to_union_joint_adv": pearson_corr(mlp_totals, union_joint_adv),
            "corr_mlp_total_to_union_synergy_joint": pearson_corr(mlp_totals, union_synergy_joint),
            "corr_hidden_late_focus_to_union_synergy_joint": pearson_corr(hidden_late_focus, union_synergy_joint),
            "corr_mlp_late_focus_to_union_synergy_joint": pearson_corr(mlp_late_focus, union_synergy_joint),
        }
    return axis_blocks


def aggregate_case_rows(case_rows: Sequence[Dict[str, object]], tail_tokens: int) -> Dict[str, object]:
    per_model_axis_rows: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in case_rows:
        per_model_axis_rows[str(row["model_id"])][str(row["axis"])].append(row)

    per_model: Dict[str, Dict[str, object]] = {}
    for model_id, axis_map in sorted(per_model_axis_rows.items()):
        per_axis = {
            axis: aggregate_axis_rows(rows, tail_tokens=tail_tokens)
            for axis, rows in sorted(axis_map.items())
        }
        per_model[model_id] = {
            "case_count": len({str(row["case_key"]) for rows in axis_map.values() for row in rows}),
            "per_axis": per_axis,
        }
    return {
        "record_type": "stage56_token_trajectory_equation_summary",
        "tail_tokens": tail_tokens,
        "case_count": len({str(row["case_key"]) for row in case_rows}),
        "model_count": len(per_model),
        "axis_stage6_link": build_axis_stage6_link(case_rows),
        "per_model": per_model,
    }


def build_equation_summary(trajectory_summary: Dict[str, object], equation_summary: Dict[str, object]) -> Dict[str, object]:
    per_model = dict(trajectory_summary.get("per_model", {}))
    closure_coefficients = dict(
        dict(dict(equation_summary.get("equations", {})).get("closure_equation", {})).get("coefficients", {})
    )
    per_model_equations: Dict[str, Dict[str, object]] = {}
    for model_id, block in per_model.items():
        axis_map = dict(block.get("per_axis", {}))
        per_model_equations[model_id] = {
            "token_trajectory_equation": "T*(axis,model)=argmax_t [HiddenShift(axis,t)+MLPDelta(axis,t)+Closure(axis)]",
            "closure_coefficients": closure_coefficients,
            "axes": {
                axis: {
                    "dominant_hidden_tail_position": dict(axis_block).get("dominant_hidden_tail_position", "tail_pos_-1"),
                    "dominant_mlp_tail_position": dict(axis_block).get("dominant_mlp_tail_position", "tail_pos_-1"),
                    "dominant_hidden_layer": dict(axis_block).get("dominant_hidden_layer", "layer_0"),
                    "dominant_mlp_layer": dict(axis_block).get("dominant_mlp_layer", "layer_0"),
                    "dominant_attention_head": dict(axis_block).get("dominant_attention_head", "layer_0_head_0"),
                }
                for axis, axis_block in sorted(axis_map.items())
            },
        }
    return {
        "record_type": "stage56_token_trajectory_equation_model",
        "tail_tokens": trajectory_summary.get("tail_tokens", 0),
        "per_model_equations": per_model_equations,
    }


def write_report(path: Path, summary: Dict[str, object], equation_summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 词元轨迹方程块",
        "",
        f"- case_count: {summary.get('case_count', 0)}",
        f"- model_count: {summary.get('model_count', 0)}",
        f"- tail_tokens: {summary.get('tail_tokens', 0)}",
        "",
        "## per_model",
    ]
    for model_id, block in dict(summary.get("per_model", {})).items():
        lines.append(f"- {model_id}")
        for axis, axis_block in dict(block.get("per_axis", {})).items():
            lines.append(
                f"  - {axis}: hidden={axis_block.get('dominant_hidden_tail_position', '')}, mlp={axis_block.get('dominant_mlp_tail_position', '')}, layer={axis_block.get('dominant_hidden_layer', '')}, head={axis_block.get('dominant_attention_head', '')}"
            )
    lines.extend(["", "## symbolic_form"])
    for model_id, row in dict(equation_summary.get("per_model_equations", {})).items():
        lines.append(f"- {model_id}: {row.get('token_trajectory_equation', '')}")
    lines.extend(["", "## axis_stage6_link"])
    for axis, row in dict(summary.get("axis_stage6_link", {})).items():
        lines.append(
            f"- {axis}: hidden->proto={safe_float(row.get('corr_hidden_total_to_proto_joint_adv')):.4f}, "
            f"hidden->synergy={safe_float(row.get('corr_hidden_total_to_union_synergy_joint')):.4f}, "
            f"mlp->synergy={safe_float(row.get('corr_mlp_total_to_union_synergy_joint')):.4f}, "
            f"hidden_late->synergy={safe_float(row.get('corr_hidden_late_focus_to_union_synergy_joint')):.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lift static ICSPB layer predictions into token trajectory equations")
    ap.add_argument(
        "--cases-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generation_gate_internal_map_20260318_1338" / "cases.jsonl"),
    )
    ap.add_argument(
        "--equation-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_equationization_20260318_2258" / "summary.json"),
    )
    ap.add_argument("--model-ids", default="Qwen/Qwen3-4B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,zai-org/GLM-4-9B-Chat-HF")
    ap.add_argument("--max-cases-per-model", type=int, default=1)
    ap.add_argument("--tail-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_token_trajectory_equation_20260318"),
    )
    return ap.parse_args()


def parse_model_ids(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    started = time.time()
    cases = read_jsonl(Path(args.cases_jsonl))
    equation_summary = read_json(Path(args.equation_summary_json))
    wanted_models = set(parse_model_ids(args.model_ids))
    selected_cases = [
        row
        for row in select_representative_cases(cases, max_cases_per_model=args.max_cases_per_model)
        if str(row.get("model_id", "")) in wanted_models
    ]

    output_rows: List[Dict[str, object]] = []
    grouped_cases: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for case in selected_cases:
        grouped_cases[str(case["model_id"])].append(case)

    for model_id, cases_for_model in sorted(grouped_cases.items()):
        model, tok, model_ref = load_model(
            model_id=model_id,
            dtype_name=args.dtype,
            local_files_only=bool(args.local_files_only),
            device=args.device,
        )
        collector = FullTokenGateCollector(model)
        try:
            for case in cases_for_model:
                term = str(case["instance_term"])
                prompts = prompt_variants(term)
                control_prompt = next(item["prompt"] for item in prompts if str(item["axis"]) == "control")
                control_trace = run_prompt_trace(model, tok, collector, str(control_prompt))
                for axis in AXES:
                    axis_rows: List[Dict[str, object]] = []
                    for item in prompts:
                        if str(item["axis"]) != axis:
                            continue
                        variant_trace = run_prompt_trace(model, tok, collector, str(item["prompt"]))
                        axis_rows.append(build_trace_delta(control_trace, variant_trace, tail_tokens=args.tail_tokens))
                    summarized = summarize_case_axis(case, axis=axis, axis_rows=axis_rows, tail_tokens=args.tail_tokens)
                    summarized["model_ref"] = model_ref
                    output_rows.append(summarized)
        finally:
            collector.close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = aggregate_case_rows(output_rows, tail_tokens=args.tail_tokens)
    summary["runtime_sec"] = float(time.time() - started)
    equation_model = build_equation_summary(summary, equation_summary)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cases.jsonl", output_rows)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "equation_summary.json", equation_model)
    write_report(out_dir / "REPORT.md", summary, equation_model)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "case_count": summary.get("case_count", 0),
                "model_count": summary.get("model_count", 0),
                "runtime_sec": summary.get("runtime_sec", 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
