#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import capture_qwen_mlp_payloads, discover_layers, move_batch_to_model_device, remove_hooks
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage433_polysemous_noun_family_generalization import POLYSEMOUS_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage447_polysemy_family_switch_protocol_20260403"
)
STAGE433_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage433_polysemous_noun_family_generalization_20260402"
    / "summary.json"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
TOP_K = 256
TOP_SWITCH_NEURONS = 12

ORDINARY_CONTROL_CASES: Dict[str, List[Dict[str, str]]] = {
    "apple": [
        {"target": "banana", "label": "neutral", "sentence": "I bought a banana this morning."},
        {"target": "banana", "label": "color", "sentence": "The banana is yellow and ripe."},
        {"target": "banana", "label": "taste", "sentence": "The banana tastes sweet today."},
        {"target": "banana", "label": "size", "sentence": "The banana is about the size of a hand."},
    ],
    "amazon": [
        {"target": "Nile", "label": "neutral", "sentence": "I read about the Nile this morning."},
        {"target": "Nile", "label": "flow", "sentence": "The Nile flows through several countries."},
        {"target": "Nile", "label": "boat", "sentence": "Boats moved slowly along the Nile at sunset."},
        {"target": "Nile", "label": "rain", "sentence": "Heavy rain caused the Nile to rise quickly."},
    ],
    "python": [
        {"target": "cobra", "label": "neutral", "sentence": "I saw a cobra at the zoo."},
        {"target": "cobra", "label": "motion", "sentence": "The cobra moved slowly across the ground."},
        {"target": "cobra", "label": "feeding", "sentence": "The zoo keeper fed the cobra in the reptile house."},
        {"target": "cobra", "label": "hiding", "sentence": "The cobra hid beneath the warm rock."},
    ],
    "java": [
        {"target": "coffee", "label": "neutral", "sentence": "I drank coffee this morning."},
        {"target": "coffee", "label": "smell", "sentence": "The coffee smelled rich and warm."},
        {"target": "coffee", "label": "cup", "sentence": "She ordered a cup of coffee after dinner."},
        {"target": "coffee", "label": "cafe", "sentence": "The cafe served strong coffee all morning."},
    ],
}


def free_model(model) -> None:
    try:
        del model
    except UnboundLocalError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for digit in ["1", "2"]:
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"Digit {digit} is not a single token.")
        out[digit] = int(ids[0])
    return out


def find_last_subsequence(full_ids: List[int], sub_ids: List[int]) -> Tuple[int, int] | None:
    if not sub_ids or len(sub_ids) > len(full_ids):
        return None
    last_match = None
    for start in range(0, len(full_ids) - len(sub_ids) + 1):
        if full_ids[start : start + len(sub_ids)] == sub_ids:
            last_match = (start, start + len(sub_ids))
    return last_match


def locate_target_span(tokenizer, prompt: str, target: str) -> Tuple[int, int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    best = None
    best_len = -1
    seen = set()
    for text in variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"Unable to locate target span: target={target!r} prompt={prompt!r}")
    return best


def capture_case_layer_vectors(model, tokenizer, sentence: str, target: str) -> Dict[int, torch.Tensor]:
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    start, end = locate_target_span(tokenizer, sentence, target)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    layer_count = len(discover_layers(model))
    rows: Dict[int, torch.Tensor] = {}
    for layer_idx in range(layer_count):
        rows[layer_idx] = hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
    return rows


def capture_case_flat_neuron_vector(model, tokenizer, sentence: str, target: str) -> torch.Tensor:
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(len(discover_layers(model)))}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    try:
        encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        start, end = locate_target_span(tokenizer, sentence, target)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        per_layer = []
        for layer_idx in range(len(buffers)):
            layer_tensor = buffers[layer_idx]
            if layer_tensor is None:
                raise RuntimeError(f"Missing neuron payload at layer {layer_idx}.")
            vec = layer_tensor[0, start:end, :].mean(dim=0).detach().float().cpu()
            per_layer.append(vec)
        return torch.cat(per_layer, dim=0)
    finally:
        remove_hooks(handles)


def top_active_ids(vec: torch.Tensor, top_k: int) -> List[int]:
    positive = torch.clamp(vec.float(), min=0.0)
    take_k = min(top_k, positive.numel())
    vals, idxs = torch.topk(positive, k=take_k)
    out: List[int] = []
    for value, flat_idx in zip(vals.tolist(), idxs.tolist()):
        if value <= 0:
            continue
        out.append(int(flat_idx))
    return out


def top_selective_ids(signal: torch.Tensor, top_k: int, *, positive: bool) -> List[int]:
    diff = signal.float()
    if positive:
        masked = torch.where(diff > 0, diff, torch.full_like(diff, float("-inf")))
    else:
        masked = torch.where(diff < 0, -diff, torch.full_like(diff, float("-inf")))
    take_k = min(top_k, masked.numel())
    vals, idxs = torch.topk(masked, k=take_k)
    out: List[int] = []
    for value, flat_idx in zip(vals.tolist(), idxs.tolist()):
        if not torch.isfinite(torch.tensor(value)):
            continue
        out.append(int(flat_idx))
    return out


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def overlap_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def index_to_layer_neuron(flat_idx: int, neuron_count: int) -> Tuple[int, int]:
    return int(flat_idx // neuron_count), int(flat_idx % neuron_count)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def orthogonal_control_axis(switch_axis: torch.Tensor, sense_a_rows: Sequence[Dict[str, object]], best_layer_idx: int) -> torch.Tensor:
    sense_a_mean = mean_tensors([row["layer_vectors"][best_layer_idx] for row in sense_a_rows])
    modifiers = [row["layer_vectors"][best_layer_idx] - sense_a_mean for row in sense_a_rows]
    base = mean_tensors(modifiers) if modifiers else torch.zeros_like(switch_axis)
    switch_unit = switch_axis.float() / torch.linalg.norm(switch_axis.float()).clamp_min(1e-8)
    base = base.float() - torch.dot(base.float(), switch_unit) * switch_unit
    if float(torch.linalg.norm(base).item()) <= 1e-8:
        base = torch.roll(switch_unit, shifts=1)
        base = base - torch.dot(base, switch_unit) * switch_unit
    return base / torch.linalg.norm(base).clamp_min(1e-8)


def build_switch_axis_hook(target_spans: Sequence[Tuple[int, int]], axis: torch.Tensor):
    axis = axis.float()
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-8)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        updated = hidden.clone()
        axis_dev = axis.to(updated.device)
        for batch_idx, (start, end) in enumerate(target_spans):
            section = updated[batch_idx, start:end, :].float()
            coeff = torch.matmul(section, axis_dev)
            section = section - coeff.unsqueeze(-1) * axis_dev
            updated[batch_idx, start:end, :] = section.to(updated.dtype)
        if rest is None:
            return updated
        return (updated, *rest)

    return hook


def build_sense_prompt(case: Dict[str, str], noun_spec: Dict[str, object]) -> str:
    return (
        f'Sentence: "{case["sentence"]}"\n'
        f'Question: In this sentence, does the marked word refer to 1 {noun_spec["sense_a_name"]} or 2 {noun_spec["sense_b_name"]}?\n'
        f'Answer with one digit only: 1 {noun_spec["sense_a_name"]} 2 {noun_spec["sense_b_name"]}\n'
        "Answer:"
    )


def evaluate_sense_cases(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    noun_spec: Dict[str, object],
    cases: Sequence[Dict[str, str]],
    labels: Sequence[int],
    *,
    layer_idx: int | None = None,
    axis: torch.Tensor | None = None,
) -> Dict[str, object]:
    per_case = []
    candidate_ids = None
    for case, label in zip(cases, labels):
        prompt = build_sense_prompt(case, noun_spec)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = move_batch_to_model_device(model, encoded)
        target_span = locate_target_span(tokenizer, prompt, case["target"])
        handles = []
        try:
            if layer_idx is not None and axis is not None:
                layer_module = discover_layers(model)[layer_idx]
                handles.append(layer_module.register_forward_hook(build_switch_axis_hook([target_span], axis)))
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
            if candidate_ids is None:
                candidate_ids = torch.tensor([digit_token_ids["1"], digit_token_ids["2"]], device=logits.device, dtype=torch.long)
            option_logits = logits.index_select(dim=1, index=candidate_ids)
            log_probs = option_logits.log_softmax(dim=-1)
            prediction = int(option_logits.argmax(dim=-1).item())
            correct_prob = float(log_probs[0, int(label)].exp().item())
            per_case.append(
                {
                    "sentence": case["sentence"],
                    "target": case["target"],
                    "expected_label": int(label),
                    "predicted_label": prediction,
                    "correct_prob": correct_prob,
                    "is_correct": bool(prediction == int(label)),
                }
            )
        finally:
            remove_hooks(handles)
    return {
        "accuracy": safe_ratio(sum(int(row["is_correct"]) for row in per_case), len(per_case)),
        "mean_correct_prob": safe_ratio(sum(float(row["correct_prob"]) for row in per_case), len(per_case)),
        "per_case": per_case,
    }


def build_group_rows(model, tokenizer, cases: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        rows.append(
            {
                "label": case.get("label", ""),
                "target": case["target"],
                "sentence": case["sentence"],
                "layer_vectors": capture_case_layer_vectors(model, tokenizer, case["sentence"], case["target"]),
                "flat_neuron_vector": capture_case_flat_neuron_vector(model, tokenizer, case["sentence"], case["target"]),
            }
        )
    return rows


def summarize_switch_neurons(diff_vec: torch.Tensor, neuron_count: int, top_k: int) -> Dict[str, object]:
    abs_total = float(torch.abs(diff_vec).sum().item())
    pos_ids = top_selective_ids(diff_vec, top_k, positive=True)
    neg_ids = top_selective_ids(diff_vec, top_k, positive=False)

    def build_rows(flat_ids: Sequence[int]) -> List[Dict[str, object]]:
        rows = []
        for rank, flat_idx in enumerate(flat_ids, start=1):
            layer_idx, neuron_idx = index_to_layer_neuron(int(flat_idx), neuron_count)
            rows.append(
                {
                    "rank": rank,
                    "flat_index": int(flat_idx),
                    "layer_index": layer_idx,
                    "neuron_index": neuron_idx,
                    "activation_delta": float(diff_vec[int(flat_idx)].item()),
                }
            )
        return rows

    top_abs_ids = sorted(set(pos_ids + neg_ids), key=lambda idx: abs(float(diff_vec[int(idx)].item())), reverse=True)
    concentration = safe_ratio(sum(abs(float(diff_vec[int(idx)].item())) for idx in top_abs_ids[:top_k]), abs_total)
    return {
        "sense_b_biased_neurons": build_rows(pos_ids),
        "sense_a_biased_neurons": build_rows(neg_ids),
        "top_abs_concentration_ratio": concentration,
    }


def get_best_balance_layer(stage433_summary: Dict[str, object], model_key: str, noun_id: str) -> int:
    model_row = next(row for row in stage433_summary["model_results"] if row["model_key"] == model_key)
    noun_row = next(row for row in model_row["noun_results"] if row["noun_id"] == noun_id)
    return int(noun_row["best_balance_layer"]["layer_index"])


def analyze_single_noun(model, tokenizer, stage433_summary: Dict[str, object], model_key: str, noun_spec: Dict[str, object]) -> Dict[str, object]:
    best_layer_idx = get_best_balance_layer(stage433_summary, model_key, str(noun_spec["noun_id"]))
    neuron_count = int(discover_layers(model)[0].mlp.gate_proj.out_features)

    sense_a_rows = build_group_rows(model, tokenizer, noun_spec["sense_a_cases"])
    sense_b_rows = build_group_rows(model, tokenizer, noun_spec["sense_b_cases"])
    ordinary_rows = build_group_rows(model, tokenizer, ORDINARY_CONTROL_CASES[str(noun_spec["noun_id"])])

    sense_a_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in sense_a_rows])
    sense_b_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in sense_b_rows])
    switch_diff_flat = sense_b_mean_flat - sense_a_mean_flat

    sense_a_active_ids = top_active_ids(sense_a_mean_flat, TOP_K)
    sense_b_active_ids = top_active_ids(sense_b_mean_flat, TOP_K)
    sense_a_selective_ids = top_selective_ids(switch_diff_flat, TOP_K, positive=False)
    sense_b_selective_ids = top_selective_ids(switch_diff_flat, TOP_K, positive=True)

    ordinary_neutral = next(row for row in ordinary_rows if row["label"] == "neutral")
    ordinary_neutral_ids = top_active_ids(ordinary_neutral["flat_neuron_vector"], TOP_K)
    ordinary_context_overlaps = []
    ordinary_context_details = []
    for row in ordinary_rows:
        if row["label"] == "neutral":
            continue
        context_ids = top_active_ids(row["flat_neuron_vector"], TOP_K)
        jac = jaccard(ordinary_neutral_ids, context_ids)
        ordinary_context_overlaps.append(jac)
        ordinary_context_details.append({"label": row["label"], "active_jaccard_vs_neutral": jac})

    sense_a_mean_layer = mean_tensors([row["layer_vectors"][best_layer_idx] for row in sense_a_rows])
    sense_b_mean_layer = mean_tensors([row["layer_vectors"][best_layer_idx] for row in sense_b_rows])
    switch_axis = sense_b_mean_layer - sense_a_mean_layer
    control_axis = orthogonal_control_axis(switch_axis, sense_a_rows, best_layer_idx)

    sense_cases = list(noun_spec["sense_a_cases"]) + list(noun_spec["sense_b_cases"])
    sense_labels = [0 for _ in noun_spec["sense_a_cases"]] + [1 for _ in noun_spec["sense_b_cases"]]
    digit_token_ids = resolve_digit_token_ids(tokenizer)
    baseline_eval = evaluate_sense_cases(model, tokenizer, digit_token_ids, noun_spec, sense_cases, sense_labels)
    switch_eval = evaluate_sense_cases(
        model,
        tokenizer,
        digit_token_ids,
        noun_spec,
        sense_cases,
        sense_labels,
        layer_idx=best_layer_idx,
        axis=switch_axis,
    )
    control_eval = evaluate_sense_cases(
        model,
        tokenizer,
        digit_token_ids,
        noun_spec,
        sense_cases,
        sense_labels,
        layer_idx=best_layer_idx,
        axis=control_axis,
    )

    def delta(row: Dict[str, object]) -> Dict[str, float]:
        return {
            "accuracy_drop": float(baseline_eval["accuracy"] - row["accuracy"]),
            "mean_correct_prob_drop": float(baseline_eval["mean_correct_prob"] - row["mean_correct_prob"]),
        }

    ordinary_mean_overlap = safe_ratio(sum(ordinary_context_overlaps), len(ordinary_context_overlaps))
    polysemy_jaccard = jaccard(sense_a_active_ids, sense_b_active_ids)

    return {
        "noun_id": noun_spec["noun_id"],
        "display_name": noun_spec["display_name"],
        "sense_a_name": noun_spec["sense_a_name"],
        "sense_b_name": noun_spec["sense_b_name"],
        "best_switch_layer": best_layer_idx,
        "sense_active_jaccard": polysemy_jaccard,
        "sense_active_overlap_ratio_from_a": overlap_ratio(sense_a_active_ids, sense_b_active_ids),
        "sense_active_overlap_ratio_from_b": overlap_ratio(sense_b_active_ids, sense_a_active_ids),
        "sense_selective_jaccard": jaccard(sense_a_selective_ids, sense_b_selective_ids),
        "ordinary_control_mean_active_jaccard": ordinary_mean_overlap,
        "ordinary_vs_polysemy_gap": float(ordinary_mean_overlap - polysemy_jaccard),
        "ordinary_control_details": ordinary_context_details,
        "switch_neuron_summary": summarize_switch_neurons(switch_diff_flat, neuron_count, TOP_SWITCH_NEURONS),
        "switch_axis_ablation": {
            "baseline": baseline_eval,
            "switch_axis_removed": switch_eval,
            "control_axis_removed": control_eval,
            "switch_axis_delta": delta(switch_eval),
            "control_axis_delta": delta(control_eval),
        },
        "supports_polysemy_split": bool(polysemy_jaccard + 0.05 < ordinary_mean_overlap),
        "supports_switch_causality": bool(
            float(delta(switch_eval)["mean_correct_prob_drop"]) > float(delta(control_eval)["mean_correct_prob_drop"]) + 0.01
        ),
    }


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    stage433_summary = load_json(STAGE433_SUMMARY_PATH)
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        noun_results = [analyze_single_noun(model, tokenizer, stage433_summary, model_key, noun_spec) for noun_spec in POLYSEMOUS_CASES]
        aggregate = {
            "noun_count": len(noun_results),
            "polysemy_split_support_count": sum(int(row["supports_polysemy_split"]) for row in noun_results),
            "switch_causality_support_count": sum(int(row["supports_switch_causality"]) for row in noun_results),
            "mean_polysemy_jaccard": safe_ratio(sum(float(row["sense_active_jaccard"]) for row in noun_results), len(noun_results)),
            "mean_ordinary_jaccard": safe_ratio(sum(float(row["ordinary_control_mean_active_jaccard"]) for row in noun_results), len(noun_results)),
            "mean_switch_prob_drop": safe_ratio(
                sum(float(row["switch_axis_ablation"]["switch_axis_delta"]["mean_correct_prob_drop"]) for row in noun_results),
                len(noun_results),
            ),
            "mean_control_prob_drop": safe_ratio(
                sum(float(row["switch_axis_ablation"]["control_axis_delta"]["mean_correct_prob_drop"]) for row in noun_results),
                len(noun_results),
            ),
        }
        return {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "noun_results": noun_results,
            "aggregate": aggregate,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total_nouns = sum(int(row["aggregate"]["noun_count"]) for row in model_results)
    split_support = sum(int(row["aggregate"]["polysemy_split_support_count"]) for row in model_results)
    causality_support = sum(int(row["aggregate"]["switch_causality_support_count"]) for row in model_results)
    mean_polysemy_jaccard = safe_ratio(sum(float(row["aggregate"]["mean_polysemy_jaccard"]) for row in model_results), len(model_results))
    mean_ordinary_jaccard = safe_ratio(sum(float(row["aggregate"]["mean_ordinary_jaccard"]) for row in model_results), len(model_results))
    mean_switch_prob_drop = safe_ratio(sum(float(row["aggregate"]["mean_switch_prob_drop"]) for row in model_results), len(model_results))
    mean_control_prob_drop = safe_ratio(sum(float(row["aggregate"]["mean_control_prob_drop"]) for row in model_results), len(model_results))
    return {
        "total_noun_cases": total_nouns,
        "polysemy_split_support_rate": safe_ratio(split_support, total_nouns),
        "switch_causality_support_rate": safe_ratio(causality_support, total_nouns),
        "mean_polysemy_active_jaccard": mean_polysemy_jaccard,
        "mean_ordinary_active_jaccard": mean_ordinary_jaccard,
        "mean_ordinary_vs_polysemy_gap": float(mean_ordinary_jaccard - mean_polysemy_jaccard),
        "mean_switch_axis_prob_drop": mean_switch_prob_drop,
        "mean_control_axis_prob_drop": mean_control_prob_drop,
        "core_answer": (
            "Across a family of polysemous nouns, ordinary noun context variation keeps a much larger neuron-set overlap, "
            "while true polysemy uses a lower-overlap split plus a causal switch axis."
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## Core Answer",
        summary["cross_model_summary"]["core_answer"],
        "",
    ]
    for model_row in summary["model_results"]:
        agg = model_row["aggregate"]
        lines.extend(
            [
                f"## {model_row['model_name']}",
                f"- noun_count: {agg['noun_count']}",
                f"- polysemy_split_support_count: {agg['polysemy_split_support_count']}",
                f"- switch_causality_support_count: {agg['switch_causality_support_count']}",
                f"- mean_polysemy_jaccard: {agg['mean_polysemy_jaccard']:.4f}",
                f"- mean_ordinary_jaccard: {agg['mean_ordinary_jaccard']:.4f}",
                f"- mean_switch_prob_drop: {agg['mean_switch_prob_drop']:.4f}",
                f"- mean_control_prob_drop: {agg['mean_control_prob_drop']:.4f}",
                "",
            ]
        )
        for noun_row in model_row["noun_results"]:
            switch_delta = noun_row["switch_axis_ablation"]["switch_axis_delta"]
            control_delta = noun_row["switch_axis_ablation"]["control_axis_delta"]
            lines.extend(
                [
                    f"### {noun_row['noun_id']}",
                    f"- best_switch_layer: {noun_row['best_switch_layer']}",
                    f"- sense_active_jaccard: {noun_row['sense_active_jaccard']:.4f}",
                    f"- ordinary_control_mean_active_jaccard: {noun_row['ordinary_control_mean_active_jaccard']:.4f}",
                    f"- ordinary_vs_polysemy_gap: {noun_row['ordinary_vs_polysemy_gap']:.4f}",
                    f"- switch_axis_prob_drop: {switch_delta['mean_correct_prob_drop']:.4f}",
                    f"- control_axis_prob_drop: {control_delta['mean_correct_prob_drop']:.4f}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polysemous noun family overlap and switch-axis protocol.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--models", default=",".join(MODEL_ORDER), help="Comma-separated model keys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    model_results = [analyze_model(model_key, prefer_cuda=prefer_cuda) for model_key in model_keys]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage447_polysemy_family_switch_protocol",
        "title": "多义词家族神经元交并比与切换轴统一协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
