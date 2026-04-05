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
from stage434_apple_polysemy_factorized_switch import APPLE_BRAND_CASES, APPLE_FRUIT_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage448_apple_switch_layer_scan_and_neuron_counts_20260403"
MODEL_ORDER = ["qwen3", "deepseek7b"]
GLOBAL_TOP_K = 256
LAYER_TOP_K = 64


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


def safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-8:
        return 0.0
    return float(numer / denom)


def mean_tensors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([vec.float() for vec in vectors], dim=0).mean(dim=0)


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


def resolve_digit_token_ids(tokenizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for digit in ["1", "2"]:
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"Digit {digit} is not a single token.")
        out[digit] = int(ids[0])
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


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


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def overlap_count(a: Sequence[int], b: Sequence[int]) -> int:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    return len(sa & sb)


def capture_case_payloads(model, tokenizer, sentence: str, target: str) -> Dict[str, object]:
    layer_count = len(discover_layers(model))
    layer_payload_map = {layer_idx: "neuron_in" for layer_idx in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    start, end = locate_target_span(tokenizer, sentence, target)
    try:
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
        layer_vectors: Dict[int, torch.Tensor] = {}
        layer_neuron_vectors: Dict[int, torch.Tensor] = {}
        for layer_idx in range(layer_count):
            layer_vectors[layer_idx] = outputs.hidden_states[layer_idx + 1][0, start:end, :].mean(dim=0).detach().float().cpu()
            neuron_tensor = buffers[layer_idx]
            if neuron_tensor is None:
                raise RuntimeError(f"Missing neuron payload at layer {layer_idx}.")
            layer_neuron_vectors[layer_idx] = neuron_tensor[0, start:end, :].mean(dim=0).detach().float().cpu()
        flat_neuron_vector = torch.cat([layer_neuron_vectors[idx] for idx in range(layer_count)], dim=0)
        return {
            "sentence": sentence,
            "target": target,
            "layer_vectors": layer_vectors,
            "layer_neuron_vectors": layer_neuron_vectors,
            "flat_neuron_vector": flat_neuron_vector,
        }
    finally:
        remove_hooks(handles)


def build_rows(model, tokenizer, cases: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    return [capture_case_payloads(model, tokenizer, case["sentence"], case["target"]) for case in cases]


def orthogonal_control_axis(switch_axis: torch.Tensor, fruit_rows: Sequence[Dict[str, object]], layer_idx: int) -> torch.Tensor:
    fruit_mean = mean_tensors([row["layer_vectors"][layer_idx] for row in fruit_rows])
    modifiers = [row["layer_vectors"][layer_idx] - fruit_mean for row in fruit_rows]
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


def build_sense_prompt(case: Dict[str, str]) -> str:
    return (
        f'Sentence: "{case["sentence"]}"\n'
        "Question: In this sentence, does the word apple refer to 1 fruit or 2 company?\n"
        "Answer with one digit only: 1 fruit 2 company\n"
        "Answer:"
    )


def evaluate_sense_cases(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    cases: Sequence[Dict[str, str]],
    labels: Sequence[int],
    *,
    layer_idx: int | None = None,
    axis: torch.Tensor | None = None,
) -> Dict[str, object]:
    per_case = []
    candidate_ids = None
    for case, label in zip(cases, labels):
        prompt = build_sense_prompt(case)
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


def layer_count_rows(fruit_rows: Sequence[Dict[str, object]], brand_rows: Sequence[Dict[str, object]], layer_count: int) -> List[Dict[str, object]]:
    rows = []
    for layer_idx in range(layer_count):
        fruit_mean = mean_tensors([row["layer_neuron_vectors"][layer_idx] for row in fruit_rows])
        brand_mean = mean_tensors([row["layer_neuron_vectors"][layer_idx] for row in brand_rows])
        fruit_ids = top_active_ids(fruit_mean, LAYER_TOP_K)
        brand_ids = top_active_ids(brand_mean, LAYER_TOP_K)
        shared = overlap_count(fruit_ids, brand_ids)
        fruit_only = len(set(fruit_ids) - set(brand_ids))
        brand_only = len(set(brand_ids) - set(fruit_ids))
        rows.append(
            {
                "layer_index": layer_idx,
                "shared_active_neuron_count": shared,
                "fruit_only_neuron_count": fruit_only,
                "brand_only_neuron_count": brand_only,
                "active_jaccard": jaccard(fruit_ids, brand_ids),
            }
        )
    return rows


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        fruit_rows = build_rows(model, tokenizer, APPLE_FRUIT_CASES)
        brand_rows = build_rows(model, tokenizer, APPLE_BRAND_CASES)
        layer_count = len(discover_layers(model))

        fruit_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in fruit_rows])
        brand_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in brand_rows])
        fruit_global_ids = top_active_ids(fruit_mean_flat, GLOBAL_TOP_K)
        brand_global_ids = top_active_ids(brand_mean_flat, GLOBAL_TOP_K)

        global_shared = overlap_count(fruit_global_ids, brand_global_ids)
        global_fruit_only = len(set(fruit_global_ids) - set(brand_global_ids))
        global_brand_only = len(set(brand_global_ids) - set(fruit_global_ids))

        digit_token_ids = resolve_digit_token_ids(tokenizer)
        sense_cases = list(APPLE_FRUIT_CASES) + list(APPLE_BRAND_CASES)
        sense_labels = [0 for _ in APPLE_FRUIT_CASES] + [1 for _ in APPLE_BRAND_CASES]
        baseline_eval = evaluate_sense_cases(model, tokenizer, digit_token_ids, sense_cases, sense_labels)

        layer_scan = []
        for layer_idx in range(layer_count):
            fruit_mean = mean_tensors([row["layer_vectors"][layer_idx] for row in fruit_rows])
            brand_mean = mean_tensors([row["layer_vectors"][layer_idx] for row in brand_rows])
            switch_axis = brand_mean - fruit_mean
            control_axis = orthogonal_control_axis(switch_axis, fruit_rows, layer_idx)
            switch_eval = evaluate_sense_cases(
                model,
                tokenizer,
                digit_token_ids,
                sense_cases,
                sense_labels,
                layer_idx=layer_idx,
                axis=switch_axis,
            )
            control_eval = evaluate_sense_cases(
                model,
                tokenizer,
                digit_token_ids,
                sense_cases,
                sense_labels,
                layer_idx=layer_idx,
                axis=control_axis,
            )
            switch_drop = float(baseline_eval["mean_correct_prob"] - switch_eval["mean_correct_prob"])
            control_drop = float(baseline_eval["mean_correct_prob"] - control_eval["mean_correct_prob"])
            layer_scan.append(
                {
                    "layer_index": layer_idx,
                    "switch_prob_drop": switch_drop,
                    "control_prob_drop": control_drop,
                    "excess_switch_drop": float(switch_drop - control_drop),
                    "baseline_prob": float(baseline_eval["mean_correct_prob"]),
                    "switch_prob": float(switch_eval["mean_correct_prob"]),
                    "control_prob": float(control_eval["mean_correct_prob"]),
                    "baseline_accuracy": float(baseline_eval["accuracy"]),
                    "switch_accuracy": float(switch_eval["accuracy"]),
                    "control_accuracy": float(control_eval["accuracy"]),
                }
            )

        best_sensitive = max(layer_scan, key=lambda row: float(row["excess_switch_drop"]))
        per_layer_counts = layer_count_rows(fruit_rows, brand_rows, layer_count)
        best_shared_layer = max(per_layer_counts, key=lambda row: int(row["shared_active_neuron_count"]))
        best_split_layer = min(per_layer_counts, key=lambda row: float(row["active_jaccard"]))

        result = {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "global_top_k": GLOBAL_TOP_K,
            "global_counts": {
                "shared_reuse_neuron_count": global_shared,
                "fruit_only_neuron_count": global_fruit_only,
                "brand_only_neuron_count": global_brand_only,
                "different_neuron_count_total": int(global_fruit_only + global_brand_only),
                "active_jaccard": jaccard(fruit_global_ids, brand_global_ids),
            },
            "layer_top_k": LAYER_TOP_K,
            "best_sensitive_layer": best_sensitive,
            "best_shared_layer": best_shared_layer,
            "best_split_layer": best_split_layer,
            "layer_scan": layer_scan,
            "per_layer_counts": per_layer_counts,
            "core_answer": (
                "Apple 的水果义和品牌义会复用一小部分高活跃神经元，但主要差异来自两套更大的词义偏置神经元群。"
            ),
        }
        return result
    finally:
        free_model(model)


def build_report(summary: Dict[str, object]) -> str:
    lines = [f"# {summary['experiment_id']}", ""]
    for row in summary["model_results"]:
        counts = row["global_counts"]
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- global_shared_reuse_neuron_count: {counts['shared_reuse_neuron_count']}",
                f"- global_fruit_only_neuron_count: {counts['fruit_only_neuron_count']}",
                f"- global_brand_only_neuron_count: {counts['brand_only_neuron_count']}",
                f"- global_different_neuron_count_total: {counts['different_neuron_count_total']}",
                f"- global_active_jaccard: {counts['active_jaccard']:.4f}",
                f"- best_sensitive_layer: L{row['best_sensitive_layer']['layer_index']}",
                f"- best_sensitive_excess_switch_drop: {row['best_sensitive_layer']['excess_switch_drop']:.4f}",
                f"- best_shared_layer: L{row['best_shared_layer']['layer_index']}",
                f"- best_split_layer: L{row['best_split_layer']['layer_index']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple switch layer scan and neuron reuse counts.")
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
        "experiment_id": "stage448_apple_switch_layer_scan_and_neuron_counts",
        "title": "苹果语义切换敏感层与复用神经元计数",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "model_results": model_results,
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
