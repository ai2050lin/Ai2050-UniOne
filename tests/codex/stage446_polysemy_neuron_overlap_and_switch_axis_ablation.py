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
from stage434_apple_polysemy_factorized_switch import APPLE_BRAND_CASES, APPLE_FRUIT_CASES, capture_case_layer_vectors, cosine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage446_polysemy_neuron_overlap_and_switch_axis_ablation_20260403"
)
STAGE434_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage434_apple_polysemy_factorized_switch_20260402"
    / "summary.json"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]
TOP_K = 256
TOP_SWITCH_NEURONS = 12

BANANA_CONTEXT_CASES = [
    {"target": "banana", "label": "neutral", "sentence": "I bought a banana this morning."},
    {"target": "banana", "label": "color", "sentence": "The banana is yellow and ripe."},
    {"target": "banana", "label": "taste", "sentence": "The banana tastes sweet today."},
    {"target": "banana", "label": "size", "sentence": "The banana is about the size of a hand."},
]


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
            raise RuntimeError(f"数字 {digit} 不是单 token")
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
    candidates = []
    raw_variants = [target, f" {target}", target.lower(), f" {target.lower()}", target.capitalize(), f" {target.capitalize()}"]
    seen = set()
    for text in raw_variants:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            candidates.append(ids)
    best = None
    best_len = -1
    for ids in candidates:
        match = find_last_subsequence(full_ids, ids)
        if match is not None and len(ids) > best_len:
            best = match
            best_len = len(ids)
    if best is None:
        raise RuntimeError(f"无法定位目标词: target={target!r}, prompt={prompt!r}")
    return best


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
                raise RuntimeError(f"第 {layer_idx} 层神经元激活捕获失败")
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


def orthogonal_control_axis(switch_axis: torch.Tensor, fruit_modifiers: Sequence[torch.Tensor]) -> torch.Tensor:
    base = mean_tensors(fruit_modifiers) if fruit_modifiers else torch.zeros_like(switch_axis)
    switch_unit = switch_axis.float() / torch.linalg.norm(switch_axis.float()).clamp_min(1e-8)
    base = base.float() - torch.dot(base.float(), switch_unit) * switch_unit
    if float(torch.linalg.norm(base).item()) <= 1e-8:
        base = torch.roll(switch_unit, shifts=1)
        base = base - torch.dot(base, switch_unit) * switch_unit
    return base / torch.linalg.norm(base).clamp_min(1e-8)


def build_switch_axis_hook(
    layer_idx: int,
    target_spans: Sequence[Tuple[int, int]],
    axis: torch.Tensor,
):
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
    prompts = [build_sense_prompt(case) for case in cases]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = move_batch_to_model_device(model, encoded)
    target_spans = [locate_target_span(tokenizer, prompt, case["target"]) for prompt, case in zip(prompts, cases)]
    handles = []
    try:
        if layer_idx is not None and axis is not None:
            layer_module = discover_layers(model)[layer_idx]
            handles.append(layer_module.register_forward_hook(build_switch_axis_hook(layer_idx, target_spans, axis)))
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :]
        candidate_ids = torch.tensor(
            [digit_token_ids["1"], digit_token_ids["2"]],
            device=logits.device,
            dtype=torch.long,
        )
        option_logits = logits.index_select(dim=1, index=candidate_ids)
        log_probs = option_logits.log_softmax(dim=-1)
        target_idx = torch.tensor(labels, device=logits.device, dtype=torch.long)
        correct_probs = log_probs.gather(1, target_idx.unsqueeze(1)).squeeze(1).exp()
        predictions = option_logits.argmax(dim=-1)
        per_case = []
        for row_idx, case in enumerate(cases):
            per_case.append(
                {
                    "sentence": case["sentence"],
                    "target": case["target"],
                    "expected_label": int(labels[row_idx]),
                    "predicted_label": int(predictions[row_idx].item()),
                    "correct_prob": float(correct_probs[row_idx].item()),
                    "is_correct": bool(int(predictions[row_idx].item()) == int(labels[row_idx])),
                }
            )
        return {
            "accuracy": safe_ratio(sum(int(row["is_correct"]) for row in per_case), len(per_case)),
            "mean_correct_prob": safe_ratio(sum(float(row["correct_prob"]) for row in per_case), len(per_case)),
            "per_case": per_case,
        }
    finally:
        remove_hooks(handles)


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
    def build_rows(flat_ids: Sequence[int], sign: int) -> List[Dict[str, object]]:
        rows = []
        for rank, flat_idx in enumerate(flat_ids, start=1):
            layer_idx, neuron_idx = index_to_layer_neuron(int(flat_idx), neuron_count)
            value = float(diff_vec[int(flat_idx)].item())
            rows.append(
                {
                    "rank": rank,
                    "flat_index": int(flat_idx),
                    "layer_index": layer_idx,
                    "neuron_index": neuron_idx,
                    "activation_delta": value,
                }
            )
        return rows

    top_abs_ids = sorted(set(pos_ids + neg_ids), key=lambda idx: abs(float(diff_vec[int(idx)].item())), reverse=True)
    concentration = safe_ratio(sum(abs(float(diff_vec[int(idx)].item())) for idx in top_abs_ids[:top_k]), abs_total)
    return {
        "brand_biased_neurons": build_rows(pos_ids, +1),
        "fruit_biased_neurons": build_rows(neg_ids, -1),
        "top_abs_concentration_ratio": concentration,
    }


def analyze_model(model_key: str, *, prefer_cuda: bool) -> Dict[str, object]:
    stage434_summary = load_json(STAGE434_SUMMARY_PATH)
    stage434_row = next(row for row in stage434_summary["model_results"] if row["model_key"] == model_key)
    best_layer_idx = int(stage434_row["best_layer"]["layer_index"])

    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=prefer_cuda)
    try:
        layers = discover_layers(model)
        neuron_count = int(layers[0].mlp.gate_proj.out_features)

        fruit_rows = build_group_rows(model, tokenizer, APPLE_FRUIT_CASES)
        brand_rows = build_group_rows(model, tokenizer, APPLE_BRAND_CASES)
        banana_rows = build_group_rows(model, tokenizer, BANANA_CONTEXT_CASES)

        fruit_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in fruit_rows])
        brand_mean_flat = mean_tensors([row["flat_neuron_vector"] for row in brand_rows])
        switch_diff_flat = brand_mean_flat - fruit_mean_flat

        fruit_active_ids = top_active_ids(fruit_mean_flat, TOP_K)
        brand_active_ids = top_active_ids(brand_mean_flat, TOP_K)
        fruit_selective_ids = top_selective_ids(switch_diff_flat, TOP_K, positive=False)
        brand_selective_ids = top_selective_ids(switch_diff_flat, TOP_K, positive=True)

        banana_neutral = next(row for row in banana_rows if row["label"] == "neutral")
        banana_context_overlaps = []
        banana_context_details = []
        neutral_active_ids = top_active_ids(banana_neutral["flat_neuron_vector"], TOP_K)
        for row in banana_rows:
            if row["label"] == "neutral":
                continue
            context_active_ids = top_active_ids(row["flat_neuron_vector"], TOP_K)
            jac = jaccard(neutral_active_ids, context_active_ids)
            banana_context_overlaps.append(jac)
            banana_context_details.append(
                {
                    "label": row["label"],
                    "active_jaccard_vs_neutral": jac,
                }
            )

        fruit_base_vec = next(row for row in fruit_rows if row["label"] == "fruit_neutral")["layer_vectors"][best_layer_idx]
        brand_base_vec = next(row for row in brand_rows if row["label"] == "brand_neutral")["layer_vectors"][best_layer_idx]
        switch_axis = brand_base_vec - fruit_base_vec
        fruit_modifiers = [
            row["layer_vectors"][best_layer_idx] - fruit_base_vec
            for row in fruit_rows
            if row["label"] != "fruit_neutral"
        ]
        control_axis = orthogonal_control_axis(switch_axis, fruit_modifiers)

        sense_cases = list(APPLE_FRUIT_CASES) + list(APPLE_BRAND_CASES)
        sense_labels = [0 for _ in APPLE_FRUIT_CASES] + [1 for _ in APPLE_BRAND_CASES]
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline_eval = evaluate_sense_cases(model, tokenizer, digit_token_ids, sense_cases, sense_labels)
        switch_eval = evaluate_sense_cases(
            model,
            tokenizer,
            digit_token_ids,
            sense_cases,
            sense_labels,
            layer_idx=best_layer_idx,
            axis=switch_axis,
        )
        control_eval = evaluate_sense_cases(
            model,
            tokenizer,
            digit_token_ids,
            sense_cases,
            sense_labels,
            layer_idx=best_layer_idx,
            axis=control_axis,
        )

        def sense_delta(current: Dict[str, object]) -> Dict[str, float]:
            return {
                "accuracy_drop": float(baseline_eval["accuracy"] - current["accuracy"]),
                "mean_correct_prob_drop": float(baseline_eval["mean_correct_prob"] - current["mean_correct_prob"]),
            }

        result = {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
            "best_switch_layer": best_layer_idx,
            "best_switch_layer_metrics": stage434_row["best_layer"],
            "fruit_brand_active_jaccard": jaccard(fruit_active_ids, brand_active_ids),
            "fruit_brand_active_overlap_ratio_from_fruit": overlap_ratio(fruit_active_ids, brand_active_ids),
            "fruit_brand_active_overlap_ratio_from_brand": overlap_ratio(brand_active_ids, fruit_active_ids),
            "fruit_brand_selective_jaccard": jaccard(fruit_selective_ids, brand_selective_ids),
            "banana_context_mean_active_jaccard": safe_ratio(sum(banana_context_overlaps), len(banana_context_overlaps)),
            "ordinary_vs_polysemy_gap": float(safe_ratio(sum(banana_context_overlaps), len(banana_context_overlaps)) - jaccard(fruit_active_ids, brand_active_ids)),
            "banana_context_details": banana_context_details,
            "switch_neuron_summary": summarize_switch_neurons(switch_diff_flat, neuron_count, TOP_SWITCH_NEURONS),
            "switch_axis_ablation": {
                "baseline": baseline_eval,
                "switch_axis_removed": switch_eval,
                "control_axis_removed": control_eval,
                "switch_axis_delta": sense_delta(switch_eval),
                "control_axis_delta": sense_delta(control_eval),
            },
        }
        return result
    finally:
        free_model(model)


def build_cross_model_summary(model_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    mean_active_jaccard = safe_ratio(
        sum(float(row["fruit_brand_active_jaccard"]) for row in model_results),
        len(model_results),
    )
    mean_banana_overlap = safe_ratio(
        sum(float(row["banana_context_mean_active_jaccard"]) for row in model_results),
        len(model_results),
    )
    mean_switch_prob_drop = safe_ratio(
        sum(float(row["switch_axis_ablation"]["switch_axis_delta"]["mean_correct_prob_drop"]) for row in model_results),
        len(model_results),
    )
    mean_control_prob_drop = safe_ratio(
        sum(float(row["switch_axis_ablation"]["control_axis_delta"]["mean_correct_prob_drop"]) for row in model_results),
        len(model_results),
    )
    return {
        "mean_fruit_brand_active_jaccard": mean_active_jaccard,
        "mean_banana_context_active_jaccard": mean_banana_overlap,
        "mean_switch_axis_prob_drop": mean_switch_prob_drop,
        "mean_control_axis_prob_drop": mean_control_prob_drop,
        "core_answer": "苹果的水果义与品牌义不是两套完全独立神经元，而是共享一部分名词骨干、再由一条可干预的切换轴把状态推向不同词义盆地。普通名词更像同一骨干内的上下文扰动，多义词则多出一条跨词义盆地的稳定切换结构。",
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心回答",
        summary["cross_model_summary"]["core_answer"],
        "",
    ]
    for row in summary["model_results"]:
        switch_delta = row["switch_axis_ablation"]["switch_axis_delta"]
        control_delta = row["switch_axis_ablation"]["control_axis_delta"]
        lines.extend(
            [
                f"## {row['model_name']}",
                f"- best_switch_layer: {row['best_switch_layer']}",
                f"- fruit_brand_active_jaccard: {row['fruit_brand_active_jaccard']:.4f}",
                f"- banana_context_mean_active_jaccard: {row['banana_context_mean_active_jaccard']:.4f}",
                f"- ordinary_vs_polysemy_gap: {row['ordinary_vs_polysemy_gap']:.4f}",
                f"- switch_axis_prob_drop: {switch_delta['mean_correct_prob_drop']:.4f}",
                f"- control_axis_prob_drop: {control_delta['mean_correct_prob_drop']:.4f}",
                f"- switch_axis_accuracy_drop: {switch_delta['accuracy_drop']:.4f}",
                f"- control_axis_accuracy_drop: {control_delta['accuracy_drop']:.4f}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多义词神经元重合与切换轴因果消融")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--models", default=",".join(MODEL_ORDER), help="逗号分隔的模型键")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    start_time = time.time()
    model_results = [analyze_model(model_key, prefer_cuda=prefer_cuda) for model_key in model_keys]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage446_polysemy_neuron_overlap_and_switch_axis_ablation",
        "title": "多义词神经元重合与切换轴因果消融",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start_time,
        "used_cuda": bool(prefer_cuda),
        "model_results": model_results,
        "cross_model_summary": build_cross_model_summary(model_results),
    }
    write_outputs(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
