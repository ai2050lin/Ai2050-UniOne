#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from qwen3_language_shared import PROJECT_ROOT, discover_layers
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model
from stage425_sentence_context_wordclass_causal import (
    SENTENCE_CASES,
    WORD_CLASSES,
    evaluate_case_batch,
    finalize_metric_totals,
    merge_metric_totals,
    resolve_digit_token_ids,
)


STAGE423_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage423_qwen3_deepseek_wordclass_layer_distribution_20260330"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage426_pronoun_minimal_causal_mechanism_20260330"
)

TARGET_CLASS = "pronoun"
PREFIX_KS = [1, 2, 4, 8, 12]
CANDIDATE_POOL = 12


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def register_mlp_layer_ablation(model, layer_indices: Sequence[int]) -> List[object]:
    handles = []
    layers = discover_layers(model)
    for layer_idx in layer_indices:
        module = layers[layer_idx].mlp.down_proj

        def make_pre_hook():
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0]
                zeroed = torch.zeros_like(hidden)
                if len(inputs) == 1:
                    return (zeroed,)
                return (zeroed, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook()))
    return handles


def register_attention_layer_ablation(model, layer_indices: Sequence[int]) -> List[object]:
    handles = []
    layers = discover_layers(model)
    for layer_idx in layer_indices:
        module = layers[layer_idx].self_attn.o_proj

        def make_hook():
            def hook(_module, _inputs, output):
                return torch.zeros_like(output)

            return hook

        handles.append(module.register_forward_hook(make_hook()))
    return handles


def register_mlp_neuron_ablation(model, neuron_specs: Sequence[Dict[str, int]]) -> List[object]:
    handles = []
    layers = discover_layers(model)
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for row in neuron_specs:
        by_layer[int(row["layer_index"])].append(int(row["neuron_index"]))

    for layer_idx, neuron_indices in by_layer.items():
        module = layers[layer_idx].mlp.down_proj
        neuron_tensor = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long)

        def make_pre_hook(indices: torch.Tensor):
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0].clone()
                hidden[..., indices.to(hidden.device)] = 0
                if len(inputs) == 1:
                    return (hidden,)
                return (hidden, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(neuron_tensor)))
    return handles


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def evaluate_all_classes(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    *,
    batch_size: int,
    handles: Sequence[object] | None = None,
) -> Dict[str, object]:
    try:
        by_class = {}
        aggregate = {
            "count": 0.0,
            "correct_prob_sum": 0.0,
            "correct_logprob_sum": 0.0,
            "accuracy_sum": 0.0,
            "margin_sum": 0.0,
        }
        for class_name in WORD_CLASSES:
            totals = {
                "count": 0.0,
                "correct_prob_sum": 0.0,
                "correct_logprob_sum": 0.0,
                "accuracy_sum": 0.0,
                "margin_sum": 0.0,
            }
            cases = SENTENCE_CASES[class_name]
            for start in range(0, len(cases), batch_size):
                chunk = evaluate_case_batch(
                    model,
                    tokenizer,
                    cases[start : start + batch_size],
                    [class_name for _ in cases[start : start + batch_size]],
                    digit_token_ids,
                )
                merge_metric_totals(totals, chunk)
                merge_metric_totals(aggregate, chunk)
            by_class[class_name] = finalize_metric_totals(totals)
        return {
            "by_class": by_class,
            "aggregate": finalize_metric_totals(aggregate),
        }
    finally:
        if handles:
            remove_hooks(handles)


def summarize_delta(baseline: Dict[str, object], current: Dict[str, object], *, label: str) -> Dict[str, object]:
    baseline_by_class = baseline["by_class"]
    current_by_class = current["by_class"]
    delta_by_class = {}
    other_prob_deltas = []
    other_acc_deltas = []
    for class_name in WORD_CLASSES:
        prob_delta = (
            float(current_by_class[class_name]["mean_correct_prob"])
            - float(baseline_by_class[class_name]["mean_correct_prob"])
        )
        acc_delta = float(current_by_class[class_name]["accuracy"]) - float(baseline_by_class[class_name]["accuracy"])
        delta_by_class[class_name] = {
            "correct_prob_delta": float(prob_delta),
            "accuracy_delta": float(acc_delta),
        }
        if class_name != TARGET_CLASS:
            other_prob_deltas.append(prob_delta)
            other_acc_deltas.append(acc_delta)
    target_prob_delta = delta_by_class[TARGET_CLASS]["correct_prob_delta"]
    target_acc_delta = delta_by_class[TARGET_CLASS]["accuracy_delta"]
    other_prob_delta_mean = sum(other_prob_deltas) / max(1, len(other_prob_deltas))
    other_acc_delta_mean = sum(other_acc_deltas) / max(1, len(other_acc_deltas))
    return {
        "label": label,
        "delta_by_class": delta_by_class,
        "target_prob_delta": float(target_prob_delta),
        "target_accuracy_delta": float(target_acc_delta),
        "other_prob_delta_mean": float(other_prob_delta_mean),
        "other_accuracy_delta_mean": float(other_acc_delta_mean),
        "specificity_gap_prob": float(target_prob_delta - other_prob_delta_mean),
        "specificity_gap_accuracy": float(target_acc_delta - other_acc_delta_mean),
    }


def build_prefix_results(
    model,
    tokenizer,
    digit_token_ids: Dict[str, int],
    baseline: Dict[str, object],
    candidate_neurons: Sequence[Dict[str, int]],
    *,
    batch_size: int,
) -> List[Dict[str, object]]:
    rows = []
    for k in PREFIX_KS:
        subset = list(candidate_neurons[: min(k, len(candidate_neurons))])
        handles = register_mlp_neuron_ablation(model, subset)
        current = evaluate_all_classes(
            model,
            tokenizer,
            digit_token_ids,
            batch_size=batch_size,
            handles=handles,
        )
        summary = summarize_delta(baseline, current, label=f"mlp_neuron_prefix_k_{len(subset)}")
        summary["subset_size"] = len(subset)
        summary["subset_neurons"] = subset
        rows.append(summary)
    return rows


def choose_minimal_subset(
    prefix_rows: Sequence[Dict[str, object]],
    layer_delta: Dict[str, object],
) -> Dict[str, object] | None:
    target_effect = abs(float(layer_delta["target_prob_delta"]))
    if target_effect <= 1e-8:
        return None
    threshold = 0.5 * target_effect
    for row in prefix_rows:
        if abs(float(row["target_prob_delta"])) >= threshold:
            return {
                "threshold_ratio": 0.5,
                "reference_layer_target_prob_delta": float(layer_delta["target_prob_delta"]),
                "selected_subset_size": int(row["subset_size"]),
                "selected_target_prob_delta": float(row["target_prob_delta"]),
                "selected_specificity_gap_prob": float(row["specificity_gap_prob"]),
                "selected_subset_neurons": row["subset_neurons"],
            }
    return None


def build_model_summary(
    model_key: str,
    stage423_summary: Dict[str, object],
    *,
    batch_size: int,
    use_cuda: bool,
) -> Dict[str, object]:
    class_stage423 = stage423_summary["models"][model_key]["classes"][TARGET_CLASS]
    mlp_top_layers = [int(row["layer_index"]) for row in class_stage423["top_layers_by_mass"][:2]]
    attention_top_layers = list(mlp_top_layers)
    candidate_neurons = [
        {
            "layer_index": int(row["layer_index"]),
            "neuron_index": int(row["neuron_index"]),
            "score": float(row["score"]),
            "effect_size": float(row["effect_size"]),
        }
        for row in class_stage423["top_neurons"][:CANDIDATE_POOL]
    ]

    model, tokenizer = load_qwen_like_model(MODEL_SPECS[model_key]["model_path"], prefer_cuda=use_cuda)
    try:
        digit_token_ids = resolve_digit_token_ids(tokenizer)
        baseline = evaluate_all_classes(model, tokenizer, digit_token_ids, batch_size=batch_size)

        mlp_layer_handles = register_mlp_layer_ablation(model, mlp_top_layers)
        mlp_layer_eval = evaluate_all_classes(
            model,
            tokenizer,
            digit_token_ids,
            batch_size=batch_size,
            handles=mlp_layer_handles,
        )
        mlp_layer_summary = summarize_delta(baseline, mlp_layer_eval, label="mlp_top_layer_ablation")
        mlp_layer_summary["ablated_layers"] = mlp_top_layers

        attention_handles = register_attention_layer_ablation(model, attention_top_layers)
        attention_eval = evaluate_all_classes(
            model,
            tokenizer,
            digit_token_ids,
            batch_size=batch_size,
            handles=attention_handles,
        )
        attention_summary = summarize_delta(baseline, attention_eval, label="attention_top_layer_ablation")
        attention_summary["ablated_layers"] = attention_top_layers

        prefix_rows = build_prefix_results(
            model,
            tokenizer,
            digit_token_ids,
            baseline,
            candidate_neurons,
            batch_size=batch_size,
        )
        minimal_subset = choose_minimal_subset(prefix_rows, mlp_layer_summary)

        return {
            "model_name": MODEL_SPECS[model_key]["model_name"],
            "model_path": str(MODEL_SPECS[model_key]["model_path"]),
            "digit_token_ids": digit_token_ids,
            "baseline": baseline,
            "pronoun_top_layers": mlp_top_layers,
            "candidate_neurons": candidate_neurons,
            "mlp_top_layer_ablation": mlp_layer_summary,
            "attention_top_layer_ablation": attention_summary,
            "mlp_prefix_subset_curve": prefix_rows,
            "minimal_prefix_subset": minimal_subset,
        }
    finally:
        free_model(model)


def build_cross_model_summary(model_payloads: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out = {}
    for model_key, payload in model_payloads.items():
        out[model_key] = {
            "mlp_top_layer_target_prob_delta": float(payload["mlp_top_layer_ablation"]["target_prob_delta"]),
            "attention_top_layer_target_prob_delta": float(payload["attention_top_layer_ablation"]["target_prob_delta"]),
            "mlp_minus_attention_target_prob_delta": float(
                payload["mlp_top_layer_ablation"]["target_prob_delta"]
                - payload["attention_top_layer_ablation"]["target_prob_delta"]
            ),
            "minimal_prefix_subset_size": (
                int(payload["minimal_prefix_subset"]["selected_subset_size"])
                if payload["minimal_prefix_subset"] is not None
                else None
            ),
        }
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        f"- 是否使用 CUDA: {summary['used_cuda']}",
        f"- 批大小: {summary['batch_size']}",
        "- 任务: 围绕代词句法角色，比较顶部层 MLP 整层消融、顶部层 attention 整层消融、前缀神经元子集消融。",
        "",
    ]
    for model_key in ["qwen3", "deepseek7b"]:
        payload = summary["models"][model_key]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 模型名: {payload['model_name']}",
                f"- 代词顶部层: {payload['pronoun_top_layers']}",
                f"- 基线代词概率: {payload['baseline']['by_class'][TARGET_CLASS]['mean_correct_prob']:.4f}",
                f"- MLP 顶部层消融 target_prob_delta: {payload['mlp_top_layer_ablation']['target_prob_delta']:+.4f}",
                f"- attention 顶部层消融 target_prob_delta: {payload['attention_top_layer_ablation']['target_prob_delta']:+.4f}",
            ]
        )
        minimal = payload["minimal_prefix_subset"]
        if minimal is None:
            lines.append("- 最小前缀子集: 未达到 50% 整层效果")
        else:
            lines.append(
                f"- 最小前缀子集: size={minimal['selected_subset_size']}, "
                f"target_prob_delta={minimal['selected_target_prob_delta']:+.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 与 DeepSeek7B 代词最小因果子集实验")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="前向批大小")
    parser.add_argument("--cpu", action="store_true", help="强制不用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    stage423_summary = load_json(STAGE423_SUMMARY_PATH)
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    start_time = time.time()

    model_payloads = {}
    for model_key in ["qwen3", "deepseek7b"]:
        model_payloads[model_key] = build_model_summary(
            model_key,
            stage423_summary,
            batch_size=int(args.batch_size),
            use_cuda=use_cuda,
        )

    elapsed = time.time() - start_time
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage426_pronoun_minimal_causal_mechanism",
        "title": "Qwen3 与 DeepSeek7B 代词最小因果子集实验",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "used_cuda": use_cuda,
        "batch_size": int(args.batch_size),
        "models": model_payloads,
        "cross_model_summary": build_cross_model_summary(model_payloads),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage426_pronoun_minimal_causal_mechanism_ready",
                "output_dir": str(output_dir),
                "used_cuda": use_cuda,
                "elapsed_seconds": elapsed,
                "qwen3_mlp_top_layer_delta": model_payloads["qwen3"]["mlp_top_layer_ablation"]["target_prob_delta"],
                "deepseek7b_mlp_top_layer_delta": model_payloads["deepseek7b"]["mlp_top_layer_ablation"][
                    "target_prob_delta"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
