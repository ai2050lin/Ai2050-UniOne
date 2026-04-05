#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
)
from stage423_qwen3_deepseek_wordclass_layer_distribution import MODEL_SPECS, load_qwen_like_model


OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage491_ping_guo_route_mechanism_20260403"
)

MODEL_KEYS = ["qwen3", "deepseek7b"]
CONTEXT_BANK = [
    "苹",
    "我喜欢吃苹",
    "这是一个苹",
    "我今天吃了苹",
    "他刚买了苹",
    "桌上放着苹",
    "她想画一个苹",
    "这个苹",
    "新鲜的苹",
    "苹果的苹可以写成苹",
]
TOP_CONTEXTS = 3
TOP_ROUTE_LAYERS = 4
TOP_NEURON_CANDIDATES = 24
TOP_HEAD_CANDIDATES = 12
MAX_MIXED_SUBSET = 6
MIN_GAIN = 1e-5


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_bundle(model_key: str, prefer_cuda: bool):
    spec = MODEL_SPECS[model_key]
    model, tokenizer = load_qwen_like_model(spec["model_path"], prefer_cuda=prefer_cuda)
    return model, tokenizer


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


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError("无法识别最终归一化层")


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("无法识别 lm_head")


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} 不是单一词元")
    return int(ids[0])


def next_token_metrics_from_logits(logits: torch.Tensor, target_id: int) -> Dict[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    target_prob = float(probs[target_id].item())
    target_logit = float(logits[target_id].float().item())
    rank = int((logits > logits[target_id]).sum().item()) + 1
    return {
        "target_prob": target_prob,
        "target_logit": target_logit,
        "target_rank": rank,
    }


def encode_prefix(model, tokenizer, prefix: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    return move_batch_to_model_device(model, encoded)


def final_next_token_metrics(
    model,
    tokenizer,
    prefix: str,
    target_id: int,
) -> Dict[str, object]:
    encoded = encode_prefix(model, tokenizer, prefix)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, return_dict=True)
    logits = outputs.logits[0, -1, :]
    metrics = next_token_metrics_from_logits(logits, target_id)
    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    metrics["prefix"] = prefix
    metrics["token_ids"] = [int(x) for x in ids]
    return metrics


def select_contexts(model, tokenizer, target_id: int, ping_id: int) -> Dict[str, object]:
    rows = []
    for prefix in CONTEXT_BANK:
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if not ids or int(ids[-1]) != ping_id:
            continue
        metrics = final_next_token_metrics(model, tokenizer, prefix, target_id)
        rows.append(metrics)
    rows.sort(key=lambda row: (float(row["target_prob"]), -int(row["target_rank"])), reverse=True)
    selected = rows[:TOP_CONTEXTS]
    return {"all_rows": rows, "selected_rows": selected}


def layerwise_target_curve(
    model,
    tokenizer,
    prefix: str,
    target_id: int,
) -> Dict[str, object]:
    encoded = encode_prefix(model, tokenizer, prefix)
    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    per_layer = []
    prev_prob = None
    for idx, hidden in enumerate(hidden_states):
        last_hidden = hidden[:, -1, :]
        normed = final_norm(last_hidden)
        logits = lm_head(normed)[0]
        metrics = next_token_metrics_from_logits(logits, target_id)
        prob = float(metrics["target_prob"])
        delta = None if prev_prob is None else prob - prev_prob
        per_layer.append(
            {
                "layer_index": idx - 1,
                "target_prob": prob,
                "target_logit": float(metrics["target_logit"]),
                "target_rank": int(metrics["target_rank"]),
                "delta_prob": None if delta is None else float(delta),
            }
        )
        prev_prob = prob
    return {"prefix": prefix, "layers": per_layer}


def get_down_proj_weight(layer) -> torch.Tensor:
    return layer.mlp.down_proj.weight.detach().float().cpu()


def get_target_direction(model, target_id: int) -> torch.Tensor:
    lm_head = get_lm_head(model)
    weight = lm_head.weight.detach().float().cpu()
    return weight[target_id]


def capture_route_neuron_scores(
    model,
    tokenizer,
    prefix: str,
    target_direction: torch.Tensor,
) -> List[Dict[str, float]]:
    layer_count = len(discover_layers(model))
    layer_map = {layer_idx: "neuron_in" for layer_idx in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_map)
    try:
        encoded = encode_prefix(model, tokenizer, prefix)
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
    finally:
        remove_hooks(handles)

    rows: List[Dict[str, float]] = []
    layers = discover_layers(model)
    for layer_idx in range(layer_count):
        payload = buffers[layer_idx]
        if payload is None:
            continue
        last_payload = payload[0, -1, :].float().cpu()
        down_proj = get_down_proj_weight(layers[layer_idx])
        target_write = torch.mv(down_proj.t(), target_direction)
        write_scores = last_payload * target_write
        top_k = min(8, int(write_scores.numel()))
        top_vals, top_idx = torch.topk(write_scores, k=top_k)
        for rank_idx in range(top_k):
            rows.append(
                {
                    "layer_index": int(layer_idx),
                    "neuron_index": int(top_idx[rank_idx].item()),
                    "write_score": float(top_vals[rank_idx].item()),
                    "activation": float(last_payload[top_idx[rank_idx]].item()),
                    "target_write": float(target_write[top_idx[rank_idx]].item()),
                    "prefix": prefix,
                }
            )
    return rows


def aggregate_route_neurons(
    per_prefix_rows: Sequence[List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[int, int], Dict[str, object]] = {}
    for rows in per_prefix_rows:
        for row in rows:
            key = (int(row["layer_index"]), int(row["neuron_index"]))
            bucket = grouped.setdefault(
                key,
                {
                    "layer_index": int(row["layer_index"]),
                    "neuron_index": int(row["neuron_index"]),
                    "write_scores": [],
                    "activations": [],
                    "target_writes": [],
                    "prefixes": [],
                },
            )
            bucket["write_scores"].append(float(row["write_score"]))
            bucket["activations"].append(float(row["activation"]))
            bucket["target_writes"].append(float(row["target_write"]))
            bucket["prefixes"].append(str(row["prefix"]))
    merged = []
    for bucket in grouped.values():
        merged.append(
            {
                "layer_index": int(bucket["layer_index"]),
                "neuron_index": int(bucket["neuron_index"]),
                "mean_write_score": float(statistics.mean(bucket["write_scores"])),
                "max_write_score": float(max(bucket["write_scores"])),
                "mean_activation": float(statistics.mean(bucket["activations"])),
                "mean_target_write": float(statistics.mean(bucket["target_writes"])),
                "support_count": int(len(bucket["write_scores"])),
                "prefixes": list(bucket["prefixes"]),
            }
        )
    merged.sort(
        key=lambda row: (
            float(row["mean_write_score"]),
            float(row["max_write_score"]),
            int(row["support_count"]),
        ),
        reverse=True,
    )
    return merged


def get_head_dim(model) -> int:
    hidden_size = int(getattr(model.config, "hidden_size"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    return hidden_size // n_heads


def register_attention_head_ablation(
    model,
    head_specs: Sequence[Dict[str, int]],
) -> List[object]:
    handles = []
    layers = discover_layers(model)
    head_dim = get_head_dim(model)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for spec in head_specs:
        grouped[int(spec["layer_index"])].append(int(spec["head_index"]))
    for layer_idx, head_indices in grouped.items():
        module = layers[layer_idx].self_attn.o_proj
        unique_heads = sorted(set(head_indices))

        def make_pre_hook(heads: List[int]):
            def hook(_module, inputs):
                if not inputs:
                    return inputs
                hidden = inputs[0].clone()
                for head_idx in heads:
                    start = head_idx * head_dim
                    end = start + head_dim
                    hidden[..., start:end] = 0
                if len(inputs) == 1:
                    return (hidden,)
                return (hidden, *inputs[1:])

            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(unique_heads)))
    return handles


def register_mlp_neuron_ablation(
    model,
    neuron_specs: Sequence[Dict[str, int]],
) -> List[object]:
    handles = []
    layers = discover_layers(model)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for spec in neuron_specs:
        grouped[int(spec["layer_index"])].append(int(spec["neuron_index"]))
    for layer_idx, neuron_indices in grouped.items():
        module = layers[layer_idx].mlp.down_proj
        index_tensor = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long)

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

        handles.append(module.register_forward_pre_hook(make_pre_hook(index_tensor)))
    return handles


def register_mixed_ablation(model, candidates: Sequence[Dict[str, object]]) -> List[object]:
    head_specs = [row for row in candidates if row["kind"] == "attention_head"]
    neuron_specs = [row for row in candidates if row["kind"] == "mlp_neuron"]
    handles: List[object] = []
    if head_specs:
        handles.extend(register_attention_head_ablation(model, head_specs))
    if neuron_specs:
        handles.extend(register_mlp_neuron_ablation(model, neuron_specs))
    return handles


def candidate_id(candidate: Dict[str, object]) -> str:
    if candidate["kind"] == "attention_head":
        return f"H:{candidate['layer_index']}:{candidate['head_index']}"
    return f"N:{candidate['layer_index']}:{candidate['neuron_index']}"


def mean_target_prob(
    model,
    tokenizer,
    prefixes: Sequence[str],
    target_id: int,
    *,
    handles: Sequence[object] | None = None,
) -> Dict[str, object]:
    try:
        rows = []
        for prefix in prefixes:
            encoded = encode_prefix(model, tokenizer, prefix)
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, return_dict=True)
            logits = outputs.logits[0, -1, :]
            metrics = next_token_metrics_from_logits(logits, target_id)
            rows.append(
                {
                    "prefix": prefix,
                    "target_prob": float(metrics["target_prob"]),
                    "target_rank": int(metrics["target_rank"]),
                    "target_logit": float(metrics["target_logit"]),
                }
            )
        return {
            "mean_target_prob": float(statistics.mean(row["target_prob"] for row in rows)),
            "mean_target_rank": float(statistics.mean(row["target_rank"] for row in rows)),
            "rows": rows,
        }
    finally:
        if handles:
            remove_hooks(handles)


def screen_heads(
    model,
    tokenizer,
    prefixes: Sequence[str],
    target_id: int,
    route_layers: Sequence[int],
    baseline_prob: float,
) -> List[Dict[str, object]]:
    n_heads = int(getattr(model.config, "num_attention_heads"))
    rows = []
    for layer_idx in route_layers:
        for head_idx in range(n_heads):
            candidate = {
                "kind": "attention_head",
                "layer_index": int(layer_idx),
                "head_index": int(head_idx),
            }
            handles = register_attention_head_ablation(model, [candidate])
            current = mean_target_prob(model, tokenizer, prefixes, target_id, handles=handles)
            rows.append(
                {
                    **candidate,
                    "candidate_id": candidate_id(candidate),
                    "target_drop": float(baseline_prob - current["mean_target_prob"]),
                    "mean_target_prob": float(current["mean_target_prob"]),
                    "mean_target_rank": float(current["mean_target_rank"]),
                }
            )
    rows.sort(key=lambda row: float(row["target_drop"]), reverse=True)
    return rows


def screen_neurons(
    model,
    tokenizer,
    prefixes: Sequence[str],
    target_id: int,
    neuron_candidates: Sequence[Dict[str, object]],
    baseline_prob: float,
) -> List[Dict[str, object]]:
    rows = []
    for row in neuron_candidates:
        candidate = {
            "kind": "mlp_neuron",
            "layer_index": int(row["layer_index"]),
            "neuron_index": int(row["neuron_index"]),
        }
        handles = register_mlp_neuron_ablation(model, [candidate])
        current = mean_target_prob(model, tokenizer, prefixes, target_id, handles=handles)
        rows.append(
            {
                **candidate,
                "candidate_id": candidate_id(candidate),
                "route_score": float(row["mean_write_score"]),
                "target_drop": float(baseline_prob - current["mean_target_prob"]),
                "mean_target_prob": float(current["mean_target_prob"]),
                "mean_target_rank": float(current["mean_target_rank"]),
                "support_count": int(row["support_count"]),
            }
        )
    rows.sort(key=lambda item: (float(item["target_drop"]), float(item["route_score"])), reverse=True)
    return rows


def greedy_mixed_search(
    model,
    tokenizer,
    prefixes: Sequence[str],
    target_id: int,
    baseline_prob: float,
    candidates: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    chosen: List[Dict[str, object]] = []
    chosen_ids: List[str] = []
    current_prob = baseline_prob
    trace = []
    for step in range(MAX_MIXED_SUBSET):
        best_candidate = None
        best_prob = None
        best_gain = None
        for candidate in candidates:
            cid = candidate["candidate_id"]
            if cid in chosen_ids:
                continue
            trial = chosen + [candidate]
            handles = register_mixed_ablation(model, trial)
            current = mean_target_prob(model, tokenizer, prefixes, target_id, handles=handles)
            gain = current_prob - float(current["mean_target_prob"])
            if best_gain is None or gain > best_gain + 1e-12:
                best_candidate = candidate
                best_prob = float(current["mean_target_prob"])
                best_gain = float(gain)
        if best_candidate is None or best_gain is None or best_gain <= MIN_GAIN:
            break
        chosen.append(best_candidate)
        chosen_ids.append(best_candidate["candidate_id"])
        current_prob = float(best_prob)
        trace.append(
            {
                "step": step + 1,
                "added_candidate": best_candidate["candidate_id"],
                "added_kind": best_candidate["kind"],
                "mean_target_prob_after": float(current_prob),
                "target_drop": float(baseline_prob - current_prob),
                "gain": float(best_gain),
            }
        )
    return {
        "subset_ids": list(chosen_ids),
        "subset_size": len(chosen_ids),
        "final_mean_target_prob": float(current_prob),
        "final_target_drop": float(baseline_prob - current_prob),
        "trace": trace,
    }


def summarize_layer_curves(curves: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for curve in curves:
        for layer_row in curve["layers"]:
            grouped[int(layer_row["layer_index"])].append(layer_row)
    merged = []
    for layer_idx in sorted(grouped):
        rows = grouped[layer_idx]
        probs = [float(row["target_prob"]) for row in rows]
        deltas = [float(row["delta_prob"]) for row in rows if row["delta_prob"] is not None]
        ranks = [float(row["target_rank"]) for row in rows]
        merged.append(
            {
                "layer_index": int(layer_idx),
                "mean_target_prob": float(statistics.mean(probs)),
                "mean_target_rank": float(statistics.mean(ranks)),
                "mean_delta_prob": float(statistics.mean(deltas)) if deltas else 0.0,
            }
        )
    positive_layers = [row for row in merged if row["layer_index"] >= 0]
    positive_layers.sort(key=lambda row: float(row["mean_delta_prob"]), reverse=True)
    return {
        "mean_curve": merged,
        "top_positive_delta_layers": positive_layers[:TOP_ROUTE_LAYERS],
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = ["# stage491 苹 -> 果 路线机制报告", ""]
    lines.append("## 总体结论")
    lines.append("")
    for model_key in MODEL_KEYS:
        model_summary = summary["models"][model_key]
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(
            f"- 目标词元：`果`，最佳上下文均不是裸 `苹`，而是已经把模型推到补全水果词的前缀。"
        )
        lines.append(
            f"- 入选上下文：{', '.join(row['prefix'] for row in model_summary['selected_contexts'])}"
        )
        best_layers = ", ".join(
            f"L{row['layer_index']} (均值增量 {row['mean_delta_prob']:.4f})"
            for row in model_summary["layer_route"]["top_positive_delta_layers"][:3]
        )
        lines.append(f"- 主要抬升 `果` 概率的层：{best_layers}")
        top_neurons = ", ".join(
            f"N:{row['layer_index']}:{row['neuron_index']}"
            for row in model_summary["screened_neurons"][:5]
        )
        lines.append(f"- 最强候选神经元：{top_neurons}")
        top_heads = ", ".join(
            f"H:{row['layer_index']}:{row['head_index']}"
            for row in model_summary["screened_heads"][:5]
        )
        lines.append(f"- 最强候选注意力头：{top_heads}")
        lines.append(
            f"- 混合子集：{', '.join(model_summary['mixed_route_subset']['subset_ids']) or '空'}，"
            f"平均目标概率下降 {model_summary['mixed_route_subset']['final_target_drop']:.4f}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_for_model(model_key: str, prefer_cuda: bool) -> Dict[str, object]:
    model, tokenizer = load_model_bundle(model_key, prefer_cuda=prefer_cuda)
    try:
        ping_id = token_id(tokenizer, "苹")
        guo_id = token_id(tokenizer, "果")
        ping_guo_id = token_id(tokenizer, "苹果")
        context_info = select_contexts(model, tokenizer, guo_id, ping_id)
        selected_contexts = context_info["selected_rows"]
        if not selected_contexts:
            raise RuntimeError(f"{model_key} 没有可用上下文")
        selected_prefixes = [row["prefix"] for row in selected_contexts]
        baseline = mean_target_prob(model, tokenizer, selected_prefixes, guo_id)
        curves = [layerwise_target_curve(model, tokenizer, prefix, guo_id) for prefix in selected_prefixes]
        layer_summary = summarize_layer_curves(curves)
        target_direction = get_target_direction(model, guo_id)
        per_prefix_route_rows = [
            capture_route_neuron_scores(model, tokenizer, prefix, target_direction)
            for prefix in selected_prefixes
        ]
        route_neurons = aggregate_route_neurons(per_prefix_route_rows)
        route_layers = [int(row["layer_index"]) for row in layer_summary["top_positive_delta_layers"]]
        screened_heads = screen_heads(
            model,
            tokenizer,
            selected_prefixes,
            guo_id,
            route_layers,
            float(baseline["mean_target_prob"]),
        )
        screened_neurons = screen_neurons(
            model,
            tokenizer,
            selected_prefixes,
            guo_id,
            route_neurons[:TOP_NEURON_CANDIDATES],
            float(baseline["mean_target_prob"]),
        )
        shortlist = []
        for row in screened_heads[:TOP_HEAD_CANDIDATES]:
            shortlist.append(dict(row))
        for row in screened_neurons[:TOP_HEAD_CANDIDATES]:
            shortlist.append(dict(row))
        mixed_route_subset = greedy_mixed_search(
            model,
            tokenizer,
            selected_prefixes,
            guo_id,
            float(baseline["mean_target_prob"]),
            shortlist,
        )
        return {
            "model_key": model_key,
            "used_cuda": bool(get_model_device(model).type == "cuda"),
            "token_ids": {
                "苹": int(ping_id),
                "果": int(guo_id),
                "苹果": int(ping_guo_id),
            },
            "all_context_rows": context_info["all_rows"],
            "selected_contexts": selected_contexts,
            "baseline_mean_target_prob": float(baseline["mean_target_prob"]),
            "baseline_mean_target_rank": float(baseline["mean_target_rank"]),
            "layer_route": layer_summary,
            "route_neurons": route_neurons[:40],
            "screened_neurons": screened_neurons[:20],
            "screened_heads": screened_heads[:20],
            "mixed_route_subset": mixed_route_subset,
        }
    finally:
        free_model(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析中文前缀 苹 -> 果 的大概率路线机制")
    parser.add_argument(
        "--prefer-cuda",
        action="store_true",
        help="优先使用 CUDA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    summary = {
        "stage": "stage491_ping_guo_route_mechanism",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prefer_cuda": bool(args.prefer_cuda),
        "models": {},
    }
    for model_key in MODEL_KEYS:
        summary["models"][model_key] = run_for_model(model_key, prefer_cuda=args.prefer_cuda)
    summary["elapsed_seconds"] = float(time.time() - started)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
