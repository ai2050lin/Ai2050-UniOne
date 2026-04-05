#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter, defaultdict
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
    / "stage492_chinese_pattern_route_atlas_20260403"
)

MODEL_KEYS = ["qwen3", "deepseek7b"]
TOP_CONTEXTS = 2
TOP_ROUTE_LAYERS = 3
TOP_NEURON_CANDIDATES = 12
TOP_HEAD_CANDIDATES = 8
MAX_MIXED_SUBSET = 4
MIN_GAIN = 1e-5

PATTERNS: List[Dict[str, object]] = [
    {
        "pattern_key": "apple",
        "family": "concrete_noun",
        "word": "苹果",
        "prefix_char": "苹",
        "target_char": "果",
        "contexts": ["新鲜的苹", "我今天吃了苹", "这是一个苹", "他刚买了苹", "苹果的苹可以写成苹"],
    },
    {
        "pattern_key": "grape",
        "family": "concrete_noun",
        "word": "葡萄",
        "prefix_char": "葡",
        "target_char": "萄",
        "contexts": ["一串葡", "新鲜的葡", "我喜欢吃葡", "她洗了一串葡", "葡萄的葡可以写成葡", "紫色的葡"],
    },
    {
        "pattern_key": "butterfly",
        "family": "concrete_noun",
        "word": "蝴蝶",
        "prefix_char": "蝴",
        "target_char": "蝶",
        "contexts": ["一只蝴", "那只蝴", "我画了一只蝴", "彩色的蝴", "蝴蝶的蝴可以写成蝴"],
    },
    {
        "pattern_key": "computer",
        "family": "device_noun",
        "word": "电脑",
        "prefix_char": "电",
        "target_char": "脑",
        "contexts": ["这台电", "他打开了电", "我买了新电", "桌上放着电", "电脑的电可以写成电", "一台电"],
    },
    {
        "pattern_key": "language",
        "family": "abstract_noun",
        "word": "语言",
        "prefix_char": "语",
        "target_char": "言",
        "contexts": ["这门语", "我正在学语", "他说的语", "语言的语可以写成语", "人类的语"],
    },
    {
        "pattern_key": "data",
        "family": "abstract_noun",
        "word": "数据",
        "prefix_char": "数",
        "target_char": "据",
        "contexts": ["这些数", "我们分析数", "这份数", "数据的数可以写成数", "统计数"],
    },
    {
        "pattern_key": "if",
        "family": "connective",
        "word": "如果",
        "prefix_char": "如",
        "target_char": "果",
        "contexts": ["条件句里常见如", "如果的如可以写成如", "这里只写到如", "我们常写如", "转折前先写如"],
    },
    {
        "pattern_key": "although",
        "family": "connective",
        "word": "虽然",
        "prefix_char": "虽",
        "target_char": "然",
        "contexts": ["转折词里常见虽", "虽然的虽可以写成虽", "这里只写到虽", "句子先写虽", "我们常说虽"],
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_model_bundle(model_key: str, prefer_cuda: bool):
    spec = MODEL_SPECS[model_key]
    model, tokenizer = load_qwen_like_model(spec["model_path"], prefer_cuda=prefer_cuda)
    return model, tokenizer


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


def encode_prefix(model, tokenizer, prefix: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    return move_batch_to_model_device(model, encoded)


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


def final_next_token_metrics(model, tokenizer, prefix: str, target_id: int) -> Dict[str, object]:
    encoded = encode_prefix(model, tokenizer, prefix)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, return_dict=True)
    logits = outputs.logits[0, -1, :]
    metrics = next_token_metrics_from_logits(logits, target_id)
    metrics["prefix"] = prefix
    metrics["token_ids"] = [int(x) for x in tokenizer(prefix, add_special_tokens=False)["input_ids"]]
    return metrics


def select_contexts(model, tokenizer, target_id: int, prefix_id: int, contexts: Sequence[str]) -> Dict[str, object]:
    rows = []
    for prefix in contexts:
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if not ids or int(ids[-1]) != prefix_id:
            continue
        rows.append(final_next_token_metrics(model, tokenizer, prefix, target_id))
    rows.sort(key=lambda row: (float(row["target_prob"]), -int(row["target_rank"])), reverse=True)
    return {"all_rows": rows, "selected_rows": rows[:TOP_CONTEXTS]}


def layerwise_target_curve(model, tokenizer, prefix: str, target_id: int) -> Dict[str, object]:
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
                "target_rank": int(metrics["target_rank"]),
                "delta_prob": None if delta is None else float(delta),
            }
        )
        prev_prob = prob
    return {"prefix": prefix, "layers": per_layer}


def summarize_layer_curves(curves: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for curve in curves:
        for row in curve["layers"]:
            grouped[int(row["layer_index"])].append(row)
    merged = []
    for layer_idx in sorted(grouped):
        rows = grouped[layer_idx]
        probs = [float(row["target_prob"]) for row in rows]
        ranks = [float(row["target_rank"]) for row in rows]
        deltas = [float(row["delta_prob"]) for row in rows if row["delta_prob"] is not None]
        merged.append(
            {
                "layer_index": int(layer_idx),
                "mean_target_prob": float(statistics.mean(probs)),
                "mean_target_rank": float(statistics.mean(ranks)),
                "mean_delta_prob": float(statistics.mean(deltas)) if deltas else 0.0,
            }
        )
    positive = [row for row in merged if row["layer_index"] >= 0]
    positive.sort(key=lambda row: float(row["mean_delta_prob"]), reverse=True)
    return {
        "mean_curve": merged,
        "top_positive_delta_layers": positive[:TOP_ROUTE_LAYERS],
    }


def get_target_direction(model, target_id: int) -> torch.Tensor:
    weight = get_lm_head(model).weight
    if getattr(weight, "is_meta", False):
        raise NotImplementedError("lm_head 仍在 meta 设备，无法直接抽取目标方向")
    return weight[target_id].detach().float().cpu()


def get_down_proj_weight(layer) -> torch.Tensor:
    return layer.mlp.down_proj.weight.detach().float().cpu()


def capture_route_neuron_scores(model, tokenizer, prefix: str, target_direction: torch.Tensor) -> List[Dict[str, float]]:
    layer_count = len(discover_layers(model))
    buffers, handles = capture_qwen_mlp_payloads(model, {layer_idx: "neuron_in" for layer_idx in range(layer_count)})
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
        top_k = min(6, int(write_scores.numel()))
        top_vals, top_idx = torch.topk(write_scores, k=top_k)
        for rank_idx in range(top_k):
            idx = int(top_idx[rank_idx].item())
            rows.append(
                {
                    "layer_index": int(layer_idx),
                    "neuron_index": idx,
                    "write_score": float(top_vals[rank_idx].item()),
                    "activation": float(last_payload[idx].item()),
                    "target_write": float(target_write[idx].item()),
                    "prefix": prefix,
                }
            )
    return rows


def aggregate_route_neurons(per_prefix_rows: Sequence[List[Dict[str, float]]]) -> List[Dict[str, float]]:
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
                    "support_count": 0,
                },
            )
            bucket["write_scores"].append(float(row["write_score"]))
            bucket["support_count"] += 1
    merged = []
    for bucket in grouped.values():
        merged.append(
            {
                "layer_index": int(bucket["layer_index"]),
                "neuron_index": int(bucket["neuron_index"]),
                "mean_write_score": float(statistics.mean(bucket["write_scores"])),
                "max_write_score": float(max(bucket["write_scores"])),
                "support_count": int(bucket["support_count"]),
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


def register_attention_head_ablation(model, head_specs: Sequence[Dict[str, int]]) -> List[object]:
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


def register_mlp_neuron_ablation(model, neuron_specs: Sequence[Dict[str, int]]) -> List[object]:
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


def mean_target_prob(model, tokenizer, prefixes: Sequence[str], target_id: int, *, handles=None) -> Dict[str, object]:
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
                }
            )
        return {
            "rows": rows,
            "mean_target_prob": float(statistics.mean(row["target_prob"] for row in rows)),
            "mean_target_rank": float(statistics.mean(row["target_rank"] for row in rows)),
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
                }
            )
    rows.sort(key=lambda row: float(row["target_drop"]), reverse=True)
    return rows


def screen_neurons(
    model,
    tokenizer,
    prefixes: Sequence[str],
    target_id: int,
    route_neurons: Sequence[Dict[str, object]],
    baseline_prob: float,
) -> List[Dict[str, object]]:
    rows = []
    for row in route_neurons[:TOP_NEURON_CANDIDATES]:
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
                "support_count": int(row["support_count"]),
            }
        )
    rows.sort(key=lambda row: (float(row["target_drop"]), float(row["route_score"])), reverse=True)
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
            if candidate["candidate_id"] in chosen_ids:
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


def classify_topology(top_head_drop: float, top_neuron_drop: float, mixed_drop: float, baseline_prob: float) -> str:
    if baseline_prob <= 0:
        return "unresolved"
    if mixed_drop / baseline_prob < 0.25:
        return "distributed_weak"
    if top_head_drop >= top_neuron_drop * 1.25:
        return "head_dominant"
    if top_neuron_drop >= top_head_drop * 1.25:
        return "neuron_dominant"
    return "mixed"


def run_pattern(model, tokenizer, pattern: Dict[str, object]) -> Dict[str, object]:
    prefix_char = str(pattern["prefix_char"])
    target_char = str(pattern["target_char"])
    word = str(pattern["word"])
    prefix_id = token_id(tokenizer, prefix_char)
    target_id = token_id(tokenizer, target_char)
    word_id = token_id(tokenizer, word)
    context_info = select_contexts(model, tokenizer, target_id, prefix_id, pattern["contexts"])
    selected_contexts = context_info["selected_rows"]
    if not selected_contexts:
        return {
            "pattern_key": pattern["pattern_key"],
            "family": pattern["family"],
            "word": word,
            "prefix_char": prefix_char,
            "target_char": target_char,
            "token_ids": {"prefix_char": prefix_id, "target_char": target_id, "word": word_id},
            "all_context_rows": context_info["all_rows"],
            "selected_contexts": [],
            "status": "no_valid_context",
        }
    prefixes = [row["prefix"] for row in selected_contexts]
    baseline = mean_target_prob(model, tokenizer, prefixes, target_id)
    curves = [layerwise_target_curve(model, tokenizer, prefix, target_id) for prefix in prefixes]
    layer_summary = summarize_layer_curves(curves)
    target_direction = get_target_direction(model, target_id)
    route_neurons = aggregate_route_neurons(
        [capture_route_neuron_scores(model, tokenizer, prefix, target_direction) for prefix in prefixes]
    )
    route_layers = [int(row["layer_index"]) for row in layer_summary["top_positive_delta_layers"]]
    screened_heads = screen_heads(model, tokenizer, prefixes, target_id, route_layers, float(baseline["mean_target_prob"]))
    screened_neurons = screen_neurons(
        model,
        tokenizer,
        prefixes,
        target_id,
        route_neurons,
        float(baseline["mean_target_prob"]),
    )
    shortlist = []
    shortlist.extend(dict(row) for row in screened_heads[:TOP_HEAD_CANDIDATES])
    shortlist.extend(dict(row) for row in screened_neurons[:TOP_HEAD_CANDIDATES])
    mixed = greedy_mixed_search(
        model,
        tokenizer,
        prefixes,
        target_id,
        float(baseline["mean_target_prob"]),
        shortlist,
    )
    top_head_drop = float(screened_heads[0]["target_drop"]) if screened_heads else 0.0
    top_neuron_drop = float(screened_neurons[0]["target_drop"]) if screened_neurons else 0.0
    topology = classify_topology(
        top_head_drop=top_head_drop,
        top_neuron_drop=top_neuron_drop,
        mixed_drop=float(mixed["final_target_drop"]),
        baseline_prob=float(baseline["mean_target_prob"]),
    )
    return {
        "pattern_key": pattern["pattern_key"],
        "family": pattern["family"],
        "word": word,
        "prefix_char": prefix_char,
        "target_char": target_char,
        "token_ids": {"prefix_char": prefix_id, "target_char": target_id, "word": word_id},
        "all_context_rows": context_info["all_rows"],
        "selected_contexts": selected_contexts,
        "baseline_mean_target_prob": float(baseline["mean_target_prob"]),
        "baseline_mean_target_rank": float(baseline["mean_target_rank"]),
        "layer_route": layer_summary,
        "route_neurons": route_neurons[:20],
        "screened_heads": screened_heads[:12],
        "screened_neurons": screened_neurons[:12],
        "mixed_route_subset": mixed,
        "topology_label": topology,
        "status": "ok",
    }


def summarize_model_patterns(pattern_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    valid = [row for row in pattern_rows if row["status"] == "ok"]
    family_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in valid:
        family_group[str(row["family"])].append(row)
    family_summary = {}
    for family, rows in family_group.items():
        topology_counter = Counter(str(row["topology_label"]) for row in rows)
        top_layers = [int(row["layer_route"]["top_positive_delta_layers"][0]["layer_index"]) for row in rows if row["layer_route"]["top_positive_delta_layers"]]
        family_summary[family] = {
            "count": int(len(rows)),
            "topology_counts": dict(topology_counter),
            "mean_baseline_prob": float(statistics.mean(float(row["baseline_mean_target_prob"]) for row in rows)),
            "mean_final_drop": float(statistics.mean(float(row["mixed_route_subset"]["final_target_drop"]) for row in rows)),
            "peak_layers": top_layers,
        }
    overall_topology = Counter(str(row["topology_label"]) for row in valid)
    return {
        "family_summary": family_summary,
        "overall_topology_counts": dict(overall_topology),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = ["# stage492 中文模式路线图谱", ""]
    lines.append("## 总结")
    lines.append("")
    for model_key in MODEL_KEYS:
        ms = summary["models"][model_key]
        lines.append(f"### {model_key}")
        lines.append("")
        lines.append(f"- 使用 CUDA（图形处理器）: `{ms['used_cuda']}`")
        lines.append(f"- 总体拓扑统计: `{json.dumps(ms['model_summary']['overall_topology_counts'], ensure_ascii=False)}`")
        for family, family_row in ms["model_summary"]["family_summary"].items():
            lines.append(
                f"- `{family}`: 平均基线概率 `{family_row['mean_baseline_prob']:.4f}`，"
                f"平均最终下降 `{family_row['mean_final_drop']:.4f}`，"
                f"拓扑 `{json.dumps(family_row['topology_counts'], ensure_ascii=False)}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立多类中文模式的路线机制图谱")
    parser.add_argument("--prefer-cuda", action="store_true", help="优先使用 CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    summary = {
        "stage": "stage492_chinese_pattern_route_atlas",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prefer_cuda": bool(args.prefer_cuda),
        "models": {},
    }
    for model_key in MODEL_KEYS:
        fallback_to_cpu = False
        model = None
        tokenizer = None
        try:
            model, tokenizer = load_model_bundle(model_key, prefer_cuda=args.prefer_cuda)
            pattern_rows = [run_pattern(model, tokenizer, pattern) for pattern in PATTERNS]
            summary["models"][model_key] = {
                "used_cuda": bool(get_model_device(model).type == "cuda"),
                "load_mode": "prefer_cuda" if args.prefer_cuda else "cpu",
                "patterns": pattern_rows,
                "model_summary": summarize_model_patterns(pattern_rows),
            }
        except NotImplementedError as exc:
            fallback_to_cpu = True
            if model is not None:
                free_model(model)
            model, tokenizer = load_model_bundle(model_key, prefer_cuda=False)
            pattern_rows = [run_pattern(model, tokenizer, pattern) for pattern in PATTERNS]
            summary["models"][model_key] = {
                "used_cuda": bool(get_model_device(model).type == "cuda"),
                "load_mode": "cpu_fallback_after_meta",
                "fallback_reason": str(exc),
                "patterns": pattern_rows,
                "model_summary": summarize_model_patterns(pattern_rows),
            }
        finally:
            if model is not None:
                free_model(model)
    summary["elapsed_seconds"] = float(time.time() - started)
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "REPORT.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
