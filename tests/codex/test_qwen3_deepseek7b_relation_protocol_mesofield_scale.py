#!/usr/bin/env python
"""
Estimate the minimal causal scale of the relation protocol meso-field in Qwen3 and DeepSeek-7B.

This script extends the existing relation-protocol head-group causal test by:
1) scanning top-k joint head ablations for k in {1, 3, 8, 16}
2) ablating top scoring layer clusters as a coarse meso-field probe

The goal is to answer a stricter question:
when does relation TT protocol start to collapse causally,
and is the smallest stable unit still larger than a tiny head group?
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn
    raise RuntimeError("Cannot find a probe-able attention module")


class HeadGroupAblator:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.handles = []
        self.active_map: Dict[int, set[int]] = {}
        self.head_dim = int(getattr(model.config, "hidden_size") // getattr(model.config, "num_attention_heads"))
        for li, layer in enumerate(self.layers):
            attn = get_attention_module(layer)
            target = attn.o_proj if hasattr(attn, "o_proj") else attn.c_proj
            self.handles.append(target.register_forward_pre_hook(self._pre_hook(li)))

    def _pre_hook(self, li: int):
        def fn(_module, inputs):
            heads = self.active_map.get(li)
            if not heads:
                return None
            x = inputs[0]
            y = x.clone()
            for head in heads:
                start = head * self.head_dim
                end = start + self.head_dim
                y[..., start:end] = 0
            return (y,)

        return fn

    def set_active_group(self, heads: Sequence[Tuple[int, int]]) -> None:
        active: Dict[int, set[int]] = defaultdict(set)
        for layer, head in heads:
            active[int(layer)].add(int(head))
        self.active_map = dict(active)

    def clear(self) -> None:
        self.active_map = {}

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def load_model(model_path: str, dtype_name: str, prefer_cuda: bool):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    dtype = getattr(torch, dtype_name)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": "auto" if want_cuda else "cpu",
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    model.config.output_attentions = True
    return model, tok


def support_families() -> Dict[str, List[str]]:
    return {
        "descriptor": ["big", "small", "large", "tiny", "hot", "cold", "short", "long", "fast", "slow"],
        "male": ["king", "man", "prince", "father", "boy", "uncle"],
        "female": ["queen", "woman", "princess", "mother", "girl", "aunt"],
        "instance": ["apple", "banana", "cat", "dog", "car", "bike", "orange", "rabbit"],
        "category": ["fruit", "animal", "vehicle", "object", "tool", "food"],
        "part": ["wheel", "leaf", "root", "handle", "door", "branch"],
        "whole": ["car", "tree", "house", "bike", "plant", "truck"],
        "cause": ["fire", "virus", "rain", "friction", "heat", "bacteria"],
        "effect": ["smoke", "disease", "flood", "motion", "burn", "infection"],
    }


def relation_specs() -> Dict[str, Dict[str, object]]:
    return {
        "gender": {
            "pairs": [("king", "queen"), ("man", "woman")],
            "endpoint_families": {"king": "male", "man": "male", "queen": "female", "woman": "female"},
        },
        "hypernym": {
            "pairs": [("apple", "fruit"), ("cat", "animal")],
            "endpoint_families": {"apple": "instance", "cat": "instance", "fruit": "category", "animal": "category"},
        },
        "antonym": {
            "pairs": [("hot", "cold"), ("big", "small")],
            "endpoint_families": {"hot": "descriptor", "cold": "descriptor", "big": "descriptor", "small": "descriptor"},
        },
        "synonym": {
            "pairs": [("big", "large"), ("small", "tiny")],
            "endpoint_families": {"big": "descriptor", "large": "descriptor", "small": "descriptor", "tiny": "descriptor"},
        },
        "meronym": {
            "pairs": [("wheel", "car"), ("leaf", "tree")],
            "endpoint_families": {"wheel": "part", "leaf": "part", "car": "whole", "tree": "whole"},
        },
        "cause_effect": {
            "pairs": [("fire", "smoke"), ("virus", "disease")],
            "endpoint_families": {"fire": "cause", "virus": "cause", "smoke": "effect", "disease": "effect"},
        },
    }


def base_prompts(word: str) -> List[str]:
    return [f"This is {word}", f"That is {word}", f"The word is {word}"]


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
    return out


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def affine_basis(xs: Sequence[np.ndarray], rank_k: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.stack(xs, axis=0).astype(np.float32)
    mu = np.mean(mat, axis=0).astype(np.float32)
    centered = mat - mu[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    k = int(min(rank_k, vh.shape[0]))
    basis = vh[:k].T.astype(np.float32) if k > 0 else np.zeros((mat.shape[1], 0), dtype=np.float32)
    return mu, basis


def affine_project(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return mu.astype(np.float32)
    centered = (x - mu).astype(np.float32)
    coeff = basis.T @ centered
    return (mu + basis @ coeff).astype(np.float32)


def residual_ratio(x: np.ndarray, mu: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    proj = affine_project(x, mu, basis)
    denom = float(np.linalg.norm(x - mu)) + eps
    return float(np.linalg.norm(x - proj) / denom)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def relation_error(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + eps
    return float(np.linalg.norm(v1 - v2) / denom)


def relation_score(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float]:
    err = relation_error(v1, v2)
    cos = cosine(v1, v2)
    score = max(0.0, cos) / (1.0 + err)
    return float(err), float(cos), float(score)


def last_token_head_topo(out, target_len: int) -> Dict[Tuple[int, int], np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        for hi in range(last_row.shape[0]):
            topo[(li, hi)] = last_row[hi].astype(np.float32)
    return topo


def last_token_topo(out, target_len: int) -> Dict[int, np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        topo[li] = last_row.reshape(-1).astype(np.float32)
    return topo


def resolve_snapshot_dir(repo_dir_name: str) -> str:
    roots = [
        Path(r"D:\develop\model\hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        snapshot_root = root / repo_dir_name / "snapshots"
        if not snapshot_root.exists():
            continue
        candidates = sorted([p for p in snapshot_root.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError(f"Cannot resolve snapshot directory for {repo_dir_name}")


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("qwen3_4b", resolve_snapshot_dir("models--Qwen--Qwen3-4B")),
        ("deepseek_7b", resolve_snapshot_dir("models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B")),
    ]


def build_relation_word_set(relation_name: str) -> List[str]:
    spec = relation_specs()[relation_name]
    support = support_families()
    endpoint_families: Dict[str, str] = spec["endpoint_families"]
    words = set(endpoint_families.keys())
    for family in endpoint_families.values():
        words.update(support[family])
    return sorted(words)


def compute_relation_condition_topo(
    model,
    tok,
    ablator: HeadGroupAblator | None,
    relation_name: str,
    group: Sequence[Tuple[int, int]] | None,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], int]:
    if ablator is not None:
        if group:
            ablator.set_active_group(group)
        else:
            ablator.clear()

    words = build_relation_word_set(relation_name)
    target_len = max(prompt_len(tok, text) for word in words for text in base_prompts(word))
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    base_topo: Dict[str, Dict[int, np.ndarray]] = {}
    for word in words:
        rows = {li: [] for li in range(n_layers)}
        for text in base_prompts(word):
            out = run_model(model, tok, text)
            topo = last_token_topo(out, target_len)
            for li in range(n_layers):
                rows[li].append(topo[li])
        base_topo[word] = {li: mean_stack(vs) for li, vs in rows.items()}
    return base_topo, target_len


def compute_relation_bridge_from_layer_topo(relation_name: str, base_topo: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, object]:
    spec = relation_specs()[relation_name]
    support = support_families()
    endpoint_families: Dict[str, str] = spec["endpoint_families"]
    n_layers = len(next(iter(base_topo.values())))

    family_basis = {family: {} for family in set(endpoint_families.values())}
    for family in family_basis:
        fam_words = support[family]
        for li in range(n_layers):
            family_basis[family][li] = affine_basis(
                [base_topo[word][li] for word in fam_words],
                rank_k=min(3, len(fam_words) - 1),
            )

    (a, b), (c, d) = spec["pairs"]
    bridge_tt = []
    endpoint_basis = []
    relation_align = []
    for li in range(n_layers):
        basis_vals = []
        for word, family in endpoint_families.items():
            mu_t, basis_t = family_basis[family][li]
            basis_vals.append(1.0 - residual_ratio(base_topo[word][li], mu_t, basis_t))
        topo_vec1 = base_topo[b][li] - base_topo[a][li]
        topo_vec2 = base_topo[d][li] - base_topo[c][li]
        _err, _cos, score = relation_score(topo_vec1, topo_vec2)
        basis_mean = float(np.mean(basis_vals))
        endpoint_basis.append(basis_mean)
        relation_align.append(float(score))
        bridge_tt.append(float(basis_mean * score))

    peak_layer = int(np.argmax(bridge_tt))
    return {
        "bridge_tt_by_layer": bridge_tt,
        "endpoint_topo_basis_by_layer": endpoint_basis,
        "relation_align_topo_by_layer": relation_align,
        "peak_layer": peak_layer,
        "peak_bridge_tt": float(bridge_tt[peak_layer]),
    }


def compute_baseline_head_scores(model, tok) -> Dict[str, object]:
    support = support_families()
    words = sorted({w for items in support.values() for w in items})
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    global_target_len = max(prompt_len(tok, text) for word in words for text in base_prompts(word))

    word_head_topo: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for word in words:
        rows: Dict[Tuple[int, int], List[np.ndarray]] = {(li, hi): [] for li in range(n_layers) for hi in range(n_heads)}
        for text in base_prompts(word):
            out = run_model(model, tok, text)
            head_topo = last_token_head_topo(out, global_target_len)
            for key, vec in head_topo.items():
                rows[key].append(vec)
        word_head_topo[word] = {key: mean_stack(vs) for key, vs in rows.items()}

    family_head_basis: Dict[str, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in support}
    for family, fam_words in support.items():
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                family_head_basis[family][key] = affine_basis(
                    [word_head_topo[word][key] for word in fam_words],
                    rank_k=min(3, len(fam_words) - 1),
                )

    relation_rows = {}
    layer_scores = {}
    for relation_name, spec in relation_specs().items():
        endpoint_families: Dict[str, str] = spec["endpoint_families"]
        (a, b), (c, d) = spec["pairs"]
        head_rows = []
        layer_totals = defaultdict(float)
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                basis_vals = []
                for word, family in endpoint_families.items():
                    mu_t, basis_t = family_head_basis[family][key]
                    basis_vals.append(1.0 - residual_ratio(word_head_topo[word][key], mu_t, basis_t))
                topo_vec1 = word_head_topo[b][key] - word_head_topo[a][key]
                topo_vec2 = word_head_topo[d][key] - word_head_topo[c][key]
                _err, _cos, score = relation_score(topo_vec1, topo_vec2)
                endpoint_basis = float(np.mean(basis_vals))
                bridge_tt = float(endpoint_basis * score)
                row = {
                    "layer": int(li),
                    "head": int(hi),
                    "endpoint_topo_basis": endpoint_basis,
                    "relation_align_topo": float(score),
                    "bridge_tt": bridge_tt,
                }
                head_rows.append(row)
                layer_totals[li] += bridge_tt
        relation_rows[relation_name] = sorted(head_rows, key=lambda row: float(row["bridge_tt"]), reverse=True)
        layer_scores[relation_name] = [
            {"layer": int(li), "bridge_tt_sum": float(layer_totals[li])}
            for li in range(n_layers)
        ]
        layer_scores[relation_name].sort(key=lambda row: float(row["bridge_tt_sum"]), reverse=True)

    return {
        "relation_rows": relation_rows,
        "layer_scores": layer_scores,
        "global_target_len": global_target_len,
    }


def pick_control_group(
    ranked_rows: Sequence[Dict[str, float]],
    group: Sequence[Tuple[int, int]],
    n_heads: int,
) -> List[Tuple[int, int]]:
    top_set = {(int(row["layer"]), int(row["head"])) for row in ranked_rows}
    per_layer_need = Counter(layer for layer, _head in group)
    rows_by_layer = defaultdict(list)
    for row in ranked_rows:
        rows_by_layer[int(row["layer"])].append(row)

    control: List[Tuple[int, int]] = []
    for layer, need in per_layer_need.items():
        candidates = [
            (int(row["layer"]), int(row["head"]))
            for row in reversed(rows_by_layer[layer])
            if (int(row["layer"]), int(row["head"])) not in top_set
        ]
        if len(candidates) < need:
            candidates.extend(
                [
                    (layer, head)
                    for head in range(n_heads)
                    if (layer, head) not in top_set and (layer, head) not in candidates
                ]
            )
        control.extend(candidates[:need])
    return control


def summarize_ablation(baseline_peak: float, ablated_peak: float, control_peak: float, eps: float = 1e-12) -> Dict[str, float]:
    return {
        "collapse_ratio": float(max(0.0, (baseline_peak - ablated_peak) / (baseline_peak + eps))),
        "control_collapse_ratio": float(max(0.0, (baseline_peak - control_peak) / (baseline_peak + eps))),
        "causal_margin": float((control_peak - ablated_peak) / (baseline_peak + eps)),
    }


def compute_ablation_case(
    model,
    tok,
    ablator: HeadGroupAblator,
    relation_name: str,
    top_group: Sequence[Tuple[int, int]],
    control_group: Sequence[Tuple[int, int]],
    baseline_bridge: Dict[str, object],
) -> Dict[str, object]:
    top_topo, _ = compute_relation_condition_topo(model, tok, ablator, relation_name, top_group)
    ctrl_topo, _ = compute_relation_condition_topo(model, tok, ablator, relation_name, control_group)
    top_bridge = compute_relation_bridge_from_layer_topo(relation_name, top_topo)
    ctrl_bridge = compute_relation_bridge_from_layer_topo(relation_name, ctrl_topo)
    baseline_peak = float(baseline_bridge["peak_bridge_tt"])
    top_peak = float(top_bridge["peak_bridge_tt"])
    ctrl_peak = float(ctrl_bridge["peak_bridge_tt"])
    summary = summarize_ablation(baseline_peak, top_peak, ctrl_peak)
    return {
        "top_group": [{"layer": int(l), "head": int(h)} for l, h in top_group],
        "control_group": [{"layer": int(l), "head": int(h)} for l, h in control_group],
        "top_group_size": int(len(top_group)),
        "control_group_size": int(len(control_group)),
        "top_group_ablation": top_bridge,
        "control_group_ablation": ctrl_bridge,
        "summary": {
            "baseline_peak_bridge_tt": baseline_peak,
            "top_group_peak_bridge_tt": top_peak,
            "control_group_peak_bridge_tt": ctrl_peak,
            "top_group_collapse_ratio": summary["collapse_ratio"],
            "control_group_collapse_ratio": summary["control_collapse_ratio"],
            "causal_margin": summary["causal_margin"],
        },
    }


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    k_values: Sequence[int],
    layer_cluster_size: int,
) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    baseline_head_scores = compute_baseline_head_scores(model, tok)
    ablator = HeadGroupAblator(model)
    relations = {}

    try:
        for relation_name in relation_specs().keys():
            baseline_topo, _ = compute_relation_condition_topo(model, tok, ablator, relation_name, None)
            baseline_bridge = compute_relation_bridge_from_layer_topo(relation_name, baseline_topo)

            ranked_rows = baseline_head_scores["relation_rows"][relation_name]
            ranked_heads = [(int(row["layer"]), int(row["head"])) for row in ranked_rows]
            k_scan = {}
            for k in k_values:
                use_k = min(int(k), len(ranked_heads))
                top_group = ranked_heads[:use_k]
                control_group = pick_control_group(ranked_rows, top_group, n_heads)
                k_scan[str(k)] = compute_ablation_case(model, tok, ablator, relation_name, top_group, control_group, baseline_bridge)

            ranked_layers = baseline_head_scores["layer_scores"][relation_name]
            top_layers = [int(row["layer"]) for row in ranked_layers[: min(layer_cluster_size, n_layers)]]
            control_layers = [int(row["layer"]) for row in ranked_layers[-min(layer_cluster_size, n_layers) :]]
            top_layer_group = [(layer, head) for layer in top_layers for head in range(n_heads)]
            control_layer_group = [(layer, head) for layer in control_layers for head in range(n_heads)]
            layer_cluster_scan = compute_ablation_case(
                model,
                tok,
                ablator,
                relation_name,
                top_layer_group,
                control_layer_group,
                baseline_bridge,
            )
            layer_cluster_scan["top_layers"] = top_layers
            layer_cluster_scan["control_layers"] = control_layers

            positive_k = [
                int(k)
                for k, row in k_scan.items()
                if float(row["summary"]["causal_margin"]) > 0
            ]
            stronger_k = [
                int(k)
                for k, row in k_scan.items()
                if float(row["summary"]["top_group_collapse_ratio"]) > float(row["summary"]["control_group_collapse_ratio"])
            ]
            relations[relation_name] = {
                "baseline": baseline_bridge,
                "ranked_heads_top20": [
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "bridge_tt": float(row["bridge_tt"]),
                        "endpoint_topo_basis": float(row["endpoint_topo_basis"]),
                        "relation_align_topo": float(row["relation_align_topo"]),
                    }
                    for row in ranked_rows[:20]
                ],
                "ranked_layers": ranked_layers,
                "k_scan": k_scan,
                "layer_cluster_scan": layer_cluster_scan,
                "mesofield_summary": {
                    "minimal_positive_margin_k": int(min(positive_k)) if positive_k else None,
                    "minimal_stronger_than_control_k": int(min(stronger_k)) if stronger_k else None,
                    "layer_cluster_margin": float(layer_cluster_scan["summary"]["causal_margin"]),
                    "layer_cluster_collapse_ratio": float(layer_cluster_scan["summary"]["top_group_collapse_ratio"]),
                },
            }
    finally:
        ablator.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_summary = {
        "k_values": [int(k) for k in k_values],
        "mean_top_group_collapse_ratio_by_k": {},
        "mean_control_group_collapse_ratio_by_k": {},
        "mean_causal_margin_by_k": {},
        "stronger_than_control_count_by_k": {},
    }
    for k in k_values:
        key = str(k)
        top_rows = [float(rel["k_scan"][key]["summary"]["top_group_collapse_ratio"]) for rel in relations.values()]
        ctrl_rows = [float(rel["k_scan"][key]["summary"]["control_group_collapse_ratio"]) for rel in relations.values()]
        margin_rows = [float(rel["k_scan"][key]["summary"]["causal_margin"]) for rel in relations.values()]
        global_summary["mean_top_group_collapse_ratio_by_k"][key] = float(np.mean(top_rows))
        global_summary["mean_control_group_collapse_ratio_by_k"][key] = float(np.mean(ctrl_rows))
        global_summary["mean_causal_margin_by_k"][key] = float(np.mean(margin_rows))
        global_summary["stronger_than_control_count_by_k"][key] = int(sum(1 for a, b in zip(top_rows, ctrl_rows) if a > b))

    layer_cluster_margins = [float(rel["layer_cluster_scan"]["summary"]["causal_margin"]) for rel in relations.values()]
    layer_cluster_collapse = [float(rel["layer_cluster_scan"]["summary"]["top_group_collapse_ratio"]) for rel in relations.values()]
    layer_cluster_ctrl = [float(rel["layer_cluster_scan"]["summary"]["control_group_collapse_ratio"]) for rel in relations.values()]
    global_summary["layer_cluster_size"] = int(layer_cluster_size)
    global_summary["mean_layer_cluster_collapse_ratio"] = float(np.mean(layer_cluster_collapse))
    global_summary["mean_layer_cluster_control_collapse_ratio"] = float(np.mean(layer_cluster_ctrl))
    global_summary["mean_layer_cluster_margin"] = float(np.mean(layer_cluster_margins))
    global_summary["layer_cluster_stronger_than_control_count"] = int(
        sum(
            1
            for rel in relations.values()
            if float(rel["layer_cluster_scan"]["summary"]["top_group_collapse_ratio"])
            > float(rel["layer_cluster_scan"]["summary"]["control_group_collapse_ratio"])
        )
    )

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "runtime_sec": float(time.time() - t0),
        },
        "relations": relations,
        "global_summary": global_summary,
    }


def parse_k_values(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Need at least one k value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate the minimal causal scale of the relation protocol meso-field")
    ap.add_argument("--dtype-qwen", type=str, default="bfloat16")
    ap.add_argument("--dtype-deepseek", type=str, default="bfloat16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--k-values", type=str, default="1,3,8,16")
    ap.add_argument("--layer-cluster-size", type=int, default=2)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json",
    )
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "k_values": k_values,
            "layer_cluster_size": int(args.layer_cluster_size),
        },
        "models": {},
    }
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_qwen if model_name == "qwen3_4b" else args.dtype_deepseek
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(
            model_name,
            model_path,
            dtype_name,
            prefer_cuda=not args.cpu_only,
            k_values=k_values,
            layer_cluster_size=args.layer_cluster_size,
        )
        results["models"][model_name] = row
        print(f"[summary] {model_name} relations={list(row['relations'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
