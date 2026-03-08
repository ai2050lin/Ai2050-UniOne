#!/usr/bin/env python
"""
Small-group causal validation for relation protocol heads in GPT-2 and Qwen3.

For each relation family:
1) take the top-k TT-carrying heads from the head atlas
2) ablate that head group
3) compare against a matched-layer control group

We measure collapse of the model-level TT bridge peak.
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


def load_head_atlas() -> Dict[str, object]:
    path = Path("tests/codex_temp/gpt2_qwen3_relation_protocol_head_atlas_20260308.json")
    return json.loads(path.read_text(encoding="utf-8"))


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def pick_group_and_control(model_head_atlas: Dict[str, object], relation_name: str, group_size: int, n_heads: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    top_heads = model_head_atlas["relations"][relation_name]["top_heads"][:group_size]
    group = [(int(row["layer"]), int(row["head"])) for row in top_heads]
    top_set = {(int(row["layer"]), int(row["head"])) for row in model_head_atlas["relations"][relation_name]["top_heads"]}
    per_layer_need = Counter(layer for layer, _head in group)
    control = []
    for layer, need in per_layer_need.items():
        count = 0
        for head in range(n_heads):
            if (layer, head) not in top_set:
                control.append((layer, head))
                count += 1
            if count >= need:
                break
    return group, control


def compute_relation_bridge(model, tok, ablator: HeadGroupAblator | None, relation_name: str, group: Sequence[Tuple[int, int]] | None) -> Dict[str, object]:
    if ablator is not None:
        if group:
            ablator.set_active_group(group)
        else:
            ablator.clear()

    spec = relation_specs()[relation_name]
    support = support_families()
    endpoint_families: Dict[str, str] = spec["endpoint_families"]
    words = sorted(set(endpoint_families.keys()) | {w for fam in endpoint_families.values() for w in support[fam]})
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

    family_basis = {family: {} for family in set(endpoint_families.values())}
    for family in family_basis:
        fam_words = support[family]
        for li in range(n_layers):
            family_basis[family][li] = affine_basis([base_topo[word][li] for word in fam_words], rank_k=min(3, len(fam_words) - 1))

    (a, b), (c, d) = spec["pairs"]
    bridge_tt = []
    for li in range(n_layers):
        basis_vals = []
        for word, family in endpoint_families.items():
            mu_t, basis_t = family_basis[family][li]
            basis_vals.append(1.0 - residual_ratio(base_topo[word][li], mu_t, basis_t))
        topo_vec1 = base_topo[b][li] - base_topo[a][li]
        topo_vec2 = base_topo[d][li] - base_topo[c][li]
        _err, _cos, score = relation_score(topo_vec1, topo_vec2)
        bridge_tt.append(float(np.mean(basis_vals) * score))

    peak_layer = int(np.argmax(bridge_tt))
    return {
        "bridge_tt_by_layer": bridge_tt,
        "peak_layer": peak_layer,
        "peak_bridge_tt": float(bridge_tt[peak_layer]),
    }


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool, group_size: int) -> Dict[str, object]:
    head_atlas = load_head_atlas()["models"][model_name]
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_heads = int(getattr(model.config, "num_attention_heads"))
    ablator = HeadGroupAblator(model)
    relations = {}

    try:
        for relation_name in relation_specs().keys():
            top_group, control_group = pick_group_and_control(head_atlas, relation_name, group_size, n_heads)
            baseline = compute_relation_bridge(model, tok, ablator, relation_name, None)
            top_ablate = compute_relation_bridge(model, tok, ablator, relation_name, top_group)
            ctrl_ablate = compute_relation_bridge(model, tok, ablator, relation_name, control_group)

            base_peak = float(baseline["peak_bridge_tt"])
            top_peak = float(top_ablate["peak_bridge_tt"])
            ctrl_peak = float(ctrl_ablate["peak_bridge_tt"])
            eps = 1e-12
            relations[relation_name] = {
                "top_group": [{"layer": int(l), "head": int(h)} for l, h in top_group],
                "control_group": [{"layer": int(l), "head": int(h)} for l, h in control_group],
                "baseline": baseline,
                "top_group_ablation": top_ablate,
                "control_group_ablation": ctrl_ablate,
                "summary": {
                    "baseline_peak_bridge_tt": base_peak,
                    "top_group_peak_bridge_tt": top_peak,
                    "control_group_peak_bridge_tt": ctrl_peak,
                    "top_group_collapse_ratio": float(max(0.0, (base_peak - top_peak) / (base_peak + eps))),
                    "control_group_collapse_ratio": float(max(0.0, (base_peak - ctrl_peak) / (base_peak + eps))),
                    "causal_margin": float((ctrl_peak - top_peak) / (base_peak + eps)),
                    "baseline_peak_layer": int(baseline["peak_layer"]),
                    "top_group_peak_layer": int(top_ablate["peak_layer"]),
                    "control_group_peak_layer": int(ctrl_ablate["peak_layer"]),
                },
            }
    finally:
        ablator.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    top_collapses = [row["summary"]["top_group_collapse_ratio"] for row in relations.values()]
    ctrl_collapses = [row["summary"]["control_group_collapse_ratio"] for row in relations.values()]
    causal_margins = [row["summary"]["causal_margin"] for row in relations.values()]
    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": int(getattr(model.config, "num_hidden_layers")),
            "n_heads": n_heads,
            "runtime_sec": float(time.time() - t0),
            "group_size": int(group_size),
        },
        "relations": relations,
        "global_summary": {
            "mean_top_group_collapse_ratio": float(np.mean(top_collapses)),
            "mean_control_group_collapse_ratio": float(np.mean(ctrl_collapses)),
            "mean_causal_margin": float(np.mean(causal_margins)),
            "stronger_than_control_count": int(sum(1 for a, b in zip(top_collapses, ctrl_collapses) if a > b)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-group causal validation for relation protocol heads in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_relation_protocol_head_group_causal_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only, group_size=args.group_size)
        results["models"][model_name] = row
        print(f"[summary] {model_name} relations={list(row['relations'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
