#!/usr/bin/env python
"""
Layerwise alignment between:
1) representation-space family bases from hidden states
2) topology-space family bases from attention matrices

Goal:
- quantify which layers are more representation-dominant
- quantify which layers are more routing/topology-dominant
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers")


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


def family_words() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "abstract": ["justice", "truth", "logic", "language", "freedom", "memory"],
    }


def prompts_for_word(word: str) -> List[str]:
    return [
        f"This is {word}",
        f"That is {word}",
        f"It is {word}",
    ]


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def compatible_words(tok) -> Dict[str, List[str]]:
    out = {}
    for family, words in family_words().items():
        groups: Dict[int, List[str]] = {}
        for word in words:
            lengths = {prompt_len(tok, t) for t in prompts_for_word(word)}
            if len(lengths) == 1:
                seq_len = next(iter(lengths))
                groups.setdefault(seq_len, []).append(word)
        if not groups:
            out[family] = []
            continue
        best_len = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
        out[family] = groups[best_len]
    return out


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
    return out


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
    res = x - proj
    denom = float(np.linalg.norm(x - mu)) + eps
    return float(np.linalg.norm(res) / denom)


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def collect_vectors(model, tok, selected: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]]]:
    layers = discover_layers(model)
    n_layers = len(layers)
    repr_rows: Dict[str, Dict[int, List[np.ndarray]]] = {}
    topo_rows: Dict[str, Dict[int, List[np.ndarray]]] = {}
    for family, words in selected.items():
        for word in words:
            repr_rows[word] = {li: [] for li in range(n_layers)}
            topo_rows[word] = {li: [] for li in range(n_layers)}
            for text in prompts_for_word(word):
                out = run_model(model, tok, text)
                for li in range(n_layers):
                    hidden = out.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
                    attn = out.attentions[li][0].detach().float().cpu().numpy().astype(np.float32).reshape(-1)
                    repr_rows[word][li].append(hidden)
                    topo_rows[word][li].append(attn)
    repr_mean = {w: {li: mean_stack(vs) for li, vs in layer_map.items()} for w, layer_map in repr_rows.items()}
    topo_mean = {w: {li: mean_stack(vs) for li, vs in layer_map.items()} for w, layer_map in topo_rows.items()}
    return repr_mean, topo_mean


def layer_family_residuals(word_map: Dict[str, Dict[int, np.ndarray]], selected: Dict[str, List[str]], n_layers: int) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {family: {} for family in selected.keys()}
    for family, words in selected.items():
        rank_k = min(3, max(1, len(words) - 1))
        for li in range(n_layers):
            mu, basis = affine_basis([word_map[w][li] for w in words], rank_k=rank_k)
            vals = [residual_ratio(word_map[w][li], mu, basis) for w in words]
            out[family][li] = float(np.mean(vals))
    return out


def mean_over_families(layer_map: Dict[str, Dict[int, float]], n_layers: int) -> List[float]:
    vals = []
    fams = list(layer_map.keys())
    for li in range(n_layers):
        vals.append(float(np.mean([layer_map[f][li] for f in fams])))
    return vals


def rank_layers(vals: Sequence[float], top_n: int, reverse: bool = False) -> List[int]:
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)
    return [int(i) for i in order[:top_n]]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    selected = compatible_words(tok)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    repr_mean, topo_mean = collect_vectors(model, tok, selected)

    repr_res = layer_family_residuals(repr_mean, selected, n_layers)
    topo_res = layer_family_residuals(topo_mean, selected, n_layers)
    repr_avg = mean_over_families(repr_res, n_layers)
    topo_avg = mean_over_families(topo_res, n_layers)
    dominance = [float(topo_avg[i] - repr_avg[i]) for i in range(n_layers)]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": n_layers,
            "n_heads": int(getattr(model.config, "num_attention_heads")),
            "runtime_sec": float(time.time() - t0),
        },
        "selected_words": selected,
        "repr_family_residual_by_layer": repr_res,
        "topo_family_residual_by_layer": topo_res,
        "repr_avg_residual_by_layer": repr_avg,
        "topo_avg_residual_by_layer": topo_avg,
        "topology_minus_repr_by_layer": dominance,
        "layer_role_summary": {
            "most_topology_dominant_layers": rank_layers(dominance, top_n=min(5, n_layers), reverse=False),
            "most_repr_dominant_layers": rank_layers(dominance, top_n=min(5, n_layers), reverse=True),
            "best_repr_layers": rank_layers(repr_avg, top_n=min(5, n_layers), reverse=False),
            "best_topology_layers": rank_layers(topo_avg, top_n=min(5, n_layers), reverse=False),
        },
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Layerwise representation-topology alignment in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_repr_topology_layer_alignment_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(
            f"[summary] {model_name} best_repr={row['layer_role_summary']['best_repr_layers']} "
            f"best_topo={row['layer_role_summary']['best_topology_layers']}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
