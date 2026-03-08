#!/usr/bin/env python
"""
Trace how relation families couple concept skeletons and topology routes
across layers in GPT-2 and Qwen3.

We focus on three relation families that were previously found to be
topology-dominant:
- synonym:      big->large, small->tiny
- meronym:      wheel->car, leaf->tree
- cause_effect: fire->smoke, virus->disease

For each layer l we measure:
1) endpoint_repr_basis(l): how stably the relation endpoints sit on their
   support concept bases in representation space.
2) endpoint_topo_basis(l): the same, but in topology space.
3) relation_align_repr(l): how well the two example pairs align as a relation
   path in representation space.
4) relation_align_topo(l): how well they align in topology space.
5) bridge_ht(l): concept skeleton in H coupled with relation structure in T.
6) bridge_hh(l): concept skeleton in H coupled with relation structure in H.
7) bridge_tt(l): concept skeleton in T coupled with relation structure in T.
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
        "descriptor": ["big", "small", "large", "tiny", "hot", "cold", "short", "long"],
        "part": ["wheel", "leaf", "root", "handle", "door", "branch"],
        "whole": ["car", "tree", "house", "bike", "plant", "truck"],
        "cause": ["fire", "virus", "rain", "friction", "heat", "bacteria"],
        "effect": ["smoke", "disease", "flood", "motion", "burn", "infection"],
    }


def relation_specs() -> Dict[str, Dict[str, object]]:
    return {
        "synonym": {
            "pairs": [("big", "large"), ("small", "tiny")],
            "endpoint_families": {
                "big": "descriptor",
                "large": "descriptor",
                "small": "descriptor",
                "tiny": "descriptor",
            },
        },
        "meronym": {
            "pairs": [("wheel", "car"), ("leaf", "tree")],
            "endpoint_families": {
                "wheel": "part",
                "leaf": "part",
                "car": "whole",
                "tree": "whole",
            },
        },
        "cause_effect": {
            "pairs": [("fire", "smoke"), ("virus", "disease")],
            "endpoint_families": {
                "fire": "cause",
                "virus": "cause",
                "smoke": "effect",
                "disease": "effect",
            },
        },
    }


def all_words() -> List[str]:
    words = set()
    for items in support_families().values():
        words.update(items)
    for spec in relation_specs().values():
        for a, b in spec["pairs"]:
            words.add(a)
            words.add(b)
    return sorted(words)


def base_prompts(word: str) -> List[str]:
    return [
        f"This is {word}",
        f"That is {word}",
        f"The word is {word}",
    ]


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
    return out


def last_token_repr(out) -> Dict[int, np.ndarray]:
    return {
        li: out.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
        for li in range(len(out.hidden_states) - 1)
    }


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


def top_layers(vals: List[float], top_n: int) -> List[int]:
    return [int(i) for i in sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:top_n]]


def stage_breakdown(vals: List[float]) -> Dict[str, object]:
    n = len(vals)
    cuts = [0, n // 3, (2 * n) // 3, n]
    names = ["early", "mid", "late"]
    out = {}
    for idx, name in enumerate(names):
        lo, hi = cuts[idx], cuts[idx + 1]
        if hi <= lo:
            out[name] = {"layer": None, "value": 0.0}
            continue
        window = vals[lo:hi]
        best_local = int(np.argmax(window))
        out[name] = {"layer": int(lo + best_local), "value": float(window[best_local])}
    return out


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    words = all_words()
    global_target_len = max(prompt_len(tok, text) for word in words for text in base_prompts(word))

    base_repr: Dict[str, Dict[int, np.ndarray]] = {}
    base_topo: Dict[str, Dict[int, np.ndarray]] = {}
    for word in words:
        repr_rows = {li: [] for li in range(n_layers)}
        topo_rows = {li: [] for li in range(n_layers)}
        for text in base_prompts(word):
            out = run_model(model, tok, text)
            rr = last_token_repr(out)
            tt = last_token_topo(out, global_target_len)
            for li in range(n_layers):
                repr_rows[li].append(rr[li])
                topo_rows[li].append(tt[li])
        base_repr[word] = {li: mean_stack(rows) for li, rows in repr_rows.items()}
        base_topo[word] = {li: mean_stack(rows) for li, rows in topo_rows.items()}

    fam_repr_basis: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in support_families()}
    fam_topo_basis: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in support_families()}
    for family, fam_words in support_families().items():
        for li in range(n_layers):
            fam_repr_basis[family][li] = affine_basis([base_repr[word][li] for word in fam_words], rank_k=min(3, len(fam_words) - 1))
            fam_topo_basis[family][li] = affine_basis([base_topo[word][li] for word in fam_words], rank_k=min(3, len(fam_words) - 1))

    relations = {}
    for rel_name, spec in relation_specs().items():
        (a, b), (c, d) = spec["pairs"]
        endpoint_families: Dict[str, str] = spec["endpoint_families"]
        endpoint_repr_basis = []
        endpoint_topo_basis = []
        relation_align_repr = []
        relation_align_topo = []
        relation_error_repr = []
        relation_error_topo = []
        relation_cos_repr = []
        relation_cos_topo = []
        bridge_ht = []
        bridge_hh = []
        bridge_tt = []

        for li in range(n_layers):
            repr_basis_vals = []
            topo_basis_vals = []
            for word, family in endpoint_families.items():
                mu_r, basis_r = fam_repr_basis[family][li]
                mu_t, basis_t = fam_topo_basis[family][li]
                repr_basis_vals.append(1.0 - residual_ratio(base_repr[word][li], mu_r, basis_r))
                topo_basis_vals.append(1.0 - residual_ratio(base_topo[word][li], mu_t, basis_t))

            repr_vec1 = base_repr[b][li] - base_repr[a][li]
            repr_vec2 = base_repr[d][li] - base_repr[c][li]
            topo_vec1 = base_topo[b][li] - base_topo[a][li]
            topo_vec2 = base_topo[d][li] - base_topo[c][li]

            repr_err, repr_cos, repr_score = relation_score(repr_vec1, repr_vec2)
            topo_err, topo_cos, topo_score = relation_score(topo_vec1, topo_vec2)

            basis_repr_val = float(np.mean(repr_basis_vals))
            basis_topo_val = float(np.mean(topo_basis_vals))
            endpoint_repr_basis.append(basis_repr_val)
            endpoint_topo_basis.append(basis_topo_val)
            relation_align_repr.append(repr_score)
            relation_align_topo.append(topo_score)
            relation_error_repr.append(repr_err)
            relation_error_topo.append(topo_err)
            relation_cos_repr.append(repr_cos)
            relation_cos_topo.append(topo_cos)
            bridge_ht.append(float(basis_repr_val * topo_score))
            bridge_hh.append(float(basis_repr_val * repr_score))
            bridge_tt.append(float(basis_topo_val * topo_score))

        topo_dominant_layers = [int(i) for i in range(n_layers) if relation_align_topo[i] > relation_align_repr[i]]
        relations[rel_name] = {
            "pairs": spec["pairs"],
            "endpoint_families": endpoint_families,
            "endpoint_repr_basis_by_layer": endpoint_repr_basis,
            "endpoint_topo_basis_by_layer": endpoint_topo_basis,
            "relation_align_repr_by_layer": relation_align_repr,
            "relation_align_topo_by_layer": relation_align_topo,
            "relation_error_repr_by_layer": relation_error_repr,
            "relation_error_topo_by_layer": relation_error_topo,
            "relation_cos_repr_by_layer": relation_cos_repr,
            "relation_cos_topo_by_layer": relation_cos_topo,
            "bridge_ht_by_layer": bridge_ht,
            "bridge_hh_by_layer": bridge_hh,
            "bridge_tt_by_layer": bridge_tt,
            "summary": {
                "best_bridge_ht_layers": top_layers(bridge_ht, min(5, n_layers)),
                "best_bridge_hh_layers": top_layers(bridge_hh, min(5, n_layers)),
                "best_bridge_tt_layers": top_layers(bridge_tt, min(5, n_layers)),
                "best_relation_topo_layers": top_layers(relation_align_topo, min(5, n_layers)),
                "best_relation_repr_layers": top_layers(relation_align_repr, min(5, n_layers)),
                "topo_dominant_layers": topo_dominant_layers,
                "topo_dominant_ratio": float(len(topo_dominant_layers) / max(1, n_layers)),
                "max_bridge_ht": float(max(bridge_ht)),
                "max_bridge_hh": float(max(bridge_hh)),
                "max_bridge_tt": float(max(bridge_tt)),
                "max_relation_align_topo": float(max(relation_align_topo)),
                "max_relation_align_repr": float(max(relation_align_repr)),
                "peak_endpoint_repr_basis": float(max(endpoint_repr_basis)),
                "peak_endpoint_topo_basis": float(max(endpoint_topo_basis)),
                "bridge_advantage_topo_minus_repr": float(max(bridge_ht) - max(bridge_hh)),
                "bridge_stage_ht": stage_breakdown(bridge_ht),
                "bridge_stage_tt": stage_breakdown(bridge_tt),
            },
        }

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
            "global_target_len": global_target_len,
            "runtime_sec": float(time.time() - t0),
        },
        "support_families": support_families(),
        "relations": relations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace relation-family coupling paths in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_relation_coupling_trace_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(f"[summary] {model_name} relations={list(row['relations'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
