#!/usr/bin/env python
"""
Extended relation-family analysis in GPT-2 and Qwen3.

Relation families:
- gender
- hypernym
- antonym
- synonym
- meronym
- cause_effect

For each relation family, we compare two aligned pairs and measure
layerwise path alignment in:
1) representation space
2) topology space
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


def relation_specs() -> Dict[str, List[Tuple[str, str]]]:
    return {
        "gender": [("king", "queen"), ("man", "woman")],
        "hypernym": [("apple", "fruit"), ("cat", "animal")],
        "antonym": [("hot", "cold"), ("big", "small")],
        "synonym": [("big", "large"), ("small", "tiny")],
        "meronym": [("wheel", "car"), ("leaf", "tree")],
        "cause_effect": [("fire", "smoke"), ("virus", "disease")],
    }


def all_words() -> List[str]:
    out = []
    for pairs in relation_specs().values():
        for a, b in pairs:
            out.extend([a, b])
    return sorted(set(out))


def prompts_for(word: str) -> List[str]:
    return [
        f"This is {word}",
        f"That is {word}",
        f"The word is {word}",
    ]


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
    return out, int(enc["input_ids"].shape[1])


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


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def relation_error(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + eps
    return float(np.linalg.norm(v1 - v2) / denom)


def best_layers(vals: List[float], top_n: int, reverse: bool = False) -> List[int]:
    return [int(i) for i in sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)[:top_n]]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))

    runs_by_word = {}
    global_target_len = 0
    for word in all_words():
        rows = []
        for text in prompts_for(word):
            out, seq_len = run_model(model, tok, text)
            rows.append((out, seq_len))
            global_target_len = max(global_target_len, seq_len)
        runs_by_word[word] = rows

    repr_concepts: Dict[str, Dict[int, np.ndarray]] = {}
    topo_concepts: Dict[str, Dict[int, np.ndarray]] = {}
    for word, runs in runs_by_word.items():
        repr_rows = {li: [] for li in range(n_layers)}
        topo_rows = {li: [] for li in range(n_layers)}
        for out, _seq_len in runs:
            rr = last_token_repr(out)
            tt = last_token_topo(out, global_target_len)
            for li in range(n_layers):
                repr_rows[li].append(rr[li])
                topo_rows[li].append(tt[li])
        repr_concepts[word] = {li: mean_stack(vs) for li, vs in repr_rows.items()}
        topo_concepts[word] = {li: mean_stack(vs) for li, vs in topo_rows.items()}

    relation_rows = {}
    for rel_name, pairs in relation_specs().items():
        (a, b), (c, d) = pairs
        repr_error = []
        topo_error = []
        repr_cos = []
        topo_cos = []
        for li in range(n_layers):
            repr_vec1 = repr_concepts[b][li] - repr_concepts[a][li]
            repr_vec2 = repr_concepts[d][li] - repr_concepts[c][li]
            topo_vec1 = topo_concepts[b][li] - topo_concepts[a][li]
            topo_vec2 = topo_concepts[d][li] - topo_concepts[c][li]
            repr_error.append(relation_error(repr_vec1, repr_vec2))
            topo_error.append(relation_error(topo_vec1, topo_vec2))
            repr_cos.append(cosine(repr_vec1, repr_vec2))
            topo_cos.append(cosine(topo_vec1, topo_vec2))

        relation_rows[rel_name] = {
            "repr_error_by_layer": repr_error,
            "topo_error_by_layer": topo_error,
            "repr_cosine_by_layer": repr_cos,
            "topo_cosine_by_layer": topo_cos,
            "summary": {
                "best_repr_layers": best_layers(repr_error, min(5, n_layers), reverse=False),
                "best_topo_layers": best_layers(topo_error, min(5, n_layers), reverse=False),
                "repr_better_layers": [int(i) for i in range(n_layers) if repr_error[i] < topo_error[i]],
                "topo_better_layers": [int(i) for i in range(n_layers) if topo_error[i] < repr_error[i]],
                "min_repr_error": float(min(repr_error)),
                "min_topo_error": float(min(topo_error)),
                "max_repr_cosine": float(max(repr_cos)),
                "max_topo_cosine": float(max(topo_cos)),
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
            "runtime_sec": float(time.time() - t0),
        },
        "global_target_len": int(global_target_len),
        "relations": relation_rows,
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extended relation-family analysis in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_extended_relation_families_20260308.json",
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
