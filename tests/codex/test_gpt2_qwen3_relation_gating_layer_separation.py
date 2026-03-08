#!/usr/bin/env python
"""
Separate relation-sensitive and gating-sensitive responses layerwise in GPT-2 and Qwen3.

We compare two contrast types on the same concept token:
1) relation contrast: category label changes, concept fixed
2) gating contrast: mode/style label changes, concept fixed

For each layer, we measure changes in:
- representation space H_l
- topology space T_l (last-token attention rows)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def concept_families() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana"],
        "animal": ["cat", "dog"],
        "abstract": ["truth", "justice"],
    }


def false_family(true_family: str) -> str:
    return {
        "fruit": "animal",
        "animal": "fruit",
        "abstract": "fruit",
    }[true_family]


def relation_prompt(label: str, word: str) -> str:
    return f"kind {label} item {word}"


def gating_prompt(mode: str, word: str) -> str:
    return f"{mode} mode item {word}"


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
    return out, enc["input_ids"].shape[1]


def last_token_topology(out, target_len: int) -> Dict[int, np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)  # [heads, seq, seq]
        last_row = arr[:, -1, :]  # [heads, seq]
        pad = target_len - last_row.shape[1]
        if pad < 0:
            raise RuntimeError("target_len smaller than sequence length")
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        topo[li] = last_row.reshape(-1).astype(np.float32)
    return topo


def last_token_repr(out) -> Dict[int, np.ndarray]:
    return {
        li: out.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
        for li in range(len(out.hidden_states) - 1)
    }


def norm_delta(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) + np.linalg.norm(b)) / 2.0 + eps
    return float(np.linalg.norm(a - b) / denom)


def top_layers(vals: List[float], top_n: int) -> List[int]:
    return [int(i) for i in sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:top_n]]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))

    relation_repr = [[] for _ in range(n_layers)]
    relation_topo = [[] for _ in range(n_layers)]
    gating_repr = [[] for _ in range(n_layers)]
    gating_topo = [[] for _ in range(n_layers)]

    for family, words in concept_families().items():
        neg_family = false_family(family)
        for word in words:
            rel_texts = [relation_prompt(family, word), relation_prompt(neg_family, word)]
            gate_texts = [gating_prompt("chat", word), gating_prompt("formal", word)]
            all_texts = rel_texts + gate_texts

            outputs = []
            lengths = []
            for text in all_texts:
                out, seq_len = run_model(model, tok, text)
                outputs.append(out)
                lengths.append(seq_len)
            target_len = int(max(lengths))

            rel_repr_vecs = [last_token_repr(outputs[0]), last_token_repr(outputs[1])]
            gate_repr_vecs = [last_token_repr(outputs[2]), last_token_repr(outputs[3])]
            rel_topo_vecs = [last_token_topology(outputs[0], target_len), last_token_topology(outputs[1], target_len)]
            gate_topo_vecs = [last_token_topology(outputs[2], target_len), last_token_topology(outputs[3], target_len)]

            for li in range(n_layers):
                relation_repr[li].append(norm_delta(rel_repr_vecs[0][li], rel_repr_vecs[1][li]))
                relation_topo[li].append(norm_delta(rel_topo_vecs[0][li], rel_topo_vecs[1][li]))
                gating_repr[li].append(norm_delta(gate_repr_vecs[0][li], gate_repr_vecs[1][li]))
                gating_topo[li].append(norm_delta(gate_topo_vecs[0][li], gate_topo_vecs[1][li]))

    relation_repr_avg = [float(np.mean(v)) for v in relation_repr]
    relation_topo_avg = [float(np.mean(v)) for v in relation_topo]
    gating_repr_avg = [float(np.mean(v)) for v in gating_repr]
    gating_topo_avg = [float(np.mean(v)) for v in gating_topo]

    relation_minus_gating_repr = [float(relation_repr_avg[i] - gating_repr_avg[i]) for i in range(n_layers)]
    relation_minus_gating_topo = [float(relation_topo_avg[i] - gating_topo_avg[i]) for i in range(n_layers)]

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
        "relation_repr_avg_by_layer": relation_repr_avg,
        "relation_topo_avg_by_layer": relation_topo_avg,
        "gating_repr_avg_by_layer": gating_repr_avg,
        "gating_topo_avg_by_layer": gating_topo_avg,
        "relation_minus_gating_repr_by_layer": relation_minus_gating_repr,
        "relation_minus_gating_topo_by_layer": relation_minus_gating_topo,
        "layer_role_summary": {
            "repr_relation_layers": top_layers(relation_minus_gating_repr, top_n=min(5, n_layers)),
            "repr_gating_layers": top_layers([-x for x in relation_minus_gating_repr], top_n=min(5, n_layers)),
            "topo_relation_layers": top_layers(relation_minus_gating_topo, top_n=min(5, n_layers)),
            "topo_gating_layers": top_layers([-x for x in relation_minus_gating_topo], top_n=min(5, n_layers)),
        },
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Separate relation and gating responses layerwise")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_relation_gating_layer_separation_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(
            f"[summary] {model_name} repr_relation={row['layer_role_summary']['repr_relation_layers']} "
            f"repr_gating={row['layer_role_summary']['repr_gating_layers']} "
            f"topo_relation={row['layer_role_summary']['topo_relation_layers']} "
            f"topo_gating={row['layer_role_summary']['topo_gating_layers']}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
