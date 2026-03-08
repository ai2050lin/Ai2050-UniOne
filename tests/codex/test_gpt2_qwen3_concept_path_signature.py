#!/usr/bin/env python
"""
Build layerwise concept path signatures for:
- apple (fruit)
- cat (animal)
- truth (abstract)

Each layer is summarized by five components:
- B_repr: closeness to family basis in representation space
- D_repr: offset concentration in representation space
- R_repr/G_repr: relation and gating sensitivity in representation space
- B_topo: closeness to family basis in topology space
- D_topo: offset concentration in topology space
- R_topo/G_topo: relation and gating sensitivity in topology space
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


def target_specs() -> List[Tuple[str, str]]:
    return [("apple", "fruit"), ("cat", "animal"), ("truth", "abstract")]


def base_prompts(word: str) -> List[str]:
    return [f"This is {word}", f"That is {word}", f"It is {word}"]


def relation_prompt(label: str, word: str) -> str:
    return f"kind {label} item {word}"


def gating_prompt(mode: str, word: str) -> str:
    return f"{mode} mode item {word}"


def false_family(true_family: str) -> str:
    return {"fruit": "animal", "animal": "fruit", "abstract": "fruit"}[true_family]


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def compatible_words(tok) -> Dict[str, List[str]]:
    out = {}
    for family, words in family_words().items():
        groups: Dict[int, List[str]] = {}
        for word in words:
            lengths = {prompt_len(tok, t) for t in base_prompts(word)}
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


def topk_energy_ratio(vec: np.ndarray, k: int, eps: float = 1e-12) -> float:
    score = np.square(vec.astype(np.float64))
    total = float(score.sum()) + eps
    kk = int(min(max(1, k), score.shape[0]))
    idx = np.argpartition(score, -kk)[-kk:]
    return float(score[idx].sum() / total)


def norm_delta(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) + np.linalg.norm(b)) / 2.0 + eps
    return float(np.linalg.norm(a - b) / denom)


def top_layers(vals: List[float], top_n: int, reverse: bool = True) -> List[int]:
    return [int(i) for i in sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)[:top_n]]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    selected = compatible_words(tok)
    n_layers = int(getattr(model.config, "num_hidden_layers"))

    family_repr_bases: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in selected}
    family_topo_bases: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in selected}

    base_repr_concepts: Dict[str, Dict[int, np.ndarray]] = {}
    base_topo_concepts: Dict[str, Dict[int, np.ndarray]] = {}

    for family, words in selected.items():
        for word in words:
            repr_rows = {li: [] for li in range(n_layers)}
            topo_rows = {li: [] for li in range(n_layers)}
            target_len = 0
            runs = []
            for text in base_prompts(word):
                out, seq_len = run_model(model, tok, text)
                runs.append((out, seq_len))
                target_len = max(target_len, seq_len)
            for out, _seq_len in runs:
                rr = last_token_repr(out)
                tt = last_token_topo(out, target_len)
                for li in range(n_layers):
                    repr_rows[li].append(rr[li])
                    topo_rows[li].append(tt[li])
            base_repr_concepts[word] = {li: mean_stack(vs) for li, vs in repr_rows.items()}
            base_topo_concepts[word] = {li: mean_stack(vs) for li, vs in topo_rows.items()}

        for li in range(n_layers):
            family_repr_bases[family][li] = affine_basis([base_repr_concepts[w][li] for w in words], rank_k=min(3, len(words) - 1))
            family_topo_bases[family][li] = affine_basis([base_topo_concepts[w][li] for w in words], rank_k=min(3, len(words) - 1))

    concepts = {}
    for word, family in target_specs():
        if word not in base_repr_concepts:
            continue
        neg_family = false_family(family)

        base_repr = base_repr_concepts[word]
        base_topo = base_topo_concepts[word]

        rel_runs = []
        gate_runs = []
        rel_target_len = 0
        gate_target_len = 0
        for text in [relation_prompt(family, word), relation_prompt(neg_family, word)]:
            out, seq_len = run_model(model, tok, text)
            rel_runs.append((out, seq_len))
            rel_target_len = max(rel_target_len, seq_len)
        for text in [gating_prompt("chat", word), gating_prompt("formal", word)]:
            out, seq_len = run_model(model, tok, text)
            gate_runs.append((out, seq_len))
            gate_target_len = max(gate_target_len, seq_len)

        rel_repr = [last_token_repr(out) for out, _ in rel_runs]
        rel_topo = [last_token_topo(out, rel_target_len) for out, _ in rel_runs]
        gate_repr = [last_token_repr(out) for out, _ in gate_runs]
        gate_topo = [last_token_topo(out, gate_target_len) for out, _ in gate_runs]

        b_repr = []
        d_repr = []
        r_repr = []
        g_repr = []
        b_topo = []
        d_topo = []
        r_topo = []
        g_topo = []
        for li in range(n_layers):
            mu_r, basis_r = family_repr_bases[family][li]
            mu_t, basis_t = family_topo_bases[family][li]

            proj_r = affine_project(base_repr[li], mu_r, basis_r)
            delta_r = (base_repr[li] - proj_r).astype(np.float32)
            proj_t = affine_project(base_topo[li], mu_t, basis_t)
            delta_t = (base_topo[li] - proj_t).astype(np.float32)

            b_repr.append(float(1.0 - residual_ratio(base_repr[li], mu_r, basis_r)))
            d_repr.append(float(topk_energy_ratio(delta_r, 32)))
            r_repr.append(float(norm_delta(rel_repr[0][li], rel_repr[1][li])))
            g_repr.append(float(norm_delta(gate_repr[0][li], gate_repr[1][li])))

            b_topo.append(float(1.0 - residual_ratio(base_topo[li], mu_t, basis_t)))
            d_topo.append(float(topk_energy_ratio(delta_t, 32)))
            r_topo.append(float(norm_delta(rel_topo[0][li], rel_topo[1][li])))
            g_topo.append(float(norm_delta(gate_topo[0][li], gate_topo[1][li])))

        concepts[word] = {
            "family": family,
            "B_repr_by_layer": b_repr,
            "D_repr_by_layer": d_repr,
            "R_repr_by_layer": r_repr,
            "G_repr_by_layer": g_repr,
            "B_topo_by_layer": b_topo,
            "D_topo_by_layer": d_topo,
            "R_topo_by_layer": r_topo,
            "G_topo_by_layer": g_topo,
            "signature_summary": {
                "repr_basis_layers": top_layers(b_repr, 5, reverse=True),
                "topo_basis_layers": top_layers(b_topo, 5, reverse=True),
                "repr_relation_layers": top_layers([r_repr[i] - g_repr[i] for i in range(n_layers)], 5, reverse=True),
                "repr_gating_layers": top_layers([g_repr[i] - r_repr[i] for i in range(n_layers)], 5, reverse=True),
                "topo_relation_layers": top_layers([r_topo[i] - g_topo[i] for i in range(n_layers)], 5, reverse=True),
                "topo_gating_layers": top_layers([g_topo[i] - r_topo[i] for i in range(n_layers)], 5, reverse=True),
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
        "selected_words": selected,
        "concepts": concepts,
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build concept path signatures in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_concept_path_signature_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(f"[summary] {model_name} concepts={list(row['concepts'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
