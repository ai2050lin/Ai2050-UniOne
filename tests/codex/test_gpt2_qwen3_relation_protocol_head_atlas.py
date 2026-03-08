#!/usr/bin/env python
"""
Build a head-level atlas for the relation protocol layer in GPT-2 and Qwen3.

Goal:
- For each relation family, find which attention heads carry the strongest
  topology-topology (TT) coupling signal.
- Measure whether these heads are shared across relation families or
  specialized to individual relation types.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
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
            "endpoint_families": {
                "king": "male",
                "man": "male",
                "queen": "female",
                "woman": "female",
            },
        },
        "hypernym": {
            "pairs": [("apple", "fruit"), ("cat", "animal")],
            "endpoint_families": {
                "apple": "instance",
                "cat": "instance",
                "fruit": "category",
                "animal": "category",
            },
        },
        "antonym": {
            "pairs": [("hot", "cold"), ("big", "small")],
            "endpoint_families": {
                "hot": "descriptor",
                "cold": "descriptor",
                "big": "descriptor",
                "small": "descriptor",
            },
        },
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


def top_rows(rows: Sequence[Dict[str, float]], k: int) -> List[Dict[str, float]]:
    picked = sorted(rows, key=lambda row: float(row["bridge_tt"]), reverse=True)[: max(1, int(k))]
    return [
        {
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "bridge_tt": float(row["bridge_tt"]),
            "endpoint_topo_basis": float(row["endpoint_topo_basis"]),
            "relation_align_topo": float(row["relation_align_topo"]),
        }
        for row in picked
    ]


def jaccard(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool, top_k: int) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    words = all_words()
    global_target_len = max(prompt_len(tok, text) for word in words for text in base_prompts(word))

    base_head_topo: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for word in words:
        rows: Dict[Tuple[int, int], List[np.ndarray]] = {(li, hi): [] for li in range(n_layers) for hi in range(n_heads)}
        for text in base_prompts(word):
            out = run_model(model, tok, text)
            head_topo = last_token_head_topo(out, global_target_len)
            for key, vec in head_topo.items():
                rows[key].append(vec)
        base_head_topo[word] = {key: mean_stack(vs) for key, vs in rows.items()}

    family_head_basis: Dict[str, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in support_families()}
    for family, fam_words in support_families().items():
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                family_head_basis[family][key] = affine_basis([base_head_topo[word][key] for word in fam_words], rank_k=min(3, len(fam_words) - 1))

    relations = {}
    head_counter = Counter()
    top_sets: Dict[str, List[Tuple[int, int]]] = {}
    for rel_name, spec in relation_specs().items():
        (a, b), (c, d) = spec["pairs"]
        endpoint_families: Dict[str, str] = spec["endpoint_families"]
        head_rows = []
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                basis_vals = []
                for word, family in endpoint_families.items():
                    mu_t, basis_t = family_head_basis[family][key]
                    basis_vals.append(1.0 - residual_ratio(base_head_topo[word][key], mu_t, basis_t))
                topo_vec1 = base_head_topo[b][key] - base_head_topo[a][key]
                topo_vec2 = base_head_topo[d][key] - base_head_topo[c][key]
                _err, _cos, score = relation_score(topo_vec1, topo_vec2)
                endpoint_basis = float(np.mean(basis_vals))
                bridge_tt = float(endpoint_basis * score)
                head_rows.append(
                    {
                        "layer": li,
                        "head": hi,
                        "endpoint_topo_basis": endpoint_basis,
                        "relation_align_topo": float(score),
                        "bridge_tt": bridge_tt,
                    }
                )

        top_heads = top_rows(head_rows, top_k)
        top_sets[rel_name] = [(row["layer"], row["head"]) for row in top_heads]
        head_counter.update(top_sets[rel_name])
        relations[rel_name] = {
            "pairs": spec["pairs"],
            "top_heads": top_heads,
            "summary": {
                "max_bridge_tt": float(max(row["bridge_tt"] for row in head_rows)),
                "mean_bridge_tt": float(np.mean([row["bridge_tt"] for row in head_rows])),
                "best_head": {"layer": int(top_heads[0]["layer"]), "head": int(top_heads[0]["head"])},
            },
        }

    overlap_matrix = {}
    rel_names = list(relation_specs().keys())
    for ra in rel_names:
        overlap_matrix[ra] = {}
        for rb in rel_names:
            overlap_matrix[ra][rb] = float(jaccard(top_sets[ra], top_sets[rb]))

    shared_heads = [
        {"layer": int(layer), "head": int(head), "frequency": int(freq)}
        for (layer, head), freq in head_counter.most_common(20)
    ]
    specialized_relation_count = sum(1 for _key, freq in head_counter.items() if freq == 1)
    reused_relation_count = sum(1 for _key, freq in head_counter.items() if freq >= 2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "global_target_len": global_target_len,
            "runtime_sec": float(time.time() - t0),
        },
        "relations": relations,
        "shared_heads": shared_heads,
        "top_head_overlap_jaccard": overlap_matrix,
        "global_summary": {
            "top_k": int(top_k),
            "unique_top_head_count": int(len(head_counter)),
            "specialized_relation_count": int(specialized_relation_count),
            "reused_relation_count": int(reused_relation_count),
            "most_shared_head": shared_heads[0] if shared_heads else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a head-level atlas for relation protocols in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_relation_protocol_head_atlas_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only, top_k=args.top_k)
        results["models"][model_name] = row
        print(f"[summary] {model_name} relations={list(row['relations'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
