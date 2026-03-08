#!/usr/bin/env python
"""
Test layerwise analogy structure for king/queen/man/woman in:
1) representation space H_l
2) topology space T_l

We check whether the displacement field is approximately reusable:
    H_king - H_man ≈ H_queen - H_woman
and likewise in topology space.
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


def words() -> List[str]:
    return ["king", "queen", "man", "woman"]


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


def analogy_error(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> float:
    residual = (a - b) - (c - d)
    denom = (np.linalg.norm(a - b) + np.linalg.norm(c - d)) / 2.0 + eps
    return float(np.linalg.norm(residual) / denom)


def candidate_ranking(pred: np.ndarray, pool: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    return sorted(((k, cosine(pred, v)) for k, v in pool.items()), key=lambda kv: kv[1], reverse=True)


def best_layers(vals: List[float], top_n: int, reverse: bool = False) -> List[int]:
    return [int(i) for i in sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)[:top_n]]


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))

    repr_concepts: Dict[str, Dict[int, np.ndarray]] = {}
    topo_concepts: Dict[str, Dict[int, np.ndarray]] = {}
    for word in words():
        runs = []
        target_len = 0
        for text in prompts_for(word):
            out, seq_len = run_model(model, tok, text)
            runs.append((out, seq_len))
            target_len = max(target_len, seq_len)
        repr_rows = {li: [] for li in range(n_layers)}
        topo_rows = {li: [] for li in range(n_layers)}
        for out, _seq_len in runs:
            rr = last_token_repr(out)
            tt = last_token_topo(out, target_len)
            for li in range(n_layers):
                repr_rows[li].append(rr[li])
                topo_rows[li].append(tt[li])
        repr_concepts[word] = {li: mean_stack(vs) for li, vs in repr_rows.items()}
        topo_concepts[word] = {li: mean_stack(vs) for li, vs in topo_rows.items()}

    repr_error = []
    topo_error = []
    repr_queen_rank = []
    topo_queen_rank = []
    repr_queen_score = []
    topo_queen_score = []
    for li in range(n_layers):
        hk = repr_concepts["king"][li]
        hq = repr_concepts["queen"][li]
        hm = repr_concepts["man"][li]
        hw = repr_concepts["woman"][li]
        tk = topo_concepts["king"][li]
        tq = topo_concepts["queen"][li]
        tm = topo_concepts["man"][li]
        tw = topo_concepts["woman"][li]

        repr_error.append(analogy_error(hk, hm, hq, hw))
        topo_error.append(analogy_error(tk, tm, tq, tw))

        repr_pred = hk - hm + hw
        topo_pred = tk - tm + tw
        repr_rank = candidate_ranking(repr_pred, {w: repr_concepts[w][li] for w in words()})
        topo_rank = candidate_ranking(topo_pred, {w: topo_concepts[w][li] for w in words()})

        repr_queen_rank.append(int([name for name, _score in repr_rank].index("queen") + 1))
        topo_queen_rank.append(int([name for name, _score in topo_rank].index("queen") + 1))
        repr_queen_score.append(float(dict(repr_rank)["queen"]))
        topo_queen_score.append(float(dict(topo_rank)["queen"]))

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
        "repr_analogy_error_by_layer": repr_error,
        "topo_analogy_error_by_layer": topo_error,
        "repr_queen_rank_by_layer": repr_queen_rank,
        "topo_queen_rank_by_layer": topo_queen_rank,
        "repr_queen_score_by_layer": repr_queen_score,
        "topo_queen_score_by_layer": topo_queen_score,
        "summary": {
            "best_repr_analogy_layers": best_layers(repr_error, min(5, n_layers), reverse=False),
            "best_topo_analogy_layers": best_layers(topo_error, min(5, n_layers), reverse=False),
            "repr_rank1_layers": [int(i) for i, r in enumerate(repr_queen_rank) if r == 1],
            "topo_rank1_layers": [int(i) for i, r in enumerate(topo_queen_rank) if r == 1],
        },
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Test analogy path structure in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_analogy_path_structure_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(
            f"[summary] {model_name} repr_best={row['summary']['best_repr_analogy_layers']} "
            f"topo_best={row['summary']['best_topo_analogy_layers']}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
