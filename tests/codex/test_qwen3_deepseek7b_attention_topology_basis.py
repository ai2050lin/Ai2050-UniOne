#!/usr/bin/env python
"""
Analyze shared bases in attention-defined topology for:
- Qwen3-4B
- DeepSeek-7B

This is the direct T-side measurement that was previously missing for
DeepSeek-7B on this machine.
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


def family_candidates() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "abstract": ["justice", "truth", "logic", "language", "freedom", "memory"],
    }


def template_texts(word: str) -> List[str]:
    return [
        f"This is {word}",
        f"That is {word}",
        f"It is {word}",
    ]


def token_count(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return int(len(ids))


def compatible_words(tok) -> Dict[str, List[str]]:
    out = {}
    for family, words in family_candidates().items():
        groups: Dict[int, List[str]] = {}
        for word in words:
            lengths = {token_count(tok, t) for t in template_texts(word)}
            if len(lengths) == 1:
                seq_len = next(iter(lengths))
                groups.setdefault(seq_len, []).append(word)
        if not groups:
            out[family] = []
            continue
        best_len = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
        out[family] = groups[best_len]
    return out


def run_attn(model, tok, text: str) -> Tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_attentions=True, return_dict=True)
    attns = []
    last_row_entropy = []
    for layer_attn in out.attentions:
        arr = layer_attn[0].detach().float().cpu().numpy().astype(np.float32)
        attns.append(arr.reshape(-1))
        last_row = arr[:, -1, :]
        ent = -(last_row * np.log(last_row + 1e-12)).sum(axis=1)
        last_row_entropy.append(float(ent.mean()))
    flat = np.concatenate(attns, axis=0).astype(np.float32)
    return flat, float(np.mean(last_row_entropy))


def mean_vector(rows: Sequence[np.ndarray]) -> np.ndarray:
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
    res = x - proj
    denom = float(np.linalg.norm(x - mu)) + eps
    return float(np.linalg.norm(res) / denom)


def topk_energy_ratio(vec: np.ndarray, k: int, eps: float = 1e-12) -> float:
    score = np.square(vec.astype(np.float64))
    total = float(score.sum()) + eps
    kk = int(min(max(1, k), score.shape[0]))
    idx = np.argpartition(score, -kk)[-kk:]
    return float(score[idx].sum() / total)


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    selected = compatible_words(tok)

    concept_topology = {}
    concept_entropy = {}
    for family, words in selected.items():
        for word in words:
            rows = []
            ents = []
            for text in template_texts(word):
                flat, ent = run_attn(model, tok, text)
                rows.append(flat)
                ents.append(ent)
            concept_topology[word] = mean_vector(rows)
            concept_entropy[word] = float(np.mean(ents))

    family_basis = {}
    family_summary = {}
    for family, words in selected.items():
        mu, basis = affine_basis([concept_topology[w] for w in words], rank_k=min(3, max(1, len(words) - 1)))
        family_basis[family] = {"mu": mu, "basis": basis}
        vals = [residual_ratio(concept_topology[w], mu, basis) for w in words]
        family_summary[family] = {
            "n_words": int(len(words)),
            "mean_topology_residual_ratio": float(np.mean(vals)),
            "max_topology_residual_ratio": float(np.max(vals)),
            "mean_last_token_entropy": float(np.mean([concept_entropy[w] for w in words])),
        }

    probe_words = [w for w in ("apple", "cat", "truth") if w in concept_topology]
    probes = {}
    for word in probe_words:
        word_family = next(f for f, ws in selected.items() if word in ws)
        x = concept_topology[word]
        fit = {}
        for family in selected.keys():
            mu = family_basis[family]["mu"]
            basis = family_basis[family]["basis"]
            proj = affine_project(x, mu, basis)
            delta = (x - proj).astype(np.float32)
            fit[family] = {
                "residual_ratio": residual_ratio(x, mu, basis),
                "delta_top32_energy_ratio": topk_energy_ratio(delta, 32),
                "delta_top128_energy_ratio": topk_energy_ratio(delta, 128),
            }
        probes[word] = {
            "family": word_family,
            "fit": fit,
            "supports_family_topology_basis": bool(
                fit[word_family]["residual_ratio"]
                < min(v["residual_ratio"] for fam, v in fit.items() if fam != word_family)
            ),
        }

    device = str(next(model.parameters()).device)
    hidden_size = int(getattr(model.config, "hidden_size"))
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": device,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "runtime_sec": float(time.time() - t0),
        },
        "selected_words": selected,
        "family_summary": family_summary,
        "probe_fits": probes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze shared bases in attention-defined topology for Qwen3 and DeepSeek7B")
    ap.add_argument("--dtype-qwen", type=str, default="bfloat16")
    ap.add_argument("--dtype-deepseek", type=str, default="bfloat16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_attention_topology_basis_20260309.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_qwen if model_name == "qwen3_4b" else args.dtype_deepseek
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(
            model_name=model_name,
            model_path=model_path,
            dtype_name=dtype_name,
            prefer_cuda=not args.cpu_only,
        )
        results["models"][model_name] = row
        fam = row["family_summary"]
        print(
            f"[summary] {model_name} "
            f"fruit={fam['fruit']['mean_topology_residual_ratio']:.4f} "
            f"animal={fam['animal']['mean_topology_residual_ratio']:.4f} "
            f"abstract={fam['abstract']['mean_topology_residual_ratio']:.4f}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
