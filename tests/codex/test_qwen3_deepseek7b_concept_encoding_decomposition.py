#!/usr/bin/env python
"""
Task block C:
Measure concept encoding decomposition on Qwen3-4B and DeepSeek-7B.

We approximate
    h_c^(l) = B_f^(l) + Delta_c^(l)

where:
- B_f is a shared family basis
- Delta_c is a sparse concept-specific offset

We also measure a simple relation-axis consistency:
    (king - man) ~= (queen - woman)
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
    return model, tok


def family_words() -> Dict[str, List[str]]:
    return {
        "fruit_family": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "male_role": ["man", "king", "father", "brother"],
        "female_role": ["woman", "queen", "mother", "sister"],
    }


def target_concepts() -> List[Tuple[str, str]]:
    return [
        ("apple", "fruit_family"),
        ("fruit", "fruit_family"),
        ("man", "male_role"),
        ("king", "male_role"),
        ("woman", "female_role"),
        ("queen", "female_role"),
    ]


def prompts(word: str) -> List[str]:
    return [f"This is {word}", f"That is {word}", f"It is {word}"]


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
    return out


def last_token_repr(out) -> Dict[int, np.ndarray]:
    return {
        li: out.hidden_states[li + 1][0, -1, :].detach().float().cpu().numpy().astype(np.float32)
        for li in range(len(out.hidden_states) - 1)
    }


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


def norm_gap(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (float(np.linalg.norm(a)) + float(np.linalg.norm(b))) / 2.0 + eps
    return float(np.linalg.norm(a - b) / denom)


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    concept_repr: Dict[str, Dict[int, np.ndarray]] = {}

    family_basis: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {name: {} for name in family_words()}
    for family, words in family_words().items():
        for word in words:
            rows = {li: [] for li in range(n_layers)}
            for text in prompts(word):
                out = run_model(model, tok, text)
                rr = last_token_repr(out)
                for li in range(n_layers):
                    rows[li].append(rr[li])
            concept_repr[word] = {li: mean_stack(vs) for li, vs in rows.items()}
        for li in range(n_layers):
            family_basis[family][li] = affine_basis([concept_repr[word][li] for word in words], rank_k=min(3, len(words) - 1))

    target_rows = {}
    true_residuals = []
    residual_margins = []
    offset_sparsities = []
    shared_energy_ratios = []
    for concept, true_family in target_concepts():
        rows = {li: [] for li in range(n_layers)}
        for text in prompts(concept):
            out = run_model(model, tok, text)
            rr = last_token_repr(out)
            for li in range(n_layers):
                rows[li].append(rr[li])
        concept_repr[concept] = {li: mean_stack(vs) for li, vs in rows.items()}

        per_layer = []
        for li in range(n_layers):
            x = concept_repr[concept][li]
            family_fits = {}
            for family in family_words():
                mu, basis = family_basis[family][li]
                proj = affine_project(x, mu, basis)
                delta = (x - proj).astype(np.float32)
                family_fits[family] = {
                    "residual_ratio": residual_ratio(x, mu, basis),
                    "offset_top32_energy_ratio": topk_energy_ratio(delta, 32),
                    "shared_norm_ratio": float(np.linalg.norm(proj) / (np.linalg.norm(x) + 1e-12)),
                }
            true_fit = family_fits[true_family]
            best_wrong = min(v["residual_ratio"] for fam, v in family_fits.items() if fam != true_family)
            per_layer.append(
                {
                    "layer": li,
                    "true_residual_ratio": float(true_fit["residual_ratio"]),
                    "margin_vs_best_wrong": float(best_wrong - true_fit["residual_ratio"]),
                    "offset_top32_energy_ratio": float(true_fit["offset_top32_energy_ratio"]),
                    "shared_norm_ratio": float(true_fit["shared_norm_ratio"]),
                    "best_wrong_residual_ratio": float(best_wrong),
                }
            )
        best_layer = min(per_layer, key=lambda row: row["true_residual_ratio"])
        true_residuals.append(float(best_layer["true_residual_ratio"]))
        residual_margins.append(float(best_layer["margin_vs_best_wrong"]))
        offset_sparsities.append(float(best_layer["offset_top32_energy_ratio"]))
        shared_energy_ratios.append(float(best_layer["shared_norm_ratio"]))
        target_rows[concept] = {
            "true_family": true_family,
            "best_layer": best_layer,
            "top_layers_by_basis_fit": sorted(per_layer, key=lambda row: row["true_residual_ratio"])[:6],
        }

    royalty_layer_rows = []
    for li in range(n_layers):
        if not all(word in concept_repr for word in ["king", "man", "queen", "woman"]):
            continue
        male_axis = concept_repr["king"][li] - concept_repr["man"][li]
        female_axis = concept_repr["queen"][li] - concept_repr["woman"][li]
        royalty_layer_rows.append(
            {
                "layer": li,
                "axis_gap": norm_gap(male_axis, female_axis),
                "axis_top64_energy": topk_energy_ratio((male_axis + female_axis) / 2.0, 64),
            }
        )
    royalty_best = min(royalty_layer_rows, key=lambda row: row["axis_gap"]) if royalty_layer_rows else {"layer": -1, "axis_gap": 1.0, "axis_top64_energy": 0.0}

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "n_layers": n_layers,
            "runtime_sec": float(time.time() - t0),
        },
        "targets": target_rows,
        "royalty_axis": {
            "best_layer": royalty_best,
            "top_layers": sorted(royalty_layer_rows, key=lambda row: row["axis_gap"])[:6],
        },
        "global_summary": {
            "mean_true_family_residual": float(np.mean(true_residuals)) if true_residuals else 0.0,
            "mean_margin_vs_best_wrong": float(np.mean(residual_margins)) if residual_margins else 0.0,
            "mean_offset_top32_energy_ratio": float(np.mean(offset_sparsities)) if offset_sparsities else 0.0,
            "mean_shared_norm_ratio": float(np.mean(shared_energy_ratios)) if shared_energy_ratios else 0.0,
            "royalty_axis_gap": float(royalty_best["axis_gap"]),
            "royalty_axis_top64_energy": float(royalty_best["axis_top64_energy"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Concept encoding decomposition for Qwen3 and DeepSeek7B")
    ap.add_argument("--dtype-qwen", type=str, default="bfloat16")
    ap.add_argument("--dtype-deepseek", type=str, default="bfloat16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_concept_encoding_decomposition_20260309.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_qwen if model_name == "qwen3_4b" else args.dtype_deepseek
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, not args.cpu_only)
        results["models"][model_name] = row
        print(
            f"[summary] {model_name} residual={row['global_summary']['mean_true_family_residual']:.4f} "
            f"margin={row['global_summary']['mean_margin_vs_best_wrong']:.4f} "
            f"royalty_gap={row['global_summary']['royalty_axis_gap']:.4f}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
