#!/usr/bin/env python
"""
Compare a single residual dictionary vs a clustered residual atlas for concept offsets.

The goal is to test whether concept-specific offsets are better explained by
a gated multi-component atlas than by one global family dictionary.
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
from sklearn.cluster import KMeans


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
    if want_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=dtype,
            device_map="cpu",
        )
    model.eval()
    return model, tok


class HiddenCollector:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.buffers: List[np.ndarray | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(li)))

    def _hook(self, li: int):
        def fn(_module, _inputs, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            self.buffers[li] = x[0, -1, :].detach().float().cpu().numpy().astype(np.float32)

        return fn

    def reset(self) -> None:
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hook outputs for layers: {miss}")
        return np.concatenate([x for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def concrete_prompts(noun: str) -> List[str]:
    return [
        f"This is {noun}",
        f"I saw {article(noun)} {noun}",
        f"The word is {noun}",
    ]


def abstract_prompts(noun: str) -> List[str]:
    return [
        f"The concept is {noun}",
        f"{noun} is an abstract idea",
        f"The word is {noun}",
    ]


def family_map() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "abstract": ["justice", "truth", "logic", "language", "freedom", "memory"],
    }


def target_specs() -> List[Tuple[str, str]]:
    return [
        ("apple", "fruit"),
        ("banana", "fruit"),
        ("cat", "animal"),
        ("dog", "animal"),
        ("justice", "abstract"),
        ("truth", "abstract"),
    ]


def prompts_for_word(word: str, family: str) -> List[str]:
    return abstract_prompts(word) if family == "abstract" else concrete_prompts(word)


def run_prompt(model, tok, text: str) -> None:
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        _ = model(**enc, use_cache=False, return_dict=True)


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


def residual_delta(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - affine_project(x, mu, basis)).astype(np.float32)


def orthonormal_dictionary(xs: Sequence[np.ndarray], max_atoms: int) -> np.ndarray:
    mat = np.stack(xs, axis=0).astype(np.float32)
    _u, _s, vh = np.linalg.svd(mat, full_matrices=False)
    k = int(min(max_atoms, vh.shape[0]))
    return vh[:k].T.astype(np.float32)


def dict_capture(delta: np.ndarray, dictionary: np.ndarray, top_k: int) -> float:
    coeff = dictionary.T @ delta.astype(np.float32)
    score = np.square(coeff.astype(np.float64))
    kk = int(min(max(1, top_k), score.shape[0]))
    idx = np.argpartition(score, -kk)[-kk:]
    return float(score[idx].sum() / (float(np.square(delta.astype(np.float64)).sum()) + 1e-12))


def pca_coords(xs: Sequence[np.ndarray], out_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.stack(xs, axis=0).astype(np.float32)
    mu = np.mean(mat, axis=0).astype(np.float32)
    centered = mat - mu[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    k = int(min(out_dim, vh.shape[0]))
    basis = vh[:k].T.astype(np.float32) if k > 0 else np.zeros((mat.shape[1], 0), dtype=np.float32)
    coords = centered @ basis if basis.shape[1] > 0 else np.zeros((mat.shape[0], 0), dtype=np.float32)
    return mu, basis, coords.astype(np.float32)


def collect_prompt_vectors(model, tok, collector: HiddenCollector) -> Dict[str, List[np.ndarray]]:
    data: Dict[str, List[np.ndarray]] = {}
    for family, words in family_map().items():
        for word in words:
            rows = []
            for text in prompts_for_word(word, family):
                collector.reset()
                run_prompt(model, tok, text)
                rows.append(collector.get_flat())
            data[word] = rows
    return data


def family_rank(family: str) -> int:
    return 3 if family == "abstract" else 4


def build_leave_one_family_basis(
    concept_means: Dict[str, np.ndarray],
    family: str,
    holdout_word: str,
) -> Tuple[np.ndarray, np.ndarray]:
    words = [w for w in family_map()[family] if w != holdout_word]
    return affine_basis([concept_means[w] for w in words], family_rank(family))


def build_residual_samples(
    prompt_vectors: Dict[str, List[np.ndarray]],
    concept_means: Dict[str, np.ndarray],
    family: str,
    holdout_word: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    mu, basis = build_leave_one_family_basis(concept_means, family, holdout_word)
    residuals = []
    for word in family_map()[family]:
        if word == holdout_word:
            continue
        for vec in prompt_vectors[word]:
            residuals.append(residual_delta(vec, mu, basis))
    return mu, basis, residuals


def build_atlas(residuals: Sequence[np.ndarray], n_clusters: int, atoms_per_cluster: int) -> Dict[str, object]:
    proj_mu, proj_basis, coords = pca_coords(residuals, out_dim=6)
    clusters = int(min(n_clusters, max(1, len(residuals))))
    if clusters == 1:
        labels = np.zeros(len(residuals), dtype=np.int64)
        centers = np.zeros((1, coords.shape[1]), dtype=np.float32)
    else:
        km = KMeans(n_clusters=clusters, random_state=0, n_init=10)
        labels = km.fit_predict(coords)
        centers = km.cluster_centers_.astype(np.float32)

    dictionaries = []
    for ci in range(clusters):
        xs = [residuals[i] for i in range(len(residuals)) if int(labels[i]) == ci]
        dictionaries.append(orthonormal_dictionary(xs, max_atoms=atoms_per_cluster))
    return {
        "proj_mu": proj_mu,
        "proj_basis": proj_basis,
        "centers": centers,
        "dictionaries": dictionaries,
    }


def atlas_capture(delta: np.ndarray, atlas: Dict[str, object], top_k: int) -> Dict[str, float]:
    dictionaries = atlas["dictionaries"]
    proj_mu = atlas["proj_mu"]
    proj_basis = atlas["proj_basis"]
    centers = atlas["centers"]

    all_caps = [dict_capture(delta, d, top_k) for d in dictionaries]
    oracle = float(max(all_caps))

    if proj_basis.shape[1] == 0:
        gated_index = 0
    else:
        coord = ((delta - proj_mu) @ proj_basis).astype(np.float32)
        dist = np.square(centers - coord[None, :]).sum(axis=1)
        gated_index = int(np.argmin(dist))
    gated = float(all_caps[gated_index])

    return {
        "oracle_top4_capture": oracle,
        "gated_top4_capture": gated,
        "gate_oracle_gap": float(oracle - gated),
    }


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    atoms_per_cluster: int,
    n_clusters: int,
) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    collector = HiddenCollector(model)
    try:
        prompt_vectors = collect_prompt_vectors(model, tok, collector)
    finally:
        collector.close()

    concept_means = {word: mean_vector(rows) for word, rows in prompt_vectors.items()}
    target_rows = []

    for word, family in target_specs():
        mu, basis, residuals = build_residual_samples(prompt_vectors, concept_means, family, word)
        delta = residual_delta(concept_means[word], mu, basis)

        global_dict = orthonormal_dictionary(residuals, max_atoms=atoms_per_cluster)
        global_top4 = dict_capture(delta, global_dict, 4)

        atlas = build_atlas(residuals, n_clusters=n_clusters, atoms_per_cluster=atoms_per_cluster)
        atlas_scores = atlas_capture(delta, atlas, top_k=4)

        target_rows.append(
            {
                "word": word,
                "family": family,
                "delta_norm": float(np.linalg.norm(delta)),
                "global_top4_capture": global_top4,
                "atlas_gated_top4_capture": atlas_scores["gated_top4_capture"],
                "atlas_oracle_top4_capture": atlas_scores["oracle_top4_capture"],
                "gated_gain_over_global": float(atlas_scores["gated_top4_capture"] - global_top4),
                "oracle_gain_over_global": float(atlas_scores["oracle_top4_capture"] - global_top4),
                "gate_oracle_gap": atlas_scores["gate_oracle_gap"],
                "supports_atlas": bool(atlas_scores["gated_top4_capture"] > global_top4 + 0.002),
            }
        )

    family_summary = {}
    for family in family_map().keys():
        fam_rows = [r for r in target_rows if r["family"] == family]
        family_summary[family] = {
            "mean_global_top4_capture": float(np.mean([r["global_top4_capture"] for r in fam_rows])),
            "mean_atlas_gated_top4_capture": float(np.mean([r["atlas_gated_top4_capture"] for r in fam_rows])),
            "mean_atlas_oracle_top4_capture": float(np.mean([r["atlas_oracle_top4_capture"] for r in fam_rows])),
            "mean_gated_gain_over_global": float(np.mean([r["gated_gain_over_global"] for r in fam_rows])),
            "mean_oracle_gain_over_global": float(np.mean([r["oracle_gain_over_global"] for r in fam_rows])),
            "mean_gate_oracle_gap": float(np.mean([r["gate_oracle_gap"] for r in fam_rows])),
            "support_rate": float(np.mean([1.0 if r["supports_atlas"] else 0.0 for r in fam_rows])),
        }

    meta = {
        "model_name": model_name,
        "model_path": model_path,
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "n_layers": int(getattr(model.config, "num_hidden_layers")),
        "n_heads": int(getattr(model.config, "num_attention_heads")),
        "n_kv_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0),
        "architecture": type(model.config).__name__,
        "device": str(next(model.parameters()).device),
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "runtime_sec": float(time.time() - t0),
    }
    return {
        "meta": meta,
        "targets": target_rows,
        "family_summary": family_summary,
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare single residual dictionary vs offset atlas")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--atoms-per-cluster", type=int, default=4)
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_offset_atlas_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        results["models"][model_name] = analyze_model(
            model_name=model_name,
            model_path=model_path,
            dtype_name=dtype_name,
            prefer_cuda=not args.cpu_only,
            atoms_per_cluster=args.atoms_per_cluster,
            n_clusters=args.n_clusters,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for model_name, row in results["models"].items():
        fam = row["family_summary"]
        print(
            f"[summary] {model_name} fruit_gain={fam['fruit']['mean_gated_gain_over_global']:.4f} "
            f"animal_gain={fam['animal']['mean_gated_gain_over_global']:.4f} "
            f"abstract_gain={fam['abstract']['mean_gated_gain_over_global']:.4f}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
