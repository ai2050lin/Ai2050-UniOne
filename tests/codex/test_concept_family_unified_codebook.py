#!/usr/bin/env python
"""
Unified concept-family codebook probe.

Goal:
- Put fruit / animal / mixed-control concepts into one real-model coordinate system.
- Extract:
  1) family shared basis dims
  2) concept-specific offset dims
  3) family separation margin and subspace margin

This is a structural probe on real hidden activations, not a causal edit test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers")


def get_probe_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        return layer.mlp.gate_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
        return layer.mlp.up_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc1"):
        return layer.mlp.fc1
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "c_fc"):
        return layer.mlp.c_fc
    raise RuntimeError("Cannot find a probe-able MLP linear module")


def load_model(model_id: str, dtype_name: str, local_only: bool):
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_only,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


class GateCollector:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.buffers: List[np.ndarray | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(get_probe_module(layer).register_forward_hook(self._hook(li)))

    def _hook(self, li: int):
        def fn(_module, _inputs, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            if x.ndim != 3:
                raise RuntimeError(f"Unexpected tensor rank at layer {li}: {tuple(x.shape)}")
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


def run_prompt(model, tok, text: str) -> None:
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        _ = model(**enc, use_cache=False, return_dict=True)


def prompt_set(concept: str) -> List[str]:
    return [
        f"This is {concept}",
        f"I saw {concept}",
        f"The word is {concept}",
    ]


def topk_abs_indices(vec: np.ndarray, k: int) -> np.ndarray:
    kk = min(max(int(k), 1), int(vec.shape[0]))
    score = np.abs(vec)
    idx = np.argpartition(score, -kk)[-kk:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(np.int64)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def index_to_layer(idx: int, d_ff: int) -> int:
    return int(idx) // int(d_ff)


def layer_hist(indices: Sequence[int], d_ff: int) -> Dict[str, int]:
    ctr = Counter(index_to_layer(int(i), d_ff) for i in indices)
    return {str(k): int(v) for k, v in sorted(ctr.items())}


def concept_groups() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "control": ["sun", "car", "justice", "computer", "river", "chair"],
    }


def mean_vector(vs: Sequence[np.ndarray]) -> np.ndarray:
    if not vs:
        raise RuntimeError("Empty vector list")
    return np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32)


def projection_energy(v: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    nv = float(np.linalg.norm(v))
    if nv < eps:
        return 0.0
    proj = basis @ v
    return float(np.dot(proj, proj) / (nv * nv))


def build_family_stats(
    family_name: str,
    members: List[str],
    centered: Dict[str, np.ndarray],
    top_k: int,
    shared_support_ratio: float,
    subspace_rank: int,
    d_ff: int,
) -> Dict[str, object]:
    member_vecs = [centered[x] for x in members]
    proto = mean_vector(member_vecs)
    proto_top = topk_abs_indices(proto, top_k).tolist()

    concept_top: Dict[str, List[int]] = {}
    freq = Counter()
    mean_abs = np.zeros_like(proto, dtype=np.float64)
    for name in members:
        vec = centered[name]
        mean_abs += np.abs(vec.astype(np.float64))
        top_idx = topk_abs_indices(vec, top_k).tolist()
        concept_top[name] = top_idx
        for i in top_idx:
            freq[int(i)] += 1
    mean_abs = mean_abs / max(1, len(members))

    min_support = max(2, int(math.ceil(len(members) * float(shared_support_ratio))))
    robust_dims = [idx for idx, cnt in freq.items() if cnt >= min_support]
    if not robust_dims:
        # Fallback: distributed families may not share exact top dims at very high support.
        robust_dims = [idx for idx, cnt in freq.items() if cnt >= 2]
    robust_dims = sorted(
        robust_dims,
        key=lambda i: (freq[int(i)], float(mean_abs[int(i)])),
        reverse=True,
    )[:top_k]

    shared_set = set(robust_dims)
    concept_specific = {}
    for name in members:
        specific = [i for i in concept_top[name] if i not in shared_set][:top_k]
        concept_specific[name] = {
            "top_specific_dims": [int(i) for i in specific[:16]],
            "layer_distribution": layer_hist(specific[:64], d_ff),
        }

    x = np.stack(member_vecs, axis=0).astype(np.float64)
    x0 = x - np.mean(x, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    rank = min(max(1, int(subspace_rank)), int(vh.shape[0]))
    basis = vh[:rank, :]

    same_energy = [projection_energy(centered[name].astype(np.float64), basis) for name in members]
    other_energy = []
    within_cos = []
    cross_cos = []
    for other_name, other_vec in centered.items():
        if other_name not in members:
            other_energy.append(projection_energy(other_vec.astype(np.float64), basis))
            cross_cos.append(cosine(proto, other_vec))
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            within_cos.append(cosine(centered[members[i]], centered[members[j]]))

    return {
        "family": family_name,
        "members": members,
        "prototype_top_dims": [int(i) for i in proto_top[:16]],
        "prototype_layer_distribution": layer_hist(proto_top[:64], d_ff),
        "robust_shared_dims": [int(i) for i in robust_dims[:24]],
        "robust_shared_layer_distribution": layer_hist(robust_dims[:64], d_ff),
        "shared_support_min": int(min_support),
        "within_family_cosine_mean": float(np.mean(within_cos) if within_cos else 0.0),
        "cross_family_cosine_mean": float(np.mean(cross_cos) if cross_cos else 0.0),
        "family_cosine_margin": float(
            (np.mean(within_cos) if within_cos else 0.0) - (np.mean(cross_cos) if cross_cos else 0.0)
        ),
        "subspace_rank": int(rank),
        "subspace_same_energy_mean": float(np.mean(same_energy) if same_energy else 0.0),
        "subspace_other_energy_mean": float(np.mean(other_energy) if other_energy else 0.0),
        "subspace_margin": float(
            (np.mean(same_energy) if same_energy else 0.0) - (np.mean(other_energy) if other_energy else 0.0)
        ),
        "concept_specific": concept_specific,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified concept-family codebook probe")
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--shared-support-ratio", type=float, default=0.50)
    ap.add_argument("--subspace-rank", type=int, default=4)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/concept_family_unified_codebook_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    groups = concept_groups()
    concept_order = [x for names in groups.values() for x in names]

    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        concept_vectors: Dict[str, np.ndarray] = {}
        for concept in concept_order:
            rows = []
            for prompt in prompt_set(concept):
                collector.reset()
                run_prompt(model, tok, prompt)
                rows.append(collector.get_flat())
            concept_vectors[concept] = mean_vector(rows)
    finally:
        collector.close()

    any_vec = next(iter(concept_vectors.values()))
    n_total_dims = int(any_vec.shape[0])
    n_layers = len(discover_layers(model))
    d_ff = int(n_total_dims // max(1, n_layers))

    global_mean = mean_vector(list(concept_vectors.values()))
    centered = {k: (v - global_mean).astype(np.float32) for k, v in concept_vectors.items()}

    family_stats = {}
    for family, members in groups.items():
        family_stats[family] = build_family_stats(
            family_name=family,
            members=members,
            centered=centered,
            top_k=args.top_k,
            shared_support_ratio=args.shared_support_ratio,
            subspace_rank=args.subspace_rank,
            d_ff=d_ff,
        )

    pairwise_families = {}
    fam_names = list(groups.keys())
    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            a = family_stats[fam_names[i]]
            b = family_stats[fam_names[j]]
            key = f"{fam_names[i]}__{fam_names[j]}"
            pairwise_families[key] = {
                "shared_dim_jaccard": float(jaccard(a["robust_shared_dims"], b["robust_shared_dims"])),
                "prototype_dim_jaccard": float(jaccard(a["prototype_top_dims"], b["prototype_top_dims"])),
                "family_margin_gap": float(float(a["family_cosine_margin"]) - float(b["family_cosine_margin"])),
            }

    spotlight = {}
    for concept in ("apple", "banana", "cat", "dog"):
        fam = "fruit" if concept in groups["fruit"] else "animal"
        top_dims = topk_abs_indices(centered[concept], args.top_k).tolist()
        shared_set = set(family_stats[fam]["robust_shared_dims"])
        spotlight[concept] = {
            "family": fam,
            "top_dims": [int(i) for i in top_dims[:16]],
            "layer_distribution": layer_hist(top_dims[:64], d_ff),
            "shared_overlap_ratio": float(
                len(set(top_dims[:64]) & shared_set) / max(1, len(set(top_dims[:64]) | shared_set))
            ),
            "specific_dims": [int(i) for i in top_dims if i not in shared_set][:16],
        }

    h = {
        "H1_fruit_shared_basis_exists": bool(len(family_stats["fruit"]["robust_shared_dims"]) >= 4),
        "H2_animal_shared_basis_exists": bool(len(family_stats["animal"]["robust_shared_dims"]) >= 4),
        "H3_family_subspace_margin_positive": bool(
            float(family_stats["fruit"]["subspace_margin"]) > 0.0
            and float(family_stats["animal"]["subspace_margin"]) > 0.0
        ),
        "H4_fruit_vs_animal_separable": bool(
            float(pairwise_families["fruit__animal"]["shared_dim_jaccard"]) < 0.20
        ),
    }

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 3),
            "model_id": args.model_id,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "n_total_dims": n_total_dims,
            "top_k": int(args.top_k),
            "shared_support_ratio": float(args.shared_support_ratio),
            "subspace_rank": int(args.subspace_rank),
        },
        "groups": groups,
        "family_stats": family_stats,
        "pairwise_families": pairwise_families,
        "spotlight_concepts": spotlight,
        "hypotheses": h,
        "interpretation": [
            "A family shared basis means many member concepts reuse a stable subset of hidden dimensions.",
            "Concept-specific dims are the offset coordinates that separate apple from banana or cat from dog.",
            "Positive subspace margin means the family can be described as a compact hidden subspace rather than isolated points.",
        ],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(
        json.dumps(
            {
                "fruit_margin": family_stats["fruit"]["subspace_margin"],
                "animal_margin": family_stats["animal"]["subspace_margin"],
                "fruit_animal_shared_jaccard": pairwise_families["fruit__animal"]["shared_dim_jaccard"],
                "apple_shared_overlap_ratio": spotlight["apple"]["shared_overlap_ratio"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
