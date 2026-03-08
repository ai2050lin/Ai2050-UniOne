#!/usr/bin/env python
"""
Compare GPT-2 and local Qwen3 on:
1) affine shared basis compactness
2) apple's offset against fruit basis
3) family basis inclusion in a larger world basis

This script stays model-agnostic by probing per-layer last-token hidden states
and flattening them into one representation vector per prompt.
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
        "vehicle": ["car", "bus", "train", "boat", "truck", "bicycle"],
        "object": ["chair", "table", "lamp", "door", "bottle", "spoon"],
        "abstract": ["justice", "truth", "logic", "language", "freedom", "memory"],
    }


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
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
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


def subspace_inclusion(source_basis: np.ndarray, target_basis: np.ndarray) -> float:
    if source_basis.shape[1] == 0 or target_basis.shape[1] == 0:
        return 0.0
    m = source_basis.T @ target_basis
    s = np.linalg.svd(m, compute_uv=False)
    return float(np.mean(np.square(s)))


def collect_concepts(model, tok, collector: HiddenCollector) -> Dict[str, np.ndarray]:
    data = {}
    fams = family_map()
    for fam, words in fams.items():
        for word in words:
            rows = []
            prompts = abstract_prompts(word) if fam == "abstract" else concrete_prompts(word)
            for text in prompts:
                collector.reset()
                run_prompt(model, tok, text)
                rows.append(collector.get_flat())
            data[word] = mean_vector(rows)
    return data


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    world_rank: int,
) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    collector = HiddenCollector(model)
    try:
        concept_vec = collect_concepts(model, tok, collector)
    finally:
        collector.close()

    fams = family_map()
    fam_basis = {}
    for fam, words in fams.items():
        rank_k = 4 if fam != "abstract" else 3
        mu, basis = affine_basis([concept_vec[w] for w in words], rank_k)
        fam_basis[fam] = {"mu": mu, "basis": basis}

    world_words = [w for fam in fams.values() for w in fam]
    world_mu, world_basis = affine_basis([concept_vec[w] for w in world_words], world_rank)

    family_compactness = {}
    for fam, words in fams.items():
        mu = fam_basis[fam]["mu"]
        basis = fam_basis[fam]["basis"]
        vals = [residual_ratio(concept_vec[w], mu, basis) for w in words]
        family_compactness[fam] = {
            "mean_residual_ratio": float(np.mean(vals)),
            "max_residual_ratio": float(np.max(vals)),
        }

    apple = concept_vec["apple"]
    apple_metrics = {}
    for fam in ("fruit", "animal", "vehicle", "object", "abstract"):
        mu = fam_basis[fam]["mu"]
        basis = fam_basis[fam]["basis"]
        proj = affine_project(apple, mu, basis)
        delta = (apple - proj).astype(np.float32)
        apple_metrics[fam] = {
            "residual_ratio": residual_ratio(apple, mu, basis),
            "delta_top64_energy_ratio": topk_energy_ratio(delta, 64),
            "delta_top256_energy_ratio": topk_energy_ratio(delta, 256),
            "delta_norm": float(np.linalg.norm(delta)),
        }

    inclusion = {}
    for fam in ("fruit", "animal", "vehicle", "object", "abstract"):
        inclusion[fam] = {
            "family_into_world": subspace_inclusion(fam_basis[fam]["basis"], world_basis),
        }

    nested = {
        "apple_vs_fruit_minus_animal_residual_gap": float(
            apple_metrics["animal"]["residual_ratio"] - apple_metrics["fruit"]["residual_ratio"]
        ),
        "apple_vs_fruit_minus_object_residual_gap": float(
            apple_metrics["object"]["residual_ratio"] - apple_metrics["fruit"]["residual_ratio"]
        ),
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
        "world_rank": int(world_rank),
        "runtime_sec": float(time.time() - t0),
    }

    return {
        "meta": meta,
        "family_compactness": family_compactness,
        "apple_affine_fit": apple_metrics,
        "family_into_world_inclusion": inclusion,
        "nested_metrics": nested,
        "hypotheses": {
            "H1_fruit_basis_compact": bool(family_compactness["fruit"]["mean_residual_ratio"] < 0.70),
            "H2_apple_closer_to_fruit_than_animal": bool(
                apple_metrics["fruit"]["residual_ratio"] < apple_metrics["animal"]["residual_ratio"]
            ),
            "H3_apple_closer_to_fruit_than_abstract": bool(
                apple_metrics["fruit"]["residual_ratio"] < apple_metrics["abstract"]["residual_ratio"]
            ),
            "H4_family_nested_in_world_basis": bool(
                min(v["family_into_world"] for v in inclusion.values()) > 0.15
            ),
            "H5_apple_offset_is_concentrated": bool(apple_metrics["fruit"]["delta_top64_energy_ratio"] > 0.30),
        },
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare GPT-2 and local Qwen3 on basis/offset hierarchy")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--world-rank", type=int, default=16)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_basis_hierarchy_compare_20260308.json",
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
            world_rank=args.world_rank,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpt2 = results["models"]["gpt2"]
    qwen = results["models"]["qwen3_4b"]
    results["cross_model_summary"] = {
        "fruit_compactness_delta_qwen_minus_gpt2": float(
            qwen["family_compactness"]["fruit"]["mean_residual_ratio"]
            - gpt2["family_compactness"]["fruit"]["mean_residual_ratio"]
        ),
        "apple_fruit_residual_delta_qwen_minus_gpt2": float(
            qwen["apple_affine_fit"]["fruit"]["residual_ratio"]
            - gpt2["apple_affine_fit"]["fruit"]["residual_ratio"]
        ),
        "apple_offset_top64_delta_qwen_minus_gpt2": float(
            qwen["apple_affine_fit"]["fruit"]["delta_top64_energy_ratio"]
            - gpt2["apple_affine_fit"]["fruit"]["delta_top64_energy_ratio"]
        ),
        "fruit_into_world_inclusion_delta_qwen_minus_gpt2": float(
            qwen["family_into_world_inclusion"]["fruit"]["family_into_world"]
            - gpt2["family_into_world_inclusion"]["fruit"]["family_into_world"]
        ),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["cross_model_summary"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
