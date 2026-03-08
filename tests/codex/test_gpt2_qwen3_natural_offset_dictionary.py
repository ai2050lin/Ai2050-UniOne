#!/usr/bin/env python
"""
Test whether concept-specific offsets become compact in a learned residual
dictionary, instead of in raw neuron coordinates.

For each target concept, we:
1) fit a leave-one-out family affine basis
2) compute the target residual delta
3) build residual dictionaries from prompt-level residual samples
4) compare matched-family vs mismatched-family dictionary capture

This stays within the current project scope: analysis only, no architecture design.
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


def topk_energy_ratio(vec: np.ndarray, k: int, eps: float = 1e-12) -> float:
    score = np.square(vec.astype(np.float64))
    total = float(score.sum()) + eps
    kk = int(min(max(1, k), score.shape[0]))
    idx = np.argpartition(score, -kk)[-kk:]
    return float(score[idx].sum() / total)


def min_k_for_energy(vec: np.ndarray, thresholds: Sequence[float]) -> Dict[str, int]:
    score = np.sort(np.square(vec.astype(np.float64)))[::-1]
    total = float(score.sum()) + 1e-12
    csum = np.cumsum(score) / total
    out = {}
    for th in thresholds:
        k = int(np.searchsorted(csum, th, side="left") + 1)
        out[f"{th:.2f}"] = k
    return out


def orthonormal_dictionary(xs: Sequence[np.ndarray], max_atoms: int) -> np.ndarray:
    if not xs:
        raise RuntimeError("Need at least one sample to build a dictionary")
    mat = np.stack(xs, axis=0).astype(np.float32)
    _u, _s, vh = np.linalg.svd(mat, full_matrices=False)
    k = int(min(max_atoms, vh.shape[0]))
    return vh[:k].T.astype(np.float32)


def dict_capture(delta: np.ndarray, dictionary: np.ndarray, top_k: int) -> float:
    if dictionary.shape[1] == 0:
        return 0.0
    coeff = dictionary.T @ delta.astype(np.float32)
    score = np.square(coeff.astype(np.float64))
    kk = int(min(max(1, top_k), score.shape[0]))
    idx = np.argpartition(score, -kk)[-kk:]
    return float(score[idx].sum() / (float(np.square(delta.astype(np.float64)).sum()) + 1e-12))


def dict_total_capture(delta: np.ndarray, dictionary: np.ndarray) -> float:
    if dictionary.shape[1] == 0:
        return 0.0
    coeff = dictionary.T @ delta.astype(np.float32)
    return float(np.square(coeff.astype(np.float64)).sum() / (float(np.square(delta.astype(np.float64)).sum()) + 1e-12))


def collect_prompt_vectors(model, tok, collector: HiddenCollector) -> Dict[str, List[np.ndarray]]:
    fams = family_map()
    data: Dict[str, List[np.ndarray]] = {}
    for fam, words in fams.items():
        for word in words:
            rows = []
            for text in prompts_for_word(word, fam):
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


def build_family_dictionary(
    prompt_vectors: Dict[str, List[np.ndarray]],
    concept_means: Dict[str, np.ndarray],
    family: str,
    holdout_word: str | None,
    max_atoms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, basis = build_leave_one_family_basis(concept_means, family, holdout_word) if holdout_word in family_map()[family] else affine_basis(
        [concept_means[w] for w in family_map()[family]],
        family_rank(family),
    )
    residuals = []
    for word in family_map()[family]:
        if holdout_word is not None and word == holdout_word:
            continue
        for vec in prompt_vectors[word]:
            residuals.append(residual_delta(vec, mu, basis))
    dictionary = orthonormal_dictionary(residuals, max_atoms=max_atoms)
    return mu, basis, dictionary


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    max_atoms: int,
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
        mu_match, basis_match = build_leave_one_family_basis(concept_means, family, word)
        delta = residual_delta(concept_means[word], mu_match, basis_match)

        matched_mu, matched_basis, matched_dict = build_family_dictionary(
            prompt_vectors=prompt_vectors,
            concept_means=concept_means,
            family=family,
            holdout_word=word,
            max_atoms=max_atoms,
        )
        matched_total = dict_total_capture(delta, matched_dict)
        matched_top = {
            "top1_capture": dict_capture(delta, matched_dict, 1),
            "top2_capture": dict_capture(delta, matched_dict, 2),
            "top4_capture": dict_capture(delta, matched_dict, 4),
            "top8_capture": dict_capture(delta, matched_dict, 8),
            "total_capture": matched_total,
        }

        wrong_stats = []
        for wrong_family in family_map().keys():
            if wrong_family == family:
                continue
            _wrong_mu, _wrong_basis, wrong_dict = build_family_dictionary(
                prompt_vectors=prompt_vectors,
                concept_means=concept_means,
                family=wrong_family,
                holdout_word=None,
                max_atoms=max_atoms,
            )
            wrong_stats.append(
                {
                    "family": wrong_family,
                    "top4_capture": dict_capture(delta, wrong_dict, 4),
                    "total_capture": dict_total_capture(delta, wrong_dict),
                }
            )

        raw_counts = min_k_for_energy(delta, thresholds=(0.5, 0.8))
        avg_wrong_top4 = float(np.mean([x["top4_capture"] for x in wrong_stats]))
        avg_wrong_total = float(np.mean([x["total_capture"] for x in wrong_stats]))
        target_rows.append(
            {
                "word": word,
                "family": family,
                "delta_norm": float(np.linalg.norm(delta)),
                "raw_top4_capture": topk_energy_ratio(delta, 4),
                "raw_top8_capture": topk_energy_ratio(delta, 8),
                "raw_top64_capture": topk_energy_ratio(delta, 64),
                "raw_top256_capture": topk_energy_ratio(delta, 256),
                "raw_min_neurons_for_50pct": int(raw_counts["0.50"]),
                "raw_min_neurons_for_80pct": int(raw_counts["0.80"]),
                "matched_dict": matched_top,
                "avg_wrong_dict_top4_capture": avg_wrong_top4,
                "avg_wrong_dict_total_capture": avg_wrong_total,
                "matched_vs_wrong_top4_gap": float(matched_top["top4_capture"] - avg_wrong_top4),
                "matched_vs_wrong_total_gap": float(matched_total - avg_wrong_total),
                "supports_natural_dict_sparse_offset": bool(
                    matched_top["top4_capture"] > avg_wrong_top4 + 0.005
                    and matched_top["top4_capture"] > topk_energy_ratio(delta, 4) * 1.5
                ),
            }
        )

    summary = {}
    for family in ("fruit", "animal", "abstract"):
        fam_rows = [r for r in target_rows if r["family"] == family]
        summary[family] = {
            "mean_raw_top4_capture": float(np.mean([r["raw_top4_capture"] for r in fam_rows])),
            "mean_raw_top256_capture": float(np.mean([r["raw_top256_capture"] for r in fam_rows])),
            "mean_matched_top4_capture": float(np.mean([r["matched_dict"]["top4_capture"] for r in fam_rows])),
            "mean_avg_wrong_top4_capture": float(np.mean([r["avg_wrong_dict_top4_capture"] for r in fam_rows])),
            "mean_gap_top4": float(np.mean([r["matched_vs_wrong_top4_gap"] for r in fam_rows])),
            "support_rate": float(np.mean([1.0 if r["supports_natural_dict_sparse_offset"] else 0.0 for r in fam_rows])),
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
        "max_atoms": int(max_atoms),
        "runtime_sec": float(time.time() - t0),
    }
    return {
        "meta": meta,
        "targets": target_rows,
        "family_summary": summary,
    }


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", r"C:\Users\27876\.cache\huggingface\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("qwen3_4b", r"C:\Users\27876\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Test natural residual dictionaries in GPT-2 and Qwen3")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--max-atoms", type=int, default=8)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_natural_offset_dictionary_20260308.json",
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
            max_atoms=args.max_atoms,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for model_name, row in results["models"].items():
        fam = row["family_summary"]
        print(f"[summary] {model_name} fruit_gap_top4={fam['fruit']['mean_gap_top4']:.4f} animal_gap_top4={fam['animal']['mean_gap_top4']:.4f} abstract_gap_top4={fam['abstract']['mean_gap_top4']:.4f}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
