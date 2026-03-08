#!/usr/bin/env python
"""
Category abstraction bridge probe.

Goal:
- Analyze whether category words like "fruit" and "animal" share
  a common abstraction-level coding pattern, even if their member families differ.

Key idea:
- For each family F:
    entity_proto(F) = mean(member concept activations)
    category_word(F) = activation of the category noun itself
    abstraction_lift(F) = category_word(F) - entity_proto(F)
- Compare abstraction_lift across families.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

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


def concept_prompt_set(noun: str) -> List[str]:
    return [
        f"This is {noun}",
        f"I saw {noun}",
        f"The word is {noun}",
    ]


def category_prompt_set(noun: str) -> List[str]:
    return [
        f"{noun} is a category",
        f"The concept of {noun}",
        f"{noun} is a kind of class",
    ]


def mean_vector(vs: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def norm_ratio(delta: np.ndarray, base: np.ndarray, eps: float = 1e-12) -> float:
    nd = float(np.linalg.norm(delta))
    nb = float(np.linalg.norm(base))
    if nb < eps:
        return 0.0
    return float(nd / nb)


def topk_abs_indices(vec: np.ndarray, k: int) -> np.ndarray:
    kk = min(max(1, int(k)), int(vec.shape[0]))
    score = np.abs(vec)
    idx = np.argpartition(score, -kk)[-kk:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(np.int64)


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


def category_families() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "vehicle": ["car", "bus", "train", "boat", "truck", "bicycle"],
        "object": ["chair", "table", "lamp", "door", "bottle", "spoon"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Category abstraction bridge probe")
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/category_abstraction_bridge_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    families = category_families()

    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        all_vectors: Dict[str, np.ndarray] = {}

        for family_name, members in families.items():
            member_rows = []
            for member in members:
                for prompt in concept_prompt_set(member):
                    collector.reset()
                    run_prompt(model, tok, prompt)
                    member_rows.append(collector.get_flat())
            all_vectors[f"entity_proto::{family_name}"] = mean_vector(member_rows)

            cat_rows = []
            for prompt in category_prompt_set(family_name):
                collector.reset()
                run_prompt(model, tok, prompt)
                cat_rows.append(collector.get_flat())
            all_vectors[f"category_word::{family_name}"] = mean_vector(cat_rows)
    finally:
        collector.close()

    any_vec = next(iter(all_vectors.values()))
    n_total_dims = int(any_vec.shape[0])
    n_layers = len(discover_layers(model))
    d_ff = int(n_total_dims // max(1, n_layers))

    entity_protos = {k.split("::", 1)[1]: v for k, v in all_vectors.items() if k.startswith("entity_proto::")}
    category_words = {k.split("::", 1)[1]: v for k, v in all_vectors.items() if k.startswith("category_word::")}
    lifts = {name: (category_words[name] - entity_protos[name]).astype(np.float32) for name in families}

    lift_top = {name: topk_abs_indices(vec, args.top_k).tolist() for name, vec in lifts.items()}
    lift_layer = {name: layer_hist(idx[:64], d_ff) for name, idx in lift_top.items()}
    lift_cos = {}
    pairwise_vals = []
    for a in families:
        row = {}
        for b in families:
            val = cosine(lifts[a], lifts[b])
            row[b] = float(val)
            if a != b:
                pairwise_vals.append(abs(val))
        lift_cos[a] = row

    freq = Counter()
    for name in families:
        for idx in lift_top[name]:
            freq[int(idx)] += 1
    shared_meta = [idx for idx, cnt in freq.items() if cnt >= 2]
    shared_meta = sorted(shared_meta, key=lambda i: freq[int(i)], reverse=True)[: args.top_k]

    family_rows = {}
    for name in families:
        proto = entity_protos[name]
        cat = category_words[name]
        lift = lifts[name]
        family_rows[name] = {
            "entity_proto_top_dims": [int(i) for i in topk_abs_indices(proto, args.top_k)[:16].tolist()],
            "category_word_top_dims": [int(i) for i in topk_abs_indices(cat, args.top_k)[:16].tolist()],
            "abstraction_lift_top_dims": [int(i) for i in lift_top[name][:16]],
            "lift_layer_distribution": lift_layer[name],
            "proto_to_category_cosine": float(cosine(proto, cat)),
            "lift_norm_ratio": float(norm_ratio(lift, proto)),
            "lift_to_shared_meta_jaccard": float(jaccard(lift_top[name][:64], shared_meta[:64])),
        }

    fruit_animal_alignment = float(lift_cos["fruit"]["animal"])
    within_concrete_same_level = float(
        np.mean([lift_cos["fruit"]["animal"], lift_cos["vehicle"]["object"]])
    )
    cross_mix = float(
        np.mean([lift_cos["fruit"]["vehicle"], lift_cos["animal"]["object"]])
    )

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 3),
            "model_id": args.model_id,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "n_total_dims": n_total_dims,
            "top_k": int(args.top_k),
        },
        "families": families,
        "family_rows": family_rows,
        "lift_cosine_matrix": lift_cos,
        "shared_meta_category_dims": {
            "dims": [int(i) for i in shared_meta[:24]],
            "layer_distribution": layer_hist(shared_meta[:64], d_ff),
        },
        "metrics": {
            "fruit_animal_lift_alignment": fruit_animal_alignment,
            "within_concrete_same_level_alignment": within_concrete_same_level,
            "cross_mix_alignment": cross_mix,
            "mean_abs_pairwise_lift_alignment": float(np.mean(pairwise_vals) if pairwise_vals else 0.0),
            "mean_lift_norm_ratio": float(np.mean([family_rows[name]["lift_norm_ratio"] for name in families])),
        },
        "hypotheses": {
            "H1_fruit_animal_share_abstraction_pattern": bool(fruit_animal_alignment > 0.10),
            "H2_meta_category_shared_dims_exist": bool(len(shared_meta) >= 4),
            "H3_abstraction_lift_nontrivial": bool(
                np.mean([family_rows[name]["lift_norm_ratio"] for name in families]) > 0.05
            ),
        },
        "interpretation": [
            "Category words may differ in semantics, yet share a common abstraction lift from members to class-level nouns.",
            "Shared meta-category dims are candidate coordinates for 'being a class noun' rather than being a specific family.",
            "If fruit and animal lift vectors align, the model reuses a common abstraction operator over different semantic domains.",
        ],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(
        json.dumps(
            {
                "fruit_animal_lift_alignment": result["metrics"]["fruit_animal_lift_alignment"],
                "shared_meta_dim_count": len(shared_meta),
                "within_concrete_same_level_alignment": result["metrics"]["within_concrete_same_level_alignment"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
