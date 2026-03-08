#!/usr/bin/env python
"""
Abstraction ladder hierarchy probe.

Goal:
- Test whether the model shows a continuous hierarchy:
    entity instances -> category words -> abstract system words
- Quantify whether the "entity->category" lift and the
  "category->abstract" lift are aligned.
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


def mean_vector(vs: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if nv < eps:
        return np.zeros_like(v)
    return (v / nv).astype(np.float32)


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


def abstract_system_words() -> List[str]:
    return ["justice", "truth", "infinity", "freedom", "logic", "memory", "reason", "language"]


def entity_prompt_set(noun: str) -> List[str]:
    return [
        f"This is {noun}",
        f"I saw {noun}",
        f"The word is {noun}",
    ]


def category_prompt_set(noun: str) -> List[str]:
    return [
        f"The concept of {noun}",
        f"{noun} is a category",
        f"{noun} is a kind of class",
    ]


def abstract_prompt_set(noun: str) -> List[str]:
    return [
        f"The concept of {noun}",
        f"{noun} is an abstract idea",
        f"{noun} is a principle",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Abstraction ladder hierarchy probe")
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/abstraction_ladder_hierarchy_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    families = category_families()
    abstract_words = abstract_system_words()

    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        entity_proto = {}
        category_word = {}
        abstract_word = {}

        for family, members in families.items():
            rows = []
            for member in members:
                for prompt in entity_prompt_set(member):
                    collector.reset()
                    run_prompt(model, tok, prompt)
                    rows.append(collector.get_flat())
            entity_proto[family] = mean_vector(rows)

            rows = []
            for prompt in category_prompt_set(family):
                collector.reset()
                run_prompt(model, tok, prompt)
                rows.append(collector.get_flat())
            category_word[family] = mean_vector(rows)

        for word in abstract_words:
            rows = []
            for prompt in abstract_prompt_set(word):
                collector.reset()
                run_prompt(model, tok, prompt)
                rows.append(collector.get_flat())
            abstract_word[word] = mean_vector(rows)
    finally:
        collector.close()

    any_vec = next(iter(entity_proto.values()))
    n_total_dims = int(any_vec.shape[0])
    n_layers = len(discover_layers(model))
    d_ff = int(n_total_dims // max(1, n_layers))

    entity_mean = mean_vector(list(entity_proto.values()))
    category_mean = mean_vector(list(category_word.values()))
    abstract_mean = mean_vector(list(abstract_word.values()))

    lift1 = (category_mean - entity_mean).astype(np.float32)
    lift2 = (abstract_mean - category_mean).astype(np.float32)
    axis1 = normalize(lift1)
    axis2 = normalize(lift2)
    global_axis = normalize((abstract_mean - entity_mean).astype(np.float32))

    entity_proj = [float(np.dot(entity_proto[name], global_axis)) for name in families]
    category_proj = [float(np.dot(category_word[name], global_axis)) for name in families]
    abstract_proj = [float(np.dot(abstract_word[name], global_axis)) for name in abstract_words]

    lift1_top = topk_abs_indices(lift1, args.top_k).tolist()
    abstract_lifts = {w: (abstract_word[w] - category_mean).astype(np.float32) for w in abstract_words}
    abstract_lift_top = {w: topk_abs_indices(v, args.top_k).tolist() for w, v in abstract_lifts.items()}
    abstract_lift_mean = mean_vector(list(abstract_lifts.values()))
    abstract_lift_mean_top = topk_abs_indices(abstract_lift_mean, args.top_k).tolist()

    pair_cos = {}
    pair_vals = []
    for w1 in abstract_words:
        row = {}
        for w2 in abstract_words:
            val = cosine(abstract_lifts[w1], abstract_lifts[w2])
            row[w2] = float(val)
            if w1 != w2:
                pair_vals.append(abs(val))
        pair_cos[w1] = row

    shared_freq = Counter()
    for w in abstract_words:
        for idx in abstract_lift_top[w]:
            shared_freq[int(idx)] += 1
    shared_abstract_dims = [idx for idx, cnt in shared_freq.items() if cnt >= 2]
    shared_abstract_dims = sorted(shared_abstract_dims, key=lambda i: shared_freq[int(i)], reverse=True)[: args.top_k]

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
        "abstract_words": abstract_words,
        "global_vectors": {
            "lift1_top_dims": [int(i) for i in lift1_top[:24]],
            "lift1_layer_distribution": layer_hist(lift1_top[:64], d_ff),
            "lift2_top_dims": [int(i) for i in abstract_lift_mean_top[:24]],
            "lift2_layer_distribution": layer_hist(abstract_lift_mean_top[:64], d_ff),
            "lift1_lift2_alignment": float(cosine(axis1, axis2)),
            "lift1_lift2_topdim_jaccard": float(jaccard(lift1_top[:64], abstract_lift_mean_top[:64])),
        },
        "projection_ladder": {
            "entity_mean_proj": float(np.mean(entity_proj)),
            "category_mean_proj": float(np.mean(category_proj)),
            "abstract_mean_proj": float(np.mean(abstract_proj)),
            "entity_proj_values": entity_proj,
            "category_proj_values": category_proj,
            "abstract_proj_values": abstract_proj,
        },
        "shared_abstract_system_dims": {
            "dims": [int(i) for i in shared_abstract_dims[:24]],
            "layer_distribution": layer_hist(shared_abstract_dims[:64], d_ff),
        },
        "abstract_word_rows": {
            w: {
                "lift_top_dims": [int(i) for i in abstract_lift_top[w][:16]],
                "lift_layer_distribution": layer_hist(abstract_lift_top[w][:64], d_ff),
                "lift_to_category_lift_jaccard": float(jaccard(abstract_lift_top[w][:64], lift1_top[:64])),
                "lift_cos_to_lift1": float(cosine(abstract_lifts[w], lift1)),
            }
            for w in abstract_words
        },
        "metrics": {
            "lift1_lift2_alignment": float(cosine(axis1, axis2)),
            "mean_abs_abstract_pair_alignment": float(np.mean(pair_vals) if pair_vals else 0.0),
            "mean_abstract_to_category_lift_jaccard": float(
                np.mean([jaccard(abstract_lift_top[w][:64], lift1_top[:64]) for w in abstract_words])
            ),
            "shared_abstract_dim_count": int(len(shared_abstract_dims)),
        },
        "hypotheses": {
            "H1_second_order_abstraction_alignment": bool(float(cosine(axis1, axis2)) > 0.10),
            "H2_projection_ladder_monotonic": bool(
                float(np.mean(entity_proj)) < float(np.mean(category_proj)) < float(np.mean(abstract_proj))
            ),
            "H3_shared_abstract_system_dims_exist": bool(len(shared_abstract_dims) >= 8),
        },
        "interpretation": [
            "If lift1 and lift2 align, the model reuses a similar operator from instances to categories and from categories to abstract system words.",
            "A monotonic projection ladder suggests abstraction is not random clustering but directional hierarchy.",
            "Shared abstract-system dims are candidate coordinates for high-level symbolic system coding.",
        ],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(
        json.dumps(
            {
                "lift1_lift2_alignment": result["metrics"]["lift1_lift2_alignment"],
                "shared_abstract_dim_count": result["metrics"]["shared_abstract_dim_count"],
                "projection_ladder": result["projection_ladder"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

