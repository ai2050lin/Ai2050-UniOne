#!/usr/bin/env python
"""
Attention abstraction router probe.

Goal:
- Measure which attention heads support:
  1) instance -> category abstraction
  2) category -> abstract-system abstraction

Method:
- Build three prompt groups: entity, category, abstract-system.
- Use final-token residual representation at the last layer as the probe space.
- Define two baseline lift axes:
    lift1 = mean(category) - mean(entity)
    lift2 = mean(abstract) - mean(category)
- Ablate one attention head at a time at the input of `o_proj`.
- Measure how much each head collapses lift1 vs lift2.
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


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn
    raise RuntimeError("Cannot find a probe-able attention module")


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


def mean_vector(vs: np.ndarray) -> np.ndarray:
    return np.mean(vs, axis=0).astype(np.float32)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if nv < eps:
        return np.zeros_like(v)
    return (v / nv).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def topk_pairs(rows: Sequence[Dict[str, float]], key: str, k: int) -> List[Dict[str, float]]:
    out = sorted(rows, key=lambda r: float(r[key]), reverse=True)[: max(1, int(k))]
    return [
        {
            "layer": int(r["layer"]),
            "head": int(r["head"]),
            key: float(r[key]),
            "cross_score": float(r["collapse_lift2"] if key == "collapse_lift1" else r["collapse_lift1"]),
            "preference": float(r["preference"]),
        }
        for r in out
    ]


def layer_hist(rows: Sequence[Dict[str, float]]) -> Dict[str, int]:
    ctr = Counter(int(r["layer"]) for r in rows)
    return {str(k): int(v) for k, v in sorted(ctr.items())}


def jaccard_head_set(a: Sequence[Dict[str, float]], b: Sequence[Dict[str, float]]) -> float:
    sa = {(int(r["layer"]), int(r["head"])) for r in a}
    sb = {(int(r["layer"]), int(r["head"])) for r in b}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def entity_prompts() -> List[str]:
    return [
        "This is apple",
        "This is banana",
        "This is cat",
        "This is dog",
        "This is car",
        "This is train",
        "This is chair",
        "This is bottle",
    ]


def category_prompts() -> List[str]:
    return [
        "The concept of fruit",
        "fruit is a category",
        "The concept of animal",
        "animal is a category",
        "The concept of vehicle",
        "vehicle is a category",
        "The concept of object",
        "object is a category",
    ]


def abstract_prompts() -> List[str]:
    return [
        "The concept of justice",
        "justice is an abstract idea",
        "The concept of truth",
        "truth is an abstract idea",
        "The concept of logic",
        "logic is an abstract idea",
        "The concept of language",
        "language is an abstract idea",
    ]


class HeadAblator:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.handles = []
        self.active_layer = -1
        self.active_head = -1
        self.head_dim = int(getattr(model.config, "hidden_size") // getattr(model.config, "num_attention_heads"))
        for li, layer in enumerate(self.layers):
            attn = get_attention_module(layer)
            target = attn.o_proj if hasattr(attn, "o_proj") else attn.c_proj
            self.handles.append(target.register_forward_pre_hook(self._pre_hook(li)))

    def _pre_hook(self, li: int):
        def fn(_module, inputs):
            if li != self.active_layer or self.active_head < 0:
                return None
            x = inputs[0]
            start = self.active_head * self.head_dim
            end = start + self.head_dim
            y = x.clone()
            y[..., start:end] = 0
            return (y,)

        return fn

    def set_active(self, layer_idx: int, head_idx: int) -> None:
        self.active_layer = int(layer_idx)
        self.active_head = int(head_idx)

    def clear(self) -> None:
        self.active_layer = -1
        self.active_head = -1

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def get_last_hidden_batch(model, tok, prompts: Sequence[str]) -> np.ndarray:
    device = next(model.parameters()).device
    enc = tok(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    positions = enc["attention_mask"].sum(dim=1) - 1
    rows = []
    for bi in range(hidden.shape[0]):
        rows.append(hidden[bi, positions[bi], :].detach().float().cpu().numpy().astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention abstraction router probe")
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=-1)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/attention_abstraction_router_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    layer_start = max(0, int(args.layer_start))
    layer_end = n_layers if int(args.layer_end) < 0 else min(n_layers, int(args.layer_end))
    if layer_start >= layer_end:
        raise ValueError("Invalid layer range")

    prompts_entity = entity_prompts()
    prompts_category = category_prompts()
    prompts_abstract = abstract_prompts()

    ablator = HeadAblator(model)
    try:
        ablator.clear()
        ent_base = mean_vector(get_last_hidden_batch(model, tok, prompts_entity))
        cat_base = mean_vector(get_last_hidden_batch(model, tok, prompts_category))
        abs_base = mean_vector(get_last_hidden_batch(model, tok, prompts_abstract))

        lift1 = (cat_base - ent_base).astype(np.float32)
        lift2 = (abs_base - cat_base).astype(np.float32)
        axis1 = normalize(lift1)
        axis2 = normalize(lift2)

        base_gap1 = float(np.dot(cat_base - ent_base, axis1))
        base_gap2 = float(np.dot(abs_base - cat_base, axis2))
        base_lift_cos = float(cosine(lift1, lift2))

        rows = []
        total = (layer_end - layer_start) * n_heads
        done = 0
        for li in range(layer_start, layer_end):
            for hi in range(n_heads):
                ablator.set_active(li, hi)
                ent_vec = mean_vector(get_last_hidden_batch(model, tok, prompts_entity))
                cat_vec = mean_vector(get_last_hidden_batch(model, tok, prompts_category))
                abs_vec = mean_vector(get_last_hidden_batch(model, tok, prompts_abstract))
                gap1 = float(np.dot(cat_vec - ent_vec, axis1))
                gap2 = float(np.dot(abs_vec - cat_vec, axis2))
                collapse1 = float(base_gap1 - gap1)
                collapse2 = float(base_gap2 - gap2)
                preference = float((collapse1 - collapse2) / (abs(collapse1) + abs(collapse2) + 1e-9))
                rows.append(
                    {
                        "layer": int(li),
                        "head": int(hi),
                        "gap1_after": float(gap1),
                        "gap2_after": float(gap2),
                        "collapse_lift1": collapse1,
                        "collapse_lift2": collapse2,
                        "preference": preference,
                    }
                )
                done += 1
                if done % 24 == 0 or done == total:
                    print(f"[scan] {done}/{total} heads")
        ablator.clear()
    finally:
        ablator.close()

    top_lift1 = topk_pairs(rows, "collapse_lift1", args.top_k)
    top_lift2 = topk_pairs(rows, "collapse_lift2", args.top_k)

    same_role = []
    split_role = []
    for r in rows:
        if float(r["collapse_lift1"]) > 0 and float(r["collapse_lift2"]) > 0:
            same_role.append(r)
        if abs(float(r["preference"])) >= 0.35:
            split_role.append(r)

    layer_scores = []
    for li in range(layer_start, layer_end):
        sub = [r for r in rows if int(r["layer"]) == li]
        mean1 = float(np.mean([float(r["collapse_lift1"]) for r in sub]))
        mean2 = float(np.mean([float(r["collapse_lift2"]) for r in sub]))
        layer_scores.append(
            {
                "layer": int(li),
                "mean_collapse_lift1": mean1,
                "mean_collapse_lift2": mean2,
            }
        )

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": args.model_id,
            "dtype": args.dtype,
            "local_files_only": bool(args.local_files_only),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "layer_range": [layer_start, layer_end],
            "probe_space": "last_layer_last_token_hidden_state",
            "runtime_sec": float(time.time() - t0),
        },
        "baseline": {
            "entity_prompt_count": len(prompts_entity),
            "category_prompt_count": len(prompts_category),
            "abstract_prompt_count": len(prompts_abstract),
            "base_gap_instance_to_category": base_gap1,
            "base_gap_category_to_abstract": base_gap2,
            "baseline_lift_alignment": base_lift_cos,
        },
        "top_heads_instance_to_category": top_lift1,
        "top_heads_category_to_abstract": top_lift2,
        "top20_overlap_jaccard": float(jaccard_head_set(top_lift1, top_lift2)),
        "top20_layer_hist_instance_to_category": layer_hist(top_lift1),
        "top20_layer_hist_category_to_abstract": layer_hist(top_lift2),
        "global_stats": {
            "scanned_head_count": int(len(rows)),
            "mean_collapse_lift1": float(np.mean([float(r["collapse_lift1"]) for r in rows])),
            "mean_collapse_lift2": float(np.mean([float(r["collapse_lift2"]) for r in rows])),
            "mean_abs_preference": float(np.mean([abs(float(r["preference"])) for r in rows])),
            "positive_both_count": int(len(same_role)),
            "specialized_head_count_abs_pref_ge_0_35": int(len(split_role)),
        },
        "layer_scores": layer_scores,
        "hypotheses": {
            "H1_some_heads_support_instance_to_category": bool(max(float(r["collapse_lift1"]) for r in rows) > 0.01),
            "H2_some_heads_support_category_to_abstract": bool(max(float(r["collapse_lift2"]) for r in rows) > 0.01),
            "H3_routes_are_partly_distinct": bool(jaccard_head_set(top_lift1, top_lift2) < 0.5),
            "H4_many_heads_are_role_specialized": bool(len(split_role) >= max(8, total // 20)),
        },
        "all_rows": rows,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["baseline"], ensure_ascii=False, indent=2))
    print(json.dumps(results["global_stats"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
