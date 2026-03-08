#!/usr/bin/env python
"""
Attention abstraction router stability probe.

Goal:
- Validate whether top abstraction-router heads keep the same role
  across multiple prompt template banks.

Input:
- A previously generated attention router JSON file.

Output:
- Per-head, per-bank collapse ratios for:
  1) instance -> category
  2) category -> abstract-system
- Aggregate role-consistency metrics.
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


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if nv < eps:
        return np.zeros_like(v)
    return (v / nv).astype(np.float32)


def mean_vector(vs: np.ndarray) -> np.ndarray:
    return np.mean(vs, axis=0).astype(np.float32)


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


def prompt_banks() -> List[Dict[str, List[str]]]:
    return [
        {
            "bank_name": "template_this_is",
            "entity": [
                "This is apple",
                "This is banana",
                "This is cat",
                "This is dog",
                "This is car",
                "This is train",
                "This is chair",
                "This is bottle",
            ],
            "category": [
                "The concept of fruit",
                "fruit is a category",
                "The concept of animal",
                "animal is a category",
                "The concept of vehicle",
                "vehicle is a category",
                "The concept of object",
                "object is a category",
            ],
            "abstract": [
                "The concept of justice",
                "justice is an abstract idea",
                "The concept of truth",
                "truth is an abstract idea",
                "The concept of logic",
                "logic is an abstract idea",
                "The concept of language",
                "language is an abstract idea",
            ],
        },
        {
            "bank_name": "template_i_saw",
            "entity": [
                "I saw an apple",
                "I saw a banana",
                "I saw a cat",
                "I saw a dog",
                "I saw a car",
                "I saw a train",
                "I saw a chair",
                "I saw a bottle",
            ],
            "category": [
                "Fruit groups many foods",
                "Animal groups many living beings",
                "Vehicle groups many transport tools",
                "Object groups many everyday things",
                "Fruit is a broad class",
                "Animal is a broad class",
                "Vehicle is a broad class",
                "Object is a broad class",
            ],
            "abstract": [
                "Justice guides many judgments",
                "Truth guides many claims",
                "Logic guides many arguments",
                "Language guides many expressions",
                "Justice is a higher principle",
                "Truth is a higher principle",
                "Logic is a higher principle",
                "Language is a higher principle",
            ],
        },
        {
            "bank_name": "template_definition",
            "entity": [
                "The word apple names a thing",
                "The word banana names a thing",
                "The word cat names a thing",
                "The word dog names a thing",
                "The word car names a thing",
                "The word train names a thing",
                "The word chair names a thing",
                "The word bottle names a thing",
            ],
            "category": [
                "Fruit is the name of a class",
                "Animal is the name of a class",
                "Vehicle is the name of a class",
                "Object is the name of a class",
                "Fruit collects many members",
                "Animal collects many members",
                "Vehicle collects many members",
                "Object collects many members",
            ],
            "abstract": [
                "Justice is the name of a principle",
                "Truth is the name of a principle",
                "Logic is the name of a principle",
                "Language is the name of a principle",
                "Justice is more abstract than fruit",
                "Truth is more abstract than object",
                "Logic is more abstract than vehicle",
                "Language is more abstract than animal",
            ],
        },
        {
            "bank_name": "template_relation",
            "entity": [
                "Apple can be one example",
                "Banana can be one example",
                "Cat can be one example",
                "Dog can be one example",
                "Car can be one example",
                "Train can be one example",
                "Chair can be one example",
                "Bottle can be one example",
            ],
            "category": [
                "Fruit includes apples and bananas",
                "Animal includes cats and dogs",
                "Vehicle includes cars and trains",
                "Object includes chairs and bottles",
                "Fruit stands above specific fruits",
                "Animal stands above specific animals",
                "Vehicle stands above specific vehicles",
                "Object stands above specific objects",
            ],
            "abstract": [
                "Justice stands above many legal categories",
                "Truth stands above many factual categories",
                "Logic stands above many reasoning categories",
                "Language stands above many symbol categories",
                "Justice organizes abstract judgment",
                "Truth organizes abstract belief",
                "Logic organizes abstract inference",
                "Language organizes abstract expression",
            ],
        },
    ]


def load_selected_heads(path: Path, top_n: int, preference_threshold: float) -> Dict[str, List[Tuple[int, int]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("all_rows", [])
    if rows:
        g1_rows = sorted(
            [x for x in rows if float(x.get("preference", 0.0)) >= preference_threshold],
            key=lambda x: float(x.get("collapse_lift1", 0.0)),
            reverse=True,
        )[:top_n]
        g2_rows = sorted(
            [x for x in rows if float(x.get("preference", 0.0)) <= -preference_threshold],
            key=lambda x: float(x.get("collapse_lift2", 0.0)),
            reverse=True,
        )[:top_n]
        if len(g1_rows) >= top_n and len(g2_rows) >= top_n:
            g1 = [(int(x["layer"]), int(x["head"])) for x in g1_rows]
            g2 = [(int(x["layer"]), int(x["head"])) for x in g2_rows]
            return {
                "lift1": g1,
                "lift2": g2,
                "meta": obj.get("meta", {}),
                "baseline": obj.get("baseline", {}),
                "selection_mode": "specialized_from_all_rows",
                "preference_threshold": float(preference_threshold),
            }
    g1 = [(int(x["layer"]), int(x["head"])) for x in obj["top_heads_instance_to_category"][:top_n]]
    g2 = [(int(x["layer"]), int(x["head"])) for x in obj["top_heads_category_to_abstract"][:top_n]]
    return {
        "lift1": g1,
        "lift2": g2,
        "meta": obj.get("meta", {}),
        "baseline": obj.get("baseline", {}),
        "selection_mode": "top_heads_fallback",
        "preference_threshold": float(preference_threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention abstraction router stability probe")
    ap.add_argument(
        "--router-json",
        type=str,
        default="tests/codex_temp/attention_abstraction_router_20260308.json",
    )
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--preference-threshold", type=float, default=0.5)
    ap.add_argument("--model-id", type=str, default="")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/attention_abstraction_router_stability_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    router_path = Path(args.router_json)
    selected = load_selected_heads(router_path, args.top_n, args.preference_threshold)
    model_id = args.model_id or selected["meta"].get("model_id") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    model, tok = load_model(model_id, args.dtype, args.local_files_only)
    ablator = HeadAblator(model)

    banks = prompt_banks()
    head_rows = []
    aggregate = {
        "lift1": [],
        "lift2": [],
    }

    try:
        for bank in banks:
            ablator.clear()
            ent_base = mean_vector(get_last_hidden_batch(model, tok, bank["entity"]))
            cat_base = mean_vector(get_last_hidden_batch(model, tok, bank["category"]))
            abs_base = mean_vector(get_last_hidden_batch(model, tok, bank["abstract"]))

            lift1 = (cat_base - ent_base).astype(np.float32)
            lift2 = (abs_base - cat_base).astype(np.float32)
            axis1 = normalize(lift1)
            axis2 = normalize(lift2)
            base_gap1 = float(np.dot(cat_base - ent_base, axis1))
            base_gap2 = float(np.dot(abs_base - cat_base, axis2))

            for role_name in ("lift1", "lift2"):
                for layer_idx, head_idx in selected[role_name]:
                    ablator.set_active(layer_idx, head_idx)
                    ent_vec = mean_vector(get_last_hidden_batch(model, tok, bank["entity"]))
                    cat_vec = mean_vector(get_last_hidden_batch(model, tok, bank["category"]))
                    abs_vec = mean_vector(get_last_hidden_batch(model, tok, bank["abstract"]))
                    gap1 = float(np.dot(cat_vec - ent_vec, axis1))
                    gap2 = float(np.dot(abs_vec - cat_vec, axis2))
                    collapse1 = float(base_gap1 - gap1)
                    collapse2 = float(base_gap2 - gap2)
                    ratio1 = float(collapse1 / max(1e-9, abs(base_gap1)))
                    ratio2 = float(collapse2 / max(1e-9, abs(base_gap2)))
                    intended = ratio1 if role_name == "lift1" else ratio2
                    cross = ratio2 if role_name == "lift1" else ratio1
                    role_ok = bool(intended > cross)
                    head_rows.append(
                        {
                            "bank_name": bank["bank_name"],
                            "role_group": role_name,
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "base_gap1": base_gap1,
                            "base_gap2": base_gap2,
                            "collapse1": collapse1,
                            "collapse2": collapse2,
                            "ratio1": ratio1,
                            "ratio2": ratio2,
                            "intended_ratio": float(intended),
                            "cross_ratio": float(cross),
                            "role_consistent": role_ok,
                        }
                    )
                    aggregate[role_name].append((float(intended), float(cross), role_ok))
            ablator.clear()
    finally:
        ablator.close()

    group_summary = {}
    for role_name in ("lift1", "lift2"):
        values = aggregate[role_name]
        intended_mean = float(np.mean([x[0] for x in values])) if values else 0.0
        cross_mean = float(np.mean([x[1] for x in values])) if values else 0.0
        consistency = float(np.mean([1.0 if x[2] else 0.0 for x in values])) if values else 0.0
        margin_mean = float(np.mean([x[0] - x[1] for x in values])) if values else 0.0
        group_summary[role_name] = {
            "mean_intended_ratio": intended_mean,
            "mean_cross_ratio": cross_mean,
            "mean_margin": margin_mean,
            "role_consistency_rate": consistency,
        }

    head_summary_map: Dict[str, Dict[str, object]] = {}
    for row in head_rows:
        key = f"{row['role_group']}::{row['layer']}::{row['head']}"
        cur = head_summary_map.get(
            key,
            {
                "role_group": row["role_group"],
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "intended_ratios": [],
                "cross_ratios": [],
                "role_flags": [],
            },
        )
        cur["intended_ratios"].append(float(row["intended_ratio"]))
        cur["cross_ratios"].append(float(row["cross_ratio"]))
        cur["role_flags"].append(bool(row["role_consistent"]))
        head_summary_map[key] = cur

    head_summary = []
    for cur in head_summary_map.values():
        intended_ratios = [float(x) for x in cur["intended_ratios"]]
        cross_ratios = [float(x) for x in cur["cross_ratios"]]
        role_flags = [bool(x) for x in cur["role_flags"]]
        head_summary.append(
            {
                "role_group": cur["role_group"],
                "layer": int(cur["layer"]),
                "head": int(cur["head"]),
                "mean_intended_ratio": float(np.mean(intended_ratios)),
                "mean_cross_ratio": float(np.mean(cross_ratios)),
                "mean_margin": float(np.mean(np.array(intended_ratios) - np.array(cross_ratios))),
                "role_consistency_rate": float(np.mean([1.0 if x else 0.0 for x in role_flags])),
            }
        )
    head_summary = sorted(
        head_summary,
        key=lambda x: (float(x["role_consistency_rate"]), float(x["mean_margin"])),
        reverse=True,
    )

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "dtype": args.dtype,
            "local_files_only": bool(args.local_files_only),
            "router_json": str(router_path),
            "top_n_per_role": int(args.top_n),
            "selection_mode": selected.get("selection_mode"),
            "preference_threshold": float(args.preference_threshold),
            "prompt_bank_count": len(banks),
            "runtime_sec": float(time.time() - t0),
        },
        "selected_heads": {
            "lift1": [{"layer": int(a), "head": int(b)} for a, b in selected["lift1"]],
            "lift2": [{"layer": int(a), "head": int(b)} for a, b in selected["lift2"]],
        },
        "group_summary": group_summary,
        "head_summary": head_summary,
        "per_bank_rows": head_rows,
        "hypotheses": {
            "H1_lift1_heads_stable": bool(group_summary["lift1"]["role_consistency_rate"] >= 0.7),
            "H2_lift2_heads_stable": bool(group_summary["lift2"]["role_consistency_rate"] >= 0.7),
            "H3_intended_margin_positive": bool(
                group_summary["lift1"]["mean_margin"] > 0 and group_summary["lift2"]["mean_margin"] > 0
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["group_summary"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
