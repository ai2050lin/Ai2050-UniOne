#!/usr/bin/env python
"""
Real-model apple sweetness channel-edit probe.

Goal:
- Test whether a small number of channel edits can reverse:
  red-apple sweeter-than-green-apple preference,
  while preserving anchor facts.

Method:
1) Use layer MLP output channels as editable units.
2) Rank channels by |mean(red) - mean(green)| at last token.
3) Apply global channel scaling on Top-k channels:
      y[..., idx] <- scale * y[..., idx]
4) Evaluate target-pair reversal rate + anchor sign retention.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def get_mlp_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "mlp"):
        return layer.mlp
    # fallback: choose the first large submodule
    for sub in layer.modules():
        if isinstance(sub, torch.nn.Module) and sub is not layer:
            return sub
    raise RuntimeError("Cannot find mlp module in layer")


def load_model(model_id: str, dtype_name: str, local_only: bool):
    if local_only:
        # Must be set before importing transformers in this process.
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


def token_id_single(tok, piece: str) -> int:
    ids = tok.encode(piece, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Piece {piece!r} is not single token: {ids}")
    return int(ids[0])


def run_scores(
    model,
    tok,
    prompts: List[str],
    sweet_id: int,
    sour_id: int,
    hook_layer: int | None = None,
    hook_idx: np.ndarray | None = None,
    hook_scale: float = 1.0,
) -> np.ndarray:
    layers = discover_layers(model)
    handle = None
    if hook_layer is not None and hook_idx is not None and hook_idx.size > 0:
        idx_t = torch.tensor(hook_idx.tolist(), dtype=torch.long)

        def fn(_module, _inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            y2 = y.clone()
            y2[..., idx_t] = y2[..., idx_t] * float(hook_scale)
            return y2

        handle = get_mlp_module(layers[int(hook_layer)]).register_forward_hook(fn)

    try:
        with torch.inference_mode():
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits
            attn = enc["attention_mask"]
            last_pos = attn.sum(dim=1) - 1
            row = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[row, last_pos, :]
            s = last_logits[:, sweet_id] - last_logits[:, sour_id]
            return s.detach().float().cpu().numpy().astype(np.float64)
    finally:
        if handle is not None:
            handle.remove()


def collect_layer_vectors(
    model,
    tok,
    prompts: List[str],
    layer_idx: int,
) -> np.ndarray:
    layers = discover_layers(model)
    bank = {"vec": None}

    def fn(_module, _inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        # [B, T, D] -> last token channel vector per sample
        bank["vec"] = y[:, -1, :].detach().float().cpu().numpy().astype(np.float64)

    handle = get_mlp_module(layers[int(layer_idx)]).register_forward_hook(fn)
    try:
        with torch.inference_mode():
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            _ = model(**enc, use_cache=False, return_dict=True)
        if bank["vec"] is None:
            raise RuntimeError("Failed to capture layer vectors")
        return bank["vec"]
    finally:
        handle.remove()


def build_prompts() -> Dict[str, List[str]]:
    target_red = [
        "A ripe red apple tastes",
        "A juicy red apple tastes",
        "A mature red apple tastes",
    ]
    target_green = [
        "An unripe green apple tastes",
        "A tart green apple tastes",
        "A raw green apple tastes",
    ]
    anchors = [
        "A ripe banana tastes",
        "A fresh lemon tastes",
        "A green grape tastes",
        "A red strawberry tastes",
        "A potato tastes",
        "Ocean water tastes",
    ]
    return {"target_red": target_red, "target_green": target_green, "anchors": anchors}


def evaluate_config(
    base_target_red: np.ndarray,
    base_target_green: np.ndarray,
    base_anchors: np.ndarray,
    new_target_red: np.ndarray,
    new_target_green: np.ndarray,
    new_anchors: np.ndarray,
) -> Dict[str, float]:
    base_gap = float(np.mean(base_target_red) - np.mean(base_target_green))
    new_gap = float(np.mean(new_target_red) - np.mean(new_target_green))
    pair_n = min(len(new_target_red), len(new_target_green))
    pair_new_order = new_target_red[:pair_n] < new_target_green[:pair_n]
    pair_base_order = base_target_red[:pair_n] < base_target_green[:pair_n]
    pair_flip_rate = float(np.mean(pair_new_order != pair_base_order)) if pair_n > 0 else 0.0

    bsign = np.where(base_anchors >= 0.0, 1, -1)
    nsign = np.where(new_anchors >= 0.0, 1, -1)
    anchor_retention = float(np.mean(bsign == nsign)) if len(anchors := base_anchors) > 0 else 0.0

    # true relative reversal from baseline direction
    if abs(base_gap) > 1e-8:
        gap_reversed_from_base = bool(base_gap * new_gap < 0.0)
    else:
        gap_reversed_from_base = bool(abs(new_gap) > 0.05)

    return {
        "base_gap": base_gap,
        "new_gap": new_gap,
        "gap_reversed_from_base": gap_reversed_from_base,
        "pair_flip_rate_from_base": pair_flip_rate,
        "anchor_retention": anchor_retention,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-model apple sweetness channel edit")
    ap.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--min-layer-tail", type=int, default=6, help="scan last N layers")
    ap.add_argument("--max-layer-candidates", type=int, default=4, help="pick top layers by |gap channel mass|")
    ap.add_argument("--k-list", type=str, default="4,8,16,32,64,128")
    ap.add_argument("--scales", type=str, default="0.0,-0.5,-1.0")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_model_apple_sweetness_channel_edit_20260307.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    layers = discover_layers(model)
    n_layers = len(layers)

    sweet_id = token_id_single(tok, " sweet")
    sour_id = token_id_single(tok, " sour")
    prompts = build_prompts()

    base_target_red = run_scores(model, tok, prompts["target_red"], sweet_id, sour_id)
    base_target_green = run_scores(model, tok, prompts["target_green"], sweet_id, sour_id)
    base_anchors = run_scores(model, tok, prompts["anchors"], sweet_id, sour_id)

    # Layer ranking by red-green activation difference mass.
    tail = max(1, int(args.min_layer_tail))
    layer_pool = list(range(max(0, n_layers - tail), n_layers))
    layer_mass: List[Tuple[int, float, np.ndarray]] = []
    for l in layer_pool:
        red_vec = collect_layer_vectors(model, tok, prompts["target_red"], l)
        green_vec = collect_layer_vectors(model, tok, prompts["target_green"], l)
        diff = np.mean(red_vec, axis=0) - np.mean(green_vec, axis=0)
        mass = float(np.linalg.norm(diff, ord=1))
        layer_mass.append((l, mass, diff))
    layer_mass.sort(key=lambda x: x[1], reverse=True)
    cand = layer_mass[: max(1, int(args.max_layer_candidates))]

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    scale_list = [float(x) for x in args.scales.split(",") if x.strip()]

    trials = []
    best = None
    strong_pair_threshold = (2.0 / 3.0) - 1e-9
    for l, _mass, diff in cand:
        order = np.argsort(np.abs(diff))[::-1]
        dim = diff.shape[0]
        for k in k_list:
            kk = min(max(1, int(k)), dim)
            idx = order[:kk]
            for sc in scale_list:
                new_red = run_scores(
                    model,
                    tok,
                    prompts["target_red"],
                    sweet_id,
                    sour_id,
                    hook_layer=l,
                    hook_idx=idx,
                    hook_scale=sc,
                )
                new_green = run_scores(
                    model,
                    tok,
                    prompts["target_green"],
                    sweet_id,
                    sour_id,
                    hook_layer=l,
                    hook_idx=idx,
                    hook_scale=sc,
                )
                new_anchors = run_scores(
                    model,
                    tok,
                    prompts["anchors"],
                    sweet_id,
                    sour_id,
                    hook_layer=l,
                    hook_idx=idx,
                    hook_scale=sc,
                )
                met = evaluate_config(
                    base_target_red=base_target_red,
                    base_target_green=base_target_green,
                    base_anchors=base_anchors,
                    new_target_red=new_red,
                    new_target_green=new_green,
                    new_anchors=new_anchors,
                )
                row = {
                    "layer": int(l),
                    "k": int(kk),
                    "scale": float(sc),
                    **met,
                }
                strong = bool(row["gap_reversed_from_base"]) and float(row["pair_flip_rate_from_base"]) >= strong_pair_threshold
                soft = bool(row["gap_reversed_from_base"]) or float(row["pair_flip_rate_from_base"]) >= strong_pair_threshold
                row["target_reversal_soft"] = soft
                row["target_reversal_strong"] = strong
                trials.append(row)
                score = (
                    (1.0 if strong else 0.0),
                    (1.0 if soft else 0.0),
                    row["pair_flip_rate_from_base"],
                    row["anchor_retention"],
                    -row["k"],
                )
                if best is None or score > best["score"]:
                    best = {"score": score, "row": row}

    # minimal k for strong / soft reversal under retention threshold.
    feasible_strong = [
        r
        for r in trials
        if bool(r["target_reversal_strong"]) and float(r["anchor_retention"]) >= 0.8
    ]
    feasible_soft = [
        r
        for r in trials
        if bool(r["target_reversal_soft"]) and float(r["anchor_retention"]) >= 0.8
    ]
    min_k_feasible_strong = min((int(r["k"]) for r in feasible_strong), default=None)
    min_k_feasible_soft = min((int(r["k"]) for r in feasible_soft), default=None)

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 3),
            "model_id": args.model_id,
            "n_layers": n_layers,
            "sweet_token_id": sweet_id,
            "sour_token_id": sour_id,
        },
        "base": {
            "red_mean": float(np.mean(base_target_red)),
            "green_mean": float(np.mean(base_target_green)),
            "base_gap_red_minus_green": float(np.mean(base_target_red) - np.mean(base_target_green)),
            "anchor_signs": [int(x) for x in np.where(base_anchors >= 0.0, 1, -1).tolist()],
        },
        "layer_candidates": [
            {"layer": int(l), "l1_diff_mass": float(m)} for l, m, _ in cand
        ],
        "search_space": {
            "k_list": k_list,
            "scales": scale_list,
            "strong_pair_threshold": strong_pair_threshold,
            "criteria": "maximize (gap_reversed, pair_reversal_rate, anchor_retention, -k)",
        },
        "best": best["row"] if best is not None else None,
        "min_k_reversal_anchor80_soft": min_k_feasible_soft,
        "min_k_reversal_anchor80_strong": min_k_feasible_strong,
        "trials": trials,
        "interpretation": [
            "This is a channel-level causal intervention, not a full weight-edit method like ROME/MEMIT.",
            "If low-k already reverses target with high retention, representation is locally editable.",
            "If only high-k works (or no feasible point), knowledge is strongly distributed at this readout path.",
        ],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    if result["best"] is not None:
        print(
            json.dumps(
                {
                    "base_gap": result["base"]["base_gap_red_minus_green"],
                    "best": result["best"],
                    "min_k_reversal_anchor80_soft": result["min_k_reversal_anchor80_soft"],
                    "min_k_reversal_anchor80_strong": result["min_k_reversal_anchor80_strong"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
