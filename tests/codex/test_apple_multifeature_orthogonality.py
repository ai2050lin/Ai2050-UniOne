#!/usr/bin/env python
"""
Apple multi-feature orthogonality probe.

Goal:
- Test whether apple-related features (color / size / text-form / sound) are represented
  as decoupled (near-orthogonal) directions in a deep neural network hidden space.
- Use real model activations from per-layer MLP probe modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DimPairs = Dict[str, List[Dict[str, str]]]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x)
    return x / n


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(int(x) for x in a.tolist())
    sb = set(int(x) for x in b.tolist())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def topk_indices(v: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 0), int(v.shape[0]))
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(v, -k)[-k:]
    idx = idx[np.argsort(v[idx])[::-1]]
    return idx.astype(np.int64)


def first_existing_attr(obj, candidates: List[str]):
    for name in candidates:
        cur = obj
        ok = True
        for p in name.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            return cur
    return None


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers for this model")


def discover_probe_module(layer: torch.nn.Module) -> torch.nn.Module:
    m = first_existing_attr(layer, ["mlp.gate_proj", "mlp.up_proj", "mlp.fc1", "mlp.c_fc"])
    if isinstance(m, torch.nn.Module):
        return m

    if hasattr(layer, "mlp"):
        for sub in layer.mlp.modules():
            if isinstance(sub, torch.nn.Linear):
                return sub

    for sub in layer.modules():
        if isinstance(sub, torch.nn.Linear):
            return sub

    raise RuntimeError("No probe-able linear module found in layer")


class LayerCollector:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.buffers: List[np.ndarray | None] = [None for _ in self.layers]
        self.handles = []
        for i, layer in enumerate(self.layers):
            probe = discover_probe_module(layer)
            self.handles.append(probe.register_forward_hook(self._hook(i)))

    def _hook(self, i: int):
        def fn(_module, _inputs, output):
            x = output
            if isinstance(x, (tuple, list)):
                x = x[0]
            if not torch.is_tensor(x):
                raise RuntimeError(f"Hook output is not tensor at layer {i}")
            if x.ndim >= 3:
                vec = x[0, -1, :]
            elif x.ndim == 2:
                vec = x[-1, :]
            else:
                vec = x.reshape(-1)
            self.buffers[i] = vec.detach().float().cpu().numpy().astype(np.float32)

        return fn

    def reset(self) -> None:
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing layer probes: {miss}")
        return np.concatenate([x for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def build_pairs() -> DimPairs:
    color = [
        {"id": "color_1", "a": "The apple is red.", "b": "The apple is green."},
        {"id": "color_2", "a": "I picked a bright red apple.", "b": "I picked a bright green apple."},
        {"id": "color_3", "a": "This apple has crimson skin.", "b": "This apple has pale green skin."},
        {"id": "color_4", "a": "A red apple is on the table.", "b": "A yellow apple is on the table."},
        {"id": "color_5", "a": "The basket contains red apples.", "b": "The basket contains green apples."},
        {"id": "color_6", "a": "Its peel is ruby red.", "b": "Its peel is lime green."},
    ]

    size = [
        {"id": "size_1", "a": "The apple is small.", "b": "The apple is large."},
        {"id": "size_2", "a": "I bought a tiny apple.", "b": "I bought a huge apple."},
        {"id": "size_3", "a": "That apple looks compact.", "b": "That apple looks oversized."},
        {"id": "size_4", "a": "A small apple fits in my palm.", "b": "A large apple fills my hand."},
        {"id": "size_5", "a": "This is a mini apple.", "b": "This is a giant apple."},
        {"id": "size_6", "a": "The apple has a petite shape.", "b": "The apple has a bulky shape."},
    ]

    text_form = [
        {"id": "text_1", "a": "The word is apple.", "b": "The word is APPLE."},
        {"id": "text_2", "a": "Spell it as a-p-p-l-e.", "b": "Spell it as A-P-P-L-E."},
        {"id": "text_3", "a": "Write apple in lowercase.", "b": "Write APPLE in uppercase."},
        {"id": "text_4", "a": "In plain text: apple.", "b": "In plain text: APPLE."},
        {"id": "text_5", "a": "The token is 'apple'.", "b": "The token is 'APPLE'."},
        {"id": "text_6", "a": "Use the string apple.", "b": "Use the string APPLE."},
    ]

    sound = [
        {"id": "sound_1", "a": "Biting the apple makes a crisp crunch sound.", "b": "Biting the apple is almost silent."},
        {"id": "sound_2", "a": "The apple crackles when chewed.", "b": "The apple makes no chewing sound."},
        {"id": "sound_3", "a": "The bite produces a sharp crunch.", "b": "The bite produces a soft mute."},
        {"id": "sound_4", "a": "You can hear a crunchy snap from the apple.", "b": "You hear nearly nothing from the apple."},
        {"id": "sound_5", "a": "The apple is audibly crisp.", "b": "The apple is acoustically quiet."},
        {"id": "sound_6", "a": "Its texture sounds crunchy.", "b": "Its texture sounds muted."},
    ]

    return {"color": color, "size": size, "text": text_form, "sound": sound}


def load_model(model_id: str, dtype_name: str, local_files_only: bool):
    # Force offline behavior to avoid any hub/network probe.
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def run_prompt(model, tok, text: str):
    device = next(model.parameters()).device
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        return model(**inp, use_cache=False, return_dict=True)


def principal_similarity(a: np.ndarray, b: np.ndarray, rank: int) -> float:
    # a/b: [n_samples, d]
    ra = min(rank, a.shape[0], a.shape[1])
    rb = min(rank, b.shape[0], b.shape[1])
    if ra <= 0 or rb <= 0:
        return 0.0
    ua, _, vta = np.linalg.svd(a, full_matrices=False)
    ub, _, vtb = np.linalg.svd(b, full_matrices=False)
    ba = vta[:ra].T
    bb = vtb[:rb].T
    qa, _ = np.linalg.qr(ba)
    qb, _ = np.linalg.qr(bb)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.max(s)) if s.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Apple multi-feature orthogonality probe")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=160)
    ap.add_argument("--subspace-rank", type=int, default=4)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = LayerCollector(model)

    dims = ["color", "size", "text", "sound"]
    pairs = build_pairs()

    diff_map: Dict[str, List[np.ndarray]] = {k: [] for k in dims}
    for dim in dims:
        for row in pairs[dim]:
            collector.reset()
            run_prompt(model, tok, row["a"])
            va = collector.get_flat()

            collector.reset()
            run_prompt(model, tok, row["b"])
            vb = collector.get_flat()

            diff_map[dim].append((va - vb).astype(np.float32))

    mean_dir: Dict[str, np.ndarray] = {}
    signatures: Dict[str, np.ndarray] = {}
    dim_stats: Dict[str, Dict[str, float]] = {}
    mat_map: Dict[str, np.ndarray] = {}

    for dim in dims:
        mat = np.stack(diff_map[dim], axis=0).astype(np.float32)
        mat_map[dim] = mat
        m = np.mean(mat, axis=0)
        mean_dir[dim] = l2_normalize(m)
        signatures[dim] = topk_indices(np.abs(m), args.top_k)

        align = [abs(cosine(x, mean_dir[dim])) for x in mat]
        dim_stats[dim] = {
            "n_pairs": float(mat.shape[0]),
            "mean_abs_alignment_to_axis": float(np.mean(align)),
            "std_abs_alignment_to_axis": float(np.std(align)),
        }

    pairwise = []
    abs_cos_vals = []
    jacc_vals = []
    psub_vals = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            di, dj = dims[i], dims[j]
            c = cosine(mean_dir[di], mean_dir[dj])
            ac = abs(c)
            jac = jaccard(signatures[di], signatures[dj])
            ps = principal_similarity(mat_map[di], mat_map[dj], rank=args.subspace_rank)
            pairwise.append(
                {
                    "pair": f"{di}__{dj}",
                    "cosine": float(c),
                    "abs_cosine": float(ac),
                    "signature_jaccard": float(jac),
                    "principal_similarity": float(ps),
                    "principal_orthogonality": float(1.0 - ps),
                }
            )
            abs_cos_vals.append(ac)
            jacc_vals.append(jac)
            psub_vals.append(ps)

    # Axis identifiability: each sample diff should align to its own axis more than others.
    total = 0
    correct = 0
    for dim in dims:
        for x in mat_map[dim]:
            sims = [abs(cosine(x, mean_dir[d])) for d in dims]
            pred = dims[int(np.argmax(np.array(sims)))]
            total += 1
            if pred == dim:
                correct += 1
    axis_ident_acc = float(correct / max(1, total))

    # Composition check A: projection of composed delta onto learned axis means.
    base_prompt = "A small green apple is silent. Write apple."
    composed_prompt = "A large red apple is crunchy. Write APPLE."
    collector.reset()
    run_prompt(model, tok, base_prompt)
    v_base = collector.get_flat()

    collector.reset()
    run_prompt(model, tok, composed_prompt)
    v_full = collector.get_flat()

    delta = (v_full - v_base).astype(np.float32)
    V = np.stack([mean_dir[d] for d in dims], axis=1).astype(np.float32)  # [D, 4]
    coef, *_ = np.linalg.lstsq(V, delta, rcond=None)
    recon = (V @ coef).astype(np.float32)

    num = float(np.sum((delta - recon) ** 2))
    den = float(np.sum(delta**2))
    r2 = float(1.0 - num / den) if den > 1e-12 else 0.0
    recon_cos = float(cosine(delta, recon))

    # Composition check B: controlled additive decomposition under identical context.
    prompts = {
        "base": "A small green apple is silent. Write apple.",
        "color_only": "A small red apple is silent. Write apple.",
        "size_only": "A large green apple is silent. Write apple.",
        "sound_only": "A small green apple is crunchy. Write apple.",
        "text_only": "A small green apple is silent. Write APPLE.",
        "full": "A large red apple is crunchy. Write APPLE.",
    }
    vecs: Dict[str, np.ndarray] = {}
    for k, p in prompts.items():
        collector.reset()
        run_prompt(model, tok, p)
        vecs[k] = collector.get_flat()

    d_color = (vecs["color_only"] - vecs["base"]).astype(np.float32)
    d_size = (vecs["size_only"] - vecs["base"]).astype(np.float32)
    d_sound = (vecs["sound_only"] - vecs["base"]).astype(np.float32)
    d_text = (vecs["text_only"] - vecs["base"]).astype(np.float32)
    d_full = (vecs["full"] - vecs["base"]).astype(np.float32)
    d_sum = (d_color + d_size + d_sound + d_text).astype(np.float32)

    num2 = float(np.sum((d_full - d_sum) ** 2))
    den2 = float(np.sum(d_full**2))
    add_r2 = float(1.0 - num2 / den2) if den2 > 1e-12 else 0.0
    add_cos = float(cosine(d_full, d_sum))

    metrics = {
        "mean_abs_pairwise_cosine": float(np.mean(abs_cos_vals)) if abs_cos_vals else 0.0,
        "mean_signature_jaccard": float(np.mean(jacc_vals)) if jacc_vals else 0.0,
        "mean_principal_similarity": float(np.mean(psub_vals)) if psub_vals else 0.0,
        "mean_principal_orthogonality": float(1.0 - np.mean(psub_vals)) if psub_vals else 0.0,
        "axis_identifiability_accuracy": axis_ident_acc,
        "compositional_r2": r2,
        "compositional_recon_cosine": recon_cos,
        "controlled_additive_r2": add_r2,
        "controlled_additive_cosine": add_cos,
    }

    hypotheses = {
        "H1_axis_decoupling": bool(
            metrics["mean_abs_pairwise_cosine"] < 0.35
            and metrics["mean_signature_jaccard"] < 0.25
            and metrics["axis_identifiability_accuracy"] > 0.60
        ),
        "H2_subspace_near_orthogonal": bool(metrics["mean_principal_orthogonality"] > 0.35),
        "H3_linear_composition_exists": bool(
            metrics["compositional_r2"] > 0.45 and metrics["compositional_recon_cosine"] > 0.70
        ),
        "H4_controlled_additivity": bool(
            metrics["controlled_additive_r2"] > 0.35 and metrics["controlled_additive_cosine"] > 0.65
        ),
    }

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "dims": dims,
        "dimension_stats": dim_stats,
        "pairwise": pairwise,
        "metrics": metrics,
        "composition": {
            "coefficients": {dims[i]: float(coef[i]) for i in range(len(dims))},
            "r2": r2,
            "recon_cosine": recon_cos,
            "controlled_additive_r2": add_r2,
            "controlled_additive_cosine": add_cos,
        },
        "hypotheses": hypotheses,
        "elapsed_sec": float(time.time() - t0),
        "notes": [
            "This test provides evidence of decoupled feature axes, not a proof of strict global orthogonality.",
            "Text-only model probes sound via language descriptions (not raw audio signal).",
        ],
    }

    collector.close()

    print("=== Apple Multi-Feature Orthogonality Probe ===")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print("hypotheses:", result["hypotheses"])

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
