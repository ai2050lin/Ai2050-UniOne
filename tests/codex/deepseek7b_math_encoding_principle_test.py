#!/usr/bin/env python
"""
Mathematical Encoding Principle Test (MEPT) for DeepSeek-7B.

This test targets micro-mechanistic principles rather than plain statistics.
It evaluates whether concept encoding approximately obeys:
1) Attribute direction invariance across entities
2) Attribute composition additivity
3) Cross-entity transportability of attribute directions
4) Low-rank structure of attribute direction families
5) Causal controllability by direction-aligned neuron ablation
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def effective_rank(eigvals: np.ndarray) -> float:
    s = float(np.sum(eigvals))
    if s <= 0:
        return 0.0
    p = eigvals / s
    h = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(h))


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        missing = [i for i, x in enumerate(self.buffers) if x is None]
        if missing:
            raise RuntimeError(f"Missing layer activations: {missing}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


def load_model(model_id: str, dtype_name: str, local_files_only: bool):
    dtype = getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
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


def token_id(tok, token_text: str) -> int:
    ids = tok(token_text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Tokenization failed: {token_text}")
    return int(ids[0])


def margin_score(model, tok, prompt: str, pos_token: str, neg_token: str) -> float:
    out = run_prompt(model, tok, prompt)
    logits = out.logits[0, -1, :].float().cpu()
    pid = token_id(tok, pos_token)
    nid = token_id(tok, neg_token)
    return float((logits[pid] - logits[nid]).item())


def register_ablation(model, flat_indices: List[int], d_ff: int):
    by_layer: Dict[int, List[int]] = {}
    for idx in flat_indices:
        li = idx // d_ff
        ni = idx % d_ff
        by_layer.setdefault(li, []).append(ni)
    handles = []
    device = next(model.parameters()).device
    for li, idxs in by_layer.items():
        module = model.model.layers[li].mlp.gate_proj
        idx_tensor = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=device)

        def _mk(local_idx):
            def _hook(_module, _inputs, output):
                out = output.clone()
                out[..., local_idx] = 0.0
                return out

            return _hook

        handles.append(module.register_forward_hook(_mk(idx_tensor)))
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


@dataclass
class AttrSpec:
    name: str
    pos: str
    neg: str


def build_entities() -> List[str]:
    return [
        "apple", "banana", "orange", "grape", "pear", "peach",
        "rabbit", "cat", "dog", "horse",
        "car", "train", "chair", "tree",
    ]


def build_attrs() -> List[AttrSpec]:
    return [
        AttrSpec("color_red", "red", "green"),
        AttrSpec("taste_sweet", "sweet", "sour"),
        AttrSpec("weight_heavy", "heavy", "light"),
        AttrSpec("size_big", "big", "small"),
    ]


def prompt_base(entity: str) -> str:
    return f"The {entity} is"


def prompt_attr(entity: str, attr: str) -> str:
    return f"The {entity} is {attr}"


def prompt_attr_pair(entity: str, a1: str, a2: str) -> str:
    return f"The {entity} is {a1} and {a2}"


def topk_idx(v: np.ndarray, k: int) -> List[int]:
    k = min(k, v.shape[0])
    if k <= 0:
        return []
    idx = np.argpartition(np.abs(v), -k)[-k:]
    idx = idx[np.argsort(np.abs(v[idx]))[::-1]]
    return [int(i) for i in idx.tolist()]


def main():
    parser = argparse.ArgumentParser(description="Mathematical Encoding Principle Test")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--apple-focus", default="apple")
    parser.add_argument("--topk-ablate", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_math_principle_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        entities = build_entities()
        attrs = build_attrs()
        n_layers = len(model.model.layers)
        d_ff = model.model.layers[0].mlp.gate_proj.out_features
        total_dim = n_layers * d_ff

        # Cache activations
        base_vec: Dict[str, np.ndarray] = {}
        attr_vec: Dict[Tuple[str, str], np.ndarray] = {}
        for e in entities:
            collector.reset()
            _ = run_prompt(model, tok, prompt_base(e))
            base_vec[e] = collector.get_flat()
            for a in attrs:
                collector.reset()
                _ = run_prompt(model, tok, prompt_attr(e, a.pos))
                attr_vec[(e, a.name)] = collector.get_flat()

        # Attribute direction d(e,a)=v(e,a)-v_base(e)
        direction: Dict[Tuple[str, str], np.ndarray] = {}
        for e in entities:
            for a in attrs:
                direction[(e, a.name)] = attr_vec[(e, a.name)] - base_vec[e]

        # 1) Invariance across entities
        invariance = {}
        for a in attrs:
            sims = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    d1 = direction[(entities[i], a.name)]
                    d2 = direction[(entities[j], a.name)]
                    sims.append(cosine(d1, d2))
            invariance[a.name] = {
                "mean_cosine": float(np.mean(sims) if sims else 0.0),
                "std_cosine": float(np.std(sims) if sims else 0.0),
            }

        # 2) Additivity: v(e,a1,a2) ?= v_base(e)+d(e,a1)+d(e,a2)
        # Use one pair for tractability
        additivity = {}
        pair_cfg = [("color_red", "taste_sweet"), ("weight_heavy", "size_big")]
        for a1, a2 in pair_cfg:
            errs = []
            norms = []
            for e in entities:
                collector.reset()
                _ = run_prompt(model, tok, prompt_attr_pair(e, attrs[[x.name for x in attrs].index(a1)].pos, attrs[[x.name for x in attrs].index(a2)].pos))
                vab = collector.get_flat()
                vhat = base_vec[e] + direction[(e, a1)] + direction[(e, a2)]
                err = float(np.linalg.norm(vab - vhat))
                denom = float(np.linalg.norm(vab) + 1e-8)
                errs.append(err / denom)
                norms.append(err)
            additivity[f"{a1}+{a2}"] = {
                "mean_relative_error": float(np.mean(errs)),
                "std_relative_error": float(np.std(errs)),
                "mean_abs_error": float(np.mean(norms)),
            }

        # 3) Transportability: v(e2,a) ?= v_base(e2)+d(e1,a)
        transport = {}
        focus = args.apple_focus
        for a in attrs:
            errs = []
            for e in entities:
                if e == focus:
                    continue
                vhat = base_vec[e] + direction[(focus, a.name)]
                vgt = attr_vec[(e, a.name)]
                err = float(np.linalg.norm(vhat - vgt) / (np.linalg.norm(vgt) + 1e-8))
                errs.append(err)
            transport[a.name] = {
                "mean_relative_error": float(np.mean(errs) if errs else 0.0),
                "std_relative_error": float(np.std(errs) if errs else 0.0),
            }

        # 4) Low-rank: for each attr, matrix D_a(entity, features)
        lowrank = {}
        for a in attrs:
            D = np.stack([direction[(e, a.name)] for e in entities], axis=0).astype(np.float32)
            Dc = D - D.mean(axis=0, keepdims=True)
            # sample-cov eigen spectrum via SVD
            _, s, _ = np.linalg.svd(Dc, full_matrices=False)
            eig = (s**2) / max(D.shape[0] - 1, 1)
            if eig.size == 0:
                lowrank[a.name] = {"effective_rank": 0.0, "k95": 0}
            else:
                csum = np.cumsum(eig) / (np.sum(eig) + 1e-12)
                k95 = int(np.searchsorted(csum, 0.95) + 1)
                lowrank[a.name] = {
                    "effective_rank": effective_rank(eig),
                    "k95": k95,
                    "n_entities": int(D.shape[0]),
                }

        # 5) Causal controllability (apple-focused)
        causal = {}
        for a in attrs:
            d = direction[(focus, a.name)]
            idx = topk_idx(d, args.topk_ablate)
            # target prompt + control prompt
            target_prompt = prompt_base(focus)
            control_entity = "banana" if focus != "banana" else "rabbit"
            control_prompt = prompt_base(control_entity)
            base_margin_target = margin_score(model, tok, target_prompt, f" {a.pos}", f" {a.neg}")
            base_margin_ctrl = margin_score(model, tok, control_prompt, f" {a.pos}", f" {a.neg}")

            h = register_ablation(model, idx, d_ff)
            try:
                abl_margin_target = margin_score(model, tok, target_prompt, f" {a.pos}", f" {a.neg}")
                abl_margin_ctrl = margin_score(model, tok, control_prompt, f" {a.pos}", f" {a.neg}")
            finally:
                remove_handles(h)

            causal[a.name] = {
                "ablate_k": len(idx),
                "target_margin_before": base_margin_target,
                "target_margin_after": abl_margin_target,
                "target_margin_delta": abl_margin_target - base_margin_target,
                "control_margin_before": base_margin_ctrl,
                "control_margin_after": abl_margin_ctrl,
                "control_margin_delta": abl_margin_ctrl - base_margin_ctrl,
                "layer_distribution": dict(sorted(Counter([i // d_ff for i in idx]).items())),
                "indices": idx[:120],
            }

        # Principle score summary
        principle = {
            "invariance_mean": float(np.mean([invariance[a.name]["mean_cosine"] for a in attrs])),
            "additivity_error_mean": float(np.mean([additivity[k]["mean_relative_error"] for k in additivity])),
            "transport_error_mean": float(np.mean([transport[a.name]["mean_relative_error"] for a in attrs])),
            "lowrank_effective_rank_mean": float(np.mean([lowrank[a.name]["effective_rank"] for a in attrs])),
            "causal_target_delta_mean": float(np.mean([causal[a.name]["target_margin_delta"] for a in attrs])),
            "causal_control_delta_mean": float(np.mean([causal[a.name]["control_margin_delta"] for a in attrs])),
        }

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "apple_focus": args.apple_focus,
                "n_entities": len(entities),
                "n_attrs": len(attrs),
                "topk_ablate": args.topk_ablate,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "total_dim": total_dim,
            },
            "invariance": invariance,
            "additivity": additivity,
            "transport": transport,
            "lowrank": lowrank,
            "causal": causal,
            "principle_summary": principle,
        }

        json_path = out_dir / "math_encoding_principle_results.json"
        md_path = out_dir / "MATH_ENCODING_PRINCIPLE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# 数学编码原理测试报告 (MEPT)", ""]
        lines.append("## 1) 原理指标总览")
        for k, v in principle.items():
            lines.append(f"- {k}: {v:.6f}")
        lines.append("")
        lines.append("## 2) 属性方向不变性")
        for a in attrs:
            x = invariance[a.name]
            lines.append(f"- {a.name}: mean_cos={x['mean_cosine']:.6f}, std={x['std_cosine']:.6f}")
        lines.append("")
        lines.append("## 3) 可加性误差")
        for k, v in additivity.items():
            lines.append(f"- {k}: mean_rel_err={v['mean_relative_error']:.6f}, std={v['std_relative_error']:.6f}")
        lines.append("")
        lines.append("## 4) 传输性误差 (apple->others)")
        for a in attrs:
            x = transport[a.name]
            lines.append(f"- {a.name}: mean_rel_err={x['mean_relative_error']:.6f}, std={x['std_relative_error']:.6f}")
        lines.append("")
        lines.append("## 5) 低秩性")
        for a in attrs:
            x = lowrank[a.name]
            lines.append(f"- {a.name}: effective_rank={x['effective_rank']:.4f}, k95={x['k95']}")
        lines.append("")
        lines.append("## 6) 因果可控性 (apple对齐)")
        for a in attrs:
            x = causal[a.name]
            lines.append(
                f"- {a.name}: target_delta={x['target_margin_delta']:+.6f}, control_delta={x['control_margin_delta']:+.6f}, layers={x['layer_distribution']}"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(
            f"[OK] principle: inv={principle['invariance_mean']:.4f}, add_err={principle['additivity_error_mean']:.4f}, trans_err={principle['transport_error_mean']:.4f}, eff_rank={principle['lowrank_effective_rank_mean']:.4f}"
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
