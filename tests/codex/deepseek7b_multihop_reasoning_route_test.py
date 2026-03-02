#!/usr/bin/env python
"""
Multi-hop reasoning route test (A->B->C) for DeepSeek 7B.

Goal:
- Build a 1-hop/2-hop/3-hop benchmark with matched invalid controls.
- Identify route-like neurons from MLP gate activations.
- Find a minimal causal subset whose ablation suppresses 3-hop selectivity.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Chain:
    a: str
    b: str
    c: str
    d: str


@dataclass
class QueryItem:
    prompt: str
    target: str
    hop: int
    valid: bool
    chain_id: int


def load_model(model_id: str, dtype_name: str, local_files_only: bool, device_map: str):
    dtype = getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
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
            raise RuntimeError(f"Missing hook outputs: {missing}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


def build_chains() -> List[Chain]:
    return [
        Chain("apple", "fruit", "plant", "alive"),
        Chain("banana", "fruit", "plant", "alive"),
        Chain("rabbit", "animal", "mammal", "alive"),
        Chain("dog", "animal", "mammal", "alive"),
        Chain("car", "vehicle", "machine", "object"),
        Chain("train", "vehicle", "machine", "object"),
        Chain("sun", "star", "celestial", "object"),
        Chain("moon", "satellite", "celestial", "object"),
    ]


def build_queries(chains: List[Chain]) -> List[QueryItem]:
    items: List[QueryItem] = []
    for i, ch in enumerate(chains):
        p1 = f"Facts: {ch.a} is a {ch.b}. Question: {ch.a} is a"
        p2 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. Question: {ch.a} is a"
        p3 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. {ch.c} is {ch.d}. Question: {ch.a} is"
        items.append(QueryItem(p1, f" {ch.b}", 1, True, i))
        items.append(QueryItem(p2, f" {ch.c}", 2, True, i))
        items.append(QueryItem(p3, f" {ch.d}", 3, True, i))

    for i, ch in enumerate(chains):
        j = (i + 3) % len(chains)
        ch2 = chains[j]
        p1 = f"Facts: {ch.a} is a {ch.b}. Question: {ch.a} is a"
        p2 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. Question: {ch.a} is a"
        p3 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. {ch.c} is {ch.d}. Question: {ch.a} is"
        items.append(QueryItem(p1, f" {ch2.b}", 1, False, i))
        items.append(QueryItem(p2, f" {ch2.c}", 2, False, i))
        items.append(QueryItem(p3, f" {ch2.d}", 3, False, i))
    return items


def eval_items_batch(
    model,
    tok,
    items: List[QueryItem],
    target_ids: List[int],
    batch_size: int,
) -> Dict[int, float]:
    device = next(model.parameters()).device
    out: Dict[int, float] = {}
    for st in range(0, len(items), batch_size):
        ed = min(st + batch_size, len(items))
        batch = items[st:ed]
        prompts = [x.prompt for x in batch]
        inp = tok(prompts, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            logits = model(**inp, use_cache=False, return_dict=True).logits.float()
        attn = inp["attention_mask"]
        last_pos = attn.sum(dim=1) - 1
        rows = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[rows, last_pos, :]
        probs = torch.softmax(last_logits, dim=-1)
        tids = torch.tensor(target_ids[st:ed], dtype=torch.long, device=probs.device)
        vals = probs[rows, tids].detach().cpu().tolist()
        for i, v in enumerate(vals):
            out[st + i] = float(v)
    return out


def group_metrics(items: List[QueryItem], vals: Dict[int, float]) -> Dict[str, float]:
    out = {}
    for hop in [1, 2, 3]:
        v_valid = [vals[i] for i, it in enumerate(items) if it.hop == hop and it.valid]
        v_invalid = [vals[i] for i, it in enumerate(items) if it.hop == hop and not it.valid]
        m_valid = float(np.mean(v_valid) if v_valid else 0.0)
        m_invalid = float(np.mean(v_invalid) if v_invalid else 0.0)
        out[f"hop{hop}_valid"] = m_valid
        out[f"hop{hop}_invalid"] = m_invalid
        out[f"hop{hop}_selectivity"] = m_valid - m_invalid
    out["route_index"] = out["hop3_selectivity"] - out["hop1_selectivity"]
    return out


def register_ablation(model, flat_indices: List[int], d_ff: int):
    by_layer = defaultdict(list)
    for idx in flat_indices:
        by_layer[idx // d_ff].append(idx % d_ff)
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


def discover_candidates(
    model,
    tok,
    collector: GateCollector,
    items: List[QueryItem],
    top_k: int,
) -> List[Dict]:
    total_dim = len(model.model.layers) * model.model.layers[0].mlp.gate_proj.out_features
    sum_h3v = np.zeros(total_dim, dtype=np.float64)
    sum_h1v = np.zeros(total_dim, dtype=np.float64)
    sum_h3i = np.zeros(total_dim, dtype=np.float64)
    n_h3v = n_h1v = n_h3i = 0

    for it in items:
        collector.reset()
        _ = run_prompt(model, tok, it.prompt)
        v = collector.get_flat()
        if it.hop == 3 and it.valid:
            sum_h3v += v
            n_h3v += 1
        if it.hop == 1 and it.valid:
            sum_h1v += v
            n_h1v += 1
        if it.hop == 3 and not it.valid:
            sum_h3i += v
            n_h3i += 1

    mu_h3v = sum_h3v / max(n_h3v, 1)
    mu_h1v = sum_h1v / max(n_h1v, 1)
    mu_h3i = sum_h3i / max(n_h3i, 1)
    score = (mu_h3v - mu_h1v) + (mu_h3v - mu_h3i)
    k = min(top_k, score.shape[0])
    idx = np.argpartition(score, -k)[-k:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return [{"flat_idx": int(i), "score": float(score[i])} for i in idx.tolist()]


def single_neuron_utilities(
    model,
    tok,
    d_ff: int,
    items: List[QueryItem],
    target_ids: List[int],
    batch_size: int,
    candidates: List[Dict],
    route_penalty: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    base_vals = eval_items_batch(model, tok, items, target_ids, batch_size)
    base_metrics = group_metrics(items, base_vals)

    rows = []
    for c in candidates:
        idx = c["flat_idx"]
        h = register_ablation(model, [idx], d_ff)
        try:
            cur_vals = eval_items_batch(model, tok, items, target_ids, batch_size)
            cur_metrics = group_metrics(items, cur_vals)
        finally:
            remove_handles(h)

        drop_h3 = base_metrics["hop3_selectivity"] - cur_metrics["hop3_selectivity"]
        drop_h1 = base_metrics["hop1_selectivity"] - cur_metrics["hop1_selectivity"]
        utility = drop_h3 - route_penalty * abs(drop_h1)
        rows.append(
            {
                **c,
                "drop_h3_selectivity": float(drop_h3),
                "drop_h1_selectivity": float(drop_h1),
                "utility": float(utility),
            }
        )
    rows.sort(key=lambda x: x["utility"], reverse=True)
    return rows, base_metrics


def eval_subset(
    model,
    tok,
    d_ff: int,
    items: List[QueryItem],
    target_ids: List[int],
    batch_size: int,
    subset: List[Dict],
) -> Dict[str, float]:
    if not subset:
        vals = eval_items_batch(model, tok, items, target_ids, batch_size)
        return group_metrics(items, vals)
    idxs = [x["flat_idx"] for x in subset]
    h = register_ablation(model, idxs, d_ff)
    try:
        vals = eval_items_batch(model, tok, items, target_ids, batch_size)
    finally:
        remove_handles(h)
    return group_metrics(items, vals)


def minimal_subset(
    model,
    tok,
    d_ff: int,
    items: List[QueryItem],
    target_ids: List[int],
    batch_size: int,
    ranked: List[Dict],
    max_subset: int,
    target_ratio: float,
):
    n = min(max_subset, len(ranked))
    if n == 0:
        base = eval_subset(model, tok, d_ff, items, target_ids, batch_size, [])
        return [], base, {"best_drop": 0.0, "goal": 0.0, "achieved": 0.0}

    base = eval_subset(model, tok, d_ff, items, target_ids, batch_size, [])
    drops = []
    mets = []
    for i in range(1, n + 1):
        sub = ranked[:i]
        m = eval_subset(model, tok, d_ff, items, target_ids, batch_size, sub)
        d = base["hop3_selectivity"] - m["hop3_selectivity"]
        drops.append(d)
        mets.append(m)

    best_i = int(np.argmax(drops))
    best_drop = float(drops[best_i])
    if best_drop <= 0:
        return [], base, {"best_drop": best_drop, "goal": 0.0, "achieved": 0.0}

    goal = best_drop * target_ratio
    chosen_i = best_i
    for i, d in enumerate(drops):
        if d >= goal:
            chosen_i = i
            break
    return ranked[: chosen_i + 1], mets[chosen_i], {
        "best_drop": best_drop,
        "goal": goal,
        "achieved": float(drops[chosen_i]),
    }


def main():
    parser = argparse.ArgumentParser(description="A->B->C multi-hop reasoning route test")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chains", type=int, default=0)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--max-subset", type=int, default=12)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--route-penalty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_multihop_route_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only, args.device_map)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    collector = GateCollector(model)
    try:
        chains = build_chains()
        if args.max_chains > 0:
            chains = chains[: args.max_chains]
        items = build_queries(chains)
        target_ids = [token_id(tok, it.target) for it in items]

        candidates = discover_candidates(model, tok, collector, items, top_k=args.candidate_k)
        ranked, base_metrics = single_neuron_utilities(
            model=model,
            tok=tok,
            d_ff=d_ff,
            items=items,
            target_ids=target_ids,
            batch_size=args.batch_size,
            candidates=candidates,
            route_penalty=args.route_penalty,
        )
        subset, subset_metrics, progress = minimal_subset(
            model=model,
            tok=tok,
            d_ff=d_ff,
            items=items,
            target_ids=target_ids,
            batch_size=args.batch_size,
            ranked=ranked,
            max_subset=args.max_subset,
            target_ratio=args.target_ratio,
        )

        idxs = [x["flat_idx"] for x in subset]
        layer_dist = dict(sorted(Counter([i // d_ff for i in idxs]).items()))

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "device_map": args.device_map,
                "batch_size": args.batch_size,
                "candidate_k": args.candidate_k,
                "max_subset": args.max_subset,
                "target_ratio": args.target_ratio,
                "route_penalty": args.route_penalty,
                "n_chains": len(chains),
                "n_queries": len(items),
            },
            "base_metrics": base_metrics,
            "candidates_layer_distribution": dict(sorted(Counter([x["flat_idx"] // d_ff for x in candidates]).items())),
            "ranked_single": ranked,
            "minimal_subset": {
                "size": len(subset),
                "layer_distribution": layer_dist,
                "neurons": [
                    {
                        "layer": int(x["flat_idx"] // d_ff),
                        "neuron": int(x["flat_idx"] % d_ff),
                        "flat_idx": int(x["flat_idx"]),
                        "utility": float(x["utility"]),
                        "drop_h3_selectivity": float(x["drop_h3_selectivity"]),
                        "drop_h1_selectivity": float(x["drop_h1_selectivity"]),
                    }
                    for x in subset
                ],
                "metrics_after_ablation": subset_metrics,
                "progress": progress,
            },
        }

        json_path = out_dir / "multihop_route_results.json"
        md_path = out_dir / "MULTIHOP_ROUTE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# Multi-hop Reasoning Route Report (A->B->C)", ""]
        lines.append("## 1) Baseline")
        for k, v in base_metrics.items():
            lines.append(f"- {k}: {v:.8f}")
        lines.append("")
        lines.append("## 2) Minimal Route-Cut Subset")
        lines.append(f"- size: {len(subset)}")
        lines.append(f"- layers: {layer_dist}")
        lines.append(f"- progress: {progress}")
        lines.append("")
        lines.append("## 3) Metrics After Ablation")
        for k, v in subset_metrics.items():
            lines.append(f"- {k}: {v:.8f}")
        lines.append("")
        lines.append("## 4) Subset Neurons")
        for n in result["minimal_subset"]["neurons"]:
            lines.append(
                f"- L{n['layer']}N{n['neuron']} util={n['utility']:.8f} "
                f"drop_h3={n['drop_h3_selectivity']:.8f} drop_h1={n['drop_h1_selectivity']:.8f}"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(
            f"[OK] base hop3_sel={base_metrics['hop3_selectivity']:.8f}, "
            f"after={subset_metrics['hop3_selectivity']:.8f}, "
            f"subset_size={len(subset)}"
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
