#!/usr/bin/env python
"""
Large-sample multi-hop route test (A->B->C) for DeepSeek 7B.

This script extends the small benchmark to >100 chains, then reports:
- global hop selectivity and route index,
- bootstrap confidence intervals,
- domain-level generalization behavior,
- minimal causal subset for cutting hop-3 routing.
"""

from __future__ import annotations

import argparse
import json
import math
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
    domain: str


@dataclass
class QueryItem:
    prompt: str
    target: str
    hop: int
    valid: bool
    chain_id: int
    domain: str


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


def build_chain_bank() -> List[Chain]:
    spec = [
        ("food_fruit", "fruit", "food", "edible", [
            "apple", "banana", "orange", "grape", "pear", "peach", "plum", "mango", "lemon", "lime", "melon", "cherry"
        ]),
        ("food_vegetable", "vegetable", "food", "edible", [
            "carrot", "potato", "tomato", "onion", "pepper", "cabbage", "spinach", "garlic", "broccoli", "pumpkin", "radish", "lettuce"
        ]),
        ("animal_mammal", "mammal", "animal", "living", [
            "dog", "cat", "rabbit", "horse", "tiger", "lion", "mouse", "sheep", "goat", "monkey", "panda", "whale"
        ]),
        ("animal_bird", "bird", "animal", "living", [
            "eagle", "sparrow", "pigeon", "crow", "duck", "goose", "owl", "falcon", "parrot", "swan", "robin", "hawk"
        ]),
        ("animal_fish", "fish", "animal", "living", [
            "salmon", "tuna", "trout", "shark", "carp", "cod", "eel", "bass", "herring", "sardine", "perch", "snapper"
        ]),
        ("animal_insect", "insect", "animal", "living", [
            "ant", "bee", "wasp", "moth", "beetle", "dragonfly", "mosquito", "butterfly", "termite", "cricket", "locust", "spider"
        ]),
        ("artifact_vehicle", "vehicle", "machine", "artifact", [
            "car", "truck", "bus", "train", "taxi", "van", "scooter", "bicycle", "motorcycle", "tram", "subway", "ship"
        ]),
        ("artifact_tool", "tool", "object", "artifact", [
            "hammer", "wrench", "saw", "drill", "chisel", "pliers", "shovel", "ladder", "anvil", "screwdriver", "rake", "tongs"
        ]),
        ("artifact_instrument", "instrument", "object", "artifact", [
            "piano", "guitar", "violin", "drum", "trumpet", "flute", "cello", "harp", "clarinet", "saxophone", "banjo", "organ"
        ]),
        ("geo_city", "city", "location", "place", [
            "paris", "london", "tokyo", "beijing", "berlin", "madrid", "rome", "moscow", "seoul", "delhi", "dubai", "chicago"
        ]),
        ("geo_country", "country", "location", "place", [
            "china", "india", "france", "germany", "brazil", "canada", "japan", "mexico", "italy", "spain", "egypt", "turkey"
        ]),
        ("cosmos_body", "planet", "celestial", "cosmic", [
            "earth", "mars", "venus", "jupiter", "saturn", "uranus", "neptune", "mercury", "pluto", "moon", "sun", "sirius"
        ]),
    ]
    out: List[Chain] = []
    for domain, b, c, d, ents in spec:
        for e in ents:
            out.append(Chain(a=e, b=b, c=c, d=d, domain=domain))
    return out


def build_queries(chains: List[Chain], rng: random.Random) -> List[QueryItem]:
    items: List[QueryItem] = []
    n = len(chains)
    if n < 2:
        raise ValueError("Need at least 2 chains to build invalid controls.")

    for i, ch in enumerate(chains):
        p1 = f"Facts: {ch.a} is a {ch.b}. Question: {ch.a} is a"
        p2 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. Question: {ch.a} is a"
        p3 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. {ch.c} is {ch.d}. Question: {ch.a} is"
        items.append(QueryItem(p1, f" {ch.b}", 1, True, i, ch.domain))
        items.append(QueryItem(p2, f" {ch.c}", 2, True, i, ch.domain))
        items.append(QueryItem(p3, f" {ch.d}", 3, True, i, ch.domain))

    base_offsets = {1: rng.randint(1, n - 1), 2: rng.randint(1, n - 1), 3: rng.randint(1, n - 1)}
    for i, ch in enumerate(chains):
        p1 = f"Facts: {ch.a} is a {ch.b}. Question: {ch.a} is a"
        p2 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. Question: {ch.a} is a"
        p3 = f"Facts: {ch.a} is a {ch.b}. {ch.b} is a {ch.c}. {ch.c} is {ch.d}. Question: {ch.a} is"
        for hop, field, prompt in [(1, "b", p1), (2, "c", p2), (3, "d", p3)]:
            j = (i + base_offsets[hop]) % n
            tries = 0
            while getattr(chains[j], field) == getattr(ch, field) and tries < n:
                j = (j + 1) % n
                tries += 1
            if getattr(chains[j], field) == getattr(ch, field):
                continue
            items.append(QueryItem(prompt, f" {getattr(chains[j], field)}", hop, False, i, ch.domain))
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


def group_metrics(items: List[QueryItem], vals: Dict[int, float], indices: List[int] | None = None) -> Dict[str, float]:
    idx_set = set(indices) if indices is not None else None
    out: Dict[str, float] = {}
    for hop in [1, 2, 3]:
        v_valid = [
            vals[i] for i, it in enumerate(items)
            if it.hop == hop and it.valid and (idx_set is None or i in idx_set)
        ]
        v_invalid = [
            vals[i] for i, it in enumerate(items)
            if it.hop == hop and not it.valid and (idx_set is None or i in idx_set)
        ]
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
    discovery_max_items: int,
    seed: int,
) -> List[Dict]:
    total_dim = len(model.model.layers) * model.model.layers[0].mlp.gate_proj.out_features
    sum_h3v = np.zeros(total_dim, dtype=np.float64)
    sum_h1v = np.zeros(total_dim, dtype=np.float64)
    sum_h3i = np.zeros(total_dim, dtype=np.float64)
    n_h3v = n_h1v = n_h3i = 0

    h3v = [it for it in items if it.hop == 3 and it.valid]
    h1v = [it for it in items if it.hop == 1 and it.valid]
    h3i = [it for it in items if it.hop == 3 and not it.valid]
    if discovery_max_items > 0:
        rng = random.Random(seed)
        each = max(1, discovery_max_items // 3)
        h3v = rng.sample(h3v, min(each, len(h3v)))
        h1v = rng.sample(h1v, min(each, len(h1v)))
        h3i = rng.sample(h3i, min(each, len(h3i)))
    seq = h3v + h1v + h3i

    for it in seq:
        collector.reset()
        _ = run_prompt(model, tok, it.prompt)
        v = collector.get_flat()
        if it.hop == 3 and it.valid:
            sum_h3v += v
            n_h3v += 1
        elif it.hop == 1 and it.valid:
            sum_h1v += v
            n_h1v += 1
        elif it.hop == 3 and not it.valid:
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


def domain_metrics(items: List[QueryItem], vals: Dict[int, float]) -> Dict[str, Dict[str, float]]:
    domains = sorted({it.domain for it in items})
    out: Dict[str, Dict[str, float]] = {}
    for d in domains:
        idx = [i for i, it in enumerate(items) if it.domain == d]
        out[d] = group_metrics(items, vals, idx)
    return out


def bootstrap_ci(
    items: List[QueryItem],
    vals: Dict[int, float],
    n_chains: int,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    key_to_idx = {}
    for i, it in enumerate(items):
        key_to_idx[(it.chain_id, it.hop, it.valid)] = i

    rng = np.random.default_rng(seed)
    stat = {"hop1_selectivity": [], "hop2_selectivity": [], "hop3_selectivity": [], "route_index": []}
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_chains, size=n_chains)
        sel = {}
        for hop in [1, 2, 3]:
            v_valid = []
            v_invalid = []
            for cid in sample:
                kv = (int(cid), hop, True)
                ki = (int(cid), hop, False)
                if kv in key_to_idx:
                    v_valid.append(vals[key_to_idx[kv]])
                if ki in key_to_idx:
                    v_invalid.append(vals[key_to_idx[ki]])
            m_valid = float(np.mean(v_valid) if v_valid else 0.0)
            m_invalid = float(np.mean(v_invalid) if v_invalid else 0.0)
            sel[f"hop{hop}_selectivity"] = m_valid - m_invalid
        sel["route_index"] = sel["hop3_selectivity"] - sel["hop1_selectivity"]
        for k in stat:
            stat[k].append(sel[k])

    ci = {}
    for k, arr in stat.items():
        a = np.asarray(arr, dtype=np.float64)
        ci[k] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
        }
    return ci


def layer_concentration(ranked: List[Dict], d_ff: int, top_n: int) -> Dict[str, float]:
    if not ranked:
        return {"entropy": 0.0, "effective_layers": 0.0}
    top = ranked[: min(top_n, len(ranked))]
    cnt = Counter([x["flat_idx"] // d_ff for x in top])
    total = float(sum(cnt.values()))
    p = [v / total for v in cnt.values()]
    ent = -sum(pi * math.log(pi + 1e-12) for pi in p)
    eff = math.exp(ent) if ent > 0 else 1.0
    return {"entropy": float(ent), "effective_layers": float(eff)}


def main():
    parser = argparse.ArgumentParser(description="Large-sample A->B->C multi-hop generalization test")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-chains", type=int, default=120)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-subset", type=int, default=6)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--route-penalty", type=float, default=0.5)
    parser.add_argument("--discovery-max-items", type=int, default=240)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_multihop_large_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only, args.device_map)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    collector = GateCollector(model)
    try:
        chains = build_chain_bank()
        rng.shuffle(chains)
        if args.max_chains > 0:
            chains = chains[: args.max_chains]
        items = build_queries(chains, rng)
        target_ids = [token_id(tok, it.target) for it in items]

        base_vals = eval_items_batch(model, tok, items, target_ids, args.batch_size)
        base_metrics = group_metrics(items, base_vals)
        base_domains = domain_metrics(items, base_vals)
        base_boot = bootstrap_ci(items, base_vals, len(chains), args.bootstrap, args.seed)

        candidates = discover_candidates(
            model=model,
            tok=tok,
            collector=collector,
            items=items,
            top_k=args.candidate_k,
            discovery_max_items=args.discovery_max_items,
            seed=args.seed,
        )
        ranked, _ = single_neuron_utilities(
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
        after_vals = eval_items_batch(model, tok, items, target_ids, args.batch_size) if not subset else None
        if subset:
            h = register_ablation(model, idxs, d_ff)
            try:
                after_vals = eval_items_batch(model, tok, items, target_ids, args.batch_size)
            finally:
                remove_handles(h)
        assert after_vals is not None
        after_domains = domain_metrics(items, after_vals)

        domain_drop = {}
        for d in base_domains:
            domain_drop[d] = {
                "hop3_selectivity_drop": float(base_domains[d]["hop3_selectivity"] - after_domains[d]["hop3_selectivity"]),
                "route_index_drop": float(base_domains[d]["route_index"] - after_domains[d]["route_index"]),
            }

        layer_conc = layer_concentration(ranked, d_ff, top_n=min(10, len(ranked)))

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "device_map": args.device_map,
                "batch_size": args.batch_size,
                "max_chains": args.max_chains,
                "candidate_k": args.candidate_k,
                "max_subset": args.max_subset,
                "target_ratio": args.target_ratio,
                "route_penalty": args.route_penalty,
                "discovery_max_items": args.discovery_max_items,
                "bootstrap": args.bootstrap,
                "n_chains": len(chains),
                "n_queries": len(items),
            },
            "base_metrics": base_metrics,
            "bootstrap_ci": base_boot,
            "domain_base_metrics": base_domains,
            "domain_drop_after_subset": domain_drop,
            "candidates_layer_distribution": dict(sorted(Counter([x["flat_idx"] // d_ff for x in candidates]).items())),
            "layer_concentration_top_candidates": layer_conc,
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

        json_path = out_dir / "multihop_large_generalization_results.json"
        md_path = out_dir / "MULTIHOP_LARGE_GENERALIZATION_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# Multi-hop Large-Sample Generalization Report", ""]
        lines.append("## 1) Global Baseline")
        for k, v in base_metrics.items():
            lines.append(f"- {k}: {v:.8f}")
        lines.append("")
        lines.append("## 2) Bootstrap CI (95%)")
        for k, c in base_boot.items():
            lines.append(
                f"- {k}: mean={c['mean']:.8f}, std={c['std']:.8f}, ci95=[{c['ci95_low']:.8f}, {c['ci95_high']:.8f}]"
            )
        lines.append("")
        lines.append("## 3) Minimal Subset")
        lines.append(f"- size: {len(subset)}")
        lines.append(f"- layers: {layer_dist}")
        lines.append(f"- progress: {progress}")
        lines.append("")
        lines.append("## 4) After-Ablation Global")
        for k, v in subset_metrics.items():
            lines.append(f"- {k}: {v:.8f}")
        lines.append("")
        lines.append("## 5) Domain Route Drops (hop3_selectivity_drop)")
        top_drop = sorted(domain_drop.items(), key=lambda x: x[1]["hop3_selectivity_drop"], reverse=True)[:10]
        for d, dv in top_drop:
            lines.append(
                f"- {d}: hop3_drop={dv['hop3_selectivity_drop']:.8f}, route_drop={dv['route_index_drop']:.8f}"
            )
        lines.append("")
        lines.append("## 6) Candidate Layer Concentration")
        lines.append(
            f"- entropy={layer_conc['entropy']:.8f}, effective_layers={layer_conc['effective_layers']:.4f}"
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(
            f"[OK] chains={len(chains)}, queries={len(items)}, "
            f"hop3_sel={base_metrics['hop3_selectivity']:.8f}, "
            f"after={subset_metrics['hop3_selectivity']:.8f}, subset_size={len(subset)}"
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()

