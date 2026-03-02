#!/usr/bin/env python
"""
Tri-Scale Micro Causal Apple Analysis (DeepSeek-7B)

Scales:
- Micro: adjectives (red / sweet / heavy ...)
- Meso: concrete taxonomy nouns (fruit / animal / vehicle ...)
- Macro: verbs + abstract nouns

Pipeline:
1) Per-scale candidate neuron discovery (apple vs non-apple contrast)
2) Single-neuron causal utility scoring
3) Minimal causal subset search per scale (prefix-optimal)
4) Cross-scale causal matrix (ablate one scale subset, evaluate all scales)
5) 100-concept tri-scale projection and regularity extraction
6) Micro causal graph edges by interventional transmission
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

SCALES = ("micro", "meso", "macro")


@dataclass
class EvalItem:
    prompt: str
    targets: List[str]


def concept_catalog_100() -> List[Tuple[str, str]]:
    rows = [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"), ("pear", "fruit"),
        ("peach", "fruit"), ("mango", "fruit"), ("lemon", "fruit"), ("strawberry", "fruit"), ("watermelon", "fruit"),
        ("rabbit", "animal"), ("cat", "animal"), ("dog", "animal"), ("horse", "animal"), ("tiger", "animal"),
        ("lion", "animal"), ("bird", "animal"), ("fish", "animal"), ("elephant", "animal"), ("monkey", "animal"),
        ("sun", "celestial"), ("moon", "celestial"), ("star", "celestial"), ("planet", "celestial"), ("comet", "celestial"),
        ("cloud", "weather"), ("rain", "weather"), ("snow", "weather"), ("wind", "weather"), ("storm", "weather"),
        ("car", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"), ("bicycle", "vehicle"), ("airplane", "vehicle"),
        ("ship", "vehicle"), ("truck", "vehicle"), ("motorcycle", "vehicle"), ("subway", "vehicle"), ("boat", "vehicle"),
        ("chair", "object"), ("table", "object"), ("bed", "object"), ("lamp", "object"), ("door", "object"),
        ("window", "object"), ("bottle", "object"), ("cup", "object"), ("spoon", "object"), ("knife", "object"),
        ("bread", "food"), ("rice", "food"), ("meat", "food"), ("soup", "food"), ("pizza", "food"),
        ("cake", "food"), ("coffee", "food"), ("tea", "food"), ("milk", "food"), ("cheese", "food"),
        ("tree", "nature"), ("flower", "nature"), ("grass", "nature"), ("forest", "nature"), ("river", "nature"),
        ("mountain", "nature"), ("ocean", "nature"), ("desert", "nature"), ("leaf", "nature"), ("seed", "nature"),
        ("child", "human"), ("teacher", "human"), ("doctor", "human"), ("student", "human"), ("parent", "human"),
        ("friend", "human"), ("king", "human"), ("queen", "human"), ("artist", "human"), ("worker", "human"),
        ("computer", "tech"), ("phone", "tech"), ("robot", "tech"), ("internet", "tech"), ("software", "tech"),
        ("hardware", "tech"), ("algorithm", "tech"), ("data", "tech"), ("number", "tech"), ("equation", "tech"),
        ("love", "abstract"), ("hate", "abstract"), ("justice", "abstract"), ("peace", "abstract"), ("war", "abstract"),
        ("music", "abstract"), ("art", "abstract"), ("history", "abstract"), ("future", "abstract"), ("memory", "abstract"),
    ]
    return rows


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


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk(li)))

    def _mk(self, li: int):
        def _hook(_module, _inputs, output):
            self.buffers[li] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hooks: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def get_selected(self, indices: List[int], d_ff: int) -> np.ndarray:
        out = []
        for idx in indices:
            li = idx // d_ff
            ni = idx % d_ff
            out.append(float(self.buffers[li][ni].item()))
        return np.array(out, dtype=np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


def token_ids(tok, targets: List[str]) -> List[int]:
    out = set()
    for t in targets:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            out.add(int(ids[0]))
    return sorted(out)


def eval_mass(model, tok, items: List[EvalItem]) -> float:
    vals = []
    for it in items:
        out = run_prompt(model, tok, it.prompt)
        logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        ids = token_ids(tok, it.targets)
        mass = float(probs[ids].sum().item()) if ids else 0.0
        vals.append(mass)
    return float(np.mean(vals) if vals else 0.0)


def mean_var(sum_t: np.ndarray, sumsq_t: np.ndarray, n: int):
    mu = sum_t / max(n, 1)
    var = sumsq_t / max(n, 1) - mu * mu
    return mu, np.maximum(var, 0.0)


def discovery_prompts(scale: str, concept: str) -> Tuple[List[str], List[str]]:
    # Positive: apple tied to scale semantics; Negative: same scale templates with non-apple concepts.
    negatives = ["banana", "rabbit", "sun", "car", "tree", "equation", "justice", "bird"]
    if scale == "micro":
        pos = [
            f"The {concept} is red and",
            f"The {concept} tastes sweet and",
            f"The {concept} feels heavy and",
            f"A ripe {concept} looks juicy and",
            f"The {concept} is fresh and",
            f"The {concept} is small and",
        ]
        neg = []
        for c in negatives:
            neg.extend(
                [
                    f"The {c} is red and",
                    f"The {c} tastes sweet and",
                    f"The {c} feels heavy and",
                    f"A {c} looks juicy and",
                ]
            )
        return pos, neg

    if scale == "meso":
        pos = [
            f"An {concept} is a fruit and",
            f"This {concept} belongs to the fruit category and",
            f"{concept} is a kind of fruit and",
            f"The category of {concept} is fruit and",
        ]
        neg = []
        for c in negatives:
            neg.extend(
                [
                    f"A {c} is a fruit and",
                    f"The category of {c} is fruit and",
                    f"This {c} belongs to the fruit category and",
                    f"{c} is a kind of fruit and",
                ]
            )
        return pos, neg

    # macro
    pos = [
        f"A {concept} can fall and",
        f"A {concept} can grow and",
        f"The {concept} symbolizes truth and",
        f"The {concept} relates to infinity and",
        f"The {concept} evokes justice and",
    ]
    neg = []
    for c in negatives:
        neg.extend(
            [
                f"A {c} can fall and",
                f"A {c} can grow and",
                f"The {c} symbolizes truth and",
                f"The {c} relates to infinity and",
            ]
        )
    return pos, neg


def scale_eval_sets(concept: str) -> Dict[str, List[EvalItem]]:
    return {
        "micro": [
            EvalItem(f"The {concept} is", [" red", " sweet", " juicy", " fresh", " heavy", " light"]),
            EvalItem(f"A ripe {concept} looks", [" red", " sweet", " juicy", " fresh"]),
            EvalItem(f"This {concept} feels", [" heavy", " light", " soft"]),
            EvalItem(f"The {concept} tastes", [" sweet", " sour", " good"]),
        ],
        "meso": [
            EvalItem(f"An {concept} is a", [" fruit"]),
            EvalItem(f"The category of {concept} is", [" fruit"]),
            EvalItem(f"This {concept} belongs to", [" fruit"]),
            EvalItem(f"{concept} is a kind of", [" fruit"]),
        ],
        "macro": [
            EvalItem(f"A {concept} can", [" fall", " grow", " rot", " move"]),
            EvalItem(f"The {concept} symbolizes", [" truth", " justice", " memory", " infinity"]),
            EvalItem(f"In stories, {concept} means", [" knowledge", " truth", " temptation"]),
            EvalItem(f"People connect {concept} with", [" memory", " meaning", " value"]),
        ],
    }


def collect_stats(model, tok, collector: GateCollector, pos: List[str], neg: List[str], total_dim: int):
    stat = {
        "pos_n": 0,
        "neg_n": 0,
        "pos_sum": np.zeros(total_dim, dtype=np.float64),
        "pos_sumsq": np.zeros(total_dim, dtype=np.float64),
        "neg_sum": np.zeros(total_dim, dtype=np.float64),
        "neg_sumsq": np.zeros(total_dim, dtype=np.float64),
    }
    for p in pos:
        collector.reset()
        _ = run_prompt(model, tok, p)
        v = collector.get_flat()
        stat["pos_n"] += 1
        stat["pos_sum"] += v
        stat["pos_sumsq"] += v * v
    for p in neg:
        collector.reset()
        _ = run_prompt(model, tok, p)
        v = collector.get_flat()
        stat["neg_n"] += 1
        stat["neg_sum"] += v
        stat["neg_sumsq"] += v * v
    return stat


def top_candidates(stat, k: int):
    mu_p, var_p = mean_var(stat["pos_sum"], stat["pos_sumsq"], stat["pos_n"])
    mu_n, var_n = mean_var(stat["neg_sum"], stat["neg_sumsq"], stat["neg_n"])
    diff = mu_p - mu_n
    z = diff / np.sqrt(0.5 * (var_p + var_n) + 1e-8)
    z = np.where(diff > 0, z, -np.inf)
    k = min(k, z.shape[0])
    idx = np.argpartition(z, -k)[-k:]
    idx = idx[np.argsort(z[idx])[::-1]]
    return [{"flat_idx": int(i), "score_z": float(z[i]), "diff": float(diff[i])} for i in idx.tolist()]


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


def register_additive(model, flat_idx: int, d_ff: int, alpha: float):
    li = flat_idx // d_ff
    ni = flat_idx % d_ff
    module = model.model.layers[li].mlp.gate_proj

    def _hook(_module, _inputs, output):
        out = output.clone()
        out[..., ni] = out[..., ni] + alpha
        return out

    return module.register_forward_hook(_hook)


def remove_handles(handles):
    for h in handles:
        h.remove()


def eval_all_scales(model, tok, eval_sets: Dict[str, List[EvalItem]]) -> Dict[str, float]:
    return {s: eval_mass(model, tok, eval_sets[s]) for s in SCALES}


def single_neuron_utilities(
    model,
    tok,
    d_ff: int,
    candidates: List[Dict],
    eval_sets: Dict[str, List[EvalItem]],
    off_penalty: float,
):
    base = eval_all_scales(model, tok, eval_sets)
    scored = {s: [] for s in SCALES}
    for c in candidates:
        idx = c["flat_idx"]
        h = register_ablation(model, [idx], d_ff)
        try:
            aft = eval_all_scales(model, tok, eval_sets)
        finally:
            remove_handles(h)
        for s in SCALES:
            drop = base[s] - aft[s]
            off = sum(abs(aft[o] - base[o]) for o in SCALES if o != s)
            util = drop - off_penalty * off
            scored[s].append({**c, "drop_target": float(drop), "off_shift": float(off), "utility": float(util)})
    for s in SCALES:
        scored[s].sort(key=lambda x: x["utility"], reverse=True)
    return scored, base


def eval_subset(model, tok, d_ff: int, subset: List[Dict], eval_sets: Dict[str, List[EvalItem]]):
    base = eval_all_scales(model, tok, eval_sets)
    if not subset:
        return {"base": base, "after": base.copy()}
    idxs = [x["flat_idx"] for x in subset]
    h = register_ablation(model, idxs, d_ff)
    try:
        aft = eval_all_scales(model, tok, eval_sets)
    finally:
        remove_handles(h)
    return {"base": base, "after": aft}


def minimal_subset_for_scale(
    model,
    tok,
    d_ff: int,
    ranked: List[Dict],
    scale: str,
    eval_sets: Dict[str, List[EvalItem]],
    max_subset: int,
    target_ratio: float,
):
    max_n = min(max_subset, len(ranked))
    if max_n == 0:
        e = eval_subset(model, tok, d_ff, [], eval_sets)
        return [], e, {"best_drop": 0.0, "goal": 0.0, "achieved": 0.0}

    drops = []
    evals = []
    for i in range(1, max_n + 1):
        sub = ranked[:i]
        ev = eval_subset(model, tok, d_ff, sub, eval_sets)
        d = ev["base"][scale] - ev["after"][scale]
        drops.append(d)
        evals.append(ev)

    best_i = int(np.argmax(drops))
    best_drop = float(drops[best_i])
    if best_drop <= 0:
        e = eval_subset(model, tok, d_ff, [], eval_sets)
        return [], e, {"best_drop": best_drop, "goal": 0.0, "achieved": 0.0}

    goal = best_drop * target_ratio
    chosen_i = best_i
    for i, d in enumerate(drops):
        if d >= goal:
            chosen_i = i
            break
    chosen = ranked[: chosen_i + 1]
    chosen_eval = evals[chosen_i]
    return chosen, chosen_eval, {"best_drop": best_drop, "goal": goal, "achieved": float(drops[chosen_i])}


def layer_dist(subset: List[Dict], d_ff: int):
    ctr = Counter([x["flat_idx"] // d_ff for x in subset])
    return dict(sorted(ctr.items()))


def subset_overlap(a: List[Dict], b: List[Dict]):
    sa = set(x["flat_idx"] for x in a)
    sb = set(x["flat_idx"] for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def causal_matrix(model, tok, d_ff: int, subsets: Dict[str, List[Dict]], eval_sets: Dict[str, List[EvalItem]]):
    base = eval_all_scales(model, tok, eval_sets)
    mat = {}
    for src in SCALES:
        idxs = [x["flat_idx"] for x in subsets[src]]
        if idxs:
            h = register_ablation(model, idxs, d_ff)
            try:
                aft = eval_all_scales(model, tok, eval_sets)
            finally:
                remove_handles(h)
        else:
            aft = base.copy()
        mat[src] = {dst: float(aft[dst] - base[dst]) for dst in SCALES}
    return {"baseline": base, "delta": mat}


def extract_graph_edges(
    model,
    tok,
    collector: GateCollector,
    d_ff: int,
    prompts: List[str],
    selected_flat: List[int],
    alpha: float,
):
    if not selected_flat:
        return []
    base = []
    for p in prompts:
        collector.reset()
        _ = run_prompt(model, tok, p)
        base.append(collector.get_selected(selected_flat, d_ff))
    base = np.stack(base, axis=0)

    edges = []
    for si, src in enumerate(selected_flat):
        h = register_additive(model, src, d_ff, alpha)
        try:
            aft = []
            for p in prompts:
                collector.reset()
                _ = run_prompt(model, tok, p)
                aft.append(collector.get_selected(selected_flat, d_ff))
            aft = np.stack(aft, axis=0)
        finally:
            h.remove()

        delta = aft - base
        src_l = src // d_ff
        for ti, tgt in enumerate(selected_flat):
            if ti == si:
                continue
            tgt_l = tgt // d_ff
            if tgt_l <= src_l:
                continue
            score = float(np.mean(np.abs(delta[:, ti])))
            if score <= 1e-7:
                continue
            edges.append(
                {
                    "src_flat": int(src),
                    "src_layer": int(src_l),
                    "src_neuron": int(src % d_ff),
                    "tgt_flat": int(tgt),
                    "tgt_layer": int(tgt_l),
                    "tgt_neuron": int(tgt % d_ff),
                    "score": score,
                }
            )
    edges.sort(key=lambda x: x["score"], reverse=True)
    return edges


def tri_projection(
    model,
    tok,
    collector: GateCollector,
    d_ff: int,
    subsets: Dict[str, List[Dict]],
    catalog: List[Tuple[str, str]],
):
    idx_by_scale = {s: [x["flat_idx"] for x in subsets[s]] for s in SCALES}
    all_idx = sorted(set(i for s in SCALES for i in idx_by_scale[s]))
    if not all_idx:
        return {"records": [], "apple_neighbors": [], "category_mean": {}}

    records = []
    for concept, cat in catalog:
        p = f"This is {concept}"
        collector.reset()
        _ = run_prompt(model, tok, p)
        vec = collector.get_selected(all_idx, d_ff)

        scale_proj = {}
        for s in SCALES:
            sidx = idx_by_scale[s]
            if not sidx:
                scale_proj[s] = 0.0
                continue
            pos = [all_idx.index(i) for i in sidx if i in all_idx]
            scale_proj[s] = float(vec[pos].mean()) if pos else 0.0

        records.append(
            {
                "concept": concept,
                "category": cat,
                "micro_proj": scale_proj["micro"],
                "meso_proj": scale_proj["meso"],
                "macro_proj": scale_proj["macro"],
            }
        )

    # nearest to apple in 3D projection
    apple = [r for r in records if r["concept"] == "apple"][0]
    for r in records:
        d = (
            (r["micro_proj"] - apple["micro_proj"]) ** 2
            + (r["meso_proj"] - apple["meso_proj"]) ** 2
            + (r["macro_proj"] - apple["macro_proj"]) ** 2
        ) ** 0.5
        r["distance_to_apple"] = float(d)
    neighbors = sorted([r for r in records if r["concept"] != "apple"], key=lambda x: x["distance_to_apple"])[:15]

    by_cat = defaultdict(list)
    for r in records:
        if r["concept"] == "apple":
            continue
        by_cat[r["category"]].append(r)
    cat_mean = {}
    for cat, arr in by_cat.items():
        cat_mean[cat] = {
            "n": len(arr),
            "mean_micro_proj": float(np.mean([x["micro_proj"] for x in arr])),
            "mean_meso_proj": float(np.mean([x["meso_proj"] for x in arr])),
            "mean_macro_proj": float(np.mean([x["macro_proj"] for x in arr])),
            "mean_distance_to_apple": float(np.mean([x["distance_to_apple"] for x in arr])),
        }
    return {"records": records, "apple_neighbors": neighbors, "category_mean": dict(sorted(cat_mean.items()))}


def main():
    parser = argparse.ArgumentParser(description="Tri-scale micro-causal apple analysis")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--concept", default="apple")
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--max-subset", type=int, default=10)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--off-penalty", type=float, default=0.5)
    parser.add_argument("--edge-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_triscale_apple_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    n_layers = len(model.model.layers)
    total_dim = d_ff * n_layers
    collector = GateCollector(model)

    try:
        eval_sets = scale_eval_sets(args.concept)
        subsets = {}
        scale_info = {}

        for scale in SCALES:
            pos, neg = discovery_prompts(scale, args.concept)
            stat = collect_stats(model, tok, collector, pos, neg, total_dim)
            cands = top_candidates(stat, args.candidate_k)
            scored, base = single_neuron_utilities(
                model=model,
                tok=tok,
                d_ff=d_ff,
                candidates=cands,
                eval_sets=eval_sets,
                off_penalty=args.off_penalty,
            )

            subset, sev, prog = minimal_subset_for_scale(
                model=model,
                tok=tok,
                d_ff=d_ff,
                ranked=scored[scale],
                scale=scale,
                eval_sets=eval_sets,
                max_subset=args.max_subset,
                target_ratio=args.target_ratio,
            )
            subsets[scale] = subset
            scale_info[scale] = {
                "candidate_layer_distribution": layer_dist(cands, d_ff),
                "subset_size": len(subset),
                "subset_layer_distribution": layer_dist(subset, d_ff),
                "subset_drop_target": float(sev["base"][scale] - sev["after"][scale]),
                "subset_off_shift": float(
                    sum(abs(sev["after"][s] - sev["base"][s]) for s in SCALES if s != scale)
                ),
                "progress": prog,
                "subset_neurons": [
                    {
                        "flat_idx": int(x["flat_idx"]),
                        "layer": int(x["flat_idx"] // d_ff),
                        "neuron": int(x["flat_idx"] % d_ff),
                        "utility": float(x["utility"]),
                        "drop_target": float(x["drop_target"]),
                        "off_shift": float(x["off_shift"]),
                    }
                    for x in subset
                ],
            }

        overlaps = {
            "micro_meso": subset_overlap(subsets["micro"], subsets["meso"]),
            "micro_macro": subset_overlap(subsets["micro"], subsets["macro"]),
            "meso_macro": subset_overlap(subsets["meso"], subsets["macro"]),
        }

        cm = causal_matrix(model, tok, d_ff, subsets, eval_sets)

        union_flat = sorted(set(i for s in SCALES for i in [x["flat_idx"] for x in subsets[s]]))
        graph_edges = extract_graph_edges(
            model=model,
            tok=tok,
            collector=collector,
            d_ff=d_ff,
            prompts=discovery_prompts("micro", args.concept)[0],
            selected_flat=union_flat,
            alpha=args.edge_alpha,
        )

        projection = tri_projection(
            model=model,
            tok=tok,
            collector=collector,
            d_ff=d_ff,
            subsets=subsets,
            catalog=concept_catalog_100(),
        )

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "concept": args.concept,
                "candidate_k": args.candidate_k,
                "max_subset": args.max_subset,
                "target_ratio": args.target_ratio,
                "off_penalty": args.off_penalty,
                "edge_alpha": args.edge_alpha,
                "n_layers": n_layers,
                "d_ff": d_ff,
            },
            "scales": scale_info,
            "subset_overlap": overlaps,
            "cross_scale_causal_matrix": cm,
            "knowledge_graph": {
                "n_nodes": len(union_flat),
                "n_edges": len(graph_edges),
                "top_edges": graph_edges[:200],
            },
            "tri_projection": projection,
        }

        json_path = out_dir / "apple_triscale_micro_causal_results.json"
        md_path = out_dir / "APPLE_TRISCALE_MICRO_CAUSAL_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# Apple 三尺度微观因果编码报告", ""]
        for s in SCALES:
            rr = scale_info[s]
            lines.append(
                f"- {s}: size={rr['subset_size']}, drop={rr['subset_drop_target']:.8f}, off={rr['subset_off_shift']:.8f}, layers={rr['subset_layer_distribution']}"
            )
        lines.append("")
        lines.append(f"- overlap: {overlaps}")
        lines.append(f"- knowledge graph: nodes={len(union_flat)}, edges={len(graph_edges)}")
        lines.append("")
        lines.append("## Cross-Scale Causal Delta")
        for src in SCALES:
            lines.append(f"- ablate {src}: {cm['delta'][src]}")
        lines.append("")
        lines.append("## Apple Neighbors in Tri-Scale Space (Top-10)")
        for r in projection["apple_neighbors"][:10]:
            lines.append(
                f"- {r['concept']} ({r['category']}): dist={r['distance_to_apple']:.6f}, micro={r['micro_proj']:.4f}, meso={r['meso_proj']:.4f}, macro={r['macro_proj']:.4f}"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        for s in SCALES:
            rr = scale_info[s]
            print(
                f"[OK] {s}: subset={rr['subset_size']} drop={rr['subset_drop_target']:.8f} off={rr['subset_off_shift']:.8f} layers={rr['subset_layer_distribution']}"
            )
        print(f"[OK] overlap={overlaps} graph_nodes={len(union_flat)} graph_edges={len(graph_edges)}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
