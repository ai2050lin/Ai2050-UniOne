#!/usr/bin/env python
"""
Micro Causal Encoding Graph (MCEG) for DeepSeek-7B.

Focus:
- concept A/B (default: apple/banana)
- role-level micro structure: entity / size / weight / fruit_superclass
- minimal causal neuron subset per role (not purely statistical)
- cross-layer knowledge graph edges from intervention transmission
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


ROLES = ("entity", "size", "weight", "fruit")


@dataclass
class EvalItem:
    prompt: str
    targets: List[str]


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
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            # collect final token
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hooks on layers: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def get_selected(self, flat_indices: List[int], d_ff: int) -> np.ndarray:
        vals = []
        for idx in flat_indices:
            li = idx // d_ff
            ni = idx % d_ff
            vals.append(float(self.buffers[li][ni].item()))
        return np.array(vals, dtype=np.float32)

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
    var = np.maximum(var, 0.0)
    return mu, var


def register_ablation(model, flat_indices: List[int], d_ff: int):
    by_layer = defaultdict(list)
    for idx in flat_indices:
        li = int(idx) // d_ff
        ni = int(idx) % d_ff
        by_layer[li].append(ni)
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


def register_additive_intervention(model, flat_idx: int, d_ff: int, alpha: float):
    li = int(flat_idx) // d_ff
    ni = int(flat_idx) % d_ff
    module = model.model.layers[li].mlp.gate_proj

    def _hook(_module, _inputs, output):
        out = output.clone()
        out[..., ni] = out[..., ni] + alpha
        return out

    return module.register_forward_hook(_hook)


def remove_handles(handles):
    for h in handles:
        h.remove()


def build_role_eval(concept: str) -> Dict[str, List[EvalItem]]:
    # role-specific evaluation prompts for one concept
    return {
        "entity": [
            EvalItem(f"This is a", [f" {concept}", f" {concept}s"]),
            EvalItem(f"I saw a", [f" {concept}", f" {concept}s"]),
            EvalItem(f"They bought a", [f" {concept}", f" {concept}s"]),
        ],
        "size": [
            EvalItem(f"A {concept} is usually", [" big", " small", " large", " tiny"]),
            EvalItem(f"The size of a {concept} is", [" small", " medium", " large"]),
            EvalItem(f"A ripe {concept} can be", [" small", " big"]),
        ],
        "weight": [
            EvalItem(f"A {concept} can feel", [" heavy", " light"]),
            EvalItem(f"The weight of a {concept} is", [" heavy", " light"]),
            EvalItem(f"Holding a {concept} feels", [" heavy", " light"]),
        ],
        "fruit": [
            EvalItem(f"A {concept} is a", [" fruit", " fruits"]),
            EvalItem(f"This {concept} belongs to", [" fruit", " fruits"]),
            EvalItem(f"People classify {concept} as", [" fruit", " fruits"]),
        ],
    }


def build_discovery_prompts(concept: str) -> List[str]:
    return [
        f"This is a {concept}",
        f"I saw a {concept}",
        f"They discussed {concept}",
        f"The item is {concept}",
    ]


def build_negative_prompts() -> List[str]:
    neg = ["rabbit", "sun", "car", "book", "tree", "cloud", "equation", "music", "chair", "river"]
    out = []
    for c in neg:
        out.extend(build_discovery_prompts(c))
    return out


def collect_activation_stats(model, tok, collector: GateCollector, pos_prompts: List[str], neg_prompts: List[str]):
    example = run_prompt(model, tok, pos_prompts[0])
    _ = example  # silence linter-like warning
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    n_layers = len(model.model.layers)
    total = d_ff * n_layers

    stat = {
        "pos_n": 0,
        "neg_n": 0,
        "pos_sum": np.zeros(total, dtype=np.float64),
        "pos_sumsq": np.zeros(total, dtype=np.float64),
        "neg_sum": np.zeros(total, dtype=np.float64),
        "neg_sumsq": np.zeros(total, dtype=np.float64),
    }

    for p in pos_prompts:
        collector.reset()
        _ = run_prompt(model, tok, p)
        v = collector.get_flat()
        stat["pos_n"] += 1
        stat["pos_sum"] += v
        stat["pos_sumsq"] += v * v

    for p in neg_prompts:
        collector.reset()
        _ = run_prompt(model, tok, p)
        v = collector.get_flat()
        stat["neg_n"] += 1
        stat["neg_sum"] += v
        stat["neg_sumsq"] += v * v
    return stat


def top_candidates_from_stats(stat, k: int):
    mu_p, var_p = mean_var(stat["pos_sum"], stat["pos_sumsq"], stat["pos_n"])
    mu_n, var_n = mean_var(stat["neg_sum"], stat["neg_sumsq"], stat["neg_n"])
    diff = mu_p - mu_n
    z = diff / np.sqrt(0.5 * (var_p + var_n) + 1e-8)
    z = np.where(diff > 0, z, -np.inf)

    k = min(k, z.shape[0])
    idx = np.argpartition(z, -k)[-k:]
    idx = idx[np.argsort(z[idx])[::-1]]
    out = []
    for i in idx.tolist():
        out.append({"flat_idx": int(i), "score_z": float(z[i]), "diff": float(diff[i])})
    return out


def single_neuron_role_scores(
    model,
    tok,
    d_ff: int,
    candidates: List[Dict],
    role_eval: Dict[str, List[EvalItem]],
    off_penalty: float,
):
    base = {r: eval_mass(model, tok, role_eval[r]) for r in ROLES}
    scored = {r: [] for r in ROLES}

    for c in candidates:
        idx = c["flat_idx"]
        h = register_ablation(model, [idx], d_ff)
        try:
            after = {r: eval_mass(model, tok, role_eval[r]) for r in ROLES}
        finally:
            remove_handles(h)

        for r in ROLES:
            drop_target = base[r] - after[r]
            off = 0.0
            for rr in ROLES:
                if rr == r:
                    continue
                off += abs(after[rr] - base[rr])
            utility = drop_target - off_penalty * off
            scored[r].append(
                {
                    **c,
                    "drop_target": float(drop_target),
                    "off_shift": float(off),
                    "utility": float(utility),
                }
            )

    for r in ROLES:
        scored[r].sort(key=lambda x: x["utility"], reverse=True)
    return scored, base


def eval_subset(model, tok, d_ff: int, subset: List[Dict], role_eval: Dict[str, List[EvalItem]]):
    base = {r: eval_mass(model, tok, role_eval[r]) for r in ROLES}
    if not subset:
        return {"base": base, "after": base.copy()}
    idxs = [x["flat_idx"] for x in subset]
    h = register_ablation(model, idxs, d_ff)
    try:
        after = {r: eval_mass(model, tok, role_eval[r]) for r in ROLES}
    finally:
        remove_handles(h)
    return {"base": base, "after": after}


def minimal_subset_for_role(
    model,
    tok,
    d_ff: int,
    ranked: List[Dict],
    role: str,
    role_eval: Dict[str, List[EvalItem]],
    max_subset: int,
    target_ratio: float,
):
    max_n = min(max_subset, len(ranked))
    if max_n == 0:
        e = eval_subset(model, tok, d_ff, [], role_eval)
        return [], e, {"best_drop": 0.0, "goal": 0.0, "achieved": 0.0}

    prefix_eval = []
    drops = []
    for i in range(1, max_n + 1):
        sub = ranked[:i]
        ev = eval_subset(model, tok, d_ff, sub, role_eval)
        drop = ev["base"][role] - ev["after"][role]
        prefix_eval.append(ev)
        drops.append(drop)

    best_i = int(np.argmax(drops))
    best_drop = float(drops[best_i])
    if best_drop <= 0:
        e = eval_subset(model, tok, d_ff, [], role_eval)
        return [], e, {"best_drop": best_drop, "goal": 0.0, "achieved": 0.0}

    goal = best_drop * target_ratio
    chosen_i = best_i
    for i, d in enumerate(drops):
        if d >= goal:
            chosen_i = i
            break
    chosen = ranked[: chosen_i + 1]
    chosen_eval = prefix_eval[chosen_i]
    achieved = float(drops[chosen_i])
    return chosen, chosen_eval, {"best_drop": best_drop, "goal": goal, "achieved": achieved}


def extract_edges_by_intervention(
    model,
    tok,
    collector: GateCollector,
    d_ff: int,
    concept_prompts: List[str],
    selected_flat: List[int],
    alpha: float,
):
    if not selected_flat:
        return []

    # Baseline activations on selected neurons
    baseline = []
    for p in concept_prompts:
        collector.reset()
        _ = run_prompt(model, tok, p)
        baseline.append(collector.get_selected(selected_flat, d_ff))
    baseline = np.stack(baseline, axis=0)  # [P, N]

    edges = []
    for si, src_idx in enumerate(selected_flat):
        h = register_additive_intervention(model, src_idx, d_ff, alpha=alpha)
        try:
            changed = []
            for p in concept_prompts:
                collector.reset()
                _ = run_prompt(model, tok, p)
                changed.append(collector.get_selected(selected_flat, d_ff))
            changed = np.stack(changed, axis=0)
        finally:
            h.remove()

        delta = changed - baseline  # [P,N]
        src_l = src_idx // d_ff
        for ti, tgt_idx in enumerate(selected_flat):
            if ti == si:
                continue
            tgt_l = tgt_idx // d_ff
            if tgt_l <= src_l:
                continue
            score = float(np.mean(np.abs(delta[:, ti])))
            if score <= 1e-7:
                continue
            edges.append(
                {
                    "src_flat": int(src_idx),
                    "src_layer": int(src_l),
                    "src_neuron": int(src_idx % d_ff),
                    "tgt_flat": int(tgt_idx),
                    "tgt_layer": int(tgt_l),
                    "tgt_neuron": int(tgt_idx % d_ff),
                    "transmission_score": score,
                }
            )
    edges.sort(key=lambda x: x["transmission_score"], reverse=True)
    return edges


def layer_dist_from_subset(subset: List[Dict], d_ff: int):
    ctr = Counter([(x["flat_idx"] // d_ff) for x in subset])
    return dict(sorted(ctr.items()))


def subset_overlap(a: List[Dict], b: List[Dict]):
    sa = set(x["flat_idx"] for x in a)
    sb = set(x["flat_idx"] for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def main():
    parser = argparse.ArgumentParser(description="Micro causal encoding graph for apple/banana")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--concept-a", default="apple")
    parser.add_argument("--concept-b", default="banana")
    parser.add_argument("--candidate-k", type=int, default=28)
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
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_micro_causal_{args.concept_a}_{args.concept_b}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    collector = GateCollector(model)

    try:
        neg_prompts = build_negative_prompts()

        concepts = {}
        for concept in [args.concept_a, args.concept_b]:
            pos_prompts = build_discovery_prompts(concept)
            role_eval = build_role_eval(concept)

            stat = collect_activation_stats(model, tok, collector, pos_prompts, neg_prompts)
            candidates = top_candidates_from_stats(stat, args.candidate_k)

            scored, baseline = single_neuron_role_scores(
                model=model,
                tok=tok,
                d_ff=d_ff,
                candidates=candidates,
                role_eval=role_eval,
                off_penalty=args.off_penalty,
            )

            role_subsets = {}
            role_metrics = {}
            for role in ROLES:
                subset, sev, prog = minimal_subset_for_role(
                    model=model,
                    tok=tok,
                    d_ff=d_ff,
                    ranked=scored[role],
                    role=role,
                    role_eval=role_eval,
                    max_subset=args.max_subset,
                    target_ratio=args.target_ratio,
                )
                role_subsets[role] = subset
                role_metrics[role] = {
                    "size": len(subset),
                    "layer_distribution": layer_dist_from_subset(subset, d_ff),
                    "progress": prog,
                    "drop_target": float(sev["base"][role] - sev["after"][role]),
                    "off_shift_sum": float(
                        sum(abs(sev["after"][r] - sev["base"][r]) for r in ROLES if r != role)
                    ),
                    "subset_neurons": [
                        {
                            "layer": int(x["flat_idx"] // d_ff),
                            "neuron": int(x["flat_idx"] % d_ff),
                            "flat_idx": int(x["flat_idx"]),
                            "utility": float(x["utility"]),
                        }
                        for x in subset
                    ],
                }

            union_flat = sorted(set(x["flat_idx"] for role in ROLES for x in role_subsets[role]))
            edges = extract_edges_by_intervention(
                model=model,
                tok=tok,
                collector=collector,
                d_ff=d_ff,
                concept_prompts=pos_prompts,
                selected_flat=union_flat,
                alpha=args.edge_alpha,
            )

            concepts[concept] = {
                "candidates_layer_distribution": layer_dist_from_subset(candidates, d_ff),
                "baseline_role_mass": baseline,
                "role_subsets": role_metrics,
                "knowledge_network": {
                    "n_nodes": len(union_flat),
                    "n_edges": len(edges),
                    "top_edges": edges[:120],
                },
            }

        # Cross concept sharing
        overlaps = {}
        for role in ROLES:
            a_sub = concepts[args.concept_a]["role_subsets"][role]["subset_neurons"]
            b_sub = concepts[args.concept_b]["role_subsets"][role]["subset_neurons"]
            a_conv = [{"flat_idx": x["flat_idx"]} for x in a_sub]
            b_conv = [{"flat_idx": x["flat_idx"]} for x in b_sub]
            overlaps[role] = subset_overlap(a_conv, b_conv)

        # Fruit-concept subset and ablation location/effect
        fruit_union = sorted(
            set(x["flat_idx"] for x in concepts[args.concept_a]["role_subsets"]["fruit"]["subset_neurons"])
            | set(x["flat_idx"] for x in concepts[args.concept_b]["role_subsets"]["fruit"]["subset_neurons"])
        )
        fruit_layer_dist = dict(sorted(Counter([idx // d_ff for idx in fruit_union]).items()))

        # Evaluate fruit-union ablation impact on both concepts and roles
        eval_a = build_role_eval(args.concept_a)
        eval_b = build_role_eval(args.concept_b)
        base_a = {r: eval_mass(model, tok, eval_a[r]) for r in ROLES}
        base_b = {r: eval_mass(model, tok, eval_b[r]) for r in ROLES}
        h = register_ablation(model, fruit_union, d_ff) if fruit_union else []
        try:
            after_a = {r: eval_mass(model, tok, eval_a[r]) for r in ROLES}
            after_b = {r: eval_mass(model, tok, eval_b[r]) for r in ROLES}
        finally:
            remove_handles(h)

        fruit_ablation_effect = {
            args.concept_a: {r: float(after_a[r] - base_a[r]) for r in ROLES},
            args.concept_b: {r: float(after_b[r] - base_b[r]) for r in ROLES},
        }

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "concept_a": args.concept_a,
                "concept_b": args.concept_b,
                "candidate_k": args.candidate_k,
                "max_subset": args.max_subset,
                "target_ratio": args.target_ratio,
                "off_penalty": args.off_penalty,
                "edge_alpha": args.edge_alpha,
                "d_ff": d_ff,
                "n_layers": len(model.model.layers),
            },
            "concepts": concepts,
            "cross_concept_shared_ratio": overlaps,
            "fruit_concept_subset": {
                "size": len(fruit_union),
                "layer_distribution": fruit_layer_dist,
            },
            "fruit_concept_ablation_effect": fruit_ablation_effect,
        }

        json_path = out_dir / "micro_causal_encoding_graph_results.json"
        md_path = out_dir / "MICRO_CAUSAL_ENCODING_GRAPH_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# 微观因果编码图 (MCEG) 报告", ""]
        lines.append(f"- 概念A: {args.concept_a}, 概念B: {args.concept_b}")
        lines.append(f"- fruit概念子集大小: {result['fruit_concept_subset']['size']}")
        lines.append(f"- fruit概念层分布: {result['fruit_concept_subset']['layer_distribution']}")
        lines.append("")
        for concept in [args.concept_a, args.concept_b]:
            lines.append(f"## {concept}")
            for role in ROLES:
                rr = concepts[concept]["role_subsets"][role]
                lines.append(
                    f"- {role}: size={rr['size']}, drop={rr['drop_target']:.8f}, off={rr['off_shift_sum']:.8f}, layers={rr['layer_distribution']}"
                )
            kn = concepts[concept]["knowledge_network"]
            lines.append(f"- knowledge network: nodes={kn['n_nodes']}, edges={kn['n_edges']}")
            lines.append("")
        lines.append("## 跨概念共享比")
        for role in ROLES:
            lines.append(f"- {role}: {overlaps[role]:.4f}")
        lines.append("")
        lines.append("## fruit概念消融效应")
        for concept in [args.concept_a, args.concept_b]:
            lines.append(f"- {concept}: {fruit_ablation_effect[concept]}")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        for concept in [args.concept_a, args.concept_b]:
            s = concepts[concept]["role_subsets"]
            print(
                f"[OK] {concept}: entity={s['entity']['size']} size={s['size']['size']} weight={s['weight']['size']} fruit={s['fruit']['size']} nodes={concepts[concept]['knowledge_network']['n_nodes']}"
            )
        print(f"[OK] fruit_shared size={len(fruit_union)} shared_ratio={overlaps}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
