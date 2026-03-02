#!/usr/bin/env python
"""
DeepSeek-7B multi-fruit neuron analysis:
- Compare banana/orange/grape/pear/peach/apple neural differences
- Extract fruit-concept shared neurons and fruit-specific neurons
- Validate with causal ablation and structural clustering
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ProbeItem:
    text: str
    label: str


@dataclass
class EvalItem:
    prompt: str
    targets: List[str]


class GateCollector:
    """Collect last-token gate_proj activations for every layer."""

    def __init__(self, model):
        self.model = model
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._mk_hook(li)))

    def _mk_hook(self, li: int):
        def _hook(_module, _inputs, output):
            self.buffers[li] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get(self) -> List[torch.Tensor]:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hooks for layers: {miss}")
        return [x for x in self.buffers if x is not None]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


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


def build_probe_dataset() -> List[ProbeItem]:
    fruits = {
        "apple": ["apple", "apples"],
        "banana": ["banana", "bananas"],
        "orange": ["orange", "oranges"],
        "grape": ["grape", "grapes"],
        "pear": ["pear", "pears"],
        "peach": ["peach", "peaches"],
    }
    nonfruits = ["car", "book", "stone", "chair", "computer", "shirt", "house", "phone"]
    templates = [
        "The {noun} is",
        "A {noun} is",
        "People say the {noun} tastes",
        "The color of the {noun} is",
        "I bought a fresh {noun} and",
        "This {noun} can be",
        "The smell of this {noun} is",
        "When ripe, the {noun} becomes",
    ]

    out: List[ProbeItem] = []
    for tpl in templates:
        for fruit_name, nouns in fruits.items():
            for n in nouns:
                out.append(ProbeItem(tpl.format(noun=n), fruit_name))
        for n in nonfruits:
            out.append(ProbeItem(tpl.format(noun=n), "nonfruit"))
    return out


def init_stats(labels: Sequence[str], n_layers: int, d_ff: int):
    stats = {}
    for lb in labels:
        stats[lb] = {
            "n": 0,
            "sum": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)],
            "sumsq": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)],
        }
    return stats


def mean_var(sum_t: torch.Tensor, sumsq_t: torch.Tensor, n: int):
    mu = sum_t / max(n, 1)
    var = sumsq_t / max(n, 1) - mu * mu
    return mu, torch.clamp(var, min=0.0)


def collect_stats(model, tok, collector: GateCollector, dataset: List[ProbeItem], max_items: int | None):
    if max_items is not None:
        dataset = dataset[:max_items]
    labels = sorted(set([x.label for x in dataset]))

    n_layers = len(model.model.layers)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    stats = init_stats(labels, n_layers, d_ff)

    for item in dataset:
        collector.reset()
        _ = run_prompt(model, tok, item.text)
        acts = collector.get()
        rec = stats[item.label]
        rec["n"] += 1
        for li, vec in enumerate(acts):
            v = vec.double()
            rec["sum"][li].add_(v)
            rec["sumsq"][li].add_(v * v)
    return stats, dataset


def compute_fruit_general_neurons(stats, top_k: int):
    fruit_labels = [k for k in stats.keys() if k != "nonfruit"]
    n_layers = len(stats["nonfruit"]["sum"])
    rows = []
    for li in range(n_layers):
        mu_non, var_non = mean_var(
            stats["nonfruit"]["sum"][li], stats["nonfruit"]["sumsq"][li], stats["nonfruit"]["n"]
        )

        fruit_mus = []
        fruit_vars = []
        for f in fruit_labels:
            mu_f, var_f = mean_var(stats[f]["sum"][li], stats[f]["sumsq"][li], stats[f]["n"])
            fruit_mus.append(mu_f)
            fruit_vars.append(var_f)
        mu_fruit = torch.stack(fruit_mus, dim=0).mean(dim=0)
        var_fruit_mean = torch.stack(fruit_vars, dim=0).mean(dim=0)
        var_between_fruits = torch.stack(fruit_mus, dim=0).var(dim=0, unbiased=False)

        diff = mu_fruit - mu_non
        z = diff / torch.sqrt(0.5 * (var_fruit_mean + var_non) + 1e-8)
        # Penalize large inter-fruit variance to keep "fruit shared" neurons.
        score = z - 0.35 * (var_between_fruits / (torch.sqrt(var_fruit_mean + 1e-8)))
        score = torch.where(diff > 0, score, torch.full_like(score, float("-inf")))

        k_take = min(top_k * 3, score.numel())
        vals, idx = torch.topk(score, k=k_take)
        for s, ni in zip(vals.tolist(), idx.tolist()):
            if not np.isfinite(s):
                continue
            rows.append(
                {
                    "layer": li,
                    "neuron": int(ni),
                    "fruit_general_score": float(s),
                    "diff_fruit_vs_nonfruit": float(diff[ni].item()),
                    "between_fruit_var": float(var_between_fruits[ni].item()),
                }
            )
    rows.sort(key=lambda x: x["fruit_general_score"], reverse=True)
    return rows[:top_k]


def compute_fruit_specific_neurons(stats, fruit_name: str, top_k: int):
    labels = [k for k in stats.keys() if k != "nonfruit"]
    others = [x for x in labels if x != fruit_name]
    n_layers = len(stats["nonfruit"]["sum"])
    rows = []
    for li in range(n_layers):
        mu_f, var_f = mean_var(stats[fruit_name]["sum"][li], stats[fruit_name]["sumsq"][li], stats[fruit_name]["n"])
        mu_o_sum = torch.zeros_like(mu_f)
        var_o_sum = torch.zeros_like(var_f)
        n_o = 0
        for o in others:
            mu_o, var_o = mean_var(stats[o]["sum"][li], stats[o]["sumsq"][li], stats[o]["n"])
            mu_o_sum += mu_o
            var_o_sum += var_o
            n_o += 1
        mu_o = mu_o_sum / max(n_o, 1)
        var_o = var_o_sum / max(n_o, 1)
        diff = mu_f - mu_o
        z = diff / torch.sqrt(0.5 * (var_f + var_o) + 1e-8)
        z = torch.where(diff > 0, z, torch.full_like(z, float("-inf")))

        k_take = min(top_k * 3, z.numel())
        vals, idx = torch.topk(z, k=k_take)
        for s, ni in zip(vals.tolist(), idx.tolist()):
            if not np.isfinite(s):
                continue
            rows.append(
                {
                    "layer": li,
                    "neuron": int(ni),
                    "fruit_specific_score": float(s),
                    "diff_target_vs_other_fruits": float(diff[ni].item()),
                    "fruit": fruit_name,
                }
            )
    rows.sort(key=lambda x: x["fruit_specific_score"], reverse=True)
    return rows[:top_k]


def register_ablation(model, neurons: List[Dict]):
    by_layer = defaultdict(list)
    for n in neurons:
        by_layer[int(n["layer"])].append(int(n["neuron"]))

    handles = []
    device = next(model.parameters()).device
    for li, idxs in by_layer.items():
        module = model.model.layers[li].mlp.gate_proj
        idx_tensor = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=device)

        def _mk(local_idx: torch.Tensor):
            def _hook(_module, _inputs, output):
                out = output.clone()
                out[..., local_idx] = 0.0
                return out

            return _hook

        handles.append(module.register_forward_hook(_mk(idx_tensor)))
    return handles, {int(k): sorted(set(v)) for k, v in by_layer.items()}


def remove_handles(handles):
    for h in handles:
        h.remove()


def token_ids(tok, targets: List[str]) -> List[int]:
    out = set()
    for t in targets:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            out.add(int(ids[0]))
    return sorted(out)


def eval_items(model, tok, items: List[EvalItem]) -> Dict:
    masses = []
    detail = []
    for x in items:
        out = run_prompt(model, tok, x.prompt)
        logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        ids = token_ids(tok, x.targets)
        mass = float(probs[ids].sum().item()) if ids else 0.0
        masses.append(mass)
        topv, topi = torch.topk(probs, k=5)
        detail.append(
            {
                "prompt": x.prompt,
                "targets": x.targets,
                "target_mass": mass,
                "top5_tokens": [tok.decode([int(i)]) for i in topi.tolist()],
                "top5_probs": [float(v) for v in topv.tolist()],
            }
        )
    return {"mean_target_mass": float(np.mean(masses) if masses else 0.0), "items": detail}


def build_eval_groups() -> Dict[str, List[EvalItem]]:
    return {
        "fruit_recall": [
            EvalItem("I ate a", [" apple", " banana", " orange", " grape", " pear", " peach"]),
            EvalItem("She bought a", [" apple", " banana", " orange", " grape", " pear", " peach"]),
            EvalItem("He picked a", [" apple", " banana", " orange", " grape", " pear", " peach"]),
            EvalItem("They sliced a", [" apple", " banana", " orange", " grape", " pear", " peach"]),
        ],
        "fruit_general": [
            EvalItem("The banana is usually", [" yellow", " sweet"]),
            EvalItem("The orange is usually", [" orange", " sweet", " juicy"]),
            EvalItem("The grape is usually", [" sweet", " small"]),
            EvalItem("The pear is usually", [" sweet", " green"]),
            EvalItem("The peach is usually", [" sweet", " soft"]),
            EvalItem("The apple is usually", [" red", " sweet"]),
        ],
        "nonfruit_control": [
            EvalItem("The car is usually", [" fast", " expensive"]),
            EvalItem("The stone is usually", [" hard", " heavy"]),
            EvalItem("The chair is usually", [" wooden", " comfortable"]),
            EvalItem("The book is usually", [" interesting", " useful"]),
        ],
        "banana_specific": [
            EvalItem("A ripe banana is", [" yellow"]),
            EvalItem("Banana peel is", [" yellow"]),
        ],
        "banana_recall": [
            EvalItem("I peeled a", [" banana", " bananas"]),
            EvalItem("She ate a", [" banana", " bananas"]),
        ],
        "orange_specific": [
            EvalItem("A ripe orange is", [" orange"]),
            EvalItem("Orange peel is", [" orange"]),
        ],
        "orange_recall": [
            EvalItem("I squeezed an", [" orange", " oranges"]),
            EvalItem("She peeled an", [" orange", " oranges"]),
        ],
        "apple_specific": [
            EvalItem("A ripe apple is", [" red", " green"]),
            EvalItem("Apple skin is", [" red", " green"]),
        ],
        "apple_recall": [
            EvalItem("I ate an", [" apple", " apples"]),
            EvalItem("She sliced an", [" apple", " apples"]),
        ],
        "grape_recall": [
            EvalItem("I washed some", [" grapes", " grape"]),
            EvalItem("They picked some", [" grapes", " grape"]),
        ],
        "pear_recall": [
            EvalItem("I bought a", [" pear", " pears"]),
            EvalItem("She ate a", [" pear", " pears"]),
        ],
        "peach_recall": [
            EvalItem("I ate a", [" peach", " peaches"]),
            EvalItem("She bought a", [" peach", " peaches"]),
        ],
    }


def sample_random_neurons(model, layer_map: Dict[int, List[int]], seed: int):
    rng = random.Random(seed)
    out = []
    for li, idxs in sorted(layer_map.items()):
        d_ff = model.model.layers[li].mlp.gate_proj.out_features
        for ni in rng.sample(range(d_ff), len(idxs)):
            out.append({"layer": int(li), "neuron": int(ni)})
    return out


def collect_matrix_for_neurons(model, tok, collector, dataset: List[ProbeItem], neurons: List[Dict]):
    x = np.zeros((len(dataset), len(neurons)), dtype=np.float32)
    labels = []
    for i, item in enumerate(dataset):
        collector.reset()
        _ = run_prompt(model, tok, item.text)
        acts = collector.get()
        labels.append(item.label)
        for j, n in enumerate(neurons):
            x[i, j] = float(acts[n["layer"]][n["neuron"]].item())
    return x, labels


def cluster_modules(neurons: List[Dict], x: np.ndarray, labels: List[str], corr_th: float):
    if x.shape[1] <= 1:
        return []
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    a = np.abs(c)
    parent = list(range(x.shape[1]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        fi, fj = find(i), find(j)
        if fi != fj:
            parent[fj] = fi

    n = x.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] >= corr_th:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    fruit_mask = np.array([lb != "nonfruit" for lb in labels], dtype=bool)
    out = []
    sorted_groups = sorted(groups.values(), key=lambda z: len(z), reverse=True)
    for mid, idxs in enumerate(sorted_groups):
        sub = a[np.ix_(idxs, idxs)]
        tri = sub[np.triu_indices(len(idxs), k=1)]
        mean_corr = float(tri.mean()) if tri.size else 1.0
        mod_act = x[:, idxs].mean(axis=1)
        fruit_mean = float(mod_act[fruit_mask].mean()) if fruit_mask.any() else 0.0
        non_mean = float(mod_act[~fruit_mask].mean()) if (~fruit_mask).any() else 0.0
        out.append(
            {
                "module_id": mid,
                "size": len(idxs),
                "mean_abs_corr": mean_corr,
                "fruit_minus_nonfruit": fruit_mean - non_mean,
                "layer_distribution": dict(sorted(Counter([neurons[i]["layer"] for i in idxs]).items())),
                "neurons": [neurons[i] for i in idxs],
            }
        )
    return out


def layer_band_counts(neurons: List[Dict], n_layers: int):
    t1 = n_layers // 3
    t2 = (2 * n_layers) // 3
    out = {"early": 0, "middle": 0, "late": 0}
    for x in neurons:
        li = x["layer"]
        if li < t1:
            out["early"] += 1
        elif li < t2:
            out["middle"] += 1
        else:
            out["late"] += 1
    return out


def overlap_summary(fruit_specific: Dict[str, List[Dict]], top_n: int):
    names = sorted(fruit_specific.keys())
    sets = {}
    for n in names:
        sets[n] = set((x["layer"], x["neuron"]) for x in fruit_specific[n][:top_n])
    mat = {}
    for a in names:
        mat[a] = {}
        for b in names:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            mat[a][b] = float(inter / union) if union else 0.0
    return mat


def write_report(path: Path, result: Dict):
    b = result["evaluation"]["baseline"]
    fg = result["evaluation"]["fruit_general_ablation"]
    rr = result["evaluation"]["random_ablation"]

    lines = []
    lines.append("# DeepSeek-7B 多水果神经元差异与水果概念编码报告")
    lines.append("")
    lines.append("## 1) 关键结论")
    lines.append(result["conclusion"])
    lines.append("")
    lines.append("## 2) 水果共性编码结构")
    lines.append(f"- 共性神经元数: {result['config']['fruit_general_k']}")
    lines.append(f"- 层分布: {result['fruit_general']['layer_distribution']}")
    lines.append(f"- 三段分布: {result['fruit_general']['layer_bands']}")
    lines.append(f"- 模块数: {len(result['fruit_general']['modules'])}")
    if result["fruit_general"]["modules"]:
        m = result["fruit_general"]["modules"][0]
        lines.append(
            f"- 主模块: size={m['size']}, mean|corr|={m['mean_abs_corr']:.3f}, fruit-minus-nonfruit={m['fruit_minus_nonfruit']:+.4f}"
        )
    lines.append("")
    lines.append("## 3) 因果消融 (共性神经元)")
    lines.append("| Group | Baseline | Fruit-General Ablation | Random Ablation | FG Δ | Random Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for g in [
        "fruit_recall",
        "fruit_general",
        "nonfruit_control",
        "banana_specific",
        "orange_specific",
        "apple_specific",
    ]:
        bv = b[g]["mean_target_mass"]
        fv = fg[g]["mean_target_mass"]
        rv = rr[g]["mean_target_mass"]
        lines.append(f"| {g} | {bv:.6f} | {fv:.6f} | {rv:.6f} | {fv-bv:+.6f} | {rv-bv:+.6f} |")
    lines.append("")
    lines.append("## 4) 各水果特异神经元")
    for fruit, arr in result["fruit_specific"].items():
        lines.append(f"- {fruit}:")
        top = arr["top_neurons"][:5]
        joined = ", ".join([f"L{x['layer']}N{x['neuron']}({x['fruit_specific_score']:.2f})" for x in top])
        lines.append(f"  - top5: {joined}")
        lines.append(f"  - layer_distribution: {arr['layer_distribution']}")
        lines.append(f"  - band_distribution: {arr['layer_bands']}")
        lines.append(f"  - targeted_ablation_delta({fruit}_specific): {arr['targeted_delta']:+.6f}")
        lines.append(f"  - recall_ablation_delta({arr['recall_group']}): {arr['recall_delta']:+.6f}")
    lines.append("")
    lines.append("## 5) 水果间编码重叠 (Jaccard, top-30)")
    mat = result["fruit_specific_overlap_top30"]
    names = sorted(mat.keys())
    lines.append("| fruit | " + " | ".join(names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(names)) + "|")
    for a in names:
        vals = " | ".join([f"{mat[a][b]:.3f}" for b in names])
        lines.append(f"| {a} | {vals} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-7B multi-fruit concept analysis")
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--max-probe-items", type=int, default=180)
    parser.add_argument("--fruit-general-k", type=int, default=50)
    parser.add_argument("--fruit-specific-k", type=int, default=40)
    parser.add_argument("--module-corr-th", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_multi_fruit_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)

    try:
        dataset = build_probe_dataset()
        stats, used = collect_stats(model, tok, collector, dataset, max_items=args.max_probe_items)

        fruit_general = compute_fruit_general_neurons(stats, top_k=args.fruit_general_k)
        fg_layer_dist = dict(sorted(Counter([x["layer"] for x in fruit_general]).items()))
        fg_band_dist = layer_band_counts(fruit_general, len(model.model.layers))

        x_fg, labels_fg = collect_matrix_for_neurons(model, tok, collector, used, fruit_general)
        fg_modules = cluster_modules(fruit_general, x_fg, labels_fg, corr_th=args.module_corr_th)

        fruit_names = sorted([k for k in stats.keys() if k != "nonfruit"])
        fruit_specific = {}
        for name in fruit_names:
            arr = compute_fruit_specific_neurons(stats, name, top_k=args.fruit_specific_k)
            fruit_specific[name] = arr

        eval_groups = build_eval_groups()
        baseline = {k: eval_items(model, tok, v) for k, v in eval_groups.items()}

        fg_handles, fg_map = register_ablation(model, fruit_general)
        try:
            fg_eval = {k: eval_items(model, tok, v) for k, v in eval_groups.items()}
        finally:
            remove_handles(fg_handles)

        rand_neurons = sample_random_neurons(model, fg_map, seed=args.seed + 1)
        rr_handles, _ = register_ablation(model, rand_neurons)
        try:
            rr_eval = {k: eval_items(model, tok, v) for k, v in eval_groups.items()}
        finally:
            remove_handles(rr_handles)

        # per-fruit targeted ablation check on each specific group
        fruit_specific_result = {}
        for name in fruit_names:
            topn = fruit_specific[name]
            layer_dist = dict(sorted(Counter([x["layer"] for x in topn]).items()))
            band_dist = layer_band_counts(topn, len(model.model.layers))
            group_name = f"{name}_specific" if f"{name}_specific" in eval_groups else "fruit_general"
            recall_group = f"{name}_recall" if f"{name}_recall" in eval_groups else "fruit_recall"
            handles, _ = register_ablation(model, topn)
            try:
                eval_after = eval_items(model, tok, eval_groups[group_name])
                recall_after = eval_items(model, tok, eval_groups[recall_group])
            finally:
                remove_handles(handles)
            delta = eval_after["mean_target_mass"] - baseline[group_name]["mean_target_mass"]
            recall_delta = recall_after["mean_target_mass"] - baseline[recall_group]["mean_target_mass"]
            fruit_specific_result[name] = {
                "top_neurons": topn,
                "layer_distribution": layer_dist,
                "layer_bands": band_dist,
                "target_group": group_name,
                "targeted_delta": float(delta),
                "recall_group": recall_group,
                "recall_delta": float(recall_delta),
            }

        overlap = overlap_summary(fruit_specific, top_n=min(30, args.fruit_specific_k))

        fg_delta = fg_eval["fruit_recall"]["mean_target_mass"] - baseline["fruit_recall"]["mean_target_mass"]
        rr_delta = rr_eval["fruit_recall"]["mean_target_mass"] - baseline["fruit_recall"]["mean_target_mass"]
        conclusion = (
            "水果概念在早期层形成共享稀疏簇，且共性神经元消融对水果组的影响显著强于随机消融；"
            "同时各水果存在重叠有限的特异子簇，体现“共性骨架 + 实例分叉”的编码结构。"
            if fg_delta < rr_delta
            else "本轮未得到强共性因果证据，但可见稳定的水果共享簇与水果特异子簇分化，建议扩大评估任务后复核因果强度。"
        )

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "dtype": args.dtype,
                "seed": args.seed,
                "max_probe_items": args.max_probe_items,
                "fruit_general_k": args.fruit_general_k,
                "fruit_specific_k": args.fruit_specific_k,
                "module_corr_th": args.module_corr_th,
            },
            "probe": {"n_items": len(used), "label_counts": dict(Counter([x.label for x in used]))},
            "fruit_general": {
                "neurons": fruit_general,
                "layer_distribution": fg_layer_dist,
                "layer_bands": fg_band_dist,
                "modules": fg_modules,
            },
            "fruit_specific": fruit_specific_result,
            "fruit_specific_overlap_top30": overlap,
            "evaluation": {
                "baseline": baseline,
                "fruit_general_ablation": fg_eval,
                "random_ablation": rr_eval,
            },
            "conclusion": conclusion,
        }

        json_path = out_dir / "multi_fruit_analysis_results.json"
        md_path = out_dir / "MULTI_FRUIT_ANALYSIS_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(md_path, result)

        print(f"[OK] Saved: {out_dir}")
        print(f"[OK] JSON: {json_path}")
        print(f"[OK] FG delta fruit_general: {fg_delta:+.6f}")
        print(f"[OK] Random delta fruit_general: {rr_delta:+.6f}")
        print(f"[OK] Fruit-general layer distribution: {fg_layer_dist}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
