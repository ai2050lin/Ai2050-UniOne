#!/usr/bin/env python
"""
DeepSeek-7B apple concept neuron ablation analysis.

Pipeline:
1) Find apple-selective MLP gate neurons (layer, neuron index).
2) Run targeted ablation and random ablation.
3) Compare token-probability mass shifts on apple/fruit/non-fruit tasks.
4) Summarize the encoding structure (layer bands + correlated neuron modules).
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
class PromptItem:
    text: str
    group: str  # apple / fruit_control / nonfruit_control
    noun: str


@dataclass
class EvalItem:
    prompt: str
    target_tokens: List[str]


class GateCollector:
    """Collects last-token MLP gate projection activations for all layers."""

    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.layers = list(model.model.layers)
        self.buffers: List[torch.Tensor | None] = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            # output: [batch, seq, intermediate_size]
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self) -> None:
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get(self) -> List[torch.Tensor]:
        missing = [i for i, x in enumerate(self.buffers) if x is None]
        if missing:
            raise RuntimeError(f"Missing layer activations for layers: {missing}")
        return [x for x in self.buffers if x is not None]

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def build_discovery_dataset() -> List[PromptItem]:
    templates = [
        "The {noun} is",
        "A {noun} is",
        "People say the {noun} tastes",
        "The color of the {noun} is",
        "I bought a fresh {noun} and",
        "This {noun} can be",
        "Everyone knows the {noun} is",
        "On the table, the {noun} looks",
        "The smell of this {noun} is",
        "When ripe, the {noun} becomes",
    ]
    apple_nouns = ["apple", "apples"]
    fruit_controls = ["banana", "orange", "grape", "pear", "peach"]
    nonfruit_controls = ["car", "stone", "book", "chair", "computer"]

    out: List[PromptItem] = []
    for t in templates:
        for n in apple_nouns:
            out.append(PromptItem(text=t.format(noun=n), group="apple", noun=n))
        for n in fruit_controls:
            out.append(PromptItem(text=t.format(noun=n), group="fruit_control", noun=n))
        for n in nonfruit_controls:
            out.append(PromptItem(text=t.format(noun=n), group="nonfruit_control", noun=n))
    return out


def build_eval_groups() -> Dict[str, List[EvalItem]]:
    return {
        "apple_attribute": [
            EvalItem("The apple is usually", [" red", " sweet", " juicy"]),
            EvalItem("An apple tastes", [" sweet", " good", " delicious"]),
            EvalItem("A ripe apple is often", [" red", " sweet", " juicy"]),
            EvalItem("Apple pie is made from", [" apples", " apple"]),
        ],
        "fruit_attribute": [
            EvalItem("The banana is usually", [" yellow", " sweet"]),
            EvalItem("An orange tastes", [" sweet", " sour", " juicy"]),
            EvalItem("A grape is usually", [" small", " sweet"]),
            EvalItem("A pear is often", [" sweet", " green"]),
        ],
        "nonfruit_attribute": [
            EvalItem("The car is usually", [" fast", " expensive", " new"]),
            EvalItem("The stone is usually", [" hard", " heavy"]),
            EvalItem("The chair is usually", [" wooden", " comfortable"]),
            EvalItem("The book is usually", [" interesting", " useful"]),
        ],
        "apple_recall": [
            EvalItem("I ate an", [" apple", " apples"]),
            EvalItem("She bought an", [" apple", " apples"]),
            EvalItem("He picked an", [" apple", " apples"]),
            EvalItem("They sliced an", [" apple", " apples"]),
        ],
    }


def load_model_and_tokenizer(model_id: str, dtype_name: str, local_files_only: bool):
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_prompt(model, tokenizer, text: str):
    device = next(model.parameters()).device
    toks = tokenizer(text, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.inference_mode():
        out = model(**toks, use_cache=False, return_dict=True)
    return out


def init_stats(groups: Sequence[str], n_layers: int, d_ff: int):
    stats = {}
    for g in groups:
        stats[g] = {
            "n": 0,
            "sum": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)],
            "sumsq": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)],
        }
    return stats


def accumulate_discovery_stats(
    model,
    tokenizer,
    collector: GateCollector,
    dataset: List[PromptItem],
    max_prompts: int | None = None,
):
    if max_prompts is not None:
        dataset = dataset[:max_prompts]

    n_layers = len(model.model.layers)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    stats = init_stats(["apple", "fruit_control", "nonfruit_control"], n_layers, d_ff)

    for item in dataset:
        collector.reset()
        _ = run_prompt(model, tokenizer, item.text)
        acts = collector.get()
        group_stats = stats[item.group]
        group_stats["n"] += 1
        for li, vec in enumerate(acts):
            v = vec.double()
            group_stats["sum"][li].add_(v)
            group_stats["sumsq"][li].add_(v * v)
    return stats, dataset


def mean_var(sum_t: torch.Tensor, sumsq_t: torch.Tensor, n: int):
    mu = sum_t / max(n, 1)
    var = (sumsq_t / max(n, 1)) - (mu * mu)
    var = torch.clamp(var, min=0.0)
    return mu, var


def discover_top_neurons(stats, top_k: int):
    n_a = stats["apple"]["n"]
    n_f = stats["fruit_control"]["n"]
    n_n = stats["nonfruit_control"]["n"]
    n_c = n_f + n_n

    records = []
    n_layers = len(stats["apple"]["sum"])
    for li in range(n_layers):
        sum_a = stats["apple"]["sum"][li]
        sumsq_a = stats["apple"]["sumsq"][li]
        sum_f = stats["fruit_control"]["sum"][li]
        sumsq_f = stats["fruit_control"]["sumsq"][li]
        sum_n = stats["nonfruit_control"]["sum"][li]
        sumsq_n = stats["nonfruit_control"]["sumsq"][li]

        mu_a, var_a = mean_var(sum_a, sumsq_a, n_a)
        mu_f, var_f = mean_var(sum_f, sumsq_f, n_f)
        mu_n, var_n = mean_var(sum_n, sumsq_n, n_n)
        mu_c, var_c = mean_var(sum_f + sum_n, sumsq_f + sumsq_n, n_c)

        diff = mu_a - mu_c
        z = diff / torch.sqrt(0.5 * (var_a + var_c) + 1e-8)
        # Keep only positive apple-selective neurons.
        z = torch.where(diff > 0, z, torch.full_like(z, float("-inf")))

        k_layer = min(top_k * 3, z.numel())
        vals, idxs = torch.topk(z, k=k_layer)
        for score, ni in zip(vals.tolist(), idxs.tolist()):
            if not np.isfinite(score):
                continue
            records.append(
                {
                    "layer": li,
                    "neuron": int(ni),
                    "score_z": float(score),
                    "diff_apple_vs_control": float(diff[ni].item()),
                    "diff_apple_vs_fruit": float((mu_a[ni] - mu_f[ni]).item()),
                    "diff_apple_vs_nonfruit": float((mu_a[ni] - mu_n[ni]).item()),
                    "mean_apple": float(mu_a[ni].item()),
                    "mean_control": float(mu_c[ni].item()),
                }
            )

    records.sort(key=lambda x: (x["score_z"], x["diff_apple_vs_control"]), reverse=True)
    return records[:top_k]


def register_ablation_hooks(model, neurons: List[Dict]):
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for r in neurons:
        by_layer[int(r["layer"])].append(int(r["neuron"]))

    handles = []
    device = next(model.parameters()).device
    for layer_idx, idxs in by_layer.items():
        idx_tensor = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=device)
        module = model.model.layers[layer_idx].mlp.gate_proj

        def _make_hook(local_idxs: torch.Tensor):
            def _hook(_module, _inputs, output):
                out = output.clone()
                out[..., local_idxs] = 0.0
                return out

            return _hook

        handles.append(module.register_forward_hook(_make_hook(idx_tensor)))

    return handles, {k: sorted(set(v)) for k, v in by_layer.items()}


def remove_handles(handles):
    for h in handles:
        h.remove()


def first_token_ids(tokenizer, target_tokens: Sequence[str]) -> List[int]:
    ids = set()
    for t in target_tokens:
        token_ids = tokenizer(t, add_special_tokens=False).input_ids
        if token_ids:
            ids.add(int(token_ids[0]))
    return sorted(ids)


def evaluate_groups(model, tokenizer, eval_groups: Dict[str, List[EvalItem]]):
    per_group = {}
    for gname, items in eval_groups.items():
        item_results = []
        masses = []
        for item in items:
            out = run_prompt(model, tokenizer, item.prompt)
            logits = out.logits[0, -1, :].float().cpu()
            probs = torch.softmax(logits, dim=-1)
            target_ids = first_token_ids(tokenizer, item.target_tokens)
            mass = float(probs[target_ids].sum().item()) if target_ids else 0.0
            masses.append(mass)

            top_vals, top_idx = torch.topk(probs, k=5)
            top_tokens = [tokenizer.decode([int(i)]) for i in top_idx.tolist()]
            item_results.append(
                {
                    "prompt": item.prompt,
                    "target_tokens": item.target_tokens,
                    "target_first_token_ids": target_ids,
                    "target_prob_mass": mass,
                    "top5_tokens": top_tokens,
                    "top5_probs": [float(v) for v in top_vals.tolist()],
                }
            )

        per_group[gname] = {
            "n_items": len(items),
            "mean_target_prob_mass": float(np.mean(masses) if masses else 0.0),
            "items": item_results,
        }
    return per_group


def quick_group_mass(model, tokenizer, items: List[EvalItem]) -> float:
    masses = []
    for item in items:
        out = run_prompt(model, tokenizer, item.prompt)
        logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        target_ids = first_token_ids(tokenizer, item.target_tokens)
        mass = float(probs[target_ids].sum().item()) if target_ids else 0.0
        masses.append(mass)
    return float(np.mean(masses) if masses else 0.0)


def causal_rank_neurons(
    model,
    tokenizer,
    candidate_neurons: List[Dict],
    eval_items: List[EvalItem],
    baseline_mass: float,
):
    ranked = []
    for r in candidate_neurons:
        handles, _ = register_ablation_hooks(model, [r])
        try:
            mass = quick_group_mass(model, tokenizer, eval_items)
        finally:
            remove_handles(handles)
        delta = mass - baseline_mass
        item = dict(r)
        item["single_ablation_group_mass"] = mass
        item["single_ablation_delta"] = delta
        ranked.append(item)
    ranked.sort(key=lambda x: x["single_ablation_delta"])  # more negative = more causal
    return ranked


def collect_selected_matrix(
    model,
    tokenizer,
    collector: GateCollector,
    dataset: List[PromptItem],
    neurons: List[Dict],
):
    m = len(dataset)
    k = len(neurons)
    x = np.zeros((m, k), dtype=np.float32)
    labels = []
    for i, item in enumerate(dataset):
        collector.reset()
        _ = run_prompt(model, tokenizer, item.text)
        acts = collector.get()
        for j, r in enumerate(neurons):
            x[i, j] = float(acts[r["layer"]][r["neuron"]].item())
        labels.append(item.group)
    return x, labels


def build_modules(neurons: List[Dict], matrix: np.ndarray, labels: List[str], corr_th: float):
    if matrix.shape[1] <= 1:
        return []

    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)

    parent = list(range(matrix.shape[1]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = matrix.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            if abs_corr[i, j] >= corr_th:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    apple_mask = np.array([lab == "apple" for lab in labels], dtype=bool)
    control_mask = ~apple_mask

    modules = []
    for mid, idxs in enumerate(sorted(groups.values(), key=lambda z: len(z), reverse=True)):
        subcorr = abs_corr[np.ix_(idxs, idxs)]
        tri = subcorr[np.triu_indices(len(idxs), k=1)]
        mean_intra = float(tri.mean()) if tri.size > 0 else 1.0
        layer_counts = Counter([neurons[i]["layer"] for i in idxs])

        mod_act = matrix[:, idxs].mean(axis=1)
        apple_mean = float(mod_act[apple_mask].mean()) if apple_mask.any() else 0.0
        control_mean = float(mod_act[control_mask].mean()) if control_mask.any() else 0.0

        modules.append(
            {
                "module_id": mid,
                "size": len(idxs),
                "mean_abs_corr": mean_intra,
                "apple_minus_control_activation": apple_mean - control_mean,
                "layer_distribution": dict(sorted(layer_counts.items())),
                "neurons": [neurons[i] for i in idxs],
            }
        )
    return modules


def summarize_layer_bands(neurons: List[Dict], n_layers: int):
    # 3-band segmentation: early / middle / late
    t1 = n_layers // 3
    t2 = (2 * n_layers) // 3
    out = {"early": 0, "middle": 0, "late": 0}
    for r in neurons:
        li = r["layer"]
        if li < t1:
            out["early"] += 1
        elif li < t2:
            out["middle"] += 1
        else:
            out["late"] += 1
    return out


def sample_random_neurons(model, layer_to_count: Dict[int, List[int]], seed: int):
    rng = random.Random(seed)
    out = []
    for li, idxs in sorted(layer_to_count.items()):
        d_ff = model.model.layers[li].mlp.gate_proj.out_features
        count = len(idxs)
        sampled = rng.sample(range(d_ff), count)
        for ni in sampled:
            out.append({"layer": li, "neuron": ni})
    return out


def write_markdown_report(path: Path, result: Dict):
    eval_base = result["evaluation"]["baseline"]
    eval_tgt = result["evaluation"]["target_ablation"]
    eval_rand = result["evaluation"]["random_ablation"]

    lines = []
    lines.append("# DeepSeek-7B 苹果概念关键神经元消融报告")
    lines.append("")
    lines.append("## 1) 实验配置")
    lines.append(f"- 模型: `{result['model_id']}`")
    lines.append(f"- 设备: `{result['device']}`")
    lines.append(f"- 发现集样本数: `{result['discovery']['n_prompts']}`")
    lines.append(f"- 候选关键神经元(top-k): `{result['config']['top_k']}`")
    lines.append(f"- 消融神经元数: `{result['config']['ablate_k']}`")
    lines.append("")
    lines.append("## 2) 关键神经元层级结构")
    lines.append(f"- 三段层分布: `{result['encoding_structure']['layer_bands']}`")
    lines.append("- 选择性Top神经元 (前10):")
    for r in result["top_neurons"][:10]:
        lines.append(
            f"  - L{r['layer']} N{r['neuron']} | z={r['score_z']:.3f} | Δapple-control={r['diff_apple_vs_control']:.4f}"
        )
    lines.append("- 因果筛选后用于组合消融的神经元 (前10):")
    for r in result["ablation_neurons"][:10]:
        delta = r.get("single_ablation_delta", 0.0)
        lines.append(
            f"  - L{r['layer']} N{r['neuron']} | single Δ({result['config']['causal_group']})={delta:+.6f}"
        )
    lines.append("")
    lines.append("## 3) 消融因果效应 (目标概率质量均值)")
    lines.append("| Group | Baseline | Target Ablation | Random Ablation | Target Δ | Random Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for g in ["apple_attribute", "apple_recall", "fruit_attribute", "nonfruit_attribute"]:
        b = eval_base[g]["mean_target_prob_mass"]
        t = eval_tgt[g]["mean_target_prob_mass"]
        r = eval_rand[g]["mean_target_prob_mass"]
        lines.append(f"| {g} | {b:.6f} | {t:.6f} | {r:.6f} | {t-b:+.6f} | {r-b:+.6f} |")
    lines.append("")
    lines.append("## 4) 编码模块 (按相关性聚类)")
    lines.append(f"- 模块数: `{len(result['encoding_structure']['modules'])}`")
    for m in result["encoding_structure"]["modules"][:5]:
        lines.append(
            f"- Module {m['module_id']}: size={m['size']}, mean|corr|={m['mean_abs_corr']:.3f}, apple-control={m['apple_minus_control_activation']:+.4f}, layers={m['layer_distribution']}"
        )
    lines.append("")
    lines.append("## 5) 结论")
    lines.append(result["conclusion"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-7B apple neuron ablation analysis")
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=60, help="Top neurons to keep after discovery")
    parser.add_argument("--ablate-k", type=int, default=20, help="How many top neurons to ablate")
    parser.add_argument(
        "--causal-rank-pool",
        type=int,
        default=40,
        help="How many top selective neurons to run single-neuron causal scan on",
    )
    parser.add_argument(
        "--causal-group",
        type=str,
        default="apple_recall",
        choices=["apple_recall", "apple_attribute"],
        help="Group used for single-neuron causal ranking",
    )
    parser.add_argument("--max-discovery-prompts", type=int, default=120)
    parser.add_argument("--module-corr-th", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_apple_ablation_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)

    try:
        discovery_dataset = build_discovery_dataset()
        stats, used_dataset = accumulate_discovery_stats(
            model=model,
            tokenizer=tokenizer,
            collector=collector,
            dataset=discovery_dataset,
            max_prompts=args.max_discovery_prompts,
        )
        top_neurons = discover_top_neurons(stats, top_k=args.top_k)
        if not top_neurons:
            raise RuntimeError("No apple-selective neurons found.")

        # Evaluation: baseline.
        eval_groups = build_eval_groups()
        baseline_eval = evaluate_groups(model, tokenizer, eval_groups)
        baseline_causal_mass = baseline_eval[args.causal_group]["mean_target_prob_mass"]

        # Optional causal ranking on a selected pool.
        causal_pool = max(1, min(args.causal_rank_pool, len(top_neurons)))
        causal_ranked = causal_rank_neurons(
            model=model,
            tokenizer=tokenizer,
            candidate_neurons=top_neurons[:causal_pool],
            eval_items=eval_groups[args.causal_group],
            baseline_mass=baseline_causal_mass,
        )
        ablate_neurons = causal_ranked[: args.ablate_k]

        # Evaluation: targeted ablation.
        tgt_handles, tgt_map = register_ablation_hooks(model, ablate_neurons)
        try:
            target_eval = evaluate_groups(model, tokenizer, eval_groups)
        finally:
            remove_handles(tgt_handles)

        # Evaluation: random ablation with matched per-layer counts.
        random_neurons = sample_random_neurons(model, tgt_map, seed=args.seed + 1)
        rnd_handles, _ = register_ablation_hooks(model, random_neurons)
        try:
            random_eval = evaluate_groups(model, tokenizer, eval_groups)
        finally:
            remove_handles(rnd_handles)

        # Encoding structure from selected neurons.
        matrix, labels = collect_selected_matrix(model, tokenizer, collector, used_dataset, ablate_neurons)
        modules = build_modules(ablate_neurons, matrix, labels, corr_th=args.module_corr_th)
        layer_bands = summarize_layer_bands(ablate_neurons, n_layers=len(model.model.layers))
        layer_counts = Counter([r["layer"] for r in ablate_neurons])

        apple_drop = (
            target_eval["apple_attribute"]["mean_target_prob_mass"]
            - baseline_eval["apple_attribute"]["mean_target_prob_mass"]
        )
        nonfruit_drop = (
            target_eval["nonfruit_attribute"]["mean_target_prob_mass"]
            - baseline_eval["nonfruit_attribute"]["mean_target_prob_mass"]
        )
        conclusion = (
            "目标神经元消融后，apple 相关目标概率下降幅度显著大于非水果组，"
            "显示出苹果概念在中后层存在稀疏、可因果干预的编码结构。"
            if apple_drop < nonfruit_drop
            else "本轮未观察到强苹果特异消融效应，建议扩大发现集并增加任务约束。"
        )

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "dtype": args.dtype,
                "seed": args.seed,
                "top_k": args.top_k,
                "ablate_k": args.ablate_k,
                "causal_rank_pool": args.causal_rank_pool,
                "causal_group": args.causal_group,
                "max_discovery_prompts": args.max_discovery_prompts,
                "module_corr_th": args.module_corr_th,
            },
            "discovery": {
                "n_prompts": len(used_dataset),
                "group_counts": dict(Counter([x.group for x in used_dataset])),
            },
            "top_neurons": top_neurons,
            "causal_ranked_neurons": causal_ranked,
            "ablation_neurons": ablate_neurons,
            "ablation_layer_distribution": dict(sorted(layer_counts.items())),
            "evaluation": {
                "baseline": baseline_eval,
                "target_ablation": target_eval,
                "random_ablation": random_eval,
            },
            "encoding_structure": {
                "layer_bands": layer_bands,
                "modules": modules,
            },
            "conclusion": conclusion,
        }

        json_path = output_dir / "apple_neuron_ablation_results.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown_report(output_dir / "APPLE_NEURON_ABLATION_REPORT.md", result)

        print(f"[OK] Results saved to: {output_dir}")
        print(f"[OK] JSON: {json_path}")
        print("[OK] Top ablation neurons (first 10):")
        for r in ablate_neurons[:10]:
            print(
                f"  - L{r['layer']} N{r['neuron']} z={r['score_z']:.3f} Δ={r['diff_apple_vs_control']:.4f}"
            )

    finally:
        collector.close()


if __name__ == "__main__":
    main()
