#!/usr/bin/env python
"""
DeepSeek-7B plasticity-efficiency benchmark.

Compare:
1) One-shot Hebbian-style prototype write (single exposure per concept)
2) Multi-step SGD readout fit on the same one-shot supports

Goal:
Estimate how many SGD steps are needed to match one-shot prototype accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class NounItem:
    noun: str
    category: str


def default_noun_catalog() -> List[NounItem]:
    rows = [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"), ("pear", "fruit"),
        ("peach", "fruit"), ("mango", "fruit"), ("lemon", "fruit"), ("strawberry", "fruit"), ("watermelon", "fruit"),
        ("rabbit", "animal"), ("cat", "animal"), ("dog", "animal"), ("horse", "animal"), ("tiger", "animal"),
        ("lion", "animal"), ("bird", "animal"), ("fish", "animal"), ("elephant", "animal"), ("monkey", "animal"),
        ("sun", "celestial"), ("moon", "celestial"), ("star", "celestial"), ("planet", "celestial"), ("comet", "celestial"),
        ("galaxy", "celestial"), ("asteroid", "celestial"), ("meteor", "celestial"), ("satellite", "celestial"), ("nebula", "celestial"),
        ("car", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"), ("bicycle", "vehicle"), ("airplane", "vehicle"),
        ("ship", "vehicle"), ("truck", "vehicle"), ("motorcycle", "vehicle"), ("subway", "vehicle"), ("boat", "vehicle"),
        ("chair", "object"), ("table", "object"), ("bed", "object"), ("lamp", "object"), ("door", "object"),
        ("window", "object"), ("bottle", "object"), ("cup", "object"), ("spoon", "object"), ("knife", "object"),
        ("tree", "nature"), ("flower", "nature"), ("grass", "nature"), ("forest", "nature"), ("river", "nature"),
        ("mountain", "nature"), ("ocean", "nature"), ("desert", "nature"), ("leaf", "nature"), ("seed", "nature"),
        ("love", "abstract"), ("hate", "abstract"), ("justice", "abstract"), ("peace", "abstract"), ("war", "abstract"),
        ("music", "abstract"), ("art", "abstract"), ("history", "abstract"), ("future", "abstract"), ("memory", "abstract"),
    ]
    return [NounItem(noun=n, category=c) for n, c in rows]


def load_nouns(path: str | None, max_nouns: int | None) -> List[NounItem]:
    if not path:
        rows = default_noun_catalog()
        return rows[:max_nouns] if max_nouns else rows

    out: List[NounItem] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"nouns file not found: {path}")

    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            noun, cat = [x.strip() for x in s.split(",", 1)]
            if noun:
                out.append(NounItem(noun=noun, category=cat or "uncategorized"))
        else:
            out.append(NounItem(noun=s, category="uncategorized"))
    if max_nouns:
        out = out[:max_nouns]
    return out


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

    def reset(self) -> None:
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hook outputs for layers: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self) -> None:
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


def support_prompt(noun: str) -> str:
    return f"This is a {noun}"


def query_prompts(noun: str) -> List[str]:
    return [
        f"I saw a {noun}",
        f"People discuss {noun}",
        f"The {noun} is often",
        f"A {noun} can be",
    ]


def select_nouns(items: List[NounItem], n_categories: int, n_per_category: int, seed: int) -> List[NounItem]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[NounItem]] = defaultdict(list)
    for x in items:
        by_cat[x.category].append(x)
    cats = sorted([c for c, rows in by_cat.items() if len(rows) >= n_per_category])
    if len(cats) < n_categories:
        raise ValueError(f"Not enough categories with >= {n_per_category} nouns. Found: {len(cats)}")
    chosen_cats = rng.sample(cats, n_categories)
    out = []
    for cat in chosen_cats:
        out.extend(rng.sample(by_cat[cat], n_per_category))
    return out


def select_nouns_fixed(items: List[NounItem], n_categories: int, n_per_category: int) -> List[NounItem]:
    by_cat: Dict[str, List[NounItem]] = defaultdict(list)
    for x in items:
        by_cat[x.category].append(x)
    cats = sorted([c for c, rows in by_cat.items() if len(rows) >= n_per_category])
    if len(cats) < n_categories:
        raise ValueError(f"Not enough categories with >= {n_per_category} nouns. Found: {len(cats)}")
    chosen_cats = cats[:n_categories]
    out = []
    for cat in chosen_cats:
        out.extend(by_cat[cat][:n_per_category])
    return out


def project_features(vectors: np.ndarray, d_proj: int, seed: int) -> np.ndarray:
    n, d = vectors.shape
    if d_proj >= d:
        return vectors.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(d, size=d_proj, replace=False)
    return vectors[:, idx]


def one_shot_hebbian_accuracy(
    support_x: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
) -> float:
    sup = F.normalize(support_x, dim=1)
    qry = F.normalize(query_x, dim=1)
    logits = qry @ sup.T
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == query_y).float().mean().item())
    return acc


def sgd_accuracy_curve(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    n_classes: int,
    steps_list: List[int],
    lr: float,
    trial_seed: int,
) -> Dict[int, float]:
    torch.manual_seed(trial_seed)
    d = support_x.shape[1]
    w = torch.zeros(d, n_classes, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)

    out: Dict[int, float] = {}
    step_max = max(steps_list)
    step_set = set(steps_list)
    for step in range(1, step_max + 1):
        logits = support_x @ w + b
        loss = F.cross_entropy(logits, support_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in step_set:
            with torch.inference_mode():
                q_logits = query_x @ w + b
                q_pred = torch.argmax(q_logits, dim=1)
                q_acc = float((q_pred == query_y).float().mean().item())
            out[step] = q_acc
    return out


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek plasticity-efficiency benchmark")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--nouns-file", default="")
    ap.add_argument("--max-nouns", type=int, default=0)
    ap.add_argument("--n-categories", type=int, default=6)
    ap.add_argument("--n-per-category", type=int, default=4)
    ap.add_argument("--selection-strategy", choices=["random", "head"], default="random")
    ap.add_argument("--proj-dim", type=int, default=4096)
    ap.add_argument("--sgd-steps", default="1,5,20,100,300")
    ap.add_argument("--sgd-lr", type=float, default=0.2)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    steps_list = sorted({int(x) for x in args.sgd_steps.split(",") if x.strip()})
    if not steps_list:
        raise ValueError("--sgd-steps must contain at least one step")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_plasticity_efficiency_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_nouns(args.nouns_file or None, args.max_nouns if args.max_nouns > 0 else None)
    nouns = (
        select_nouns_fixed(catalog, args.n_categories, args.n_per_category)
        if args.selection_strategy == "head"
        else select_nouns(catalog, args.n_categories, args.n_per_category, args.seed)
    )
    label_map = {x.noun: i for i, x in enumerate(nouns)}

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        support_rows = []
        support_labels = []
        query_rows = []
        query_labels = []

        for item in nouns:
            collector.reset()
            _ = run_prompt(model, tok, support_prompt(item.noun))
            support_rows.append(collector.get_flat())
            support_labels.append(label_map[item.noun])

            for p in query_prompts(item.noun):
                collector.reset()
                _ = run_prompt(model, tok, p)
                query_rows.append(collector.get_flat())
                query_labels.append(label_map[item.noun])

        support_np = np.stack(support_rows, axis=0).astype(np.float32)
        query_np = np.stack(query_rows, axis=0).astype(np.float32)
        d_in = support_np.shape[1]

        support_proj = project_features(support_np, args.proj_dim, args.seed + 11)
        query_proj = project_features(query_np, args.proj_dim, args.seed + 11)

        support_x = torch.from_numpy(support_proj)
        query_x = torch.from_numpy(query_proj)
        support_y = torch.tensor(support_labels, dtype=torch.long)
        query_y = torch.tensor(query_labels, dtype=torch.long)

        hebbian_acc = one_shot_hebbian_accuracy(support_x, query_x, query_y)

        trial_curves = []
        for i in range(args.n_trials):
            seed_i = args.seed + 1000 + i
            curve = sgd_accuracy_curve(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y,
                n_classes=len(nouns),
                steps_list=steps_list,
                lr=args.sgd_lr,
                trial_seed=seed_i,
            )
            trial_curves.append({"trial": i, "seed": seed_i, "curve": curve})

        step_stats = {}
        for s in steps_list:
            vals = [float(t["curve"].get(s, 0.0)) for t in trial_curves]
            m, sd = mean_std(vals)
            step_stats[s] = {
                "mean_acc": m,
                "std_acc": sd,
                "values": vals,
            }

        steps_to_match = None
        for s in steps_list:
            if step_stats[s]["mean_acc"] >= hebbian_acc - 1e-8:
                steps_to_match = s
                break
        efficiency_ratio = float(steps_to_match) if steps_to_match is not None else math.inf

        result = {
            "model_id": args.model_id,
            "runtime_sec": float(time.time() - t0),
            "config": {
                "nouns_file": args.nouns_file or "",
                "max_nouns": int(args.max_nouns),
                "n_categories": args.n_categories,
                "n_per_category": args.n_per_category,
                "selection_strategy": args.selection_strategy,
                "n_classes": len(nouns),
                "proj_dim": int(min(args.proj_dim, d_in)),
                "d_in": int(d_in),
                "sgd_steps": steps_list,
                "sgd_lr": float(args.sgd_lr),
                "n_trials": int(args.n_trials),
            },
            "concepts": [{"noun": x.noun, "category": x.category, "label": label_map[x.noun]} for x in nouns],
            "hebbian_one_shot_acc": float(hebbian_acc),
            "sgd_step_stats": {int(k): v for k, v in step_stats.items()},
            "steps_to_match_hebbian": steps_to_match,
            "efficiency_ratio_steps_vs_one_shot": float(efficiency_ratio) if math.isfinite(efficiency_ratio) else None,
            "trial_curves": trial_curves,
        }

        json_path = out_dir / "plasticity_efficiency_benchmark.json"
        md_path = out_dir / "PLASTICITY_EFFICIENCY_BENCHMARK_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Plasticity Efficiency Benchmark",
            "",
            "## Setup",
            f"- Classes: {len(nouns)} ({args.n_categories} categories x {args.n_per_category} nouns)",
            f"- Feature dim (input): {d_in}",
            f"- Feature dim (projected): {min(args.proj_dim, d_in)}",
            f"- SGD steps tested: {steps_list}",
            f"- SGD trials: {args.n_trials}",
            "",
            "## Result",
            f"- Hebbian one-shot accuracy: {hebbian_acc:.6f}",
        ]
        for s in steps_list:
            st = step_stats[s]
            lines.append(f"- SGD step={s}: mean_acc={st['mean_acc']:.6f}, std={st['std_acc']:.6f}")
        lines.append(
            f"- Steps to match Hebbian: {steps_to_match if steps_to_match is not None else 'not reached'}"
        )
        lines.append(
            f"- Efficiency ratio (steps vs one-shot): {efficiency_ratio if math.isfinite(efficiency_ratio) else 'inf'}"
        )
        lines.extend(
            [
                "",
                "## Interpretation",
                "- If Steps to match Hebbian > 1, one-shot write is more update-efficient than repeated SGD.",
                "- This benchmark evaluates readout-level plasticity efficiency on frozen DeepSeek features.",
            ]
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(f"[OK] JSON: {json_path}")
        print(f"[OK] Report: {md_path}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
