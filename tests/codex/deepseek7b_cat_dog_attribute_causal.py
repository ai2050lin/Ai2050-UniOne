#!/usr/bin/env python
"""
Cat vs Dog attribute dimension split on DeepSeek-7B.

Goal:
- For each attribute (petness/aggression/speed), derive minimal causal neuron subsets
  for cat and dog separately.
- Use causal objective rather than pure statistics:
  maximize target-attribute degradation while minimizing off-target damage.
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
class PromptItem:
    text: str
    label: str


@dataclass
class EvalItem:
    prompt: str
    targets: List[str]


ATTRS = ("petness", "aggression", "speed")


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
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return _hook

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get(self) -> List[torch.Tensor]:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hooks: {miss}")
        return [x for x in self.buffers if x is not None]

    def close(self):
        for h in self.handles:
            h.remove()


def mean_var(sum_t: torch.Tensor, sumsq_t: torch.Tensor, n: int):
    mu = sum_t / max(n, 1)
    var = sumsq_t / max(n, 1) - mu * mu
    return mu, torch.clamp(var, min=0.0)


def build_discovery_prompts() -> Dict[str, Dict[str, List[PromptItem]]]:
    # Per attribute: positive (attribute-focused) vs negative (other attributes)
    d = {
        "petness": {
            "pos": [
                PromptItem("A cat is a common household pet and is very", "cat"),
                PromptItem("A dog is a common household pet and is very", "dog"),
                PromptItem("People keep a cat as a companion because it is", "cat"),
                PromptItem("People keep a dog as a companion because it is", "dog"),
                PromptItem("At home, a cat behaves like a family pet and feels", "cat"),
                PromptItem("At home, a dog behaves like a family pet and feels", "dog"),
            ],
            "neg": [
                PromptItem("A cat can attack prey and become", "cat"),
                PromptItem("A dog can attack intruders and become", "dog"),
                PromptItem("A cat can run quickly and stay", "cat"),
                PromptItem("A dog can run quickly and stay", "dog"),
                PromptItem("A wild cat can be dangerous and", "cat"),
                PromptItem("A guard dog can be dangerous and", "dog"),
            ],
        },
        "aggression": {
            "pos": [
                PromptItem("A cat in a fight can be", "cat"),
                PromptItem("A dog in a fight can be", "dog"),
                PromptItem("A tiger-like cat can become", "cat"),
                PromptItem("A guard dog can become", "dog"),
                PromptItem("When threatened, a cat may act", "cat"),
                PromptItem("When threatened, a dog may act", "dog"),
            ],
            "neg": [
                PromptItem("A cat as a household pet is usually", "cat"),
                PromptItem("A dog as a household pet is usually", "dog"),
                PromptItem("A cat running fast can stay", "cat"),
                PromptItem("A dog running fast can stay", "dog"),
                PromptItem("A cat at home looks", "cat"),
                PromptItem("A dog at home looks", "dog"),
            ],
        },
        "speed": {
            "pos": [
                PromptItem("A cat can move very", "cat"),
                PromptItem("A dog can move very", "dog"),
                PromptItem("During a chase, a cat is", "cat"),
                PromptItem("During a chase, a dog is", "dog"),
                PromptItem("When running, a cat becomes", "cat"),
                PromptItem("When running, a dog becomes", "dog"),
            ],
            "neg": [
                PromptItem("A cat as a pet is very", "cat"),
                PromptItem("A dog as a pet is very", "dog"),
                PromptItem("An angry cat can be", "cat"),
                PromptItem("An angry dog can be", "dog"),
                PromptItem("A calm cat at home is", "cat"),
                PromptItem("A calm dog at home is", "dog"),
            ],
        },
    }
    return d


def build_eval_groups() -> Dict[str, Dict[str, List[EvalItem]]]:
    # Attribute-specific recall/attribute prediction tasks for cat and dog.
    return {
        "petness": {
            "cat_target": [
                EvalItem("A cat is a good", [" pet", " companion"]),
                EvalItem("A house cat is usually", [" friendly", " gentle", " domestic"]),
                EvalItem("People keep a cat as a", [" pet", " companion"]),
                EvalItem("A cat at home feels", [" calm", " safe", " friendly"]),
            ],
            "dog_target": [
                EvalItem("A dog is a good", [" pet", " companion"]),
                EvalItem("A house dog is usually", [" friendly", " loyal", " domestic"]),
                EvalItem("People keep a dog as a", [" pet", " companion"]),
                EvalItem("A dog at home feels", [" calm", " safe", " friendly"]),
            ],
            "control": [
                EvalItem("A cat in battle can be", [" aggressive", " violent"]),
                EvalItem("A dog in battle can be", [" aggressive", " violent"]),
                EvalItem("A cat in a race is", [" fast", " quick"]),
                EvalItem("A dog in a race is", [" fast", " quick"]),
            ],
        },
        "aggression": {
            "cat_target": [
                EvalItem("An angry cat can be", [" aggressive", " dangerous", " violent"]),
                EvalItem("A fighting cat is", [" aggressive", " fierce"]),
                EvalItem("A threatened cat may become", [" aggressive", " dangerous"]),
                EvalItem("A wild cat can be", [" dangerous", " aggressive"]),
            ],
            "dog_target": [
                EvalItem("An angry dog can be", [" aggressive", " dangerous", " violent"]),
                EvalItem("A guard dog is", [" aggressive", " fierce"]),
                EvalItem("A threatened dog may become", [" aggressive", " dangerous"]),
                EvalItem("A wild dog can be", [" dangerous", " aggressive"]),
            ],
            "control": [
                EvalItem("A cat at home is a", [" pet", " companion"]),
                EvalItem("A dog at home is a", [" pet", " companion"]),
                EvalItem("A cat in a race is", [" fast", " quick"]),
                EvalItem("A dog in a race is", [" fast", " quick"]),
            ],
        },
        "speed": {
            "cat_target": [
                EvalItem("A running cat is", [" fast", " quick", " agile"]),
                EvalItem("A cat in a chase moves", [" fast", " quickly"]),
                EvalItem("A cat sprint is very", [" fast", " quick"]),
                EvalItem("A cat can react very", [" fast", " quickly"]),
            ],
            "dog_target": [
                EvalItem("A running dog is", [" fast", " quick", " agile"]),
                EvalItem("A dog in a chase moves", [" fast", " quickly"]),
                EvalItem("A dog sprint is very", [" fast", " quick"]),
                EvalItem("A dog can react very", [" fast", " quickly"]),
            ],
            "control": [
                EvalItem("A cat at home is a", [" pet", " companion"]),
                EvalItem("A dog at home is a", [" pet", " companion"]),
                EvalItem("An angry cat can be", [" aggressive", " dangerous"]),
                EvalItem("An angry dog can be", [" aggressive", " dangerous"]),
            ],
        },
    }


def token_ids(tok, targets: List[str]) -> List[int]:
    out = set()
    for t in targets:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            out.add(int(ids[0]))
    return sorted(out)


def eval_items(model, tok, items: List[EvalItem]) -> float:
    vals = []
    for it in items:
        out = run_prompt(model, tok, it.prompt)
        logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        ids = token_ids(tok, it.targets)
        mass = float(probs[ids].sum().item()) if ids else 0.0
        vals.append(mass)
    return float(np.mean(vals) if vals else 0.0)


def register_ablation(model, neurons: List[Dict]):
    by_layer = defaultdict(list)
    for n in neurons:
        by_layer[int(n["layer"])].append(int(n["neuron"]))
    handles = []
    device = next(model.parameters()).device
    for li, idxs in by_layer.items():
        idx_tensor = torch.tensor(sorted(set(idxs)), dtype=torch.long, device=device)
        module = model.model.layers[li].mlp.gate_proj

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


def collect_attr_stats(model, tok, collector: GateCollector, prompts: Dict[str, List[PromptItem]]):
    n_layers = len(model.model.layers)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    stats = {
        "pos": {"n": 0, "sum": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)], "sumsq": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)]},
        "neg": {"n": 0, "sum": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)], "sumsq": [torch.zeros(d_ff, dtype=torch.float64) for _ in range(n_layers)]},
    }

    for key in ("pos", "neg"):
        for item in prompts[key]:
            collector.reset()
            _ = run_prompt(model, tok, item.text)
            acts = collector.get()
            stats[key]["n"] += 1
            for li, vec in enumerate(acts):
                v = vec.double()
                stats[key]["sum"][li].add_(v)
                stats[key]["sumsq"][li].add_(v * v)
    return stats


def discover_candidates(stats, top_k: int):
    rows = []
    n_layers = len(stats["pos"]["sum"])
    for li in range(n_layers):
        mu_p, var_p = mean_var(stats["pos"]["sum"][li], stats["pos"]["sumsq"][li], stats["pos"]["n"])
        mu_n, var_n = mean_var(stats["neg"]["sum"][li], stats["neg"]["sumsq"][li], stats["neg"]["n"])
        diff = mu_p - mu_n
        z = diff / torch.sqrt(0.5 * (var_p + var_n) + 1e-8)
        z = torch.where(diff > 0, z, torch.full_like(z, float("-inf")))
        k_take = min(top_k * 4, z.numel())
        vals, idx = torch.topk(z, k=k_take)
        for s, ni in zip(vals.tolist(), idx.tolist()):
            if not np.isfinite(s):
                continue
            rows.append(
                {
                    "layer": li,
                    "neuron": int(ni),
                    "score": float(s),
                    "diff": float(diff[ni].item()),
                }
            )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]


def evaluate_for_subset(model, tok, eval_cfg: Dict[str, List[EvalItem]], subset: List[Dict], species: str):
    base_target = eval_items(model, tok, eval_cfg[f"{species}_target"])
    base_control = eval_items(model, tok, eval_cfg["control"])

    if not subset:
        return {
            "target_before": base_target,
            "target_after": base_target,
            "control_before": base_control,
            "control_after": base_control,
            "delta_target": 0.0,
            "delta_control": 0.0,
        }

    h = register_ablation(model, subset)
    try:
        after_target = eval_items(model, tok, eval_cfg[f"{species}_target"])
        after_control = eval_items(model, tok, eval_cfg["control"])
    finally:
        remove_handles(h)
    return {
        "target_before": base_target,
        "target_after": after_target,
        "control_before": base_control,
        "control_after": after_control,
        "delta_target": after_target - base_target,
        "delta_control": after_control - base_control,
    }


def score_single_neurons(
    model,
    tok,
    eval_cfg: Dict[str, List[EvalItem]],
    candidates: List[Dict],
    species: str,
    off_penalty: float,
):
    base_target = eval_items(model, tok, eval_cfg[f"{species}_target"])
    base_control = eval_items(model, tok, eval_cfg["control"])

    out = []
    for c in candidates:
        h = register_ablation(model, [c])
        try:
            tgt = eval_items(model, tok, eval_cfg[f"{species}_target"])
            ctrl = eval_items(model, tok, eval_cfg["control"])
        finally:
            remove_handles(h)

        drop_target = base_target - tgt
        off_shift = abs(ctrl - base_control)
        utility = drop_target - off_penalty * off_shift
        rec = dict(c)
        rec.update(
            {
                "drop_target": float(drop_target),
                "off_shift": float(off_shift),
                "utility": float(utility),
            }
        )
        out.append(rec)

    out.sort(key=lambda x: x["utility"], reverse=True)
    return out, {"base_target": base_target, "base_control": base_control}


def greedy_minimal_subset(
    model,
    tok,
    eval_cfg: Dict[str, List[EvalItem]],
    ranked: List[Dict],
    species: str,
    fullset_size: int,
    target_ratio: float,
):
    max_n = min(fullset_size, len(ranked))
    if max_n <= 0:
        empty_eval = evaluate_for_subset(model, tok, eval_cfg, [], species)
        return [], empty_eval, {"goal_drop": 0.0, "achieved_drop": 0.0, "best_drop": 0.0}

    prefix_evals = []
    prefix_drops = []
    for i in range(1, max_n + 1):
        subset = ranked[:i]
        ev = evaluate_for_subset(model, tok, eval_cfg, subset, species)
        drop = ev["target_before"] - ev["target_after"]
        prefix_evals.append(ev)
        prefix_drops.append(drop)

    best_idx = int(np.argmax(prefix_drops))
    best_drop = float(prefix_drops[best_idx])
    if best_drop <= 0:
        empty_eval = evaluate_for_subset(model, tok, eval_cfg, [], species)
        return [], empty_eval, {"goal_drop": 0.0, "achieved_drop": 0.0, "best_drop": best_drop}

    goal = best_drop * target_ratio
    chosen_idx = best_idx
    for i, d in enumerate(prefix_drops):
        if d >= goal:
            chosen_idx = i
            break

    chosen = ranked[: chosen_idx + 1]
    chosen_eval = prefix_evals[chosen_idx]
    achieved = float(prefix_drops[chosen_idx])
    return chosen, chosen_eval, {"goal_drop": goal, "achieved_drop": achieved, "best_drop": best_drop}


def overlap_ratio(a: List[Dict], b: List[Dict]):
    sa = set((x["layer"], x["neuron"]) for x in a)
    sb = set((x["layer"], x["neuron"]) for x in b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def layer_distribution(rows: List[Dict]):
    return dict(sorted(Counter([x["layer"] for x in rows]).items()))


def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog attribute causal neuron subsets")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--fullset-size", type=int, default=12)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--off-penalty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_cat_dog_attr_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)

    try:
        discovery = build_discovery_prompts()
        eval_groups = build_eval_groups()
        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": None,
            "config": {
                "candidate_k": args.candidate_k,
                "fullset_size": args.fullset_size,
                "target_ratio": args.target_ratio,
                "off_penalty": args.off_penalty,
            },
            "attributes": {},
        }

        for attr in ATTRS:
            attr_stats = collect_attr_stats(model, tok, collector, discovery[attr])
            candidates = discover_candidates(attr_stats, top_k=args.candidate_k)
            eval_cfg = eval_groups[attr]

            attr_out = {
                "candidates": candidates,
                "candidate_layer_distribution": layer_distribution(candidates),
                "cat": {},
                "dog": {},
            }

            for species in ("cat", "dog"):
                ranked, base = score_single_neurons(
                    model=model,
                    tok=tok,
                    eval_cfg=eval_cfg,
                    candidates=candidates,
                    species=species,
                    off_penalty=args.off_penalty,
                )
                subset, subset_eval, progress = greedy_minimal_subset(
                    model=model,
                    tok=tok,
                    eval_cfg=eval_cfg,
                    ranked=ranked,
                    species=species,
                    fullset_size=args.fullset_size,
                    target_ratio=args.target_ratio,
                )
                attr_out[species] = {
                    "baseline": base,
                    "ranked_single": ranked,
                    "minimal_subset": subset,
                    "minimal_subset_size": len(subset),
                    "minimal_subset_layer_distribution": layer_distribution(subset),
                    "subset_eval": subset_eval,
                    "progress": progress,
                }

            attr_out["cat_dog_subset_overlap"] = overlap_ratio(
                attr_out["cat"]["minimal_subset"], attr_out["dog"]["minimal_subset"]
            )
            result["attributes"][attr] = attr_out

        result["runtime_sec"] = float(time.time() - t0)

        # Global summary
        summary = {}
        for attr in ATTRS:
            a = result["attributes"][attr]
            cat_drop = a["cat"]["subset_eval"]["target_before"] - a["cat"]["subset_eval"]["target_after"]
            dog_drop = a["dog"]["subset_eval"]["target_before"] - a["dog"]["subset_eval"]["target_after"]
            summary[attr] = {
                "cat_subset_size": a["cat"]["minimal_subset_size"],
                "dog_subset_size": a["dog"]["minimal_subset_size"],
                "cat_target_drop": float(cat_drop),
                "dog_target_drop": float(dog_drop),
                "overlap": a["cat_dog_subset_overlap"],
            }
        result["summary"] = summary

        json_path = out_dir / "cat_dog_attribute_causal_results.json"
        md_path = out_dir / "CAT_DOG_ATTRIBUTE_CAUSAL_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# Cat vs Dog 属性最小因果神经元子集报告", ""]
        for attr in ATTRS:
            a = result["attributes"][attr]
            cat = a["cat"]
            dog = a["dog"]
            cat_drop = cat["subset_eval"]["target_before"] - cat["subset_eval"]["target_after"]
            dog_drop = dog["subset_eval"]["target_before"] - dog["subset_eval"]["target_after"]
            lines.append(f"## {attr}")
            lines.append(f"- 候选层分布: {a['candidate_layer_distribution']}")
            lines.append(
                f"- cat 最小子集: size={cat['minimal_subset_size']}, drop={cat_drop:.8f}, layer_dist={cat['minimal_subset_layer_distribution']}"
            )
            lines.append(
                f"- dog 最小子集: size={dog['minimal_subset_size']}, drop={dog_drop:.8f}, layer_dist={dog['minimal_subset_layer_distribution']}"
            )
            lines.append(f"- cat/dog 子集重叠(Jaccard): {a['cat_dog_subset_overlap']:.4f}")
            lines.append(
                "- cat top subset: "
                + ", ".join([f"L{x['layer']}N{x['neuron']}" for x in cat["minimal_subset"][:10]])
            )
            lines.append(
                "- dog top subset: "
                + ", ".join([f"L{x['layer']}N{x['neuron']}" for x in dog["minimal_subset"][:10]])
            )
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        for attr in ATTRS:
            s = summary[attr]
            print(
                f"[OK] {attr}: cat_size={s['cat_subset_size']}, dog_size={s['dog_subset_size']}, cat_drop={s['cat_target_drop']:.8f}, dog_drop={s['dog_target_drop']:.8f}, overlap={s['overlap']:.4f}"
            )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
