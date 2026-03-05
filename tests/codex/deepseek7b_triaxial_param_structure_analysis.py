#!/usr/bin/env python
"""
Tri-axial parameter-structure analysis for concept encoding.

Axes per concept:
1) micro_attr: attribute-driven encoding (e.g., apple->red/round)
2) same_type: sibling contrast encoding (e.g., apple vs banana/pineapple)
3) super_type: parent-category encoding (e.g., apple->fruit/food)

Focus:
- Extract causal neuron subsets per axis
- Report parameter-dimension motifs (gate/up/down dominant dims)
- Compare apple vs cat, and fruit-group vs animal-group structural overlap
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ConceptBundle:
    concept: str
    micro_attrs: List[str]
    siblings: List[str]
    parents: List[str]
    group: str


def default_bundles() -> List[ConceptBundle]:
    return [
        ConceptBundle("apple", ["red", "round"], ["banana", "pineapple"], ["fruit", "food"], "fruit"),
        ConceptBundle("banana", ["yellow", "long"], ["apple", "pineapple"], ["fruit", "food"], "fruit"),
        ConceptBundle("pineapple", ["spiky", "sweet"], ["apple", "banana"], ["fruit", "food"], "fruit"),
        ConceptBundle("cat", ["furry", "agile"], ["dog", "rabbit"], ["animal", "mammal"], "animal"),
        ConceptBundle("dog", ["loyal", "fast"], ["cat", "rabbit"], ["animal", "mammal"], "animal"),
    ]


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


class LastTokenGateAblation:
    def __init__(self, model, layer_to_neurons: Dict[int, List[int]]):
        self.handles = []
        for li, neurons in layer_to_neurons.items():
            if not neurons:
                continue
            layer = model.model.layers[li].mlp.gate_proj
            self.handles.append(layer.register_forward_hook(self._mk_hook(neurons)))

    @staticmethod
    def _mk_hook(neurons: List[int]):
        idx_cpu = torch.tensor(neurons, dtype=torch.long)

        def _hook(_module, _inputs, output):
            out = output.clone()
            idx = idx_cpu.to(out.device)
            out[:, -1, idx] = 0.0
            return out

        return _hook

    def close(self):
        for h in self.handles:
            h.remove()


def index_to_layer_neuron(idx: int, d_ff: int) -> Tuple[int, int]:
    return idx // d_ff, idx % d_ff


def flat_indices_to_layer_map(indices: Sequence[int], d_ff: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        li, ni = index_to_layer_neuron(int(idx), d_ff)
        out[li].append(ni)
    return {k: sorted(set(v)) for k, v in out.items()}


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


def token_id_for_word(tok, word: str) -> int | None:
    if any("\u4e00" <= ch <= "\u9fff" for ch in word):
        ids = tok.encode(word, add_special_tokens=False)
    else:
        ids = tok.encode(" " + word, add_special_tokens=False)
        if not ids:
            ids = tok.encode(word, add_special_tokens=False)
    return int(ids[0]) if ids else None


def logprob_target(model, tok, prefix: str, target_id: int) -> float:
    out = run_prompt(model, tok, prefix)
    logits = out.logits[0, -1, :].detach().float()
    lp = torch.log_softmax(logits, dim=-1)
    return float(lp[target_id].item())


def mean_activation(collector: GateCollector, model, tok, prompts: List[str]) -> np.ndarray:
    xs = []
    for p in prompts:
        collector.reset()
        _ = run_prompt(model, tok, p)
        xs.append(collector.get_flat())
    if not xs:
        raise ValueError("No prompts for activation extraction")
    return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)


def topk(vec: np.ndarray, k: int) -> np.ndarray:
    k = min(k, vec.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(vec, -k)[-k:]
    return idx[np.argsort(vec[idx])[::-1]]


def axis_prompts(bundle: ConceptBundle) -> Dict[str, Dict[str, List[str]]]:
    c = bundle.concept
    # Keep templates simple and stable.
    micro_pos = [f"A {c} is {a}" for a in bundle.micro_attrs]
    micro_neg = [f"A {c} is metallic", f"A {c} is electric"]

    same_pos = [f"This is a {c}", f"I saw a {c}"]
    same_neg = [f"This is a {s}" for s in bundle.siblings] + [f"I saw a {s}" for s in bundle.siblings]

    super_pos = [f"A {c} is a {p}" for p in bundle.parents]
    super_neg = [f"A {c} is a vehicle", f"A {c} is an emotion"]

    return {
        "micro_attr": {"pos": micro_pos, "neg": micro_neg},
        "same_type": {"pos": same_pos, "neg": same_neg},
        "super_type": {"pos": super_pos, "neg": super_neg},
    }


def axis_eval_targets(bundle: ConceptBundle) -> Dict[str, List[Tuple[str, str]]]:
    c = bundle.concept
    return {
        "micro_attr": [(f"A {c} is", a) for a in bundle.micro_attrs],
        "same_type": [("This is a", c)],
        "super_type": [(f"A {c} is a", p) for p in bundle.parents],
    }


def top_dims(vec: torch.Tensor, k: int = 6) -> Dict[str, List[Dict[str, float]]]:
    v = vec.detach().float().cpu().numpy()
    # positive dominant dims
    pos_idx = np.argsort(v)[-k:][::-1]
    neg_idx = np.argsort(v)[:k]
    return {
        "pos": [{"dim": int(i), "value": float(v[i])} for i in pos_idx],
        "neg": [{"dim": int(i), "value": float(v[i])} for i in neg_idx],
    }


def neuron_param_signature(model, layer: int, neuron: int, k_dim: int) -> Dict[str, object]:
    mlp = model.model.layers[layer].mlp
    gate_row = mlp.gate_proj.weight[neuron, :]
    up_row = mlp.up_proj.weight[neuron, :]
    down_col = mlp.down_proj.weight[:, neuron]
    cos = float(torch.nn.functional.cosine_similarity(gate_row.float(), up_row.float(), dim=0).item())
    return {
        "layer": int(layer),
        "neuron": int(neuron),
        "gate_norm": float(torch.norm(gate_row.float()).item()),
        "up_norm": float(torch.norm(up_row.float()).item()),
        "down_norm": float(torch.norm(down_col.float()).item()),
        "gate_up_alignment": cos,
        "gate_dims": top_dims(gate_row, k=k_dim),
        "up_dims": top_dims(up_row, k=k_dim),
        "down_dims": top_dims(down_col, k=k_dim),
    }


def summarize_param_motifs(signatures: List[Dict[str, object]], top_n: int = 12) -> Dict[str, object]:
    layer_hist = Counter()
    gate_dims = Counter()
    down_dims = Counter()
    for s in signatures:
        layer_hist[int(s["layer"])] += 1
        for d in s["gate_dims"]["pos"]:
            gate_dims[int(d["dim"])] += 1
        for d in s["down_dims"]["pos"]:
            down_dims[int(d["dim"])] += 1
    return {
        "layer_hist": dict(sorted((int(k), int(v)) for k, v in layer_hist.items())),
        "dominant_gate_input_dims": [{"dim": int(k), "count": int(v)} for k, v in gate_dims.most_common(top_n)],
        "dominant_down_output_dims": [{"dim": int(k), "count": int(v)} for k, v in down_dims.most_common(top_n)],
    }


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tri-axial parameter structure analysis")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--axis-candidate-k", type=int, default=12)
    ap.add_argument("--axis-subset-k", type=int, default=4)
    ap.add_argument("--param-top-dim-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_triaxial_param_structure_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = default_bundles()
    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        d_ff = model.model.layers[0].mlp.gate_proj.out_features

        concept_axes = {}
        for bundle in bundles:
            axis_data = {}
            prompts = axis_prompts(bundle)
            eval_targets = axis_eval_targets(bundle)
            for axis in ["micro_attr", "same_type", "super_type"]:
                pos = prompts[axis]["pos"]
                neg = prompts[axis]["neg"]
                pos_mean = mean_activation(collector, model, tok, pos)
                neg_mean = mean_activation(collector, model, tok, neg)
                diff = pos_mean - neg_mean
                cands = topk(diff, args.axis_candidate_k)

                causal_rows = []
                for idx in cands.tolist():
                    li, ni = index_to_layer_neuron(int(idx), d_ff)
                    drops = []
                    for prefix, target_word in eval_targets[axis]:
                        tid = token_id_for_word(tok, target_word)
                        if tid is None:
                            continue
                        base = logprob_target(model, tok, prefix, tid)
                        ablator = LastTokenGateAblation(model, {li: [ni]})
                        try:
                            abl = logprob_target(model, tok, prefix, tid)
                        finally:
                            ablator.close()
                        drops.append(base - abl)
                    mean_drop = float(np.mean(drops)) if drops else 0.0
                    causal_rows.append({"flat_index": int(idx), "layer": int(li), "neuron": int(ni), "mean_logprob_drop": mean_drop})

                causal_rows = sorted(causal_rows, key=lambda x: x["mean_logprob_drop"], reverse=True)
                subset = [r for r in causal_rows[: args.axis_subset_k] if r["mean_logprob_drop"] > 0]
                if not subset:
                    subset = causal_rows[: args.axis_subset_k]

                param_signatures = [
                    neuron_param_signature(model, int(r["layer"]), int(r["neuron"]), args.param_top_dim_k) for r in subset
                ]
                motifs = summarize_param_motifs(param_signatures, top_n=12)
                axis_data[axis] = {
                    "pos_prompts": pos,
                    "neg_prompts": neg,
                    "candidate_neurons": causal_rows,
                    "causal_subset": subset,
                    "param_signatures": param_signatures,
                    "motifs": motifs,
                }
            concept_axes[bundle.concept] = {
                "group": bundle.group,
                "micro_attrs": bundle.micro_attrs,
                "siblings": bundle.siblings,
                "parents": bundle.parents,
                "axes": axis_data,
            }

        # Apple vs Cat axis-by-axis structural comparison
        def axis_struct(concept: str, axis: str):
            ax = concept_axes[concept]["axes"][axis]
            neuron_ids = [int(x["flat_index"]) for x in ax["causal_subset"]]
            layers = [int(x["layer"]) for x in ax["causal_subset"]]
            gate_dims = [int(x["dim"]) for x in ax["motifs"]["dominant_gate_input_dims"]]
            down_dims = [int(x["dim"]) for x in ax["motifs"]["dominant_down_output_dims"]]
            return neuron_ids, layers, gate_dims, down_dims

        apple_cat = {}
        for axis in ["micro_attr", "same_type", "super_type"]:
            a = axis_struct("apple", axis)
            c = axis_struct("cat", axis)
            apple_cat[axis] = {
                "neuron_jaccard": jaccard(a[0], c[0]),
                "layer_jaccard": jaccard(a[1], c[1]),
                "gate_dim_jaccard": jaccard(a[2], c[2]),
                "down_dim_jaccard": jaccard(a[3], c[3]),
            }

        # Group-wise shared parameter dims
        group_shared = {}
        by_group = defaultdict(list)
        for c, rec in concept_axes.items():
            by_group[rec["group"]].append(c)
        for g, concepts in by_group.items():
            g_out = {}
            for axis in ["micro_attr", "same_type", "super_type"]:
                dim_sets = []
                for c in concepts:
                    dims = [int(x["dim"]) for x in concept_axes[c]["axes"][axis]["motifs"]["dominant_gate_input_dims"]]
                    dim_sets.append(set(dims))
                common = set.intersection(*dim_sets) if dim_sets else set()
                g_out[axis] = {
                    "concepts": concepts,
                    "common_gate_input_dims": sorted(int(x) for x in common),
                }
            group_shared[g] = g_out

        result = {
            "model_id": args.model_id,
            "runtime_sec": float(time.time() - t0),
            "config": {
                "axis_candidate_k": args.axis_candidate_k,
                "axis_subset_k": args.axis_subset_k,
                "param_top_dim_k": args.param_top_dim_k,
            },
            "concept_axes": concept_axes,
            "comparisons": {
                "apple_vs_cat": apple_cat,
                "group_shared": group_shared,
            },
        }

        json_path = out_dir / "triaxial_param_structure.json"
        md_path = out_dir / "TRIAXIAL_PARAM_STRUCTURE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Tri-axial Parameter Structure Report",
            "",
            "## Concepts",
            f"- {', '.join([b.concept for b in bundles])}",
            "",
            "## Apple vs Cat (Axis-level Structural Overlap)",
        ]
        for axis, v in apple_cat.items():
            lines.append(
                f"- {axis}: neuron={v['neuron_jaccard']:.4f}, layer={v['layer_jaccard']:.4f}, "
                f"gate_dim={v['gate_dim_jaccard']:.4f}, down_dim={v['down_dim_jaccard']:.4f}"
            )

        lines.extend(["", "## Group Shared Gate Input Dims"])
        for g, axes in group_shared.items():
            lines.append(f"- group={g}")
            for axis, rec in axes.items():
                lines.append(f"  - {axis}: common_gate_dims={rec['common_gate_input_dims']}")

        lines.extend(["", "## Per-Concept Axis Signatures"])
        for c, rec in concept_axes.items():
            lines.append(f"- concept={c} (group={rec['group']})")
            for axis in ["micro_attr", "same_type", "super_type"]:
                ax = rec["axes"][axis]
                subset = ax["causal_subset"]
                subset_ids = [f"L{int(x['layer'])}N{int(x['neuron'])}" for x in subset]
                lines.append(f"  - axis={axis}, subset={subset_ids}")
                lines.append(
                    f"    gate_dims={ [d['dim'] for d in ax['motifs']['dominant_gate_input_dims'][:8]] }, "
                    f"down_dims={ [d['dim'] for d in ax['motifs']['dominant_down_output_dims'][:8]] }"
                )

        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(f"[OK] JSON: {json_path}")
        print(f"[OK] Report: {md_path}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
