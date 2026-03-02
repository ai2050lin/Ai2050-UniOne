#!/usr/bin/env python
"""
DeepSeek-7B apple vs 100 concepts encoding comparison.

What this script does:
1) Build micro-level concept signatures from MLP gate neurons.
2) Compare apple with 100 concepts (incl. banana/rabbit/sun).
3) Find structural regularities across concept categories.
4) Run causal ablation with apple signature neurons.
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
class ConceptItem:
    concept: str
    category: str


@dataclass
class EvalItem:
    prompt: str
    targets: List[str]


def concept_catalog() -> List[ConceptItem]:
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
    return [ConceptItem(concept=x[0], category=x[1]) for x in rows]


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
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hook outputs for layers: {miss}")
        arr = np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)
        return arr

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


def concept_prompts(concept: str) -> List[str]:
    return [
        f"This is a {concept}",
        f"I saw a {concept}",
        f"People discuss {concept}",
    ]


def index_to_layer_neuron(idx: int, d_ff: int) -> Tuple[int, int]:
    return idx // d_ff, idx % d_ff


def layer_counts_from_indices(indices: List[int], d_ff: int) -> Dict[int, int]:
    ctr = Counter([index_to_layer_neuron(i, d_ff)[0] for i in indices])
    return dict(sorted(ctr.items()))


def topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    k = min(k, vec.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(vec, -k)[-k:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    return idx


def layerwise_topk_indices(vec: np.ndarray, n_layers: int, d_ff: int, k_per_layer: int) -> List[np.ndarray]:
    out = []
    for li in range(n_layers):
        st = li * d_ff
        ed = st + d_ff
        local = vec[st:ed]
        top_local = topk_indices(local, k_per_layer)
        out.append(top_local + st)
    return out


def layerwise_jaccard(a_layers: List[np.ndarray], b_layers: List[np.ndarray]) -> Dict[str, float]:
    vals = []
    shared_total = 0
    for a, b in zip(a_layers, b_layers):
        sa = set(a.tolist())
        sb = set(b.tolist())
        if sa or sb:
            vals.append(len(sa & sb) / len(sa | sb))
        else:
            vals.append(0.0)
        shared_total += len(sa & sb)
    return {
        "mean": float(np.mean(vals) if vals else 0.0),
        "max": float(np.max(vals) if vals else 0.0),
        "shared_total": int(shared_total),
    }


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(a.tolist())
    sb = set(b.tolist())
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


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


def register_ablation(model, flat_indices: List[int], d_ff: int):
    by_layer = defaultdict(list)
    for idx in flat_indices:
        li, ni = index_to_layer_neuron(int(idx), d_ff)
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


def remove_handles(handles):
    for h in handles:
        h.remove()


def build_causal_eval_sets() -> Dict[str, List[EvalItem]]:
    return {
        "apple_attr": [
            EvalItem("The apple is usually", [" red", " sweet", " juicy"]),
            EvalItem("A ripe apple is", [" red", " sweet"]),
            EvalItem("Apple pie uses", [" apples", " apple"]),
        ],
        "banana_attr": [
            EvalItem("The banana is usually", [" yellow", " sweet"]),
            EvalItem("A ripe banana is", [" yellow", " sweet"]),
            EvalItem("Banana peel is", [" yellow"]),
        ],
        "rabbit_attr": [
            EvalItem("A rabbit is usually", [" small", " fast", " furry"]),
            EvalItem("Rabbits can move", [" fast", " quickly"]),
            EvalItem("A rabbit looks", [" cute", " small"]),
        ],
        "sun_attr": [
            EvalItem("The sun is usually", [" bright", " hot", " yellow"]),
            EvalItem("Sunlight is", [" bright", " warm"]),
            EvalItem("The sun looks", [" bright", " yellow"]),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Apple vs 100 concepts encoding comparison")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--top-signature-k", type=int, default=120)
    parser.add_argument("--layer-top-k", type=int, default=64)
    parser.add_argument("--ablate-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_apple_100_compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        concepts = concept_catalog()
        concept_names = [c.concept for c in concepts]
        categories = {c.concept: c.category for c in concepts}

        n_layers = len(model.model.layers)
        d_ff = model.model.layers[0].mlp.gate_proj.out_features
        total_neurons = n_layers * d_ff

        sums = {c.concept: np.zeros(total_neurons, dtype=np.float64) for c in concepts}
        counts = {c.concept: 0 for c in concepts}

        # Extraction
        for ci in concepts:
            for p in concept_prompts(ci.concept):
                collector.reset()
                _ = run_prompt(model, tok, p)
                flat = collector.get_flat()
                sums[ci.concept] += flat
                counts[ci.concept] += 1

        mat = np.zeros((len(concepts), total_neurons), dtype=np.float32)
        for i, c in enumerate(concepts):
            mat[i] = (sums[c.concept] / max(counts[c.concept], 1)).astype(np.float32)

        # Apple signature
        apple_idx = concept_names.index("apple")
        apple_vec = mat[apple_idx]
        mean_vec = mat.mean(axis=0)
        std_vec = mat.std(axis=0) + 1e-8
        apple_z = (apple_vec - mean_vec) / std_vec
        apple_sig = topk_indices(apple_z, args.top_signature_k)
        apple_sig_set = set(apple_sig.tolist())
        apple_layer_top = layerwise_topk_indices(apple_vec, n_layers, d_ff, args.layer_top_k)

        # Per-concept metrics vs apple
        apple_norm = float(np.linalg.norm(apple_vec) + 1e-8)
        row_norms = np.linalg.norm(mat, axis=1) + 1e-8
        cos = (mat @ apple_vec) / (row_norms * apple_norm)

        per_concept = []
        topk_map = {}
        layer_top_map = {}
        for i, name in enumerate(concept_names):
            vec = mat[i]
            top_idx = topk_indices(vec, args.top_signature_k)
            topk_map[name] = top_idx
            layer_top = layerwise_topk_indices(vec, n_layers, d_ff, args.layer_top_k)
            layer_top_map[name] = layer_top
            jac = jaccard(apple_sig, top_idx)
            layer_j = layerwise_jaccard(apple_layer_top, layer_top)
            proj = float(vec[apple_sig].mean())
            layer_overlap = layer_counts_from_indices(list(set(top_idx.tolist()) & apple_sig_set), d_ff)
            per_concept.append(
                {
                    "concept": name,
                    "category": categories[name],
                    "cosine_to_apple": float(cos[i]),
                    "jaccard_topk_with_apple": jac,
                    "layerwise_jaccard_mean": layer_j["mean"],
                    "layerwise_jaccard_max": layer_j["max"],
                    "layerwise_shared_total": layer_j["shared_total"],
                    "apple_signature_activation": proj,
                    "shared_neuron_count": int(len(set(top_idx.tolist()) & apple_sig_set)),
                    "shared_layer_distribution": layer_overlap,
                }
            )

        per_concept.sort(
            key=lambda x: (x["layerwise_jaccard_mean"], x["cosine_to_apple"], x["apple_signature_activation"]),
            reverse=True,
        )

        # Category regularity
        by_cat = defaultdict(list)
        for rec in per_concept:
            if rec["concept"] == "apple":
                continue
            by_cat[rec["category"]].append(rec)
        category_summary = {}
        for cat, arr in by_cat.items():
            category_summary[cat] = {
                "n": len(arr),
                "mean_cosine": float(np.mean([x["cosine_to_apple"] for x in arr])),
                "mean_jaccard_topk": float(np.mean([x["jaccard_topk_with_apple"] for x in arr])),
                "mean_layerwise_jaccard": float(np.mean([x["layerwise_jaccard_mean"] for x in arr])),
                "mean_shared_count": float(np.mean([x["shared_neuron_count"] for x in arr])),
            }

        # Pair details: banana/rabbit/sun
        pair_details = {}
        for name in ["banana", "rabbit", "sun"]:
            ti = topk_map[name]
            layer_info = layerwise_jaccard(apple_layer_top, layer_top_map[name])
            shared = sorted(list(set(apple_sig.tolist()) & set(ti.tolist())))
            only_apple = [x for x in apple_sig.tolist() if x not in set(ti.tolist())][:30]
            only_other = [x for x in ti.tolist() if x not in apple_sig_set][:30]
            pair_details[name] = {
                "jaccard_topk": jaccard(apple_sig, ti),
                "layerwise_jaccard_mean": layer_info["mean"],
                "layerwise_jaccard_max": layer_info["max"],
                "layerwise_shared_total": layer_info["shared_total"],
                "shared_count": len(shared),
                "shared_top": [
                    {
                        "layer": index_to_layer_neuron(idx, d_ff)[0],
                        "neuron": index_to_layer_neuron(idx, d_ff)[1],
                        "apple_z": float(apple_z[idx]),
                        "apple_value": float(apple_vec[idx]),
                        "other_value": float(mat[concept_names.index(name)][idx]),
                    }
                    for idx in shared[:30]
                ],
                "only_apple_top": [
                    {
                        "layer": index_to_layer_neuron(idx, d_ff)[0],
                        "neuron": index_to_layer_neuron(idx, d_ff)[1],
                        "apple_z": float(apple_z[idx]),
                    }
                    for idx in only_apple
                ],
                "only_other_top": [
                    {
                        "layer": index_to_layer_neuron(idx, d_ff)[0],
                        "neuron": index_to_layer_neuron(idx, d_ff)[1],
                        "other_value": float(mat[concept_names.index(name)][idx]),
                    }
                    for idx in only_other
                ],
            }

        # Causal check: ablate apple signature subset
        ablate_idx = apple_sig[: args.ablate_k].tolist()
        eval_sets = build_causal_eval_sets()
        baseline = {k: eval_mass(model, tok, v) for k, v in eval_sets.items()}
        h = register_ablation(model, ablate_idx, d_ff)
        try:
            ablated = {k: eval_mass(model, tok, v) for k, v in eval_sets.items()}
        finally:
            remove_handles(h)
        causal = {}
        for k in eval_sets.keys():
            causal[k] = {
                "baseline": baseline[k],
                "ablated": ablated[k],
                "delta": ablated[k] - baseline[k],
            }

        # Build regularity findings
        top_similar = [x for x in per_concept if x["concept"] != "apple"][:10]
        bottom_similar = sorted(
            [x for x in per_concept if x["concept"] != "apple"],
            key=lambda x: (x["layerwise_jaccard_mean"], x["cosine_to_apple"]),
        )[:10]

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "top_signature_k": args.top_signature_k,
                "layer_top_k": args.layer_top_k,
                "ablate_k": args.ablate_k,
                "n_concepts": len(concepts),
                "n_layers": n_layers,
                "d_ff": d_ff,
                "total_neurons": total_neurons,
            },
            "apple_signature": {
                "indices": [int(x) for x in apple_sig.tolist()],
                "layer_distribution": layer_counts_from_indices([int(x) for x in apple_sig.tolist()], d_ff),
            },
            "per_concept": per_concept,
            "category_summary": dict(sorted(category_summary.items())),
            "pair_details": pair_details,
            "causal_ablation": causal,
            "regularities": {
                "top_similar_to_apple": top_similar,
                "bottom_similar_to_apple": bottom_similar,
            },
        }

        json_path = out_dir / "apple_100_concepts_compare_results.json"
        md_path = out_dir / "APPLE_100_CONCEPTS_COMPARE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# Apple vs 100 Concepts 编码结构比较报告", ""]
        lines.append("## 1) Apple 签名结构")
        lines.append(f"- top-k 签名神经元: {args.top_signature_k}")
        lines.append(f"- 层分布: {result['apple_signature']['layer_distribution']}")
        lines.append("")
        lines.append("## 2) 与 Apple 最相似概念 (Top-10)")
        for x in top_similar:
            lines.append(
                f"- {x['concept']} ({x['category']}): layer-jaccard={x['layerwise_jaccard_mean']:.4f}, cosine={x['cosine_to_apple']:.4f}, layer-shared={x['layerwise_shared_total']}"
            )
        lines.append("")
        lines.append("## 3) 与 Apple 最不相似概念 (Bottom-10 by layer-jaccard)")
        for x in bottom_similar:
            lines.append(
                f"- {x['concept']} ({x['category']}): layer-jaccard={x['layerwise_jaccard_mean']:.4f}, cosine={x['cosine_to_apple']:.4f}, layer-shared={x['layerwise_shared_total']}"
            )
        lines.append("")
        lines.append("## 4) 重点对比")
        for name in ["banana", "rabbit", "sun"]:
            p = pair_details[name]
            lines.append(
                f"- apple vs {name}: layer-jaccard={p['layerwise_jaccard_mean']:.4f}, layer-shared={p['layerwise_shared_total']}, strict-jaccard={p['jaccard_topk']:.4f}"
            )
        lines.append("")
        lines.append("## 5) 因果消融 (apple signature top-40)")
        lines.append("| group | baseline | ablated | delta |")
        lines.append("|---|---:|---:|---:|")
        for k, v in causal.items():
            lines.append(f"| {k} | {v['baseline']:.8f} | {v['ablated']:.8f} | {v['delta']:+.8f} |")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(f"[OK] JSON: {json_path}")
        print(f"[OK] Top similar concepts: {[x['concept'] for x in top_similar[:5]]}")
        print(f"[OK] Pair jaccard apple-banana={pair_details['banana']['jaccard_topk']:.4f}, apple-rabbit={pair_details['rabbit']['jaccard_topk']:.4f}, apple-sun={pair_details['sun']['jaccard_topk']:.4f}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
