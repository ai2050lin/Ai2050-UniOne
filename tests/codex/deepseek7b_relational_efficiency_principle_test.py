#!/usr/bin/env python
"""
Relational Efficiency Principle Test (REPT) for DeepSeek-7B.

Purpose:
Probe whether a deep network compresses a complex concept-relation graph into
an efficient micro-code with sparse routing neurons and low-dimensional geometry.

Core tests:
1) Graph-Geometry Alignment: graph distance vs neural distance (Spearman)
2) Low-Rank Compression: how much relation geometry survives in top-k latent dims
3) Sparse Basis Efficiency: few neurons carrying most relational discrimination mass
4) Causal Bridge Routing: intervention on bridge neurons should increase
   neighbor-vs-nonneighbor alignment for anchor concepts.
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
class TripleNode:
    subj: str
    subj_cat: str
    rel: str
    obj: str
    prompt: str


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

    def get_flat(self) -> np.ndarray:
        miss = [i for i, x in enumerate(self.buffers) if x is None]
        if miss:
            raise RuntimeError(f"Missing hooks: {miss}")
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


def relation_phrase(rel: str) -> str:
    m = {
        "is_a": "is a",
        "has_color": "has color",
        "has_property": "is",
        "has_taste": "tastes",
        "can": "can",
        "belongs_to": "belongs to",
        "related_to": "relates to",
    }
    return m[rel]


def build_relational_nodes() -> List[TripleNode]:
    # Designed to create a connected multi-domain knowledge graph.
    nodes: List[TripleNode] = []

    def add(subj: str, cat: str, rel: str, obj: str):
        p = f"Knowledge {subj} {relation_phrase(rel)} {obj}"
        nodes.append(TripleNode(subj=subj, subj_cat=cat, rel=rel, obj=obj, prompt=p))

    # Fruits
    fruit_color = {"apple": "red", "banana": "yellow", "orange": "orange", "grape": "purple", "pear": "green", "peach": "pink"}
    for s, c in fruit_color.items():
        add(s, "fruit", "is_a", "fruit")
        add(s, "fruit", "has_color", c)
        add(s, "fruit", "has_taste", "sweet")
        add(s, "fruit", "belongs_to", "plant")
        add(s, "fruit", "can", "grow")

    # Animals
    animals = ["rabbit", "cat", "dog", "horse", "tiger", "lion"]
    for s in animals:
        add(s, "animal", "is_a", "animal")
        add(s, "animal", "has_property", "alive")
        add(s, "animal", "can", "move")
        add(s, "animal", "can", "run")
        add(s, "animal", "belongs_to", "nature")

    # Vehicles
    vehicles = ["car", "bus", "train", "bicycle", "airplane", "boat"]
    for s in vehicles:
        add(s, "vehicle", "is_a", "vehicle")
        add(s, "vehicle", "has_property", "mechanical")
        add(s, "vehicle", "can", "move")
        add(s, "vehicle", "belongs_to", "transport")

    # Celestial + nature links
    cel = ["sun", "moon", "star", "planet"]
    for s in cel:
        add(s, "celestial", "is_a", "celestial_body")
        add(s, "celestial", "has_property", "bright")
        add(s, "celestial", "can", "shine")
        add(s, "celestial", "belongs_to", "universe")

    # Abstract
    abstracts = ["justice", "truth", "memory", "infinity", "freedom", "logic"]
    for s in abstracts:
        add(s, "abstract", "is_a", "concept")
        add(s, "abstract", "related_to", "meaning")
        add(s, "abstract", "related_to", "value")

    return nodes


def build_graph(nodes: List[TripleNode]) -> np.ndarray:
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = nodes[i], nodes[j]
            linked = False
            if ni.subj == nj.subj:
                linked = True
            if ni.obj == nj.obj:
                linked = True
            if ni.rel == nj.rel:
                linked = True
            if ni.subj_cat == nj.subj_cat:
                linked = True
            # lexical bridge
            if ni.obj == nj.subj or nj.obj == ni.subj:
                linked = True
            if linked:
                A[i, j] = 1
                A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def shortest_path_dist(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    inf = 10**9
    D = np.full((n, n), inf, dtype=np.int32)
    D[A > 0] = 1
    np.fill_diagonal(D, 0)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D.astype(np.float32)


def pairwise_cos_dist(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    S = np.clip(S, -1.0, 1.0)
    return 1.0 - S


def rankdata(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    # simple rank without tie-averaging (stable enough for this test)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def graph_geometry_alignment(D_graph: np.ndarray, D_neural: np.ndarray) -> Dict[str, float]:
    n = D_graph.shape[0]
    tri = np.triu_indices(n, k=1)
    x = D_graph[tri]
    y = D_neural[tri]
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return {"spearman": 0.0}
    return {"spearman": spearman(x[mask], y[mask])}


def compression_alignment(X: np.ndarray, D_graph: np.ndarray, k_list: List[int]) -> Dict[str, float]:
    # Dual PCA in sample space
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T  # N x N
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    scores = eigvecs * np.sqrt(eigvals + 1e-12)  # N x N

    out = {}
    for k in k_list:
        kk = min(k, scores.shape[1])
        Z = scores[:, :kk]
        Dz = pairwise_cos_dist(Z)
        out[f"spearman_k{k}"] = graph_geometry_alignment(D_graph, Dz)["spearman"]
    # rank summary
    p = eigvals / (np.sum(eigvals) + 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    csum = np.cumsum(p)
    k95 = int(np.searchsorted(csum, 0.95) + 1)
    out["effective_rank_sample"] = eff_rank
    out["k95_sample"] = k95
    return out


def eta2_by_label(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    ss_total = np.sum((X - mu) ** 2, axis=0) + 1e-12
    ss_between = np.zeros_like(ss_total)
    for c in np.unique(labels):
        idx = labels == c
        if np.sum(idx) <= 0:
            continue
        mu_c = X[idx].mean(axis=0)
        ss_between += np.sum(idx) * (mu_c - mu) ** 2
    eta = ss_between / ss_total
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
    return eta


def sparse_basis_efficiency(X: np.ndarray, rel_labels: np.ndarray, cat_labels: np.ndarray) -> Dict:
    eta_rel = eta2_by_label(X, rel_labels)
    eta_cat = eta2_by_label(X, cat_labels)
    bridge = np.sqrt(np.maximum(eta_rel, 0.0) * np.maximum(eta_cat, 0.0))
    order = np.argsort(bridge)[::-1]
    sorted_mass = bridge[order]
    total = float(np.sum(sorted_mass) + 1e-12)
    csum = np.cumsum(sorted_mass) / total
    k80 = int(np.searchsorted(csum, 0.8) + 1)
    k90 = int(np.searchsorted(csum, 0.9) + 1)
    return {
        "k80_bridge_mass": k80,
        "k90_bridge_mass": k90,
        "bridge_mass_total": total,
        "top_bridge_indices": [int(i) for i in order[:256].tolist()],
        "top_bridge_scores": [float(x) for x in sorted_mass[:256].tolist()],
        "eta_rel_mean": float(np.mean(eta_rel)),
        "eta_cat_mean": float(np.mean(eta_cat)),
    }


def register_additive(model, flat_idx: int, d_ff: int, alpha: float):
    li = flat_idx // d_ff
    ni = flat_idx % d_ff
    module = model.model.layers[li].mlp.gate_proj

    def _hook(_module, _inputs, output):
        out = output.clone()
        out[..., ni] = out[..., ni] + alpha
        return out

    return module.register_forward_hook(_hook)


def causal_bridge_routing(
    model,
    tok,
    collector: GateCollector,
    d_ff: int,
    nodes: List[TripleNode],
    X: np.ndarray,
    A: np.ndarray,
    bridge_indices: List[int],
    alpha: float,
    anchor_names: List[str],
) -> Dict:
    # anchor node indices by subject
    anchors = []
    for i, n in enumerate(nodes):
        if n.subj in anchor_names and n.rel == "is_a":
            anchors.append(i)
    if not anchors:
        anchors = list(range(min(8, len(nodes))))

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def quality(vec: np.ndarray, i: int) -> float:
        nbr = np.where(A[i] > 0)[0]
        non = np.where(A[i] == 0)[0]
        non = non[non != i]
        if len(nbr) == 0 or len(non) == 0:
            return 0.0
        v = vec / (np.linalg.norm(vec) + 1e-12)
        sim_n = float(np.mean(Xn[nbr] @ v))
        # downsample non-neighbors for balance
        if len(non) > len(nbr) * 3:
            non = non[: len(nbr) * 3]
        sim_o = float(np.mean(Xn[non] @ v))
        return sim_n - sim_o

    per_neuron = []
    for idx in bridge_indices:
        effects = []
        h = register_additive(model, idx, d_ff, alpha=alpha)
        try:
            for ai in anchors:
                prompt = nodes[ai].prompt
                collector.reset()
                _ = run_prompt(model, tok, prompt)
                v_int = collector.get_flat()
                q1 = quality(v_int, ai)
                q0 = quality(X[ai], ai)
                effects.append(q1 - q0)
        finally:
            h.remove()
        per_neuron.append({"flat_idx": int(idx), "mean_routing_gain": float(np.mean(effects) if effects else 0.0)})

    per_neuron.sort(key=lambda x: x["mean_routing_gain"], reverse=True)
    gains = [x["mean_routing_gain"] for x in per_neuron]
    return {
        "anchor_count": len(anchors),
        "mean_gain": float(np.mean(gains) if gains else 0.0),
        "std_gain": float(np.std(gains) if gains else 0.0),
        "top_neurons": per_neuron[:64],
    }


def main():
    parser = argparse.ArgumentParser(description="Relational Efficiency Principle Test")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--bridge-top-m", type=int, default=12)
    parser.add_argument("--edge-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_relational_efficiency_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    d_ff = model.model.layers[0].mlp.gate_proj.out_features
    collector = GateCollector(model)
    try:
        nodes = build_relational_nodes()
        A = build_graph(nodes)
        Dg = shortest_path_dist(A)

        # Extract node vectors
        X = []
        for n in nodes:
            collector.reset()
            _ = run_prompt(model, tok, n.prompt)
            X.append(collector.get_flat())
        X = np.stack(X, axis=0).astype(np.float32)

        Dn = pairwise_cos_dist(X)
        align_full = graph_geometry_alignment(Dg, Dn)
        compress = compression_alignment(X, Dg, k_list=[4, 8, 16, 24, 32, 48])

        rel_vocab = sorted(set(n.rel for n in nodes))
        cat_vocab = sorted(set(n.subj_cat for n in nodes))
        rel_labels = np.array([rel_vocab.index(n.rel) for n in nodes], dtype=np.int32)
        cat_labels = np.array([cat_vocab.index(n.subj_cat) for n in nodes], dtype=np.int32)

        sparse = sparse_basis_efficiency(X, rel_labels, cat_labels)
        bridge_idx = sparse["top_bridge_indices"][: args.bridge_top_m]

        routing = causal_bridge_routing(
            model=model,
            tok=tok,
            collector=collector,
            d_ff=d_ff,
            nodes=nodes,
            X=X,
            A=A,
            bridge_indices=bridge_idx,
            alpha=args.edge_alpha,
            anchor_names=["apple", "banana", "rabbit", "car", "sun"],
        )

        result = {
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "runtime_sec": float(time.time() - t0),
            "config": {
                "n_nodes": len(nodes),
                "n_layers": len(model.model.layers),
                "d_ff": d_ff,
                "bridge_top_m": args.bridge_top_m,
                "edge_alpha": args.edge_alpha,
            },
            "graph_geometry_alignment": align_full,
            "compression_alignment": compress,
            "sparse_basis_efficiency": sparse,
            "causal_bridge_routing": routing,
            "node_summary": {
                "subject_category_counts": dict(sorted(Counter([n.subj_cat for n in nodes]).items())),
                "relation_counts": dict(sorted(Counter([n.rel for n in nodes]).items())),
                "edge_density": float(np.sum(A) / max(A.shape[0] * (A.shape[0] - 1), 1)),
            },
        }

        json_path = out_dir / "relational_efficiency_principle_results.json"
        md_path = out_dir / "RELATIONAL_EFFICIENCY_PRINCIPLE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# 关系网络高效编码原理测试报告 (REPT)", ""]
        lines.append("## 1) 图几何对齐")
        lines.append(f"- Spearman(graph_dist, neural_dist): {align_full['spearman']:.6f}")
        lines.append("")
        lines.append("## 2) 压缩后对齐")
        for k in [4, 8, 16, 24, 32, 48]:
            lines.append(f"- k={k}: spearman={compress[f'spearman_k{k}']:.6f}")
        lines.append(f"- sample effective_rank: {compress['effective_rank_sample']:.6f}")
        lines.append(f"- sample k95: {compress['k95_sample']}")
        lines.append("")
        lines.append("## 3) 稀疏基效率")
        lines.append(f"- k80 bridge mass: {sparse['k80_bridge_mass']}")
        lines.append(f"- k90 bridge mass: {sparse['k90_bridge_mass']}")
        lines.append(f"- eta_rel_mean: {sparse['eta_rel_mean']:.8f}")
        lines.append(f"- eta_cat_mean: {sparse['eta_cat_mean']:.8f}")
        lines.append("")
        lines.append("## 4) 因果桥接路由")
        lines.append(f"- mean routing gain: {routing['mean_gain']:.8f}")
        lines.append(f"- std routing gain: {routing['std_gain']:.8f}")
        lines.append("- top bridge neurons:")
        for x in routing["top_neurons"][:20]:
            li = x["flat_idx"] // d_ff
            ni = x["flat_idx"] % d_ff
            lines.append(f"  - L{li}N{ni}: gain={x['mean_routing_gain']:.8f}")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Saved: {out_dir}")
        print(
            f"[OK] align={align_full['spearman']:.4f} "
            f"k16={compress['spearman_k16']:.4f} k32={compress['spearman_k32']:.4f} "
            f"k80={sparse['k80_bridge_mass']} routing_gain={routing['mean_gain']:.6f}"
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
