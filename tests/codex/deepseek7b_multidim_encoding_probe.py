#!/usr/bin/env python
"""
DeepSeek-7B multi-dimension encoding probe.

目标：
1) 用成对对照提示词隔离三种维度：风格 / 逻辑 / 句法；
2) 提取每个维度的关键神经元集合（gate激活差分）与layer影响谱；
3) 计算维度间重叠、层级相关、维度特异性，形成可复核结果。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_contrast_pairs() -> Dict[str, List[Dict[str, str]]]:
    style_subjects = [
        ("apple", "an apple"),
        ("gravity", "gravity"),
        ("cat", "a cat"),
        ("democracy", "democracy"),
        ("language", "language"),
        ("neural network", "a neural network"),
        ("economy", "the economy"),
        ("quantum theory", "quantum theory"),
        ("photosynthesis", "photosynthesis"),
        ("memory", "memory"),
        ("ethics", "ethics"),
        ("algorithm", "an algorithm"),
    ]
    style_pairs = []
    for idx, (topic, topic_phrase) in enumerate(style_subjects):
        style_pairs.append(
            {
                "id": f"style_chat_vs_paper_{idx}",
                "a": f"User: Explain {topic} briefly.\nAssistant: {topic_phrase} is",
                "b": f"In formal academic writing, {topic_phrase} is",
            }
        )
        style_pairs.append(
            {
                "id": f"style_casual_vs_formal_{idx}",
                "a": f"Quick note: {topic_phrase} means",
                "b": f"Formal definition: {topic_phrase} denotes",
            }
        )
        style_pairs.append(
            {
                "id": f"style_conversation_vs_report_{idx}",
                "a": f"Let's chat: why does {topic} matter? It matters because",
                "b": f"Technical report tone: the significance of {topic} is that",
            }
        )

    logic_rules = [
        ("All fruits are edible", "apple", "a fruit", "edible"),
        ("All mammals are animals", "dog", "a mammal", "an animal"),
        ("All metals conduct electricity", "copper", "a metal", "conductive"),
        ("All planets orbit stars", "earth", "a planet", "orbiting a star"),
        ("All triangles have three sides", "this shape", "a triangle", "three-sided"),
        ("All birds lay eggs", "sparrow", "a bird", "egg-laying"),
        ("All neurons transmit signals", "this neuron", "a neuron", "signal-transmitting"),
        ("All doctors are trained professionals", "she", "a doctor", "a trained professional"),
        ("All prime numbers greater than 2 are odd", "11", "a prime number >2", "odd"),
        ("All rivers flow downhill", "that river", "a river", "downhill-flowing"),
        ("All vaccines aim to prime immunity", "this vaccine", "a vaccine", "immunity-priming"),
        ("All photons are massless", "a photon", "a photon", "massless"),
    ]
    logic_pairs = []
    for idx, (rule, ent, ent_type, concl) in enumerate(logic_rules):
        logic_pairs.append(
            {
                "id": f"logic_valid_vs_invalid_{idx}",
                "a": f"Premise: {rule}. Premise: {ent} is {ent_type}. Therefore {ent} is",
                "b": f"Premise: {rule}. Premise: {ent} is {ent_type}. Therefore {ent} is not",
            }
        )
        logic_pairs.append(
            {
                "id": f"logic_consistent_vs_contradict_{idx}",
                "a": f"Rule: {rule}. Fact: {ent} is {ent_type}. Conclusion: {ent} should be {concl} and is",
                "b": f"Rule: {rule}. Fact: {ent} is {ent_type}. Conclusion: {ent} should be {concl} but is not",
            }
        )
        logic_pairs.append(
            {
                "id": f"logic_chain_vs_break_{idx}",
                "a": f"Given {rule}, and given {ent} is {ent_type}, infer a coherent conclusion: {ent} is",
                "b": f"Given {rule}, and given {ent} is {ent_type}, infer a contradictory conclusion: {ent} is not",
            }
        )

    syntax_active_passive = [
        ("scientist", "solved", "problem"),
        ("dog", "chased", "cat"),
        ("engineer", "designed", "bridge"),
        ("teacher", "praised", "student"),
        ("chef", "prepared", "meal"),
        ("artist", "painted", "portrait"),
        ("doctor", "treated", "patient"),
        ("team", "won", "match"),
        ("writer", "published", "book"),
        ("researcher", "verified", "hypothesis"),
        ("pilot", "landed", "aircraft"),
        ("programmer", "optimized", "code"),
    ]
    syntax_pairs = []
    for idx, (subj, verb, obj) in enumerate(syntax_active_passive):
        syntax_pairs.append(
            {
                "id": f"syntax_active_vs_passive_{idx}",
                "a": f"The {subj} {verb} the {obj}. The result was",
                "b": f"The {obj} was {verb} by the {subj}. The result was",
            }
        )
        syntax_pairs.append(
            {
                "id": f"syntax_clause_order_{idx}",
                "a": f"Because the {subj} {verb} the {obj}, the report stated that",
                "b": f"The report stated that because the {subj} {verb} the {obj},",
            }
        )
        syntax_pairs.append(
            {
                "id": f"syntax_relative_clause_{idx}",
                "a": f"The {obj} that the {subj} {verb} yesterday was",
                "b": f"The {subj}, who {verb} the {obj} yesterday, was",
            }
        )

    return {"style": style_pairs, "logic": logic_pairs, "syntax": syntax_pairs}


def sample_pairs(pairs: List[Dict[str, str]], max_pairs: int, seed: int, dim_name: str) -> List[Dict[str, str]]:
    if max_pairs <= 0 or max_pairs >= len(pairs):
        return pairs
    dim_seed = sum(ord(ch) for ch in dim_name)
    rng = random.Random(seed + dim_seed + 20260305)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    keep = sorted(idx[:max_pairs])
    return [pairs[i] for i in keep]


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
        return np.concatenate([x.numpy() for x in self.buffers if x is not None], axis=0).astype(np.float32)

    def close(self):
        for h in self.handles:
            h.remove()


def load_model(model_id: str, dtype_name: str, local_files_only: bool):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
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


def topk_indices(vec: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 0), vec.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(vec, -k)[-k:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    return idx


def split_layer_profile(flat_vec: np.ndarray, d_ff: int, n_layers: int) -> List[float]:
    out = []
    for li in range(n_layers):
        st = li * d_ff
        ed = st + d_ff
        out.append(float(flat_vec[st:ed].sum()))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.size == 0 or xb.size == 0:
        return 0.0
    xa = xa - xa.mean()
    xb = xb - xb.mean()
    da = float(np.linalg.norm(xa))
    db = float(np.linalg.norm(xb))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(np.dot(xa, xb) / (da * db))


def energy_rank_stats(mat: np.ndarray) -> Dict[str, float]:
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig, a_min=0.0, a_max=None)
    s = float(eig.sum())
    if s <= 1e-12:
        return {"top1_energy_ratio": 0.0, "top2_energy_ratio": 0.0, "participation_ratio": 0.0}
    es = np.sort(eig)[::-1]
    top1 = float(es[:1].sum() / s)
    top2 = float(es[: min(2, len(es))].sum() / s)
    pr = float((s**2) / (float((eig**2).sum()) + 1e-12))
    return {"top1_energy_ratio": top1, "top2_energy_ratio": top2, "participation_ratio": pr}


def analyze_dimension(
    dim_name: str,
    pairs: List[Dict[str, str]],
    model,
    tok,
    collector: GateCollector,
    top_k: int,
    d_ff: int,
    n_layers: int,
) -> Dict[str, object]:
    pair_rows = []
    deltas = []
    abs_deltas = []
    for p in pairs:
        collector.reset()
        run_prompt(model, tok, p["a"])
        act_a = collector.get_flat()
        collector.reset()
        run_prompt(model, tok, p["b"])
        act_b = collector.get_flat()
        delta = act_a - act_b
        deltas.append(delta)
        abs_deltas.append(np.abs(delta))
        pair_rows.append(
            {
                "id": p["id"],
                "a": p["a"],
                "b": p["b"],
                "delta_l2": float(np.linalg.norm(delta)),
                "delta_mean_abs": float(np.mean(np.abs(delta))),
            }
        )

    delta_mat = np.stack(deltas, axis=0) if deltas else np.zeros((0, d_ff * n_layers), dtype=np.float32)
    abs_mat = np.stack(abs_deltas, axis=0) if abs_deltas else np.zeros((0, d_ff * n_layers), dtype=np.float32)
    mean_abs = np.mean(abs_mat, axis=0) if abs_mat.size else np.zeros((d_ff * n_layers,), dtype=np.float32)
    top_idx = topk_indices(mean_abs, top_k)
    top_rows = []
    for idx in top_idx.tolist():
        li = int(idx // d_ff)
        ni = int(idx % d_ff)
        top_rows.append({"flat_index": int(idx), "layer": li, "neuron": ni, "mean_abs_delta": float(mean_abs[idx])})

    layer_profile = split_layer_profile(mean_abs, d_ff, n_layers)
    profile_sum = float(sum(layer_profile)) + 1e-12
    layer_profile_norm = [float(x / profile_sum) for x in layer_profile]

    pair_cos = []
    for i in range(delta_mat.shape[0]):
        for j in range(i + 1, delta_mat.shape[0]):
            pair_cos.append(cosine(delta_mat[i], delta_mat[j]))

    rank_stats = energy_rank_stats(delta_mat) if delta_mat.shape[0] > 0 else {}

    return {
        "dimension": dim_name,
        "n_pairs": len(pairs),
        "pairs": pair_rows,
        "mean_pair_delta_l2": float(np.mean([x["delta_l2"] for x in pair_rows])) if pair_rows else 0.0,
        "mean_pair_delta_abs": float(np.mean([x["delta_mean_abs"] for x in pair_rows])) if pair_rows else 0.0,
        "pair_delta_cosine_mean": float(np.mean(pair_cos)) if pair_cos else 0.0,
        "layer_profile_abs_delta_sum": layer_profile,
        "layer_profile_abs_delta_norm": layer_profile_norm,
        "generic_top_neurons": top_rows,
        "top_neuron_indices": [int(x["flat_index"]) for x in top_rows],
        "rank_stats": rank_stats,
        "delta_mat": delta_mat,
        "mean_abs_vec": mean_abs,
    }


def build_report(payload: Dict[str, object]) -> str:
    dims = payload["dimensions"]
    cross = payload["cross_dimension"]
    lines = [
        "# 深度神经网络多维编码探针报告",
        "",
        "## 实验目标",
        "- 分析风格/逻辑/句法三个维度在参数激活空间中的编码分布与分工。",
        "",
        "## 维度摘要",
    ]
    for dim in ["style", "logic", "syntax"]:
        d = dims[dim]
        rs = d.get("rank_stats", {})
        lines.append(
            f"- {dim}: n_pairs={d['n_pairs']}, mean_delta_l2={d['mean_pair_delta_l2']:.4f}, "
            f"pair_cos_mean={d['pair_delta_cosine_mean']:.4f}, "
            f"top1_energy={float(rs.get('top1_energy_ratio', 0.0)):.4f}, "
            f"pr={float(rs.get('participation_ratio', 0.0)):.4f}"
        )

    lines.extend(["", "## 维度间关系"])
    for k, row in cross.items():
        lines.append(
            f"- {k}: top_neuron_jaccard={row['top_neuron_jaccard']:.4f}, "
            f"layer_profile_corr={row['layer_profile_corr']:.4f}"
        )

    lines.extend(["", "## 维度特异性"])
    for k, row in payload["specificity"].items():
        lines.append(
            f"- {k}: own_mean={row['own_mean_abs_delta_on_top']:.6f}, "
            f"other_mean={row['other_mean_abs_delta_on_top']:.6f}, "
            f"margin={row['specificity_margin']:.6f}"
        )

    lines.extend(
        [
            "",
            "## 解释",
            "- 若 top_neuron_jaccard 低且 specificity_margin 为正，说明维度存在相对分工编码。",
            "- 若 layer_profile_corr 较高，说明不同维度共享部分层级通道（可能为通用语义骨架）。",
            "- 若 top1_energy 高且 participation_ratio 低，说明该维编码更接近低秩控制方向。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_specific_top_neurons(
    dims_out: Dict[str, Dict[str, object]],
    top_k: int,
    d_ff: int,
    dim_names: Sequence[str],
) -> None:
    vecs = {d: np.asarray(dims_out[d]["mean_abs_vec"], dtype=np.float32) for d in dim_names}
    for dim in dim_names:
        others = [vecs[x] for x in dim_names if x != dim]
        other_mean = np.mean(np.stack(others, axis=0), axis=0) if others else np.zeros_like(vecs[dim])
        spec_vec = vecs[dim] - other_mean
        idx = topk_indices(spec_vec, top_k)
        rows = []
        for i in idx.tolist():
            li = int(i // d_ff)
            ni = int(i % d_ff)
            rows.append(
                {
                    "flat_index": int(i),
                    "layer": li,
                    "neuron": ni,
                    "specific_score": float(spec_vec[i]),
                    "mean_abs_this_dim": float(vecs[dim][i]),
                    "mean_abs_other_dims": float(other_mean[i]),
                }
            )
        dims_out[dim]["specific_top_neurons"] = rows
        dims_out[dim]["top_neuron_indices"] = [int(x["flat_index"]) for x in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=160)
    parser.add_argument("--max-pairs-per-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"tempdata/deepseek7b_multidim_encoding_probe_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = build_contrast_pairs()
    for k in list(all_pairs.keys()):
        all_pairs[k] = sample_pairs(all_pairs[k], max(1, int(args.max_pairs_per_dim)), int(args.seed), k)

    model, tok = load_model(args.model_id, args.dtype, args.local_files_only)
    collector = GateCollector(model)
    try:
        n_layers = len(model.model.layers)
        d_ff = int(model.model.layers[0].mlp.gate_proj.out_features)

        dims_out = {}
        dim_names = ["style", "logic", "syntax"]
        for dim in ["style", "logic", "syntax"]:
            dims_out[dim] = analyze_dimension(
                dim_name=dim,
                pairs=all_pairs[dim],
                model=model,
                tok=tok,
                collector=collector,
                top_k=args.top_k,
                d_ff=d_ff,
                n_layers=n_layers,
            )

        build_specific_top_neurons(dims_out, int(args.top_k), d_ff, dim_names)

        cross = {}
        for i in range(len(dim_names)):
            for j in range(i + 1, len(dim_names)):
                a = dim_names[i]
                b = dim_names[j]
                ka = dims_out[a]["top_neuron_indices"]
                kb = dims_out[b]["top_neuron_indices"]
                pa = dims_out[a]["layer_profile_abs_delta_norm"]
                pb = dims_out[b]["layer_profile_abs_delta_norm"]
                cross[f"{a}__{b}"] = {
                    "top_neuron_jaccard": jaccard(ka, kb),
                    "layer_profile_corr": pearson(pa, pb),
                }

        specificity = {}
        for dim in dim_names:
            sel = np.asarray(dims_out[dim]["top_neuron_indices"], dtype=np.int64)
            own_vals = []
            other_vals = []
            for d2 in dim_names:
                mat = dims_out[d2]["delta_mat"]
                if mat.shape[0] == 0 or sel.size == 0:
                    continue
                v = np.mean(np.abs(mat[:, sel]), axis=1).tolist()
                if d2 == dim:
                    own_vals.extend([float(x) for x in v])
                else:
                    other_vals.extend([float(x) for x in v])
            own_mean = float(np.mean(own_vals)) if own_vals else 0.0
            other_mean = float(np.mean(other_vals)) if other_vals else 0.0
            specificity[dim] = {
                "own_mean_abs_delta_on_top": own_mean,
                "other_mean_abs_delta_on_top": other_mean,
                "specificity_margin": float(own_mean - other_mean),
            }

        result = {
            "model_id": args.model_id,
            "runtime_config": {
                "top_k": int(args.top_k),
                "max_pairs_per_dim": int(args.max_pairs_per_dim),
                "seed": int(args.seed),
                "n_layers": n_layers,
                "d_ff": d_ff,
                "total_neurons": int(n_layers * d_ff),
            },
            "dimensions": {
                k: {
                    kk: vv
                    for kk, vv in dims_out[k].items()
                    if kk not in {"delta_mat", "mean_abs_vec"}
                }
                for k in dim_names
            },
            "cross_dimension": cross,
            "specificity": specificity,
        }

        json_path = out_dir / "multidim_encoding_probe.json"
        md_path = out_dir / "MULTIDIM_ENCODING_PROBE_REPORT.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(build_report(result), encoding="utf-8")
        print(json.dumps({"json": json_path.as_posix(), "markdown": md_path.as_posix()}, ensure_ascii=False))
    finally:
        collector.close()


if __name__ == "__main__":
    main()
