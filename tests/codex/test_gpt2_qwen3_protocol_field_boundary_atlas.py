#!/usr/bin/env python
"""
Build a broader concept-level boundary atlas for protocol-field calls.

Compared with the earlier boundary script on apple/cat/truth, this atlas:
- expands to 9 concepts across 3 field families
- recomputes U(c, tau, l, h)
- scans the minimal causal boundary per concept

Goal:
- estimate k*(c, tau) as a distribution, not a single anecdote
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def resolve_snapshot_dir(repo_dir_name: str) -> str:
    roots = [
        Path(r"D:\develop\model\hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        snapshot_root = root / repo_dir_name / "snapshots"
        if not snapshot_root.exists():
            continue
        candidates = sorted([p for p in snapshot_root.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError(f"Cannot resolve snapshot directory for {repo_dir_name}")


def default_model_specs() -> List[Tuple[str, str]]:
    return [
        ("gpt2", resolve_snapshot_dir("models--gpt2")),
        ("qwen3_4b", resolve_snapshot_dir("models--Qwen--Qwen3-4B")),
    ]


def load_model(model_path: str, dtype_name: str, prefer_cuda: bool):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    dtype = getattr(torch, dtype_name)
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
        "device_map": "auto" if want_cuda else "cpu",
        "attn_implementation": "eager",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    model.config.output_attentions = True
    return model, tok


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot discover transformer layers")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn
    raise RuntimeError("Cannot find a probe-able attention module")


class HeadGroupAblator:
    def __init__(self, model):
        self.layers = discover_layers(model)
        self.handles = []
        self.active_map: Dict[int, set[int]] = {}
        self.head_dim = int(getattr(model.config, "hidden_size") // getattr(model.config, "num_attention_heads"))
        for li, layer in enumerate(self.layers):
            attn = get_attention_module(layer)
            target = attn.o_proj if hasattr(attn, "o_proj") else attn.c_proj
            self.handles.append(target.register_forward_pre_hook(self._pre_hook(li)))

    def _pre_hook(self, li: int):
        def fn(_module, inputs):
            heads = self.active_map.get(li)
            if not heads:
                return None
            x = inputs[0]
            y = x.clone()
            for head in heads:
                start = head * self.head_dim
                end = start + self.head_dim
                y[..., start:end] = 0
            return (y,)

        return fn

    def set_active_group(self, heads: Sequence[Tuple[int, int]]) -> None:
        active: Dict[int, set[int]] = defaultdict(set)
        for layer, head in heads:
            active[int(layer)].add(int(head))
        self.active_map = dict(active)

    def clear(self) -> None:
        self.active_map = {}

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def family_words() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange"],
        "animal": ["cat", "dog", "rabbit"],
        "abstract": ["truth", "justice", "logic"],
    }


def all_targets() -> List[Tuple[str, str]]:
    rows = []
    for field, words in family_words().items():
        for word in words:
            rows.append((word, field))
    return rows


def base_prompts(word: str) -> List[str]:
    return [f"This is {word}", f"That is {word}", f"It is {word}"]


def protocol_prompts(field: str, word: str) -> List[str]:
    return [
        f"kind {field} item {word}",
        f"class {field} item {word}",
        f"group {field} item {word}",
    ]


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_attentions=True, return_dict=True)
    return out, int(enc["input_ids"].shape[1])


def mean_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def affine_basis(xs: Sequence[np.ndarray], rank_k: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.stack(xs, axis=0).astype(np.float32)
    mu = np.mean(mat, axis=0).astype(np.float32)
    centered = mat - mu[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    k = int(min(rank_k, vh.shape[0]))
    basis = vh[:k].T.astype(np.float32) if k > 0 else np.zeros((mat.shape[1], 0), dtype=np.float32)
    return mu, basis


def affine_project(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return mu.astype(np.float32)
    centered = (x - mu).astype(np.float32)
    coeff = basis.T @ centered
    return (mu + basis @ coeff).astype(np.float32)


def residual_ratio(x: np.ndarray, mu: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> float:
    proj = affine_project(x, mu, basis)
    denom = float(np.linalg.norm(x - mu)) + eps
    return float(np.linalg.norm(x - proj) / denom)


def norm_delta(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (float(np.linalg.norm(a)) + float(np.linalg.norm(b))) / 2.0 + eps
    return float(np.linalg.norm(a - b) / denom)


def last_token_head_topo(out, target_len: int) -> Dict[Tuple[int, int], np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        for hi in range(last_row.shape[0]):
            topo[(li, hi)] = last_row[hi].astype(np.float32)
    return topo


def cumulative_head_count(rows: Sequence[Dict[str, float]], threshold: float) -> int:
    total = float(sum(max(0.0, float(row["usage_score"])) for row in rows))
    if total <= 1e-12:
        return 0
    acc = 0.0
    for idx, row in enumerate(rows, start=1):
        acc += max(0.0, float(row["usage_score"]))
        if acc / total >= threshold:
            return idx
    return len(rows)


def build_control_group(top_rows: Sequence[Dict[str, float]], use_k: int, n_heads: int) -> List[Tuple[int, int]]:
    top_group = [(int(row["layer"]), int(row["head"])) for row in top_rows[:use_k]]
    top_set = {(int(row["layer"]), int(row["head"])) for row in top_rows}
    need = Counter(layer for layer, _head in top_group)
    control = []
    for layer, count in need.items():
        picked = 0
        for head in range(n_heads):
            if (layer, head) not in top_set:
                control.append((layer, head))
                picked += 1
            if picked >= count:
                break
    return control


def compute_field_scores(
    model,
    tok,
    ablator: HeadGroupAblator | None,
    concept: str,
    fields: Sequence[str],
    group: Sequence[Tuple[int, int]] | None,
    fixed_fit_selectivity: Dict[str, Dict[Tuple[int, int], float]],
) -> Dict[str, float]:
    if ablator is not None:
        if group:
            ablator.set_active_group(group)
        else:
            ablator.clear()

    target_len = max(
        prompt_len(tok, text)
        for field in fields
        for text in protocol_prompts(field, concept)
    )
    prompt_runs_by_field = {}
    for field in fields:
        rows = []
        for text in protocol_prompts(field, concept):
            out, _seq_len = run_model(model, tok, text)
            rows.append(last_token_head_topo(out, target_len))
        prompt_runs_by_field[field] = {key: mean_stack([row[key] for row in rows]) for key in rows[0].keys()}

    field_scores = {}
    for field in fields:
        total = 0.0
        for key in prompt_runs_by_field[field].keys():
            negative_mean = mean_stack([prompt_runs_by_field[other][key] for other in fields if other != field])
            protocol_delta = norm_delta(prompt_runs_by_field[field][key], negative_mean)
            total += float(fixed_fit_selectivity[field][key] * protocol_delta)
        field_scores[field] = float(total)
    return field_scores


def analyze_model(
    model_name: str,
    model_path: str,
    dtype_name: str,
    prefer_cuda: bool,
    k_values: Sequence[int],
    collapse_threshold: float,
) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    fields = list(family_words().keys())
    targets = all_targets()

    base_head_topo: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for concept, _true_field in targets:
        rows: Dict[Tuple[int, int], List[np.ndarray]] = {(li, hi): [] for li in range(n_layers) for hi in range(n_heads)}
        target_len = max(prompt_len(tok, text) for text in base_prompts(concept))
        for text in base_prompts(concept):
            out, _seq_len = run_model(model, tok, text)
            topo = last_token_head_topo(out, target_len)
            for key, vec in topo.items():
                rows[key].append(vec)
        base_head_topo[concept] = {key: mean_stack(vs) for key, vs in rows.items()}

    family_head_basis: Dict[str, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]] = {field: {} for field in fields}
    for field, words in family_words().items():
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                family_head_basis[field][key] = affine_basis(
                    [base_head_topo[word][key] for word in words],
                    rank_k=min(3, len(words) - 1),
                )

    ablator = HeadGroupAblator(model)
    concepts = {}
    try:
        for concept, true_field in targets:
            field_scores = {}
            total_usage_by_field = {}
            for field in fields:
                head_rows = []
                layer_usage = [0.0 for _ in range(n_layers)]
                prompt_runs_by_field = {}
                target_len = max(prompt_len(tok, text) for text in protocol_prompts(field, concept))
                rows = []
                for text in protocol_prompts(field, concept):
                    out, _seq_len = run_model(model, tok, text)
                    rows.append(last_token_head_topo(out, target_len))
                prompt_runs_by_field[field] = {key: mean_stack([row[key] for row in rows]) for key in rows[0].keys()}

                other_runs = {}
                for other_field in fields:
                    if other_field == field:
                        continue
                    other_target_len = max(prompt_len(tok, text) for text in protocol_prompts(other_field, concept))
                    other_rows = []
                    for text in protocol_prompts(other_field, concept):
                        out, _seq_len = run_model(model, tok, text)
                        other_rows.append(last_token_head_topo(out, other_target_len))
                    other_runs[other_field] = {key: mean_stack([row[key] for row in other_rows]) for key in other_rows[0].keys()}

                for li in range(n_layers):
                    for hi in range(n_heads):
                        key = (li, hi)
                        fit_scores = {}
                        for candidate_field in fields:
                            mu_t, basis_t = family_head_basis[candidate_field][key]
                            fit_scores[candidate_field] = float(1.0 - residual_ratio(base_head_topo[concept][key], mu_t, basis_t))
                        this_fit = fit_scores[field]
                        other_best = max(fit_scores[f] for f in fields if f != field)
                        fit_selectivity = float(max(0.0, this_fit - other_best))
                        negative_mean = mean_stack([other_runs[f][key] for f in other_runs.keys()])
                        protocol_delta = float(norm_delta(prompt_runs_by_field[field][key], negative_mean))
                        usage_score = float(fit_selectivity * protocol_delta)
                        layer_usage[li] += usage_score
                        head_rows.append(
                            {
                                "layer": int(li),
                                "head": int(hi),
                                "usage_score": usage_score,
                                "fit_selectivity": fit_selectivity,
                                "protocol_delta": protocol_delta,
                            }
                        )
                head_rows.sort(key=lambda row: float(row["usage_score"]), reverse=True)
                total_usage = float(sum(float(row["usage_score"]) for row in head_rows))
                total_usage_by_field[field] = total_usage
                field_scores[field] = {
                    "total_usage": total_usage,
                    "top_heads": head_rows[:32],
                    "layer_usage_by_layer": [float(x) for x in layer_usage],
                    "mass_summary": {
                        "heads_for_50pct_mass": int(cumulative_head_count(head_rows, 0.5)),
                        "heads_for_80pct_mass": int(cumulative_head_count(head_rows, 0.8)),
                    },
                }

            ranked_fields = sorted(total_usage_by_field.items(), key=lambda kv: kv[1], reverse=True)
            preferred_field = ranked_fields[0][0]
            true_top_rows = field_scores[true_field]["top_heads"]
            fixed_fit_selectivity = {
                field: {
                    (int(row["layer"]), int(row["head"])): float(row["fit_selectivity"])
                    for row in field_scores[field]["top_heads"]
                }
                for field in fields
            }
            for field in fields:
                for li in range(n_layers):
                    for hi in range(n_heads):
                        fixed_fit_selectivity[field].setdefault((li, hi), 0.0)

            baseline_scores = compute_field_scores(model, tok, ablator, concept, fields, None, fixed_fit_selectivity)
            baseline_margin = float(baseline_scores[true_field] - max(v for f, v in baseline_scores.items() if f != true_field))
            scans = {}
            for k in k_values:
                use_k = min(int(k), len(true_top_rows))
                top_group = [(int(row["layer"]), int(row["head"])) for row in true_top_rows[:use_k]]
                control_group = build_control_group(true_top_rows, use_k, n_heads)
                top_scores = compute_field_scores(model, tok, ablator, concept, fields, top_group, fixed_fit_selectivity)
                ctrl_scores = compute_field_scores(model, tok, ablator, concept, fields, control_group, fixed_fit_selectivity)
                top_margin = float(top_scores[true_field] - max(v for f, v in top_scores.items() if f != true_field))
                ctrl_margin = float(ctrl_scores[true_field] - max(v for f, v in ctrl_scores.items() if f != true_field))
                scans[str(k)] = {
                    "summary": {
                        "baseline_margin": baseline_margin,
                        "top_margin": top_margin,
                        "control_margin": ctrl_margin,
                        "top_collapse_ratio": float(max(0.0, (baseline_margin - top_margin) / (baseline_margin + 1e-12))),
                        "control_collapse_ratio": float(max(0.0, (baseline_margin - ctrl_margin) / (baseline_margin + 1e-12))),
                        "causal_margin": float((ctrl_margin - top_margin) / (baseline_margin + 1e-12)),
                    },
                }

            passing_k = [
                int(k)
                for k, row in scans.items()
                if float(row["summary"]["top_collapse_ratio"]) >= collapse_threshold
                and float(row["summary"]["top_collapse_ratio"]) > float(row["summary"]["control_collapse_ratio"])
            ]
            concepts[concept] = {
                "true_field": true_field,
                "preferred_field": preferred_field,
                "preferred_field_matches_truth": bool(preferred_field == true_field),
                "field_scores": field_scores,
                "baseline_margin": baseline_margin,
                "k_scan": scans,
                "boundary_summary": {
                    "collapse_threshold": float(collapse_threshold),
                    "minimal_boundary_k": int(min(passing_k)) if passing_k else None,
                    "best_k_by_causal_margin": int(
                        max(scans.items(), key=lambda kv: kv[1]["summary"]["causal_margin"])[0]
                    ),
                },
            }
    finally:
        ablator.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hist = {}
    by_field = {field: [] for field in fields}
    preferred_match_count = 0
    heads50 = []
    heads80 = []
    for concept, row in concepts.items():
        value = row["boundary_summary"]["minimal_boundary_k"]
        hist["none" if value is None else str(value)] = hist.get("none" if value is None else str(value), 0) + 1
        by_field[row["true_field"]].append(value)
        preferred_match_count += int(row["preferred_field_matches_truth"])
        mass = row["field_scores"][row["true_field"]]["mass_summary"]
        heads50.append(int(mass["heads_for_50pct_mass"]))
        heads80.append(int(mass["heads_for_80pct_mass"]))

    per_field_boundary = {}
    for field, values in by_field.items():
        counter = Counter("none" if value is None else str(value) for value in values)
        per_field_boundary[field] = dict(counter)

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "concept_count": int(len(concepts)),
            "runtime_sec": float(time.time() - t0),
        },
        "concepts": concepts,
        "global_summary": {
            "k_values": [int(x) for x in k_values],
            "collapse_threshold": float(collapse_threshold),
            "minimal_boundary_histogram": hist,
            "preferred_field_match_rate": float(preferred_match_count / max(1, len(concepts))),
            "mean_heads_for_50pct_mass": float(np.mean(heads50)) if heads50 else 0.0,
            "mean_heads_for_80pct_mass": float(np.mean(heads80)) if heads80 else 0.0,
            "boundary_histogram_by_true_field": per_field_boundary,
        },
    }


def parse_k_values(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Need at least one k value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Build concept-level boundary atlas for protocol fields")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--k-values", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--collapse-threshold", type=float, default=0.1)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_protocol_field_boundary_atlas_20260308.json",
    )
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(
            model_name,
            model_path,
            dtype_name,
            prefer_cuda=not args.cpu_only,
            k_values=k_values,
            collapse_threshold=float(args.collapse_threshold),
        )
        results["models"][model_name] = row
        print(
            f"[summary] {model_name} match_rate={row['global_summary']['preferred_field_match_rate']:.3f} "
            f"boundary_hist={row['global_summary']['minimal_boundary_histogram']}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
