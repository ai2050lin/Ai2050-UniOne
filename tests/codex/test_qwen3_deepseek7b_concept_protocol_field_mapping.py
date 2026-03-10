#!/usr/bin/env python
"""
Map concept-to-protocol-field usage U(c, tau, l, h) for:
- Qwen3-4B
- DeepSeek-7B

Goal:
- stop asking only "which head is strongest"
- ask "which head-group / layer-group region does a concept call
  when it enters a protocol field"

Here tau is a protocol field family in:
- fruit
- animal
- abstract

We define a head-level usage score:
    U(c, tau, l, h) = S(c, tau, l, h) * P(c, tau, l, h)

where:
- S = basis-fit selectivity of concept c to field tau at head (l, h)
- P = prompt-induced protocol activation gap for tau at head (l, h)
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
        ("qwen3_4b", resolve_snapshot_dir("models--Qwen--Qwen3-4B")),
        ("deepseek_7b", resolve_snapshot_dir("models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B")),
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


def family_words() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange", "grape", "pear", "lemon"],
        "animal": ["cat", "dog", "rabbit", "horse", "tiger", "bird"],
        "abstract": ["justice", "truth", "logic", "language", "freedom", "memory"],
    }


def target_specs() -> List[Tuple[str, str]]:
    return [("apple", "fruit"), ("cat", "animal"), ("truth", "abstract")]


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


def compatible_words(tok) -> Dict[str, List[str]]:
    out = {}
    for family, words in family_words().items():
        groups: Dict[int, List[str]] = {}
        for word in words:
            lengths = {prompt_len(tok, text) for text in base_prompts(word)}
            if len(lengths) == 1:
                seq_len = next(iter(lengths))
                groups.setdefault(seq_len, []).append(word)
        if not groups:
            out[family] = []
            continue
        best_len = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
        out[family] = groups[best_len]
    return out


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


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    selected_words = compatible_words(tok)
    fields = list(selected_words.keys())

    base_head_topo: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for family, words in selected_words.items():
        for word in words:
            rows: Dict[Tuple[int, int], List[np.ndarray]] = {(li, hi): [] for li in range(n_layers) for hi in range(n_heads)}
            target_len = max(prompt_len(tok, text) for text in base_prompts(word))
            for text in base_prompts(word):
                out, _seq_len = run_model(model, tok, text)
                head_topo = last_token_head_topo(out, target_len)
                for key, vec in head_topo.items():
                    rows[key].append(vec)
            base_head_topo[word] = {key: mean_stack(vs) for key, vs in rows.items()}

    family_head_basis: Dict[str, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]] = {field: {} for field in fields}
    for field, words in selected_words.items():
        for li in range(n_layers):
            for hi in range(n_heads):
                key = (li, hi)
                family_head_basis[field][key] = affine_basis(
                    [base_head_topo[word][key] for word in words],
                    rank_k=min(3, len(words) - 1),
                )

    concepts = {}
    for concept, true_field in target_specs():
        if concept not in base_head_topo:
            continue

        prompt_runs_by_field: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
        protocol_target_len = max(
            prompt_len(tok, text)
            for field in fields
            for text in protocol_prompts(field, concept)
        )
        for field in fields:
            rows: Dict[Tuple[int, int], List[np.ndarray]] = {(li, hi): [] for li in range(n_layers) for hi in range(n_heads)}
            for text in protocol_prompts(field, concept):
                out, _seq_len = run_model(model, tok, text)
                topo = last_token_head_topo(out, protocol_target_len)
                for key, vec in topo.items():
                    rows[key].append(vec)
            prompt_runs_by_field[field] = {key: mean_stack(vs) for key, vs in rows.items()}

        concept_entry = {
            "true_field": true_field,
            "field_scores": {},
        }
        total_usage_by_field = {}
        for field in fields:
            head_rows = []
            layer_usage = [0.0 for _ in range(n_layers)]
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

                    negative_vecs = [prompt_runs_by_field[f][key] for f in fields if f != field]
                    negative_mean = mean_stack(negative_vecs)
                    protocol_delta = float(norm_delta(prompt_runs_by_field[field][key], negative_mean))
                    usage_score = float(fit_selectivity * protocol_delta)
                    layer_usage[li] += usage_score
                    head_rows.append(
                        {
                            "layer": int(li),
                            "head": int(hi),
                            "usage_score": usage_score,
                            "fit_score": float(this_fit),
                            "fit_selectivity": fit_selectivity,
                            "protocol_delta": protocol_delta,
                        }
                    )

            head_rows.sort(key=lambda row: float(row["usage_score"]), reverse=True)
            total_usage = float(sum(float(row["usage_score"]) for row in head_rows))
            total_usage_by_field[field] = total_usage
            layer_rows = [
                {"layer": int(li), "usage_score": float(layer_usage[li])}
                for li in range(n_layers)
            ]
            layer_rows.sort(key=lambda row: float(row["usage_score"]), reverse=True)
            concept_entry["field_scores"][field] = {
                "total_usage": total_usage,
                "top_heads": head_rows[:20],
                "top_layers": layer_rows[:8],
                "layer_usage_by_layer": [float(x) for x in layer_usage],
                "mass_summary": {
                    "top8_head_mass_ratio": float(
                        sum(float(row["usage_score"]) for row in head_rows[:8]) / (total_usage + 1e-12)
                    ),
                    "top16_head_mass_ratio": float(
                        sum(float(row["usage_score"]) for row in head_rows[:16]) / (total_usage + 1e-12)
                    ),
                    "heads_for_50pct_mass": int(cumulative_head_count(head_rows, 0.5)),
                    "heads_for_80pct_mass": int(cumulative_head_count(head_rows, 0.8)),
                },
            }

        ranked_fields = sorted(total_usage_by_field.items(), key=lambda kv: kv[1], reverse=True)
        best_field, best_value = ranked_fields[0]
        second_value = ranked_fields[1][1] if len(ranked_fields) > 1 else 0.0
        concept_entry["summary"] = {
            "preferred_field": best_field,
            "preferred_field_matches_truth": bool(best_field == true_field),
            "best_total_usage": float(best_value),
            "margin_vs_second": float(best_value - second_value),
            "ranked_fields": [
                {"field": field, "total_usage": float(value)}
                for field, value in ranked_fields
            ],
        }
        concepts[concept] = concept_entry

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "hidden_size": int(getattr(model.config, "hidden_size")),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "runtime_sec": float(time.time() - t0),
            "fields": fields,
        },
        "selected_words": selected_words,
        "concepts": concepts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Map concept-to-protocol-field usage U(c,tau,l,h) for Qwen3 and DeepSeek7B")
    ap.add_argument("--dtype-qwen", type=str, default="bfloat16")
    ap.add_argument("--dtype-deepseek", type=str, default="bfloat16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_concept_protocol_field_mapping_20260309.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_qwen if model_name == "qwen3_4b" else args.dtype_deepseek
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        print(f"[summary] {model_name} concepts={list(row['concepts'].keys())}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
