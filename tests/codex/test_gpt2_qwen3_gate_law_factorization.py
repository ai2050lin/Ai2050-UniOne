#!/usr/bin/env python
"""
Empirically factorize the gate law G in GPT-2 and Qwen3.

Question:
- Is gating response closer to a low-rank, low-dimensional control law
  driven by a small set of factors?

Method:
- Build prompts over:
  1) mode
  2) task
  3) family / concept
- Measure per-layer, per-head topology delta against a base prompt.
- Fit a linear factor model from prompt factors to head-level gate deltas.
- Report:
  - effective rank
  - mean head-level R^2
  - factor-group importance
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


def family_words() -> Dict[str, List[str]]:
    return {
        "fruit": ["apple", "banana", "orange"],
        "animal": ["cat", "dog", "rabbit"],
        "abstract": ["truth", "justice", "logic"],
    }


def modes() -> List[str]:
    return ["chat", "formal", "logic", "story"]


def tasks() -> List[str]:
    return ["describe", "classify", "compare"]


def base_prompt(word: str) -> str:
    return f"This is {word}"


def factor_prompt(mode: str, task: str, word: str) -> str:
    return f"{mode} {task} item {word}"


def prompt_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def selected_words() -> Dict[str, List[str]]:
    # Prompt length can vary across tokenizers. We pad attention rows later, so
    # there is no need to discard words just because token counts differ.
    return {family: list(words) for family, words in family_words().items()}


def run_model(model, tok, text: str):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, use_cache=False, output_attentions=True, return_dict=True)
    return out, int(enc["input_ids"].shape[1])


def last_token_head_rows(out, target_len: int) -> Dict[int, np.ndarray]:
    topo = {}
    for li, attn in enumerate(out.attentions):
        arr = attn[0].detach().float().cpu().numpy().astype(np.float32)
        last_row = arr[:, -1, :]
        pad = target_len - last_row.shape[1]
        if pad > 0:
            last_row = np.pad(last_row, ((0, 0), (0, pad)), mode="constant")
        topo[li] = last_row.astype(np.float32)
    return topo


def norm_delta_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)) / 2.0 + eps
    return (np.linalg.norm(a - b, axis=1) / denom).astype(np.float32)


def effective_rank_80(mat: np.ndarray, eps: float = 1e-12) -> int:
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return 0
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    _u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    energy = np.square(s.astype(np.float64))
    total = float(np.sum(energy)) + eps
    csum = np.cumsum(energy) / total
    return int(np.searchsorted(csum, 0.8) + 1)


def build_design_matrix(samples: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    feature_rows = []
    mode_list = modes()
    task_list = tasks()
    fam_list = ["fruit", "animal", "abstract"]
    groups = {"bias": [0]}
    cursor = 1
    groups["mode"] = list(range(cursor, cursor + len(mode_list)))
    cursor += len(mode_list)
    groups["task"] = list(range(cursor, cursor + len(task_list)))
    cursor += len(task_list)
    groups["family"] = list(range(cursor, cursor + len(fam_list)))
    cursor += len(fam_list)
    groups["concreteness"] = [cursor]

    for sample in samples:
        row = [1.0]
        for mode in mode_list:
            row.append(1.0 if sample["mode"] == mode else 0.0)
        for task in task_list:
            row.append(1.0 if sample["task"] == task else 0.0)
        for family in fam_list:
            row.append(1.0 if sample["family"] == family else 0.0)
        is_concrete = 0.0 if sample["family"] == "abstract" else 1.0
        row.append(is_concrete)
        feature_rows.append(row)

    return np.asarray(feature_rows, dtype=np.float64), groups


def fit_linear_r2(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y)))) + eps
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return float(r2), coef


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    selected = selected_words()
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))

    base_rows = {}
    target_len_map = {}
    for family, words in selected.items():
        for word in words:
            text = base_prompt(word)
            out, seq_len = run_model(model, tok, text)
            base_rows[word] = last_token_head_rows(out, seq_len)
            target_len_map[word] = int(seq_len)

    samples = []
    delta_by_layer: Dict[int, List[np.ndarray]] = {li: [] for li in range(n_layers)}
    for family, words in selected.items():
        for word in words:
            target_len = target_len_map[word]
            for mode in modes():
                for task in tasks():
                    out, seq_len = run_model(model, tok, factor_prompt(mode, task, word))
                    rows = last_token_head_rows(out, max(target_len, seq_len))
                    base = base_rows[word]
                    sample = {
                        "family": family,
                        "word": word,
                        "mode": mode,
                        "task": task,
                        "delta_by_layer": {},
                    }
                    for li in range(n_layers):
                        base_rows_layer = base[li]
                        cur_rows = rows[li]
                        if cur_rows.shape[1] != base_rows_layer.shape[1]:
                            max_len = max(cur_rows.shape[1], base_rows_layer.shape[1])
                            if cur_rows.shape[1] < max_len:
                                cur_rows = np.pad(cur_rows, ((0, 0), (0, max_len - cur_rows.shape[1])), mode="constant")
                            if base_rows_layer.shape[1] < max_len:
                                base_rows_layer = np.pad(base_rows_layer, ((0, 0), (0, max_len - base_rows_layer.shape[1])), mode="constant")
                        head_delta = norm_delta_rows(cur_rows, base_rows_layer)
                        sample["delta_by_layer"][li] = head_delta
                        delta_by_layer[li].append(head_delta)
                    samples.append(sample)

    if not samples:
        raise RuntimeError(f"No usable samples for {model_name}")

    x, feature_groups = build_design_matrix(samples)
    layer_reports = []
    for li in range(n_layers):
        y_mat = np.stack([sample["delta_by_layer"][li] for sample in samples], axis=0).astype(np.float64)
        head_r2 = []
        coefs = []
        for hi in range(n_heads):
            r2, coef = fit_linear_r2(x, y_mat[:, hi])
            head_r2.append(r2)
            coefs.append(coef)
        coef_mat = np.stack(coefs, axis=0)
        group_norms = {}
        for group_name, idxs in feature_groups.items():
            if group_name == "bias":
                continue
            group_norms[group_name] = float(np.linalg.norm(coef_mat[:, idxs]))
        dominant_group = max(group_norms.items(), key=lambda kv: kv[1])[0] if group_norms else None
        top_heads = sorted(
            [
                {
                    "head": int(hi),
                    "r2": float(head_r2[hi]),
                    "mean_gate_delta": float(np.mean(y_mat[:, hi])),
                }
                for hi in range(n_heads)
            ],
            key=lambda row: row["r2"],
            reverse=True,
        )[:8]
        layer_reports.append(
            {
                "layer": int(li),
                "effective_rank_80pct": int(effective_rank_80(y_mat)),
                "mean_head_r2": float(np.mean(head_r2)),
                "median_head_r2": float(np.median(head_r2)),
                "mean_gate_delta": float(np.mean(y_mat)),
                "group_norms": group_norms,
                "dominant_group": dominant_group,
                "top_predictable_heads": top_heads,
            }
        )

    global_summary = {
        "mean_effective_rank_80pct": float(np.mean([row["effective_rank_80pct"] for row in layer_reports])),
        "mean_head_r2": float(np.mean([row["mean_head_r2"] for row in layer_reports])),
        "median_head_r2": float(np.median([row["median_head_r2"] for row in layer_reports])),
        "dominant_group_histogram": {},
        "best_gate_law_layers": [
            int(row["layer"])
            for row in sorted(layer_reports, key=lambda row: (row["mean_head_r2"], -row["effective_rank_80pct"]), reverse=True)[: min(6, n_layers)]
        ],
    }
    hist = {}
    for row in layer_reports:
        name = row["dominant_group"] or "unknown"
        hist[name] = hist.get(name, 0) + 1
    global_summary["dominant_group_histogram"] = hist

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
            "sample_count": int(len(samples)),
            "runtime_sec": float(time.time() - t0),
            "modes": modes(),
            "tasks": tasks(),
        },
        "selected_words": selected,
        "layer_reports": layer_reports,
        "global_summary": global_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Empirically factorize the gate law G")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_gate_law_factorization_20260308.json",
    )
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}
    for model_name, model_path in default_model_specs():
        dtype_name = args.dtype_gpt2 if model_name == "gpt2" else args.dtype_qwen
        print(f"[run] {model_name} from {model_path}")
        row = analyze_model(model_name, model_path, dtype_name, prefer_cuda=not args.cpu_only)
        results["models"][model_name] = row
        gs = row["global_summary"]
        print(
            f"[summary] {model_name} mean_rank80={gs['mean_effective_rank_80pct']:.2f} "
            f"mean_head_r2={gs['mean_head_r2']:.4f} best_layers={gs['best_gate_law_layers']}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
