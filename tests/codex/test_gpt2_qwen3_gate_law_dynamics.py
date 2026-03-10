#!/usr/bin/env python
"""
Probe whether the gate law G has layer-to-layer dynamics.

Question:
- Can G^(l+1) be predicted from:
  1) prompt factors alone
  2) prompt factors + G^(l)

If the second model substantially improves prediction, G is not only
factorizable, but also has a learnable inter-layer recursion.
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


def selected_words() -> Dict[str, List[str]]:
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
        row.extend(1.0 if sample["mode"] == mode else 0.0 for mode in mode_list)
        row.extend(1.0 if sample["task"] == task else 0.0 for task in task_list)
        row.extend(1.0 if sample["family"] == family else 0.0 for family in fam_list)
        row.append(0.0 if sample["family"] == "abstract" else 1.0)
        feature_rows.append(row)

    return np.asarray(feature_rows, dtype=np.float64), groups


def fit_linear_r2(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y)))) + eps
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return float(r2), coef


def collect_samples(model, tok, n_layers: int) -> List[Dict[str, object]]:
    base_rows = {}
    target_len_map = {}
    for family, words in selected_words().items():
        for word in words:
            out, seq_len = run_model(model, tok, base_prompt(word))
            base_rows[word] = last_token_head_rows(out, seq_len)
            target_len_map[word] = int(seq_len)

    samples = []
    for family, words in selected_words().items():
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
                        sample["delta_by_layer"][li] = norm_delta_rows(cur_rows, base_rows_layer)
                    samples.append(sample)
    return samples


def analyze_model(model_name: str, model_path: str, dtype_name: str, prefer_cuda: bool) -> Dict[str, object]:
    t0 = time.time()
    model, tok = load_model(model_path, dtype_name, prefer_cuda)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    n_heads = int(getattr(model.config, "num_attention_heads"))
    samples = collect_samples(model, tok, n_layers)
    if not samples:
        raise RuntimeError(f"No usable samples for {model_name}")

    x_factor, _feature_groups = build_design_matrix(samples)
    transition_reports = []
    for li in range(n_layers - 1):
        cur_mat = np.stack([sample["delta_by_layer"][li] for sample in samples], axis=0).astype(np.float64)
        next_mat = np.stack([sample["delta_by_layer"][li + 1] for sample in samples], axis=0).astype(np.float64)
        x_full = np.concatenate([x_factor, cur_mat], axis=1)
        factor_r2 = []
        full_r2 = []
        gains = []
        for hi in range(n_heads):
            r2_factor, _ = fit_linear_r2(x_factor, next_mat[:, hi])
            r2_full, _ = fit_linear_r2(x_full, next_mat[:, hi])
            factor_r2.append(r2_factor)
            full_r2.append(r2_full)
            gains.append(float(r2_full - r2_factor))
        transition_reports.append(
            {
                "transition": f"{li}->{li + 1}",
                "layer_from": int(li),
                "layer_to": int(li + 1),
                "mean_factor_only_r2": float(np.mean(factor_r2)),
                "mean_full_r2": float(np.mean(full_r2)),
                "mean_recurrence_gain": float(np.mean(gains)),
                "median_recurrence_gain": float(np.median(gains)),
                "positive_gain_head_count": int(sum(1 for v in gains if v > 0.01)),
                "top_gain_heads": [
                    {
                        "head": int(hi),
                        "factor_only_r2": float(factor_r2[hi]),
                        "full_r2": float(full_r2[hi]),
                        "gain": float(gains[hi]),
                    }
                    for hi in np.argsort(np.asarray(gains))[::-1][: min(8, n_heads)]
                ],
            }
        )

    global_summary = {
        "mean_factor_only_r2": float(np.mean([row["mean_factor_only_r2"] for row in transition_reports])),
        "mean_full_r2": float(np.mean([row["mean_full_r2"] for row in transition_reports])),
        "mean_recurrence_gain": float(np.mean([row["mean_recurrence_gain"] for row in transition_reports])),
        "positive_gain_transition_count": int(sum(1 for row in transition_reports if row["mean_recurrence_gain"] > 0.02)),
        "best_transitions": [
            row["transition"]
            for row in sorted(transition_reports, key=lambda row: row["mean_recurrence_gain"], reverse=True)[: min(8, len(transition_reports))]
        ],
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "meta": {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "sample_count": int(len(samples)),
            "runtime_sec": float(time.time() - t0),
            "modes": modes(),
            "tasks": tasks(),
        },
        "transition_reports": transition_reports,
        "global_summary": global_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe layer-to-layer dynamics of gate law G")
    ap.add_argument("--dtype-gpt2", type=str, default="float32")
    ap.add_argument("--dtype-qwen", type=str, default="float16")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/gpt2_qwen3_gate_law_dynamics_20260308.json",
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
            f"[summary] {model_name} factor_r2={gs['mean_factor_only_r2']:.4f} "
            f"full_r2={gs['mean_full_r2']:.4f} gain={gs['mean_recurrence_gain']:.4f} "
            f"best={gs['best_transitions'][:4]}"
        )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
