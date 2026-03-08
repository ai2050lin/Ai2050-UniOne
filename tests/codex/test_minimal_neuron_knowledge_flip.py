#!/usr/bin/env python
"""
Minimal-neuron knowledge flip probe.

Question:
- Can we reverse a local fact pattern like:
  "apple:red->sweet, apple:green->not sweet"
  to the opposite with only a few neuron edits?

Design:
- Regime A (disentangled): fact features are directly readable.
- Regime B (entangled): the same facts are mixed by a dense orthogonal transform.
- We solve a constrained local edit and then sparsify to Top-k neuron deltas.

Output:
- JSON with Top-k curves and minimal k under retention thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def predict_sign(logits: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = np.ones_like(logits, dtype=np.int64)
    out[logits < -eps] = -1
    return out


def build_fact_space(seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    entities = ["apple", "banana", "grape", "lemon", "pear", "orange"]
    colors = ["red", "green", "yellow", "purple", "blue"]
    facts = [(e, c) for e in entities for c in colors]
    n = len(facts)
    index = {f: i for i, f in enumerate(facts)}

    # One latent feature per fact (local table-like latent basis).
    g = np.eye(n, dtype=np.float64)

    # Base knowledge logits (with non-zero margins to measure retention robustly).
    w_lat = rng.normal(0.0, 0.20, size=n)
    w_lat[index[("apple", "red")]] = +2.0
    w_lat[index[("apple", "green")]] = -2.0
    w_lat[index[("banana", "yellow")]] = +1.5
    w_lat[index[("grape", "green")]] = +1.2
    w_lat[index[("grape", "purple")]] = +1.1
    w_lat[index[("lemon", "yellow")]] = -1.4
    bias = 0.0

    t_idx = np.array(
        [index[("apple", "red")], index[("apple", "green")]],
        dtype=np.int64,
    )
    y_target = np.array([-1, +1], dtype=np.int64)  # reverse the original logic
    anchor_mask = np.ones(n, dtype=bool)
    anchor_mask[t_idx] = False

    return {
        "rng": rng,
        "facts": facts,
        "g": g,
        "w_lat": w_lat,
        "bias": bias,
        "target_idx": t_idx,
        "target_labels": y_target,
        "anchor_mask": anchor_mask,
    }


def run_disentangled(case: Dict[str, object]) -> Dict[str, object]:
    g = case["g"]
    w_lat = case["w_lat"].copy()
    t_idx = case["target_idx"]
    y_target = case["target_labels"]
    anchor_mask = case["anchor_mask"]
    bias = float(case["bias"])

    base_logits = g @ w_lat + bias
    base_pred = predict_sign(base_logits)

    edited = w_lat.copy()
    edited[t_idx[0]] *= -1.0
    edited[t_idx[1]] *= -1.0
    logits_new = g @ edited + bias
    pred_new = predict_sign(logits_new)

    return {
        "mode": "disentangled",
        "k_fixed_edit": 2,
        "target_flip_success": float(np.mean(pred_new[t_idx] == y_target)),
        "anchor_retention": float(np.mean(pred_new[anchor_mask] == base_pred[anchor_mask])),
        "l2_edit_norm": float(np.linalg.norm(edited - w_lat)),
    }


def solve_dense_local_edit(
    h: np.ndarray,
    w0: np.ndarray,
    bias: float,
    target_idx: np.ndarray,
    target_labels: np.ndarray,
    anchor_mask: np.ndarray,
    margin: float,
    lambda_anchor: float,
    ridge: float,
) -> np.ndarray:
    z0 = h @ w0 + bias
    a_t = h[target_idx]
    a_a = h[anchor_mask]

    # Force both target facts to cross to the opposite side with margin.
    desired = target_labels.astype(np.float64) * float(margin)
    b = desired - z0[target_idx]

    m = a_t.T @ a_t + float(lambda_anchor) * (a_a.T @ a_a) + float(ridge) * np.eye(h.shape[1])
    rhs = a_t.T @ b
    delta = np.linalg.solve(m, rhs)
    return delta


def run_entangled(
    case: Dict[str, object],
    margin: float,
    lambda_anchor: float,
    ridge: float,
    k_grid: List[int],
) -> Dict[str, object]:
    rng = case["rng"]
    g = case["g"]
    w_lat = case["w_lat"]
    t_idx = case["target_idx"]
    y_target = case["target_labels"]
    anchor_mask = case["anchor_mask"]
    bias = float(case["bias"])

    n = g.shape[1]
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    h = g @ q
    w0 = q.T @ w_lat

    base_logits = h @ w0 + bias
    base_pred = predict_sign(base_logits)

    dense_delta = solve_dense_local_edit(
        h=h,
        w0=w0,
        bias=bias,
        target_idx=t_idx,
        target_labels=y_target,
        anchor_mask=anchor_mask,
        margin=margin,
        lambda_anchor=lambda_anchor,
        ridge=ridge,
    )

    rows: List[Dict[str, float]] = []
    k_seen = set()
    for k in k_grid:
        kk = int(min(max(k, 1), n))
        if kk in k_seen:
            continue
        k_seen.add(kk)
        top = np.argsort(np.abs(dense_delta))[::-1][:kk]
        delta_k = np.zeros_like(dense_delta)
        delta_k[top] = dense_delta[top]
        logits_new = h @ (w0 + delta_k) + bias
        pred_new = predict_sign(logits_new)
        rows.append(
            {
                "k": kk,
                "target_flip_success": float(np.mean(pred_new[t_idx] == y_target)),
                "anchor_retention": float(np.mean(pred_new[anchor_mask] == base_pred[anchor_mask])),
                "l2_edit_norm": float(np.linalg.norm(delta_k)),
            }
        )

    def minimal_k(threshold: float) -> int | None:
        for r in rows:
            if r["target_flip_success"] >= 1.0 and r["anchor_retention"] >= threshold:
                return int(r["k"])
        return None

    return {
        "mode": "entangled",
        "dense_nonzero_count": int(np.sum(np.abs(dense_delta) > 1e-10)),
        "dense_l2_norm": float(np.linalg.norm(dense_delta)),
        "curve": rows,
        "minimal_k_for_anchor_95": minimal_k(0.95),
        "minimal_k_for_anchor_100": minimal_k(1.00),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal-neuron local knowledge flip probe")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.8)
    ap.add_argument("--lambda-anchor", type=float, default=20.0)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/minimal_neuron_knowledge_flip_20260307.json",
    )
    args = ap.parse_args()

    case = build_fact_space(seed=args.seed)
    disentangled = run_disentangled(case)
    entangled = run_entangled(
        case=case,
        margin=args.margin,
        lambda_anchor=args.lambda_anchor,
        ridge=args.ridge,
        k_grid=[1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 128],
    )

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "margin": float(args.margin),
            "lambda_anchor": float(args.lambda_anchor),
            "ridge": float(args.ridge),
        },
        "target_rule": {
            "before": [
                "apple:red -> sweet",
                "apple:green -> not_sweet",
            ],
            "after": [
                "apple:red -> not_sweet",
                "apple:green -> sweet",
            ],
        },
        "results": {
            "disentangled": disentangled,
            "entangled": entangled,
        },
        "interpretation": [
            "When encoding is axis-disentangled, a tiny local edit can flip target facts with near-zero collateral.",
            "When encoding is strongly mixed, exact local rewrite usually needs many neuron parameters, not just a few.",
            "So 'few-neuron precise rewrite' is conditional on representational locality/disentanglement.",
        ],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(
        json.dumps(
            {
                "disentangled_k": disentangled["k_fixed_edit"],
                "disentangled_anchor_retention": disentangled["anchor_retention"],
                "entangled_min_k_95": entangled["minimal_k_for_anchor_95"],
                "entangled_min_k_100": entangled["minimal_k_for_anchor_100"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
