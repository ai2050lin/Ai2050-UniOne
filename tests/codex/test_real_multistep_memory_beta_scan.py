#!/usr/bin/env python
"""
Scan the slow-memory decay constant beta for long-horizon real multi-step closure.

Baseline:
- trace_gated_local

Variants:
- trace_anchor_local(beta): trace_gated_local plus a slow memory anchor
  m_t = beta * m_(t-1) + (1 - beta) * h_t
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_real_multistep_agi_closure_memory_boost_scan as memory_scan


def anchor_config(beta: float) -> Dict[str, float]:
    return {
        "use_trace": 1.0,
        "use_slow_memory": 1.0,
        "memory_decay": float(beta),
        "stability": 0.16,
        "phase1_replay_stride": 1.0,
        "phase2_replay_stride": 3.0,
    }


def run_with_config(
    cfg: Dict[str, float],
    length: int,
    seed: int,
    noise: float,
    dropout_p: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = memory_scan.MemoryLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13,
        lr_enc=0.045,
        lr_rec=0.055,
        use_trace=bool(cfg["use_trace"]),
        use_slow_memory=bool(cfg["use_slow_memory"]),
        memory_decay=float(cfg["memory_decay"]),
        stability=float(cfg["stability"]),
        seed=seed,
    )

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = memory_scan.episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = memory_scan.episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in memory_scan.episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = memory_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

    if bool(cfg["use_trace"]):
        model.consolidate(phase1_memory)

    phase2_rows = memory_scan.episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    phase1_stride = int(cfg["phase1_replay_stride"])
    phase2_stride = int(cfg["phase2_replay_stride"])
    for idx, (states, targets) in enumerate(phase2_rows):
        model.train_episode(states, targets)
        if phase1_stride > 0 and idx % phase1_stride == 0:
            mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
            model.train_episode(mem_states, mem_targets)
        if phase2_stride > 0 and idx % phase2_stride == 0:
            abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
            model.train_episode(abs_states, abs_targets)

    phase2_eval = memory_scan.evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18)
    retention_eval = memory_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)
    overall_eval = memory_scan.evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18)

    return {
        "phase1_episode_success": phase1_eval["episode_success"],
        "phase2_episode_success": phase2_eval["episode_success"],
        "retention_after_phase2": retention_eval["episode_success"],
        "retention_drop": float(phase1_eval["episode_success"] - retention_eval["episode_success"]),
        "overall_tool_accuracy": overall_eval["tool_accuracy"],
        "overall_route_accuracy": overall_eval["route_accuracy"],
        "overall_final_accuracy": overall_eval["final_accuracy"],
        "overall_episode_success": overall_eval["episode_success"],
    }


def summarize_system(rows_by_length: Dict[int, List[Dict[str, float]]], lengths: List[int]) -> Dict[str, object]:
    per_length = {}
    closure_curve = []
    retention_curve = []
    for length in lengths:
        summary = memory_scan.summarize_runs(rows_by_length[length])
        score = memory_scan.closure_score(summary)
        per_length[str(length)] = {
            "summary": summary,
            "real_closure_score": score,
        }
        closure_curve.append(score)
        retention_curve.append(float(summary["retention_after_phase2"]["mean"]))

    return {
        "per_length": per_length,
        "global_summary": {
            "lengths": lengths,
            "closure_curve": closure_curve,
            "retention_curve": retention_curve,
            "closure_decay_slope": memory_scan.fit_slope(lengths, closure_curve),
            "retention_decay_slope": memory_scan.fit_slope(lengths, retention_curve),
            "closure_relative_drop": memory_scan.relative_drop(closure_curve),
            "retention_relative_drop": memory_scan.relative_drop(retention_curve),
            "mean_closure_score": float(np.mean(closure_curve)),
            "mean_retention_score": float(np.mean(retention_curve)),
        },
    }


def beta_key(beta: float) -> str:
    return f"{beta:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan slow-memory beta for real multi-step AGI closure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--beta-values", type=float, nargs="+", default=[0.50, 0.70, 0.80, 0.86, 0.92, 0.96])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_beta_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    beta_values = [float(v) for v in args.beta_values]

    trace_rows_by_length: Dict[int, List[Dict[str, float]]] = {length: [] for length in lengths}
    for length in lengths:
        for offset in range(int(args.num_seeds)):
            seed = int(args.seed) + offset
            trace_rows_by_length[length].append(
                memory_scan.run_for_length("trace_gated_local", length, seed, float(args.noise), float(args.dropout_p))
            )
    trace_reference = summarize_system(trace_rows_by_length, lengths)

    betas: Dict[str, object] = {}
    ranking = []
    max_length = max(lengths)
    for beta in beta_values:
        rows_by_length: Dict[int, List[Dict[str, float]]] = {length: [] for length in lengths}
        cfg = anchor_config(beta)
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                rows_by_length[length].append(run_with_config(cfg, length, seed, float(args.noise), float(args.dropout_p)))
        beta_summary = summarize_system(rows_by_length, lengths)

        gains_vs_trace = {}
        gain_area = 0.0
        for length in lengths:
            anchor_row = beta_summary["per_length"][str(length)]
            trace_row = trace_reference["per_length"][str(length)]
            closure_gain = float(anchor_row["real_closure_score"] - trace_row["real_closure_score"])
            retention_gain = float(
                anchor_row["summary"]["retention_after_phase2"]["mean"] - trace_row["summary"]["retention_after_phase2"]["mean"]
            )
            gain_area += closure_gain
            gains_vs_trace[str(length)] = {
                "closure_gain_vs_trace": closure_gain,
                "retention_gain_vs_trace": retention_gain,
            }

        beta_id = beta_key(beta)
        betas[beta_id] = {
            "config": cfg,
            **beta_summary,
            "gains_vs_trace": gains_vs_trace,
            "global_comparison": {
                "advantage_area_over_trace": float(gain_area),
                "final_length_gain_vs_trace": float(gains_vs_trace[str(max_length)]["closure_gain_vs_trace"]),
                "final_length_retention_gain_vs_trace": float(gains_vs_trace[str(max_length)]["retention_gain_vs_trace"]),
            },
        }
        ranking.append(
            {
                "beta": beta,
                "mean_closure_score": float(beta_summary["global_summary"]["mean_closure_score"]),
                "mean_retention_score": float(beta_summary["global_summary"]["mean_retention_score"]),
                "closure_relative_drop": float(beta_summary["global_summary"]["closure_relative_drop"]),
                "advantage_area_over_trace": float(gain_area),
                "max_length_score": float(beta_summary["per_length"][str(max_length)]["real_closure_score"]),
                "max_length_gain_vs_trace": float(gains_vs_trace[str(max_length)]["closure_gain_vs_trace"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    best_mean = max(ranking, key=lambda row: row["mean_closure_score"])
    best_max = max(ranking, key=lambda row: row["max_length_score"])
    best_gain = max(ranking, key=lambda row: row["advantage_area_over_trace"])
    best_retention = max(ranking, key=lambda row: row["mean_retention_score"])
    trace_mean = float(trace_reference["global_summary"]["mean_closure_score"])
    trace_max = float(trace_reference["per_length"][str(max_length)]["real_closure_score"])

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "lengths": lengths,
            "beta_values": beta_values,
            "runtime_sec": float(time.time() - t0),
        },
        "trace_reference": trace_reference,
        "betas": betas,
        "ranking": ranking,
        "best_betas": {
            "best_mean_closure": best_mean,
            "best_max_length": best_max,
            "best_advantage_area": best_gain,
            "best_mean_retention": best_retention,
        },
        "hypotheses": {
            "H1_some_beta_beats_trace_on_average": bool(best_mean["mean_closure_score"] > trace_mean),
            "H2_some_beta_beats_trace_at_max_length": bool(best_max["max_length_score"] > trace_max),
            "H3_best_beta_is_not_always_the_slowest": bool(float(best_mean["beta"]) < max(beta_values)),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best_betas"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
