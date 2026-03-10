#!/usr/bin/env python
"""
Scan gate temperature tau_g for context-gated multi-timescale memory readout.
The goal is to test whether longer tasks prefer harder gate selection or softer
mixture over memory timescales.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_real_multistep_agi_closure_memory_boost_scan as memory_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan


def softmax_with_temperature(vec: np.ndarray, tau_g: float) -> np.ndarray:
    tau = max(float(tau_g), 1e-4)
    x = vec.astype(np.float64) / tau
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / np.sum(ex)).astype(np.float32)


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "trace_gated_local": {
            "betas": [],
            "use_gate": False,
            "gate_temperature": None,
            "stability": 0.12,
            "phase1_replay_stride": 2.0,
            "phase2_replay_stride": 5.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "gate_temperature": None,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "gated_triple_tau_035": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 0.35,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_050": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 0.50,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_070": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 0.70,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_100": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 1.00,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_140": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 1.40,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_200": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "gate_temperature": 2.00,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
    }


def attach_temperature_gate(model: gated_scan.ContextGatedMemoryLearner, tau_g: float) -> None:
    def gates(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]):
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            return ones, np.zeros(len(memories), dtype=np.float32)
        w_gate, b_gate = self._gate_params(head_name)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        gate_logits = (h @ w_gate + b_gate + 0.05 * mem_strength).astype(np.float32)
        return softmax_with_temperature(gate_logits, tau_g), gate_logits

    model.gates = types.MethodType(gates, model)


def evaluate_system(
    model: gated_scan.ContextGatedMemoryLearner,
    families: List[str],
    length: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    repeats: int,
) -> Dict[str, float]:
    return gated_scan.evaluate_system(model, families, length, rng, noise, dropout_p, repeats)


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    model = gated_scan.ContextGatedMemoryLearner(
        input_dim=13,
        hidden_dim=6,
        lr_head=0.13 * float(cfg["head_lr_scale"]),
        lr_enc=0.045,
        lr_rec=0.055,
        gate_lr=float(cfg["gate_lr"]),
        use_trace=True,
        memory_betas=[float(beta) for beta in cfg["betas"]],
        use_gate=bool(cfg["use_gate"]),
        stability=float(cfg["stability"]),
        seed=seed,
    )
    if cfg["use_gate"] and cfg["gate_temperature"] is not None:
        attach_temperature_gate(model, float(cfg["gate_temperature"]))

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = memory_scan.episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = memory_scan.episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in memory_scan.episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)

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

    phase2_eval = evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18)
    retention_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)
    overall_eval = evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18)

    return {
        "phase1_episode_success": phase1_eval["episode_success"],
        "phase2_episode_success": phase2_eval["episode_success"],
        "retention_after_phase2": retention_eval["episode_success"],
        "retention_drop": float(phase1_eval["episode_success"] - retention_eval["episode_success"]),
        "overall_tool_accuracy": overall_eval["tool_accuracy"],
        "overall_route_accuracy": overall_eval["route_accuracy"],
        "overall_final_accuracy": overall_eval["final_accuracy"],
        "overall_episode_success": overall_eval["episode_success"],
        "gate_entropy": overall_eval["gate_entropy"],
        "gate_peak": overall_eval["gate_peak"],
    }


def tau_from_system_name(name: str) -> float | None:
    if not name.startswith("gated_triple_tau_"):
        return None
    return float(name.split("_")[-1]) / 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate temperature scan for context-gated multi-timescale memory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_gate_temperature_scan_20260308.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    systems = {}
    ranking = []
    max_length = max(lengths)

    for system_name, cfg in system_configs().items():
        rows_by_length = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                seed = int(args.seed) + offset
                rows_by_length[length].append(run_for_length(system_name, length, seed, float(args.noise), float(args.dropout_p)))
        systems[system_name] = gated_scan.summarize_system(rows_by_length, lengths)
        systems[system_name]["config"] = cfg
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "tau_g": tau_from_system_name(system_name),
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "mean_gate_entropy": float(g["mean_gate_entropy"]),
                "mean_gate_peak": float(g["mean_gate_peak"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    tau_rows = [row for row in ranking if row["tau_g"] is not None]
    tau_rows_by_tau = sorted(tau_rows, key=lambda row: float(row["tau_g"]))
    single_ref = systems["single_anchor_beta_086"]
    trace_ref = systems["trace_gated_local"]
    tau_100 = systems["gated_triple_tau_100"]

    per_length_gain = {}
    for length in lengths:
        best_tau_row = max(
            tau_rows,
            key=lambda row: float(systems[row["system"]]["per_length"][str(length)]["real_closure_score"]),
        )
        best_tau_name = str(best_tau_row["system"])
        per_length_gain[str(length)] = {
            "best_tau_system": best_tau_name,
            "best_tau_g": float(best_tau_row["tau_g"]),
            "best_tau_score": float(systems[best_tau_name]["per_length"][str(length)]["real_closure_score"]),
            "gain_vs_tau_100": float(
                systems[best_tau_name]["per_length"][str(length)]["real_closure_score"]
                - tau_100["per_length"][str(length)]["real_closure_score"]
            ),
            "gain_vs_single_anchor": float(
                systems[best_tau_name]["per_length"][str(length)]["real_closure_score"]
                - single_ref["per_length"][str(length)]["real_closure_score"]
            ),
        }

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "lengths": lengths,
            "runtime_sec": float(time.time() - t0),
        },
        "systems": systems,
        "ranking": ranking,
        "tau_rows": tau_rows_by_tau,
        "best": {
            "best_mean_closure": max(tau_rows, key=lambda row: row["mean_closure_score"]),
            "best_max_length": max(tau_rows, key=lambda row: row["max_length_score"]),
            "best_mean_retention": max(tau_rows, key=lambda row: row["mean_retention_score"]),
            "best_gate_selectivity": max(tau_rows, key=lambda row: row["mean_gate_peak"]),
            "best_gate_entropy": min(tau_rows, key=lambda row: row["mean_gate_entropy"]),
        },
        "gains": {
            "per_length_best_tau": per_length_gain,
            "best_mean_vs_single_anchor": float(
                max(tau_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                - single_ref["global_summary"]["mean_closure_score"]
            ),
            "best_max_vs_single_anchor": float(
                max(tau_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                - single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "best_mean_vs_trace": float(
                max(tau_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                - trace_ref["global_summary"]["mean_closure_score"]
            ),
            "best_max_vs_tau_100": float(
                max(tau_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                - tau_100["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
        "hypotheses": {
            "H1_some_tau_beats_single_anchor_on_average": bool(
                max(tau_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                > single_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_some_tau_beats_single_anchor_at_max_length": bool(
                max(tau_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                > single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H3_best_max_length_tau_is_harder_than_1": bool(
                float(max(tau_rows, key=lambda row: row["max_length_score"])["tau_g"]) < 1.0
            ),
            "H4_higher_temperature_increases_gate_entropy": bool(
                tau_rows_by_tau[-1]["mean_gate_entropy"] > tau_rows_by_tau[0]["mean_gate_entropy"]
            ),
            "H5_harder_temperature_increases_gate_peak": bool(
                tau_rows_by_tau[0]["mean_gate_peak"] > tau_rows_by_tau[-1]["mean_gate_peak"]
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
