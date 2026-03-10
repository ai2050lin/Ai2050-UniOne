#!/usr/bin/env python
"""
Task block A upgrade:
explicit phase-state controller plus hierarchical memory routing.

The goal is to test whether a stronger, head-aware controller can turn the
hierarchical-state advantage into a stable ultra-long-horizon gain over the
single-anchor baseline.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_real_multistep_memory_gate_temperature_scan as temperature_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan
import test_real_multistep_memory_hierarchical_state_scan as hierarchical_scan
import test_real_multistep_memory_ultra_long_horizon_temperature_scan as ultra_scan


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "policy": "none",
            "hidden_dim": 6,
            "state_mode": "base",
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "phase_controller_h8": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "phase_controller",
            "hidden_dim": 8,
            "state_mode": "hierarchical",
            "stability": 0.19,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 0.96,
            "gate_lr": 0.060,
        },
        "phase_controller_h10": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "phase_controller",
            "hidden_dim": 10,
            "state_mode": "hierarchical",
            "stability": 0.20,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 0.98,
            "gate_lr": 0.055,
        },
        "phase_controller_tail_lock": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "phase_controller_tail_lock",
            "hidden_dim": 8,
            "state_mode": "hierarchical",
            "stability": 0.18,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 1.0,
            "head_lr_scale": 0.95,
            "gate_lr": 0.058,
        },
    }


def state_input_dim(state_mode: str) -> int:
    if state_mode == "hierarchical":
        return 42
    if state_mode == "segment":
        return 26
    return 13


def gate_prior(head_name: str, remain_ratio: float, progress: float, tail_lock: bool) -> np.ndarray:
    if head_name == "tool":
        prior = np.array([0.74, 0.19, 0.07], dtype=np.float32)
        if remain_ratio < 0.35:
            prior = np.array([0.62, 0.25, 0.13], dtype=np.float32)
    elif head_name == "route":
        prior = np.array([0.14, 0.58, 0.28], dtype=np.float32)
        if remain_ratio < 0.40:
            prior = np.array([0.10, 0.42, 0.48], dtype=np.float32)
    else:
        prior = np.array([0.06, 0.22, 0.72], dtype=np.float32)
        if progress > 0.85:
            prior = np.array([0.03, 0.12, 0.85], dtype=np.float32)
        if tail_lock and progress > 0.90:
            prior = np.array([0.02, 0.08, 0.90], dtype=np.float32)
    return prior / np.sum(prior)


def attach_phase_controller(
    model: gated_scan.ContextGatedMemoryLearner,
    policy: str,
    length: int,
) -> None:
    model._tau_history = []  # type: ignore[attr-defined]
    model._tau_call_index = 0  # type: ignore[attr-defined]
    total_steps = int(length)
    tail_lock = policy == "phase_controller_tail_lock"

    def gates(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]):
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            return ones, np.zeros(len(memories), dtype=np.float32)

        w_gate, b_gate = self._gate_params(head_name)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        gate_logits = (h @ w_gate + b_gate + 0.06 * mem_strength).astype(np.float32)

        step_idx = int(self._tau_call_index % total_steps)
        remain_ratio = max(total_steps - step_idx - 1, 0) / max(total_steps - 1, 1)
        progress = step_idx / max(total_steps - 1, 1)
        tau_g = ultra_scan.tau_from_policy("joint_ultra_oracle", head_name, length, step_idx, total_steps)

        phase_slice = h[-3:] if h.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
        phase_focus = float(np.max(np.abs(phase_slice)))
        hierarchical_energy = float(np.mean(np.abs(h[-19:-3]))) if h.shape[0] >= 19 else float(np.mean(np.abs(h)))
        tau_g = float(np.clip(tau_g - 0.08 * phase_focus - 0.06 * hierarchical_energy, 0.72, 1.36))
        learned = temperature_scan.softmax_with_temperature(gate_logits, tau_g)
        prior = gate_prior(head_name, remain_ratio, progress, tail_lock)
        blend = 0.52 * prior + 0.48 * learned
        blend = (blend / np.sum(blend)).astype(np.float32)

        self._tau_history.append(float(tau_g))
        self._tau_call_index += 1
        return blend, gate_logits

    model.gates = types.MethodType(gates, model)


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    state_mode = str(cfg["state_mode"])
    model = gated_scan.ContextGatedMemoryLearner(
        input_dim=state_input_dim(state_mode),
        hidden_dim=int(cfg["hidden_dim"]),
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
    if cfg["use_gate"]:
        attach_phase_controller(model, str(cfg["policy"]), length)

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = hierarchical_scan.episode_pool(
        phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode
    )
    phase2_memory = hierarchical_scan.episode_pool(
        phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode
    )

    for states, targets in hierarchical_scan.episode_pool(
        phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode
    ):
        model.train_episode(states, targets)
    phase1_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)

    model.consolidate(phase1_memory)

    phase2_rows = hierarchical_scan.episode_pool(
        phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode
    )
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

    tau_history = list(getattr(model, "_tau_history", []))
    phase2_eval = hierarchical_scan.evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
    retention_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
    overall_eval = hierarchical_scan.evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
    tau_history.extend(list(getattr(model, "_tau_history", [])))

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
        "tau_mean": float(np.mean(tau_history)) if tau_history else 0.0,
        "tau_std": float(np.std(tau_history)) if tau_history else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-state controller on ultra-long horizons")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_phase_state_controller_20260309.json",
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
                rows_by_length[length].append(
                    run_for_length(system_name, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p))
                )
        systems[system_name] = gated_scan.summarize_system(rows_by_length, lengths)
        systems[system_name]["config"] = cfg
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    base = systems["single_anchor_beta_086"]["per_length"][str(max_length)]["real_closure_score"]
    best_controller = max(
        [row for row in ranking if row["system"] != "single_anchor_beta_086"],
        key=lambda row: row["max_length_score"],
    )

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
        "best_controller": best_controller,
        "gains": {
            "best_controller_vs_single_anchor_at_max_length": float(best_controller["max_length_score"] - base),
        },
        "hypotheses": {
            "H1_phase_controller_beats_single_anchor_at_max_length": bool(best_controller["max_length_score"] > base),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["best_controller"], ensure_ascii=False, indent=2))
    print(json.dumps(results["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(results["hypotheses"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
