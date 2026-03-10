#!/usr/bin/env python
"""
Scan longer-horizon dynamic gate-temperature policies with explicit use of
task length, current phase, and remaining steps.
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
import test_real_multistep_memory_gate_temperature_scan as temperature_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "policy": "none",
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "gated_triple_tau_100": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "fixed_100",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_length_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "length_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_remaining_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "remaining_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_long_horizon": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_long_horizon",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_softroute_hardtail": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_softroute_hardtail",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
    }


def policy_description(policy: str) -> str:
    if policy == "fixed_100":
        return "固定 tau_g = 1.0。"
    if policy == "length_adaptive":
        return "短任务更软，长任务回到 tau=1.0。"
    if policy == "remaining_adaptive":
        return "随剩余步数变化，距离终点越远越软。"
    if policy == "joint_long_horizon":
        return "同时使用任务长度、当前阶段和剩余步数调温度。"
    if policy == "joint_softroute_hardtail":
        return "长保持链的 route 更软，接近 final 时更硬。"
    return "单锚点基线。"


def tau_from_policy(
    policy: str,
    head_name: str,
    length: int,
    step_idx: int,
    total_steps: int,
) -> float:
    remaining = max(total_steps - step_idx - 1, 0)
    remain_ratio = remaining / max(total_steps - 1, 1)
    progress = step_idx / max(total_steps - 1, 1)

    if policy == "fixed_100":
        return 1.0
    if policy == "length_adaptive":
        if length <= 8:
            return 1.4
        if length >= 16:
            return 0.95
        return 1.0
    if policy == "remaining_adaptive":
        base = 0.95 if length >= 16 else 1.05
        tau = base + 0.45 * remain_ratio - 0.12 * progress
        if head_name == "final":
            tau -= 0.08
        return float(np.clip(tau, 0.78, 1.45))
    if policy == "joint_long_horizon":
        if head_name == "tool":
            base = 1.28 if length >= 16 else 1.16
        elif head_name == "route":
            base = 1.22 if length >= 16 else 1.10
        else:
            base = 0.92 if length >= 16 else 0.96
        tau = base + 0.30 * remain_ratio - 0.10 * progress
        return float(np.clip(tau, 0.78, 1.42))
    if policy == "joint_softroute_hardtail":
        if head_name == "tool":
            tau = 1.18 if length >= 16 else 1.08
        elif head_name == "route":
            tau = 1.30 if remain_ratio > 0.50 else 1.05
            if length >= 16:
                tau += 0.06
        else:
            tau = 0.86 if progress > 0.85 else 0.92
        tau += 0.18 * remain_ratio - 0.08 * progress
        return float(np.clip(tau, 0.75, 1.45))
    return 1.0


def attach_policy_gate(
    model: gated_scan.ContextGatedMemoryLearner,
    policy: str,
    length: int,
) -> None:
    model._tau_history = []  # type: ignore[attr-defined]
    model._tau_call_index = 0  # type: ignore[attr-defined]
    total_steps = int(length)

    def gates(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]):
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            return ones, np.zeros(len(memories), dtype=np.float32)
        w_gate, b_gate = self._gate_params(head_name)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        gate_logits = (h @ w_gate + b_gate + 0.05 * mem_strength).astype(np.float32)
        step_idx = int(self._tau_call_index % total_steps)
        tau_g = tau_from_policy(policy, head_name, length, step_idx, total_steps)
        self._tau_history.append(float(tau_g))
        self._tau_call_index += 1
        return temperature_scan.softmax_with_temperature(gate_logits, tau_g), gate_logits

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
    if cfg["use_gate"]:
        attach_policy_gate(model, str(cfg["policy"]), length)

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

    tau_history = list(getattr(model, "_tau_history", []))
    phase2_eval = evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18)
    retention_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18)
    overall_eval = evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18)
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
    ap = argparse.ArgumentParser(description="Long-horizon joint dynamic gate-temperature scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 10, 12, 16, 20])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_long_horizon_joint_temperature_scan_20260309.json",
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
        systems[system_name]["policy_description"] = policy_description(str(cfg["policy"]))
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "policy": str(cfg["policy"]),
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "mean_gate_entropy": float(g["mean_gate_entropy"]),
                "mean_gate_peak": float(g["mean_gate_peak"]),
                "mean_tau": float(np.mean(g["tau_mean_curve"])) if "tau_mean_curve" in g else 0.0,
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    dynamic_rows = [row for row in ranking if row["policy"] not in {"none", "fixed_100"}]
    fixed_ref = systems["gated_triple_tau_100"]
    single_ref = systems["single_anchor_beta_086"]

    per_length_best = {}
    for length in lengths:
        best_row = max(
            dynamic_rows,
            key=lambda row: float(systems[row["system"]]["per_length"][str(length)]["real_closure_score"]),
        )
        per_length_best[str(length)] = {
            "best_system": best_row["system"],
            "best_policy": best_row["policy"],
            "best_score": float(systems[best_row["system"]]["per_length"][str(length)]["real_closure_score"]),
            "gain_vs_fixed_tau_100": float(
                systems[best_row["system"]]["per_length"][str(length)]["real_closure_score"]
                - fixed_ref["per_length"][str(length)]["real_closure_score"]
            ),
            "gain_vs_single_anchor": float(
                systems[best_row["system"]]["per_length"][str(length)]["real_closure_score"]
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
        "dynamic_rows": dynamic_rows,
        "best": {
            "best_mean_dynamic": max(dynamic_rows, key=lambda row: row["mean_closure_score"]),
            "best_max_dynamic": max(dynamic_rows, key=lambda row: row["max_length_score"]),
            "best_retention_dynamic": max(dynamic_rows, key=lambda row: row["mean_retention_score"]),
            "best_decay_dynamic": min(dynamic_rows, key=lambda row: row["closure_relative_drop"]),
        },
        "gains": {
            "per_length_best_dynamic": per_length_best,
            "best_dynamic_mean_vs_fixed_tau_100": float(
                max(dynamic_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                - fixed_ref["global_summary"]["mean_closure_score"]
            ),
            "best_dynamic_max_vs_fixed_tau_100": float(
                max(dynamic_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                - fixed_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "best_dynamic_mean_vs_single_anchor": float(
                max(dynamic_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                - single_ref["global_summary"]["mean_closure_score"]
            ),
            "best_dynamic_max_vs_single_anchor": float(
                max(dynamic_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                - single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
        "hypotheses": {
            "H1_some_joint_policy_beats_fixed_tau_100_on_average": bool(
                max(dynamic_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                > fixed_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_some_joint_policy_beats_fixed_tau_100_at_max_length": bool(
                max(dynamic_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                > fixed_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H3_some_joint_policy_beats_single_anchor_at_max_length": bool(
                max(dynamic_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                > single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H4_joint_long_horizon_beats_length_adaptive_on_average": bool(
                systems["gated_triple_tau_joint_long_horizon"]["global_summary"]["mean_closure_score"]
                > systems["gated_triple_tau_length_adaptive"]["global_summary"]["mean_closure_score"]
            ),
            "H5_joint_softroute_hardtail_reduces_decay_vs_fixed": bool(
                systems["gated_triple_tau_joint_softroute_hardtail"]["global_summary"]["closure_relative_drop"]
                < fixed_ref["global_summary"]["closure_relative_drop"]
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
