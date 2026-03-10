#!/usr/bin/env python
"""
Compare dynamic gate-temperature policies against the best fixed tau_g baseline.
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
            "tau_policy": "none",
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "gated_triple_tau_100": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "tau_policy": "fixed_100",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_length_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "tau_policy": "length_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_phase_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "tau_policy": "phase_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_uncertainty_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "tau_policy": "uncertainty_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_length_phase_adaptive": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "tau_policy": "length_phase_adaptive",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
    }


def tau_from_policy(policy: str, head_name: str, h: np.ndarray, memories: List[np.ndarray], length: int) -> float:
    if policy == "fixed_100":
        return 1.0
    if policy == "length_adaptive":
        return 1.4 if length <= 8 else 1.0
    if policy == "phase_adaptive":
        return {"tool": 1.35, "route": 1.05, "final": 0.95}.get(head_name, 1.0)
    if policy == "length_phase_adaptive":
        if length <= 8:
            return {"tool": 1.45, "route": 1.35, "final": 1.10}.get(head_name, 1.2)
        return {"tool": 1.10, "route": 1.00, "final": 0.90}.get(head_name, 1.0)
    if policy == "uncertainty_adaptive":
        if not memories:
            return 1.0
        strengths = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        relative_dispersion = float(np.std(strengths) / (np.mean(strengths) + 1e-6))
        h_norm = float(np.linalg.norm(h))
        h_term = min(h_norm / 2.5, 1.0)
        base = 1.05 if length >= 10 else 1.20
        tau = base + 0.35 * relative_dispersion - 0.15 * h_term
        return float(np.clip(tau, 0.75, 1.45))
    return 1.0


def policy_description(policy: str) -> str:
    if policy == "fixed_100":
        return "固定 tau_g = 1.0。"
    if policy == "length_adaptive":
        return "短任务更软，L<=8 用 tau=1.4，长任务回到 tau=1.0。"
    if policy == "phase_adaptive":
        return "按阶段调温度：tool 更软，final 更硬。"
    if policy == "uncertainty_adaptive":
        return "按记忆离散度和隐藏态范数调温度，不确定性高时更软。"
    if policy == "length_phase_adaptive":
        return "同时按任务长度和阶段调温度，短任务整体更软，长任务尾段更硬。"
    return "无门控温度策略。"


def attach_dynamic_gate(
    model: gated_scan.ContextGatedMemoryLearner,
    tau_policy: str,
    length: int,
) -> None:
    model._tau_history = []  # type: ignore[attr-defined]

    def gates(self, head_name: str, h: np.ndarray, memories: List[np.ndarray]):
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            return ones, np.zeros(len(memories), dtype=np.float32)
        w_gate, b_gate = self._gate_params(head_name)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        gate_logits = (h @ w_gate + b_gate + 0.05 * mem_strength).astype(np.float32)
        tau_g = tau_from_policy(tau_policy, head_name, h, memories, length)
        self._tau_history.append(float(tau_g))
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
        attach_dynamic_gate(model, str(cfg["tau_policy"]), length)

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
    tau_mean = float(np.mean(tau_history)) if tau_history else 0.0
    tau_std = float(np.std(tau_history)) if tau_history else 0.0

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
        "tau_mean": tau_mean,
        "tau_std": tau_std,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic gate-temperature scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_dynamic_temperature_scan_20260308.json",
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
        systems[system_name]["policy_description"] = policy_description(str(cfg["tau_policy"]))
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "policy": str(cfg["tau_policy"]),
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "mean_gate_entropy": float(g["mean_gate_entropy"]),
                "mean_gate_peak": float(g["mean_gate_peak"]),
                "tau_mean": float(np.mean(g["tau_mean_curve"])) if "tau_mean_curve" in g else 0.0,
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    dynamic_rows = [row for row in ranking if row["system"].startswith("gated_triple_tau_") and row["policy"] != "fixed_100"]
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
            "best_gate_selective_dynamic": max(dynamic_rows, key=lambda row: row["mean_gate_peak"]),
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
        },
        "hypotheses": {
            "H1_some_dynamic_beats_fixed_tau_100_on_average": bool(
                max(dynamic_rows, key=lambda row: row["mean_closure_score"])["mean_closure_score"]
                > fixed_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_some_dynamic_beats_fixed_tau_100_at_max_length": bool(
                max(dynamic_rows, key=lambda row: row["max_length_score"])["max_length_score"]
                > fixed_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H3_length_adaptive_beats_fixed_tau_100_on_average": bool(
                systems["gated_triple_tau_length_adaptive"]["global_summary"]["mean_closure_score"]
                > fixed_ref["global_summary"]["mean_closure_score"]
            ),
            "H4_length_phase_adaptive_beats_single_anchor_on_average": bool(
                systems["gated_triple_tau_length_phase_adaptive"]["global_summary"]["mean_closure_score"]
                > single_ref["global_summary"]["mean_closure_score"]
            ),
            "H5_uncertainty_adaptive_improves_retention_vs_fixed": bool(
                systems["gated_triple_tau_uncertainty_adaptive"]["global_summary"]["mean_retention_score"]
                > fixed_ref["global_summary"]["mean_retention_score"]
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
