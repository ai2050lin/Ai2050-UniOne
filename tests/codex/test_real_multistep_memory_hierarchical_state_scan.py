#!/usr/bin/env python
"""
Task block A:
Scan whether a hierarchical state representation can recover ultra-long
horizon performance beyond the earlier single segment-summary state s_t.

We extend the previous segment-summary setting with two extra slow variables:
- segment summary s_t: recent local summary
- global summary g_t: episode-level cumulative summary
- phase latent z_t: smoothed stage code

The effective gate policy becomes:
    tau_g = tau_g(L, phase_t, remaining_t, s_t, z_t)
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_real_multistep_agi_closure_memory_boost_scan as memory_scan
import test_real_multistep_memory_gate_temperature_scan as temperature_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan
import test_real_multistep_memory_segment_summary_scan as segment_scan
import test_real_multistep_memory_ultra_long_horizon_temperature_scan as ultra_scan


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "policy": "none",
            "state_mode": "base",
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "gated_triple_tau_joint_ultra_oracle_segment_summary": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_ultra_oracle",
            "state_mode": "segment",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_ultra_oracle_hierarchical_state": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_ultra_oracle",
            "state_mode": "hierarchical",
            "stability": 0.18,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.92,
            "gate_lr": 0.082,
        },
        "gated_triple_tau_joint_hierarchical_state": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_long_horizon",
            "state_mode": "hierarchical",
            "stability": 0.18,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.92,
            "gate_lr": 0.082,
        },
        "gated_triple_tau_joint_tailfocus_hierarchical_state": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_softroute_hardtail",
            "state_mode": "hierarchical",
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.94,
            "gate_lr": 0.078,
        },
    }


def policy_description(policy: str, state_mode: str) -> str:
    if state_mode == "base":
        return "单锚点基线，不使用显式分层状态。"
    if state_mode == "segment":
        return "联合温度律 + 段级摘要状态 s_t。"
    return "联合温度律 + 段级摘要 s_t + 全局摘要 g_t + 阶段隐变量 z_t。"


def augment_states_with_hierarchical_state(
    states: List[np.ndarray],
    segment_span: int = 6,
) -> List[np.ndarray]:
    anchor = states[0].astype(np.float32)
    rows: List[np.ndarray] = []
    cumulative_sum = np.zeros_like(anchor, dtype=np.float32)
    phase_latent = np.zeros(3, dtype=np.float32)
    for idx, state in enumerate(states):
        state = state.astype(np.float32)
        cumulative_sum += state
        if idx == 0:
            segment_summary = np.zeros_like(state, dtype=np.float32)
            global_summary = np.zeros_like(state, dtype=np.float32)
        else:
            start = max(0, idx - segment_span)
            recent = np.mean(np.stack(states[start:idx], axis=0), axis=0).astype(np.float32)
            cumulative = (cumulative_sum - state) / float(idx)
            segment_summary = (0.55 * anchor + 0.30 * recent + 0.15 * cumulative).astype(np.float32)
            global_summary = (0.35 * anchor + 0.65 * cumulative).astype(np.float32)
        stage_code = state[-3:].astype(np.float32)
        phase_latent = (0.72 * phase_latent + 0.28 * stage_code).astype(np.float32)
        rows.append(np.concatenate([state, segment_summary, global_summary, phase_latent], axis=0))
    return rows


def build_episode(
    rng: np.random.Generator,
    concept: str,
    length: int,
    noise: float,
    dropout_p: float,
    state_mode: str,
) -> Tuple[List[np.ndarray], List[Tuple[str, int]]]:
    states, targets = memory_scan.build_episode(rng, concept, length, noise, dropout_p)
    if state_mode == "segment":
        states = segment_scan.augment_states_with_segment_summary(states)
    elif state_mode == "hierarchical":
        states = augment_states_with_hierarchical_state(states)
    return states, targets


def episode_pool(
    families: List[str],
    length: int,
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    state_mode: str,
) -> List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in memory_scan.base.CONCEPTS[family]:
                rows.append(build_episode(rng, concept, length, noise, dropout_p, state_mode))
    rng.shuffle(rows)
    return rows


def evaluate_system(
    model: gated_scan.ContextGatedMemoryLearner,
    families: List[str],
    length: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    repeats: int,
    state_mode: str,
) -> Dict[str, float]:
    tool_correct = 0
    route_correct = 0
    final_correct = 0
    episode_correct = 0
    route_total = 0
    total = 0
    gate_entropy = []
    gate_peak = []
    for _ in range(repeats):
        for family in memory_scan.base.CONCEPTS.keys():
            if family not in families:
                continue
            for concept in memory_scan.base.CONCEPTS[family]:
                states, targets = build_episode(rng, concept, length, noise, dropout_p, state_mode)
                preds, gate_rows = model.predict_episode(states, targets)
                step_ok = []
                for head_name, pred_idx, target_idx in preds:
                    ok = int(pred_idx == target_idx)
                    if head_name == "tool":
                        tool_correct += ok
                    elif head_name == "route":
                        route_correct += ok
                        route_total += 1
                    else:
                        final_correct += ok
                    step_ok.append(ok)
                for row in gate_rows:
                    gate_entropy.append(float(row["entropy"]))
                    gate_peak.append(float(row["peak"]))
                episode_correct += int(all(step_ok))
                total += 1
    return {
        "tool_accuracy": float(tool_correct / max(1, total)),
        "route_accuracy": float(route_correct / max(1, route_total)),
        "final_accuracy": float(final_correct / max(1, total)),
        "episode_success": float(episode_correct / max(1, total)),
        "gate_entropy": float(np.mean(gate_entropy)) if gate_entropy else 0.0,
        "gate_peak": float(np.mean(gate_peak)) if gate_peak else 0.0,
    }


def attach_hierarchical_policy_gate(
    model: gated_scan.ContextGatedMemoryLearner,
    policy: str,
    length: int,
    state_mode: str,
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
        tau_g = ultra_scan.tau_from_policy(policy, head_name, length, step_idx, total_steps)
        if state_mode == "hierarchical":
            phase_slice = h[-3:] if h.shape[0] >= 3 else np.zeros(3, dtype=np.float32)
            phase_focus = float(np.max(np.abs(phase_slice)))
            long_memory = float(np.mean(np.abs(h[-16:]))) if h.shape[0] >= 16 else float(np.mean(np.abs(h)))
            tau_g = float(np.clip(tau_g - 0.10 * phase_focus - 0.08 * long_memory, 0.72, 1.42))
        self._tau_history.append(float(tau_g))
        self._tau_call_index += 1
        return temperature_scan.softmax_with_temperature(gate_logits, tau_g), gate_logits

    model.gates = types.MethodType(gates, model)


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    state_mode = str(cfg["state_mode"])
    input_dim = 13
    if state_mode == "segment":
        input_dim = 26
    elif state_mode == "hierarchical":
        input_dim = 42
    model = gated_scan.ContextGatedMemoryLearner(
        input_dim=input_dim,
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
        attach_hierarchical_policy_gate(model, str(cfg["policy"]), length, state_mode)

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode)
    phase2_memory = episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode)

    for states, targets in episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode):
        model.train_episode(states, targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)

    model.consolidate(phase1_memory)

    phase2_rows = episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p, state_mode=state_mode)
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
    phase2_eval = evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
    retention_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
    overall_eval = evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18, state_mode=state_mode)
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
    ap = argparse.ArgumentParser(description="Hierarchical-state scan on ultra-long horizons")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_hierarchical_state_scan_20260309.json",
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
        systems[system_name]["policy_description"] = policy_description(str(cfg["policy"]), str(cfg["state_mode"]))
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "policy": str(cfg["policy"]),
                "state_mode": str(cfg["state_mode"]),
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_retention_score": float(g["mean_retention_score"]),
                "closure_relative_drop": float(g["closure_relative_drop"]),
                "max_length_score": float(systems[system_name]["per_length"][str(max_length)]["real_closure_score"]),
                "mean_gate_entropy": float(g["mean_gate_entropy"]),
                "mean_gate_peak": float(g["mean_gate_peak"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_closure_score"], reverse=True)
    single_ref = systems["single_anchor_beta_086"]
    segment_ref = systems["gated_triple_tau_joint_ultra_oracle_segment_summary"]
    hier_ref = systems["gated_triple_tau_joint_ultra_oracle_hierarchical_state"]

    per_length_gains = {}
    for length in lengths:
        per_length_gains[str(length)] = {
            "hierarchical_vs_segment": float(
                hier_ref["per_length"][str(length)]["real_closure_score"]
                - segment_ref["per_length"][str(length)]["real_closure_score"]
            ),
            "hierarchical_vs_single_anchor": float(
                hier_ref["per_length"][str(length)]["real_closure_score"]
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
        "gains": {
            "per_length": per_length_gains,
            "hierarchical_mean_vs_segment": float(
                hier_ref["global_summary"]["mean_closure_score"] - segment_ref["global_summary"]["mean_closure_score"]
            ),
            "hierarchical_max_vs_segment": float(
                hier_ref["per_length"][str(max_length)]["real_closure_score"]
                - segment_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "hierarchical_max_vs_single_anchor": float(
                hier_ref["per_length"][str(max_length)]["real_closure_score"]
                - single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
        "best": {
            "best_mean_system": max(ranking, key=lambda row: row["mean_closure_score"]),
            "best_max_system": max(ranking, key=lambda row: row["max_length_score"]),
        },
        "hypotheses": {
            "H1_hierarchical_beats_segment_on_average": bool(
                hier_ref["global_summary"]["mean_closure_score"] > segment_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_hierarchical_beats_segment_at_max_length": bool(
                hier_ref["per_length"][str(max_length)]["real_closure_score"] > segment_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H3_hierarchical_beats_single_anchor_at_max_length": bool(
                hier_ref["per_length"][str(max_length)]["real_closure_score"] > single_ref["per_length"][str(max_length)]["real_closure_score"]
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
