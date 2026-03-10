#!/usr/bin/env python
"""
Test whether an explicit segment-summary state s_t can recover ultra-long
horizon performance on top of joint gate-temperature laws.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import test_real_multistep_agi_closure_memory_boost_scan as memory_scan
import test_real_multistep_memory_gate_temperature_scan as temperature_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan
import test_real_multistep_memory_ultra_long_horizon_temperature_scan as ultra_scan


def system_configs() -> Dict[str, Dict[str, object]]:
    return {
        "single_anchor_beta_086": {
            "betas": [0.86],
            "use_gate": False,
            "policy": "none",
            "use_segment_summary": False,
            "stability": 0.16,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 3.0,
            "head_lr_scale": 1.0,
            "gate_lr": 0.0,
        },
        "gated_triple_tau_joint_long_horizon": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_long_horizon",
            "use_segment_summary": False,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_long_horizon_segment_summary": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_long_horizon",
            "use_segment_summary": True,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_ultra_oracle": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_ultra_oracle",
            "use_segment_summary": False,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
        "gated_triple_tau_joint_ultra_oracle_segment_summary": {
            "betas": [0.50, 0.80, 0.92],
            "use_gate": True,
            "policy": "joint_ultra_oracle",
            "use_segment_summary": True,
            "stability": 0.17,
            "phase1_replay_stride": 1.0,
            "phase2_replay_stride": 2.0,
            "head_lr_scale": 0.90,
            "gate_lr": 0.075,
        },
    }


def policy_description(system_name: str, policy: str, use_segment_summary: bool) -> str:
    base = ultra_scan.policy_description(policy) if policy != "none" else "单锚点基线。"
    if use_segment_summary:
        return f"{base} 并显式接入段级摘要状态 s_t。"
    return base


def augment_states_with_segment_summary(states: List[np.ndarray], segment_span: int = 6) -> List[np.ndarray]:
    anchor = states[0].astype(np.float32)
    rows: List[np.ndarray] = []
    for idx, state in enumerate(states):
        if idx == 0:
            summary = np.zeros_like(state, dtype=np.float32)
        else:
            start = max(0, idx - segment_span)
            recent = np.mean(np.stack(states[start:idx], axis=0), axis=0).astype(np.float32)
            cumulative = np.mean(np.stack(states[:idx], axis=0), axis=0).astype(np.float32)
            summary = (0.55 * anchor + 0.30 * recent + 0.15 * cumulative).astype(np.float32)
        rows.append(np.concatenate([state.astype(np.float32), summary], axis=0))
    return rows


def attach_policy_gate(model: gated_scan.ContextGatedMemoryLearner, policy: str, length: int) -> None:
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
        self._tau_history.append(float(tau_g))
        self._tau_call_index += 1
        return temperature_scan.softmax_with_temperature(gate_logits, tau_g), gate_logits

    model.gates = types.MethodType(gates, model)


def build_episode(
    rng: np.random.Generator,
    concept: str,
    length: int,
    noise: float,
    dropout_p: float,
    use_segment_summary: bool,
) -> Tuple[List[np.ndarray], List[Tuple[str, int]]]:
    states, targets = memory_scan.build_episode(rng, concept, length, noise, dropout_p)
    if use_segment_summary:
        states = augment_states_with_segment_summary(states)
    return states, targets


def episode_pool(
    families: List[str],
    length: int,
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    use_segment_summary: bool,
) -> List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]:
    rows = []
    for _ in range(repeats):
        for family in families:
            for concept in memory_scan.base.CONCEPTS[family]:
                rows.append(build_episode(rng, concept, length, noise, dropout_p, use_segment_summary))
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
    use_segment_summary: bool,
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
                states, targets = build_episode(rng, concept, length, noise, dropout_p, use_segment_summary)
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


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    cfg = system_configs()[system_name]
    rng = np.random.default_rng(seed)
    use_segment_summary = bool(cfg["use_segment_summary"])
    input_dim = 26 if use_segment_summary else 13
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
        attach_policy_gate(model, str(cfg["policy"]), length)

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p, use_segment_summary=use_segment_summary)
    phase2_memory = episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p, use_segment_summary=use_segment_summary)

    for states, targets in episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p, use_segment_summary=use_segment_summary):
        model.train_episode(states, targets)
    phase1_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, use_segment_summary=use_segment_summary)

    model.consolidate(phase1_memory)

    phase2_rows = episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p, use_segment_summary=use_segment_summary)
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
    phase2_eval = evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18, use_segment_summary=use_segment_summary)
    retention_eval = evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, use_segment_summary=use_segment_summary)
    overall_eval = evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18, use_segment_summary=use_segment_summary)
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
    ap = argparse.ArgumentParser(description="Segment-summary scan on ultra-long horizons")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/real_multistep_memory_segment_summary_scan_20260309.json",
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
        systems[system_name]["policy_description"] = policy_description(str(system_name), str(cfg["policy"]), bool(cfg["use_segment_summary"]))
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "policy": str(cfg["policy"]),
                "use_segment_summary": bool(cfg["use_segment_summary"]),
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
    joint_ref = systems["gated_triple_tau_joint_long_horizon"]
    joint_seg_ref = systems["gated_triple_tau_joint_long_horizon_segment_summary"]
    ultra_ref = systems["gated_triple_tau_joint_ultra_oracle"]
    ultra_seg_ref = systems["gated_triple_tau_joint_ultra_oracle_segment_summary"]

    per_length_gains = {}
    for length in lengths:
        per_length_gains[str(length)] = {
            "joint_segment_vs_joint": float(
                joint_seg_ref["per_length"][str(length)]["real_closure_score"]
                - joint_ref["per_length"][str(length)]["real_closure_score"]
            ),
            "ultra_segment_vs_ultra": float(
                ultra_seg_ref["per_length"][str(length)]["real_closure_score"]
                - ultra_ref["per_length"][str(length)]["real_closure_score"]
            ),
            "best_segment_vs_single": float(
                max(
                    joint_seg_ref["per_length"][str(length)]["real_closure_score"],
                    ultra_seg_ref["per_length"][str(length)]["real_closure_score"],
                )
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
            "joint_segment_mean_vs_joint": float(
                joint_seg_ref["global_summary"]["mean_closure_score"] - joint_ref["global_summary"]["mean_closure_score"]
            ),
            "joint_segment_max_vs_joint": float(
                joint_seg_ref["per_length"][str(max_length)]["real_closure_score"]
                - joint_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "ultra_segment_mean_vs_ultra": float(
                ultra_seg_ref["global_summary"]["mean_closure_score"] - ultra_ref["global_summary"]["mean_closure_score"]
            ),
            "ultra_segment_max_vs_ultra": float(
                ultra_seg_ref["per_length"][str(max_length)]["real_closure_score"]
                - ultra_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "best_segment_max_vs_single": float(
                max(
                    joint_seg_ref["per_length"][str(max_length)]["real_closure_score"],
                    ultra_seg_ref["per_length"][str(max_length)]["real_closure_score"],
                )
                - single_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
        },
        "best": {
            "best_mean_segment_system": max(
                [row for row in ranking if row["use_segment_summary"]],
                key=lambda row: row["mean_closure_score"],
            ),
            "best_max_segment_system": max(
                [row for row in ranking if row["use_segment_summary"]],
                key=lambda row: row["max_length_score"],
            ),
        },
        "hypotheses": {
            "H1_joint_segment_beats_joint_on_average": bool(
                joint_seg_ref["global_summary"]["mean_closure_score"] > joint_ref["global_summary"]["mean_closure_score"]
            ),
            "H2_joint_segment_beats_joint_at_max_length": bool(
                joint_seg_ref["per_length"][str(max_length)]["real_closure_score"] > joint_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H3_ultra_segment_beats_ultra_on_average": bool(
                ultra_seg_ref["global_summary"]["mean_closure_score"] > ultra_ref["global_summary"]["mean_closure_score"]
            ),
            "H4_ultra_segment_beats_ultra_at_max_length": bool(
                ultra_seg_ref["per_length"][str(max_length)]["real_closure_score"] > ultra_ref["per_length"][str(max_length)]["real_closure_score"]
            ),
            "H5_some_segment_system_beats_single_anchor_at_max_length": bool(
                max(
                    joint_seg_ref["per_length"][str(max_length)]["real_closure_score"],
                    ultra_seg_ref["per_length"][str(max_length)]["real_closure_score"],
                ) > single_ref["per_length"][str(max_length)]["real_closure_score"]
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
