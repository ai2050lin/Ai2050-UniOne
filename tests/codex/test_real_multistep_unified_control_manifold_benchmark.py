#!/usr/bin/env python
"""
Real multi-step benchmark with a unified low-dimensional control manifold.

This extends the learnable hierarchical state machine with:
- a compact control manifold that modulates all heads jointly
- rollback + recovery on perturbed real multi-step episodes
- externalized evaluation focused on episode success, recovery, retention,
  and long-horizon stability on real task episodes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_real_multistep_agi_closure_memory_boost_scan as memory_scan
import test_real_multistep_memory_gated_multiscale_scan as gated_scan
import test_real_multistep_memory_hierarchical_state_scan as hierarchical_scan
import test_real_multistep_memory_learnable_state_machine as learnable_scan


ROOT = Path(__file__).resolve().parents[2]


class UnifiedControlManifoldLearner(learnable_scan.LearnableStateMachineLearner):
    def __post_init__(self) -> None:
        super().__post_init__()
        rng = np.random.default_rng(self.seed + 71)
        gate_count = max(1, len(self.memory_betas))
        self.control_dim = 4
        self.w_control_tool = rng.normal(scale=0.05, size=(self.control_dim, gate_count)).astype(np.float32)
        self.w_control_route = rng.normal(scale=0.05, size=(self.control_dim, gate_count)).astype(np.float32)
        self.w_control_final = rng.normal(scale=0.05, size=(self.control_dim, gate_count)).astype(np.float32)
        self.b_control_tool = np.zeros(gate_count, dtype=np.float32)
        self.b_control_route = np.zeros(gate_count, dtype=np.float32)
        self.b_control_final = np.zeros(gate_count, dtype=np.float32)
        self.rollback_threshold = 0.57
        self.recovery_scale = 1.0
        self.feedback_ema = 0.5
        self.retention_ema = 0.5

    def _control_params(self, head_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if head_name == "tool":
            return self.w_control_tool, self.b_control_tool
        if head_name == "route":
            return self.w_control_route, self.b_control_route
        return self.w_control_final, self.b_control_final

    def _set_control_params(self, head_name: str, w_control: np.ndarray, b_control: np.ndarray) -> None:
        if head_name == "tool":
            self.w_control_tool, self.b_control_tool = w_control, b_control
        elif head_name == "route":
            self.w_control_route, self.b_control_route = w_control, b_control
        else:
            self.w_control_final, self.b_control_final = w_control, b_control

    @staticmethod
    def control_features(state: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        phase = state[-3:].astype(np.float32)
        segment = state[13:26] if state.shape[0] >= 26 else state
        global_summary = state[26:39] if state.shape[0] >= 39 else state
        progress = step_idx / max(total_steps - 1, 1)
        return np.array(
            [
                float(np.mean(np.abs(phase))),
                float(np.mean(np.abs(segment))),
                float(np.mean(np.abs(global_summary))),
                float(progress),
            ],
            dtype=np.float32,
        )

    def gates_with_state(
        self,
        head_name: str,
        h: np.ndarray,
        memories: List[np.ndarray],
        state: np.ndarray,
        step_idx: int,
        total_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(memories) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        if len(memories) == 1 or not self.use_gate:
            ones = np.ones(len(memories), dtype=np.float32)
            phase, summary = self.state_features(state)
            return ones, np.zeros(len(memories), dtype=np.float32), phase, summary

        w_gate, b_gate = self._gate_params(head_name)
        w_phase, w_summary, b_phase = self._phase_params(head_name)
        w_control, b_control = self._control_params(head_name)
        phase, summary = self.state_features(state)
        control = self.control_features(state, step_idx, total_steps)
        mem_strength = np.asarray([float(np.linalg.norm(memory)) for memory in memories], dtype=np.float32)
        remain_ratio = max(total_steps - step_idx - 1, 0) / max(total_steps - 1, 1)
        dynamic_bias = np.array([0.16 * remain_ratio, 0.08 * (1.0 - abs(control[-1] - 0.5)), -0.10 * control[-1]], dtype=np.float32)
        gate_logits = (
            h @ w_gate
            + phase @ w_phase
            + summary @ w_summary
            + control @ w_control
            + b_gate
            + b_phase
            + b_control
            + 0.05 * mem_strength
            + self.recovery_scale * dynamic_bias
        ).astype(np.float32)
        gates = gated_scan.softmax(gate_logits)
        return gates, gate_logits, phase, summary

    def train_episode(self, states: List[np.ndarray], head_targets: List[Tuple[str, int]]) -> None:
        prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        prev_memories = [np.zeros(self.hidden_dim, dtype=np.float32) for _ in self.memory_betas]
        traces = []
        total_steps = len(states)

        for step_idx, (state, (head_name, target_idx)) in enumerate(zip(states, head_targets)):
            _pre, h = self.step_hidden(state, prev_h)
            memories = self.update_memories(prev_memories, h)
            gates, gate_logits, phase, summary = self.gates_with_state(head_name, h, memories, state, step_idx, total_steps)
            control = self.control_features(state, step_idx, total_steps)
            logits, contributions = self.logits(head_name, h, memories, gates)
            probs = gated_scan.base.softmax(logits)
            w_head, w_mem_list, _bias = self._head_params(head_name)
            target_vec = gated_scan.base.one_hot(int(target_idx), logits.shape[0])
            err = (target_vec - probs).astype(np.float32)
            traces.append(
                {
                    "head": head_name,
                    "state": state,
                    "prev_h": prev_h.copy(),
                    "h": h,
                    "memories": [memory.copy() for memory in memories],
                    "gates": gates.copy(),
                    "phase": phase.copy(),
                    "summary": summary.copy(),
                    "control": control.copy(),
                    "contributions": [c.copy() for c in contributions],
                    "err": err,
                }
            )
            prev_h = h
            prev_memories = memories

        for trace in traces:
            head_name = trace["head"]
            h = trace["h"]
            memories = trace["memories"]
            gates = trace["gates"]
            contributions = trace["contributions"]
            err = trace["err"]
            w_head, w_mem_list, bias = self._head_params(head_name)
            w_head = w_head + self.lr_head * np.outer(h, err).astype(np.float32)
            updated_mem = []
            for idx, (memory, w_mem) in enumerate(zip(memories, w_mem_list)):
                updated_mem.append(w_mem + self.lr_head * gates[idx] * np.outer(memory, err).astype(np.float32))
            bias = bias + self.lr_head * err.astype(np.float32)
            self._set_head_params(head_name, w_head, updated_mem, bias)

            gate_grad_logits = np.zeros(len(gates), dtype=np.float32)
            if self.use_gate and len(gates) > 1:
                gate_signal = np.asarray([float(np.dot(contrib, err)) for contrib in contributions], dtype=np.float32)
                gate_grad_logits = (gates * (gate_signal - float(np.sum(gates * gate_signal)))).astype(np.float32)
                w_gate, b_gate = self._gate_params(head_name)
                w_phase, w_summary, b_phase = self._phase_params(head_name)
                w_control, b_control = self._control_params(head_name)
                w_gate = w_gate + self.gate_lr * np.outer(h, gate_grad_logits).astype(np.float32)
                b_gate = b_gate + self.gate_lr * gate_grad_logits.astype(np.float32)
                w_phase = w_phase + self.gate_lr * np.outer(trace["phase"], gate_grad_logits).astype(np.float32)
                w_summary = w_summary + self.gate_lr * np.outer(trace["summary"], gate_grad_logits).astype(np.float32)
                b_phase = b_phase + 0.5 * self.gate_lr * gate_grad_logits.astype(np.float32)
                w_control = w_control + self.gate_lr * np.outer(trace["control"], gate_grad_logits).astype(np.float32)
                b_control = b_control + 0.5 * self.gate_lr * gate_grad_logits.astype(np.float32)
                self._set_gate_params(head_name, w_gate, b_gate)
                self._set_phase_params(head_name, w_phase, w_summary, b_phase)
                self._set_control_params(head_name, w_control, b_control)

            if self.use_trace:
                grad_h = (w_head @ err).astype(np.float32)
                for gate_value, beta, w_mem in zip(gates, self.memory_betas, updated_mem):
                    grad_h = grad_h + gate_value * (1.0 - beta) * (w_mem @ err).astype(np.float32)
                if self.use_gate and len(gates) > 1:
                    w_gate, _ = self._gate_params(head_name)
                    grad_h = grad_h + (w_gate @ gate_grad_logits).astype(np.float32)
                grad_pre = (grad_h * (1.0 - np.square(h))).astype(np.float32)
                grad_enc = np.outer(trace["state"], grad_pre).astype(np.float32)
                grad_rec = np.outer(trace["prev_h"], grad_pre).astype(np.float32)
                self.w_enc = self._apply_regularized_update(self.w_enc, grad_enc, self.enc_ref, self.enc_importance, self.lr_enc)
                self.w_rec = self._apply_regularized_update(self.w_rec, grad_rec, self.rec_ref, self.rec_importance, self.lr_rec)

    def update_feedback(self, success: bool, recovered: bool, retention_signal: float, confidence: float) -> None:
        reward = 1.0 if success else -1.0
        if recovered:
            reward += 0.35
        self.feedback_ema = 0.92 * self.feedback_ema + 0.08 * max(0.0, reward)
        self.retention_ema = 0.92 * self.retention_ema + 0.08 * retention_signal
        self.recovery_scale = float(np.clip(self.recovery_scale + 0.04 * ((0.0 if success else 1.0) + 0.3 * (1.0 - confidence)), 0.7, 1.8))
        self.rollback_threshold = float(np.clip(self.rollback_threshold + 0.02 * ((0.0 if success else 1.0) - 0.6 * confidence), 0.42, 0.68))


def clone_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def snapshot_model(model) -> Dict[str, object]:
    return {key: clone_value(value) for key, value in model.__dict__.items()}


def restore_model(model, state: Dict[str, object]) -> None:
    model.__dict__.clear()
    model.__dict__.update({key: clone_value(value) for key, value in state.items()})


def build_model(system_name: str, seed: int):
    if system_name == "single_anchor_beta_086":
        cfg = learnable_scan.system_configs()["single_anchor_beta_086"]
        return learnable_scan.build_model(cfg, seed)
    if system_name == "learnable_state_machine_h12":
        cfg = learnable_scan.system_configs()["learnable_state_machine_h12"]
        return learnable_scan.build_model(cfg, seed)
    if system_name == "unified_control_manifold_h12":
        return UnifiedControlManifoldLearner(
            input_dim=42,
            hidden_dim=12,
            lr_head=0.11,
            lr_enc=0.045,
            lr_rec=0.055,
            gate_lr=0.050,
            use_trace=True,
            memory_betas=[0.50, 0.80, 0.92],
            use_gate=True,
            stability=0.26,
            seed=seed,
        )
    raise KeyError(system_name)


def hierarchical_episode_pool(
    families: List[str],
    length: int,
    repeats: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
) -> List[Tuple[List[np.ndarray], List[Tuple[str, int]]]]:
    return hierarchical_scan.episode_pool(
        families,
        length,
        repeats=repeats,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
        state_mode="hierarchical",
    )


def predict_episode_stats(model, states: List[np.ndarray], targets: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int]], Dict[str, float]]:
    prev_h = np.zeros(model.hidden_dim, dtype=np.float32)
    prev_memories = [np.zeros(model.hidden_dim, dtype=np.float32) for _ in getattr(model, "memory_betas", [])]
    rows = []
    confidences = []
    gate_entropy = []
    gate_peak = []
    total_steps = len(states)

    for step_idx, (state, (head_name, target_idx)) in enumerate(zip(states, targets)):
        _pre, h = model.step_hidden(state, prev_h)
        memories = model.update_memories(prev_memories, h)
        if hasattr(model, "gates_with_state"):
            gates, _gate_logits, _phase, _summary = model.gates_with_state(head_name, h, memories, state, step_idx, total_steps)
        else:
            gates, _gate_logits = model.gates(head_name, h, memories)
        logits, _contribs = model.logits(head_name, h, memories, gates)
        probs = gated_scan.base.softmax(logits)
        rows.append((head_name, int(np.argmax(probs)), int(target_idx)))
        confidences.append(float(np.max(probs)))
        if len(gates) > 0:
            gate_entropy.append(float(gated_scan.normalized_entropy(gates)))
            gate_peak.append(float(np.max(gates)))
        prev_h = h
        prev_memories = memories

    return rows, {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "min_confidence": float(np.min(confidences)) if confidences else 0.0,
        "mean_gate_entropy": float(np.mean(gate_entropy)) if gate_entropy else 0.0,
        "mean_gate_peak": float(np.mean(gate_peak)) if gate_peak else 0.0,
    }


def perturb_episode(states: List[np.ndarray], rng: np.random.Generator, noise_scale: float, dropout_p: float) -> List[np.ndarray]:
    rows = []
    for state in states:
        x = state.astype(np.float32).copy()
        x = x + rng.normal(scale=noise_scale, size=x.shape[0]).astype(np.float32)
        if dropout_p > 0.0:
            mask = (rng.random(x.shape[0]) >= dropout_p).astype(np.float32)
            x = x * mask
        rows.append(x.astype(np.float32))
    return rows


def episode_success(rows: List[Tuple[str, int, int]]) -> float:
    return float(all(pred_idx == target_idx for _head_name, pred_idx, target_idx in rows))


def evaluate_recovery(
    model,
    families: List[str],
    length: int,
    rng: np.random.Generator,
    noise: float,
    dropout_p: float,
    repeats: int,
    phase1_memory: List[Tuple[List[np.ndarray], List[Tuple[str, int]]]],
    phase2_memory: List[Tuple[List[np.ndarray], List[Tuple[str, int]]]],
    enable_recovery: bool,
) -> Dict[str, float]:
    perturbed_success = 0
    recovered_success = 0
    rollback_attempts = 0
    rollback_successes = 0
    total = 0
    mean_confidence = []

    for _ in range(repeats):
        for family in families:
            for concept in memory_scan.base.CONCEPTS[family]:
                states, targets = hierarchical_scan.build_episode(
                    rng,
                    concept,
                    length,
                    noise,
                    dropout_p,
                    state_mode="hierarchical",
                )
                perturbed_states = perturb_episode(states, rng, noise_scale=noise * 0.40, dropout_p=min(0.25, dropout_p + 0.05))
                snapshot = snapshot_model(model)
                preds, stats = predict_episode_stats(model, perturbed_states, targets)
                success = episode_success(preds)
                perturbed_success += int(success)
                final_success = success
                recovered = False
                confidence = float(stats["mean_confidence"])

                threshold = float(getattr(model, "rollback_threshold", 0.52))
                trigger = (not success) or confidence < threshold
                if enable_recovery and trigger:
                    rollback_attempts += 1
                    restore_model(model, snapshot)
                    mem1 = phase1_memory[rng.integers(0, len(phase1_memory))]
                    model.train_episode(mem1[0], mem1[1])
                    model.train_episode(mem1[0], mem1[1])
                    if getattr(model, "recovery_scale", 1.0) < 1.35:
                        mem2 = phase2_memory[rng.integers(0, len(phase2_memory))]
                        model.train_episode(mem2[0], mem2[1])
                    if hasattr(model, "recovery_scale"):
                        model.recovery_scale = float(np.clip(model.recovery_scale + 0.05, 0.7, 1.8))
                    preds_retry, stats_retry = predict_episode_stats(model, perturbed_states, targets)
                    final_success = bool(episode_success(preds_retry))
                    recovered = final_success
                    rollback_successes += int(recovered)
                    confidence = float(stats_retry["mean_confidence"])

                recovered_success += int(final_success)
                mean_confidence.append(confidence)
                if hasattr(model, "update_feedback"):
                    model.update_feedback(success=final_success, recovered=recovered, retention_signal=1.0 if final_success else 0.0, confidence=confidence)
                total += 1

    return {
        "perturbed_episode_success": float(perturbed_success / max(1, total)),
        "recovered_episode_success": float(recovered_success / max(1, total)),
        "rollback_trigger_rate": float(rollback_attempts / max(1, total)),
        "rollback_recovery_rate": float(rollback_successes / max(1, rollback_attempts)),
        "recovery_mean_confidence": float(np.mean(mean_confidence)) if mean_confidence else 0.0,
    }


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = build_model(system_name, seed)

    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]

    phase1_memory = hierarchical_episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = hierarchical_episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in hierarchical_episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")

    model.consolidate(phase1_memory)

    phase2_rows = hierarchical_episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (states, targets) in enumerate(phase2_rows):
        model.train_episode(states, targets)
        if idx % 1 == 0:
            mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
            model.train_episode(mem_states, mem_targets)
        if system_name == "unified_control_manifold_h12":
            if idx % 3 == 0:
                mem_states, mem_targets = phase1_memory[(idx + 1) % len(phase1_memory)]
                model.train_episode(mem_states, mem_targets)
        elif idx % 1 == 0 and system_name != "single_anchor_beta_086":
            abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
            model.train_episode(abs_states, abs_targets)

    phase2_eval = hierarchical_scan.evaluate_system(model, phase2_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    retention_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    overall_eval = hierarchical_scan.evaluate_system(model, all_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    recovery_eval = evaluate_recovery(
        model=model,
        families=all_families,
        length=length,
        rng=rng,
        noise=noise,
        dropout_p=dropout_p,
        repeats=12,
        phase1_memory=phase1_memory,
        phase2_memory=phase2_memory,
        enable_recovery=(system_name == "unified_control_manifold_h12"),
    )

    unified_real_task_score = float(
        0.18 * overall_eval["tool_accuracy"]
        + 0.18 * overall_eval["route_accuracy"]
        + 0.18 * overall_eval["final_accuracy"]
        + 0.16 * overall_eval["episode_success"]
        + 0.12 * retention_eval["episode_success"]
        + 0.10 * recovery_eval["perturbed_episode_success"]
        + 0.08 * recovery_eval["rollback_recovery_rate"]
    )

    row = {
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
        "perturbed_episode_success": recovery_eval["perturbed_episode_success"],
        "recovered_episode_success": recovery_eval["recovered_episode_success"],
        "rollback_trigger_rate": recovery_eval["rollback_trigger_rate"],
        "rollback_recovery_rate": recovery_eval["rollback_recovery_rate"],
        "recovery_mean_confidence": recovery_eval["recovery_mean_confidence"],
        "unified_real_task_score": unified_real_task_score,
    }
    if hasattr(model, "rollback_threshold"):
        row["final_rollback_threshold"] = float(model.rollback_threshold)
    else:
        row["final_rollback_threshold"] = 0.0
    if hasattr(model, "recovery_scale"):
        row["final_recovery_scale"] = float(model.recovery_scale)
    else:
        row["final_recovery_scale"] = 1.0
    return row


def summarize_system(rows_by_length: Dict[int, List[Dict[str, float]]], lengths: List[int]) -> Dict[str, object]:
    per_length = {}
    closure_curve = []
    unified_curve = []
    recovery_curve = []
    retention_curve = []
    for length in lengths:
        summary = memory_scan.summarize_runs(rows_by_length[length])
        closure = memory_scan.closure_score(summary)
        unified = float(summary["unified_real_task_score"]["mean"])
        recovery = float(summary["rollback_recovery_rate"]["mean"])
        retention = float(summary["retention_after_phase2"]["mean"])
        per_length[str(length)] = {
            "summary": summary,
            "real_closure_score": closure,
            "unified_real_task_score": unified,
        }
        closure_curve.append(closure)
        unified_curve.append(unified)
        recovery_curve.append(recovery)
        retention_curve.append(retention)

    return {
        "per_length": per_length,
        "global_summary": {
            "lengths": lengths,
            "closure_curve": closure_curve,
            "unified_curve": unified_curve,
            "recovery_curve": recovery_curve,
            "retention_curve": retention_curve,
            "mean_closure_score": float(np.mean(closure_curve)),
            "mean_unified_score": float(np.mean(unified_curve)),
            "mean_recovery_rate": float(np.mean(recovery_curve)),
            "mean_retention_score": float(np.mean(retention_curve)),
            "closure_relative_drop": memory_scan.relative_drop(closure_curve),
            "unified_relative_drop": memory_scan.relative_drop(unified_curve),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real multi-step unified control manifold benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/real_multistep_unified_control_manifold_benchmark_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    lengths = [int(v) for v in args.lengths]
    max_length = max(lengths)
    systems = {}
    ranking = []

    for system_name in ["single_anchor_beta_086", "learnable_state_machine_h12", "unified_control_manifold_h12"]:
        rows_by_length = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                rows_by_length[length].append(
                    run_for_length(system_name, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p))
                )
        systems[system_name] = summarize_system(rows_by_length, lengths)
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_closure_score": float(g["mean_closure_score"]),
                "mean_unified_score": float(g["mean_unified_score"]),
                "mean_recovery_rate": float(g["mean_recovery_rate"]),
                "max_length_unified_score": float(systems[system_name]["per_length"][str(max_length)]["unified_real_task_score"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_unified_score"], reverse=True)
    single = systems["single_anchor_beta_086"]["per_length"][str(max_length)]
    learnable = systems["learnable_state_machine_h12"]["per_length"][str(max_length)]
    unified = systems["unified_control_manifold_h12"]["per_length"][str(max_length)]

    payload = {
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
        "best_system": ranking[0],
        "gains": {
            "unified_vs_single_at_max_length": float(unified["unified_real_task_score"] - single["unified_real_task_score"]),
            "unified_vs_learnable_at_max_length": float(unified["unified_real_task_score"] - learnable["unified_real_task_score"]),
            "unified_recovery_gain_vs_learnable": float(
                unified["summary"]["rollback_recovery_rate"]["mean"] - learnable["summary"]["rollback_recovery_rate"]["mean"]
            ),
            "unified_episode_gain_vs_learnable": float(
                unified["summary"]["overall_episode_success"]["mean"] - learnable["summary"]["overall_episode_success"]["mean"]
            ),
        },
        "headline_metrics": {
            "max_length_unified_score": float(unified["unified_real_task_score"]),
            "max_length_episode_success": float(unified["summary"]["overall_episode_success"]["mean"]),
            "max_length_recovery_rate": float(unified["summary"]["rollback_recovery_rate"]["mean"]),
            "max_length_retention": float(unified["summary"]["retention_after_phase2"]["mean"]),
        },
        "hypotheses": {
            "H1_unified_beats_single_at_max_length": bool(unified["unified_real_task_score"] > single["unified_real_task_score"]),
            "H2_unified_beats_learnable_at_max_length": bool(unified["unified_real_task_score"] > learnable["unified_real_task_score"]),
            "H3_unified_improves_recovery_vs_learnable": bool(
                unified["summary"]["rollback_recovery_rate"]["mean"] > learnable["summary"]["rollback_recovery_rate"]["mean"]
            ),
        },
        "project_readout": {
            "summary": "这一版把低维统一控制流形接回真实多步 episode，统一调制阶段状态、记忆门控和失败回退，并把评价继续外部化到真实任务成功率、恢复率、保留率和长程稳定性。",
            "next_question": "如果 unified control manifold 在真实多步任务上也能稳定领先，下一步就该把这套控制流形继续压缩成更小结构，并接回真实工具接口与脑侧约束。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_system"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

