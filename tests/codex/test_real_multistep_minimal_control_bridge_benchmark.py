#!/usr/bin/env python
"""
Real multi-step benchmark that compresses the unified control manifold into a
smaller bridge tied back to real tasks and brain-side constraints.
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
import test_real_multistep_unified_control_manifold_benchmark as unified_scan


class MinimalControlBridgeLearner(unified_scan.UnifiedControlManifoldLearner):
    def __post_init__(self) -> None:
        super().__post_init__()
        rng = np.random.default_rng(self.seed + 131)
        gate_count = max(1, len(self.memory_betas))
        self.control_dim = 2
        self.w_control_tool = rng.normal(scale=0.06, size=(self.control_dim, gate_count)).astype(np.float32)
        self.w_control_route = rng.normal(scale=0.06, size=(self.control_dim, gate_count)).astype(np.float32)
        self.w_control_final = rng.normal(scale=0.06, size=(self.control_dim, gate_count)).astype(np.float32)
        self.b_control_tool = np.zeros(gate_count, dtype=np.float32)
        self.b_control_route = np.zeros(gate_count, dtype=np.float32)
        self.b_control_final = np.zeros(gate_count, dtype=np.float32)
        self.basis_mix = 0.54
        self.control_smooth_ema = 0.52

    def control_features(self, state: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        phase = state[-3:].astype(np.float32)
        segment = state[13:26] if state.shape[0] >= 26 else state
        global_summary = state[26:39] if state.shape[0] >= 39 else state
        progress = step_idx / max(total_steps - 1, 1)
        phase_strength = float(np.mean(np.abs(phase)))
        segment_strength = float(np.mean(np.abs(segment)))
        global_strength = float(np.mean(np.abs(global_summary)))
        shared_core = (
            self.basis_mix * (0.42 * segment_strength + 0.40 * global_strength)
            + (1.0 - self.basis_mix) * phase_strength
        )
        protocol_gate = (
            0.42 * progress
            + 0.24 * (1.0 - self.feedback_ema)
            + 0.18 * (1.0 - self.retention_ema)
            + 0.16 * abs(segment_strength - global_strength)
        )
        return np.array([float(np.clip(shared_core, 0.0, 1.4)), float(np.clip(protocol_gate, 0.0, 1.4))], dtype=np.float32)

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
        progress = step_idx / max(total_steps - 1, 1)
        dynamic_bias = np.array(
            [
                0.16 * remain_ratio + 0.10 * control[0],
                0.07 * (1.0 - abs(progress - 0.5)) - 0.08 * control[1],
                0.12 * control[1] - 0.06 * progress,
            ],
            dtype=np.float32,
        )
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

    def update_feedback(self, success: bool, recovered: bool, retention_signal: float, confidence: float) -> None:
        super().update_feedback(success, recovered, retention_signal, confidence)
        delta = 0.035 * (retention_signal - (0.0 if success else 1.0))
        self.basis_mix = float(np.clip(self.basis_mix + delta, 0.42, 0.66))
        smooth_target = 0.68 if (success or recovered) else 0.40
        self.control_smooth_ema = 0.90 * self.control_smooth_ema + 0.10 * smooth_target
        self.rollback_threshold = float(np.clip(self.rollback_threshold - 0.006 * retention_signal, 0.40, 0.66))


def build_model(system_name: str, seed: int):
    if system_name == "single_anchor_beta_086":
        cfg = learnable_scan.system_configs()["single_anchor_beta_086"]
        return learnable_scan.build_model(cfg, seed)
    if system_name == "learnable_state_machine_h12":
        cfg = learnable_scan.system_configs()["learnable_state_machine_h12"]
        return learnable_scan.build_model(cfg, seed)
    if system_name == "unified_control_manifold_h12":
        return unified_scan.build_model(system_name, seed)
    if system_name == "minimal_control_bridge_h12":
        return MinimalControlBridgeLearner(
            input_dim=42,
            hidden_dim=12,
            lr_head=0.108,
            lr_enc=0.045,
            lr_rec=0.055,
            gate_lr=0.052,
            use_trace=True,
            memory_betas=[0.50, 0.80, 0.92],
            use_gate=True,
            stability=0.28,
            seed=seed,
        )
    raise KeyError(system_name)


def predict_episode_stats(model, states: List[np.ndarray], targets: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int]], Dict[str, object]]:
    prev_h = np.zeros(model.hidden_dim, dtype=np.float32)
    prev_memories = [np.zeros(model.hidden_dim, dtype=np.float32) for _ in getattr(model, "memory_betas", [])]
    rows = []
    confidences = []
    step_success = []
    gate_entropy = []
    gate_peak = []
    controls = []
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
        pred_idx = int(np.argmax(probs))
        rows.append((head_name, pred_idx, int(target_idx)))
        confidences.append(float(np.max(probs)))
        step_success.append(int(pred_idx == int(target_idx)))
        if len(gates) > 0:
            gate_entropy.append(float(gated_scan.normalized_entropy(gates)))
            gate_peak.append(float(np.max(gates)))
        if hasattr(model, "control_features"):
            controls.append(model.control_features(state, step_idx, total_steps).astype(np.float32))
        prev_h = h
        prev_memories = memories

    control_delta = 0.0
    progress_alignment = 0.5
    shared_core_stability = 0.5
    if len(controls) >= 2:
        control_array = np.stack(controls, axis=0)
        control_delta = float(np.mean(np.abs(np.diff(control_array, axis=0))))
        progress = np.linspace(0.0, 1.0, num=len(controls), dtype=np.float32)
        protocol = control_array[:, 1]
        if float(np.std(protocol)) > 1e-6:
            corr = float(np.corrcoef(progress, protocol)[0, 1])
            progress_alignment = float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))
        shared_core_stability = float(np.clip(1.0 - float(np.std(control_array[:, 0])), 0.0, 1.0))

    return rows, {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_gate_entropy": float(np.mean(gate_entropy)) if gate_entropy else 0.0,
        "mean_gate_peak": float(np.mean(gate_peak)) if gate_peak else 0.0,
        "step_confidences": confidences,
        "step_success": step_success,
        "control_delta": control_delta,
        "progress_alignment": progress_alignment,
        "shared_core_stability": shared_core_stability,
    }


def find_trigger_step(step_success: List[int], step_confidences: List[float], threshold: float) -> int:
    for idx, (ok, confidence) in enumerate(zip(step_success, step_confidences)):
        if (not ok) or (confidence < threshold):
            return idx
    if not step_confidences:
        return 0
    return int(np.argmin(np.asarray(step_confidences, dtype=np.float32)))


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
) -> Dict[str, object]:
    perturbed_success = 0
    recovered_success = 0
    rollback_attempts = 0
    rollback_successes = 0
    total = 0
    mean_confidence = []
    control_delta_rows = []
    progress_alignment_rows = []
    shared_core_rows = []
    attempts_by_step = np.zeros(length, dtype=np.float64)
    recoveries_by_step = np.zeros(length, dtype=np.float64)
    confidence_before_sum = np.zeros(length, dtype=np.float64)
    confidence_after_sum = np.zeros(length, dtype=np.float64)

    for _ in range(repeats):
        for family in families:
            for concept in memory_scan.base.CONCEPTS[family]:
                states, targets = hierarchical_scan.build_episode(rng, concept, length, noise, dropout_p, state_mode="hierarchical")
                perturbed_states = unified_scan.perturb_episode(states, rng, noise_scale=noise * 0.40, dropout_p=min(0.25, dropout_p + 0.05))
                snapshot = unified_scan.snapshot_model(model)
                preds, stats = predict_episode_stats(model, perturbed_states, targets)
                success = unified_scan.episode_success(preds)
                perturbed_success += int(success)
                final_success = success
                recovered = False
                confidence = float(stats["mean_confidence"])
                threshold = float(getattr(model, "rollback_threshold", 0.52))
                trigger = (not success) or confidence < threshold

                if trigger:
                    trigger_step = find_trigger_step(stats["step_success"], stats["step_confidences"], threshold)
                    attempts_by_step[trigger_step] += 1.0
                    confidence_before_sum[trigger_step] += float(stats["step_confidences"][trigger_step])
                else:
                    trigger_step = -1

                if enable_recovery and trigger:
                    rollback_attempts += 1
                    unified_scan.restore_model(model, snapshot)
                    mem1 = phase1_memory[rng.integers(0, len(phase1_memory))]
                    model.train_episode(mem1[0], mem1[1])
                    model.train_episode(mem1[0], mem1[1])
                    if getattr(model, "recovery_scale", 1.0) < 1.38:
                        mem2 = phase2_memory[rng.integers(0, len(phase2_memory))]
                        model.train_episode(mem2[0], mem2[1])
                    if hasattr(model, "recovery_scale"):
                        model.recovery_scale = float(np.clip(model.recovery_scale + 0.05, 0.7, 1.9))
                    preds_retry, stats_retry = predict_episode_stats(model, perturbed_states, targets)
                    final_success = bool(unified_scan.episode_success(preds_retry))
                    recovered = final_success
                    rollback_successes += int(recovered)
                    confidence = float(stats_retry["mean_confidence"])
                    if trigger_step >= 0:
                        recoveries_by_step[trigger_step] += float(recovered)
                        confidence_after_sum[trigger_step] += float(stats_retry["step_confidences"][trigger_step])

                recovered_success += int(final_success)
                mean_confidence.append(confidence)
                control_delta_rows.append(float(stats["control_delta"]))
                progress_alignment_rows.append(float(stats["progress_alignment"]))
                shared_core_rows.append(float(stats["shared_core_stability"]))
                if hasattr(model, "update_feedback"):
                    model.update_feedback(success=final_success, recovered=recovered, retention_signal=1.0 if final_success else 0.0, confidence=confidence)
                total += 1

    trigger_curve = (attempts_by_step / float(max(1, total))).tolist()
    recovery_curve = np.divide(recoveries_by_step, np.maximum(1.0, attempts_by_step)).tolist()
    confidence_before_curve = np.divide(confidence_before_sum, np.maximum(1.0, attempts_by_step)).tolist()
    confidence_after_curve = np.divide(confidence_after_sum, np.maximum(1.0, attempts_by_step)).tolist()
    brain_alignment = float(
        np.clip(
            0.38 * (1.0 - float(np.mean(control_delta_rows) if control_delta_rows else 0.0))
            + 0.34 * float(np.mean(progress_alignment_rows) if progress_alignment_rows else 0.0)
            + 0.28 * float(np.mean(shared_core_rows) if shared_core_rows else 0.0),
            0.0,
            1.0,
        )
    )

    return {
        "perturbed_episode_success": float(perturbed_success / max(1, total)),
        "recovered_episode_success": float(recovered_success / max(1, total)),
        "rollback_trigger_rate": float(rollback_attempts / max(1, total)),
        "rollback_recovery_rate": float(rollback_successes / max(1, rollback_attempts)),
        "recovery_mean_confidence": float(np.mean(mean_confidence)) if mean_confidence else 0.0,
        "brain_alignment_score": brain_alignment,
        "timeline": {
            "positions": [idx + 1 for idx in range(length)],
            "trigger_rate_curve": [float(v) for v in trigger_curve],
            "recovery_rate_curve": [float(v) for v in recovery_curve],
            "confidence_before_curve": [float(v) for v in confidence_before_curve],
            "confidence_after_curve": [float(v) for v in confidence_after_curve],
        },
    }


def evaluate_control_bridge(model, families: List[str], length: int, rng: np.random.Generator, noise: float, dropout_p: float, repeats: int) -> Dict[str, float]:
    control_deltas = []
    progress_alignments = []
    shared_core_stabilities = []
    gate_peaks = []

    for _ in range(repeats):
        for family in families:
            for concept in memory_scan.base.CONCEPTS[family]:
                states, targets = hierarchical_scan.build_episode(rng, concept, length, noise, dropout_p, state_mode="hierarchical")
                _preds, stats = predict_episode_stats(model, states, targets)
                control_deltas.append(float(stats["control_delta"]))
                progress_alignments.append(float(stats["progress_alignment"]))
                shared_core_stabilities.append(float(stats["shared_core_stability"]))
                gate_peaks.append(float(stats["mean_gate_peak"]))

    brain_alignment = float(
        np.clip(
            0.34 * (1.0 - float(np.mean(control_deltas) if control_deltas else 0.0))
            + 0.34 * float(np.mean(progress_alignments) if progress_alignments else 0.0)
            + 0.20 * float(np.mean(shared_core_stabilities) if shared_core_stabilities else 0.0)
            + 0.12 * float(np.mean(gate_peaks) if gate_peaks else 0.0),
            0.0,
            1.0,
        )
    )
    return {
        "brain_alignment_score": brain_alignment,
        "control_smoothness": float(np.clip(1.0 - float(np.mean(control_deltas) if control_deltas else 0.0), 0.0, 1.0)),
        "progress_alignment": float(np.mean(progress_alignments)) if progress_alignments else 0.0,
        "shared_core_stability": float(np.mean(shared_core_stabilities)) if shared_core_stabilities else 0.0,
    }


def run_for_length(system_name: str, length: int, seed: int, noise: float, dropout_p: float) -> Tuple[Dict[str, float], Dict[str, object]]:
    rng = np.random.default_rng(seed)
    model = build_model(system_name, seed)
    phase1_families = ["fruit", "animal"]
    phase2_families = ["abstract"]
    all_families = ["fruit", "animal", "abstract"]
    phase1_memory = unified_scan.hierarchical_episode_pool(phase1_families, length, repeats=8, rng=rng, noise=noise, dropout_p=dropout_p)
    phase2_memory = unified_scan.hierarchical_episode_pool(phase2_families, length, repeats=10, rng=rng, noise=noise, dropout_p=dropout_p)

    for states, targets in unified_scan.hierarchical_episode_pool(phase1_families, length, repeats=44, rng=rng, noise=noise, dropout_p=dropout_p):
        model.train_episode(states, targets)
    phase1_eval = hierarchical_scan.evaluate_system(model, phase1_families, length, rng, noise, dropout_p, repeats=18, state_mode="hierarchical")
    model.consolidate(phase1_memory)

    phase2_rows = unified_scan.hierarchical_episode_pool(phase2_families, length, repeats=62, rng=rng, noise=noise, dropout_p=dropout_p)
    for idx, (states, targets) in enumerate(phase2_rows):
        model.train_episode(states, targets)
        mem_states, mem_targets = phase1_memory[idx % len(phase1_memory)]
        model.train_episode(mem_states, mem_targets)
        if system_name in {"unified_control_manifold_h12", "minimal_control_bridge_h12"}:
            if idx % 3 == 0:
                extra_states, extra_targets = phase1_memory[(idx + 1) % len(phase1_memory)]
                model.train_episode(extra_states, extra_targets)
            if idx % 4 == 0:
                abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
                model.train_episode(abs_states, abs_targets)
        elif system_name != "single_anchor_beta_086":
            abs_states, abs_targets = phase2_memory[idx % len(phase2_memory)]
            model.train_episode(abs_states, abs_targets)

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
        enable_recovery=(system_name in {"unified_control_manifold_h12", "minimal_control_bridge_h12"}),
    )
    bridge_eval = evaluate_control_bridge(model, all_families, length, rng, noise, dropout_p, repeats=10)
    control_dim = float(getattr(model, "control_dim", 1))
    control_compaction = float(np.clip(1.0 - 0.20 * (control_dim - 1.0), 0.0, 1.0))
    real_task_score = float(
        0.18 * overall_eval["tool_accuracy"]
        + 0.18 * overall_eval["route_accuracy"]
        + 0.18 * overall_eval["final_accuracy"]
        + 0.16 * overall_eval["episode_success"]
        + 0.12 * retention_eval["episode_success"]
        + 0.10 * recovery_eval["perturbed_episode_success"]
        + 0.08 * recovery_eval["rollback_recovery_rate"]
    )
    control_bridge_score = float(0.28 * real_task_score + 0.22 * recovery_eval["rollback_recovery_rate"] + 0.18 * retention_eval["episode_success"] + 0.16 * bridge_eval["brain_alignment_score"] + 0.16 * control_compaction)

    return {
        "retention_after_phase2": retention_eval["episode_success"],
        "overall_tool_accuracy": overall_eval["tool_accuracy"],
        "overall_route_accuracy": overall_eval["route_accuracy"],
        "overall_final_accuracy": overall_eval["final_accuracy"],
        "overall_episode_success": overall_eval["episode_success"],
        "gate_entropy": overall_eval["gate_entropy"],
        "gate_peak": overall_eval["gate_peak"],
        "rollback_recovery_rate": recovery_eval["rollback_recovery_rate"],
        "rollback_trigger_rate": recovery_eval["rollback_trigger_rate"],
        "recovery_mean_confidence": recovery_eval["recovery_mean_confidence"],
        "real_task_score": real_task_score,
        "brain_alignment_score": bridge_eval["brain_alignment_score"],
        "control_smoothness": bridge_eval["control_smoothness"],
        "progress_alignment": bridge_eval["progress_alignment"],
        "shared_core_stability": bridge_eval["shared_core_stability"],
        "control_compaction": control_compaction,
        "control_bridge_score": control_bridge_score,
        "control_dim": control_dim,
    }, recovery_eval["timeline"]


def summarize_timeline(timelines: List[Dict[str, object]]) -> Dict[str, object]:
    if not timelines:
        return {"positions": [], "trigger_rate_curve": [], "recovery_rate_curve": [], "confidence_before_curve": [], "confidence_after_curve": []}
    keys = ["trigger_rate_curve", "recovery_rate_curve", "confidence_before_curve", "confidence_after_curve"]
    payload = {"positions": timelines[0]["positions"]}
    for key in keys:
        stacked = np.asarray([timeline[key] for timeline in timelines], dtype=np.float64)
        payload[key] = [float(v) for v in np.mean(stacked, axis=0)]
    return payload


def summarize_system(rows_by_length: Dict[int, List[Dict[str, float]]], timelines_by_length: Dict[int, List[Dict[str, object]]], lengths: List[int]) -> Dict[str, object]:
    per_length = {}
    bridge_curve = []
    recovery_curve = []
    retention_curve = []
    brain_curve = []
    compaction_curve = []
    for length in lengths:
        summary = memory_scan.summarize_runs(rows_by_length[length])
        bridge_curve.append(float(summary["control_bridge_score"]["mean"]))
        recovery_curve.append(float(summary["rollback_recovery_rate"]["mean"]))
        retention_curve.append(float(summary["retention_after_phase2"]["mean"]))
        brain_curve.append(float(summary["brain_alignment_score"]["mean"]))
        compaction_curve.append(float(summary["control_compaction"]["mean"]))
        per_length[str(length)] = {
            "summary": summary,
            "control_bridge_score": float(summary["control_bridge_score"]["mean"]),
            "timeline": summarize_timeline(timelines_by_length[length]),
        }
    return {
        "per_length": per_length,
        "global_summary": {
            "lengths": lengths,
            "bridge_curve": bridge_curve,
            "recovery_curve": recovery_curve,
            "retention_curve": retention_curve,
            "brain_curve": brain_curve,
            "compaction_curve": compaction_curve,
            "mean_bridge_score": float(np.mean(bridge_curve)),
            "mean_recovery_rate": float(np.mean(recovery_curve)),
            "mean_retention_score": float(np.mean(retention_curve)),
            "mean_brain_alignment": float(np.mean(brain_curve)),
            "mean_control_compaction": float(np.mean(compaction_curve)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Real multi-step minimal control bridge benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.34)
    ap.add_argument("--dropout-p", type=float, default=0.14)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 28, 32])
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/real_multistep_minimal_control_bridge_benchmark_20260310.json")
    args = ap.parse_args()

    lengths = [int(v) for v in args.lengths]
    max_length = max(lengths)
    t0 = time.time()
    systems = {}
    ranking = []

    for system_name in ["single_anchor_beta_086", "learnable_state_machine_h12", "unified_control_manifold_h12", "minimal_control_bridge_h12"]:
        rows_by_length = {length: [] for length in lengths}
        timelines_by_length = {length: [] for length in lengths}
        for length in lengths:
            for offset in range(int(args.num_seeds)):
                row, timeline = run_for_length(system_name, length, int(args.seed) + offset, float(args.noise), float(args.dropout_p))
                rows_by_length[length].append(row)
                timelines_by_length[length].append(timeline)
        systems[system_name] = summarize_system(rows_by_length, timelines_by_length, lengths)
        g = systems[system_name]["global_summary"]
        ranking.append(
            {
                "system": system_name,
                "mean_bridge_score": float(g["mean_bridge_score"]),
                "mean_recovery_rate": float(g["mean_recovery_rate"]),
                "mean_brain_alignment": float(g["mean_brain_alignment"]),
                "max_length_bridge_score": float(systems[system_name]["per_length"][str(max_length)]["control_bridge_score"]),
            }
        )

    ranking.sort(key=lambda row: row["mean_bridge_score"], reverse=True)
    single = systems["single_anchor_beta_086"]["per_length"][str(max_length)]
    unified = systems["unified_control_manifold_h12"]["per_length"][str(max_length)]
    minimal = systems["minimal_control_bridge_h12"]["per_length"][str(max_length)]
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
            "minimal_vs_single_at_max_length": float(minimal["control_bridge_score"] - single["control_bridge_score"]),
            "minimal_vs_unified_at_max_length": float(minimal["control_bridge_score"] - unified["control_bridge_score"]),
            "minimal_recovery_delta_vs_unified": float(minimal["summary"]["rollback_recovery_rate"]["mean"] - unified["summary"]["rollback_recovery_rate"]["mean"]),
            "minimal_brain_gain_vs_unified": float(minimal["summary"]["brain_alignment_score"]["mean"] - unified["summary"]["brain_alignment_score"]["mean"]),
            "minimal_compaction_gain_vs_unified": float(minimal["summary"]["control_compaction"]["mean"] - unified["summary"]["control_compaction"]["mean"]),
        },
        "headline_metrics": {
            "max_length_bridge_score": float(minimal["control_bridge_score"]),
            "max_length_recovery_rate": float(minimal["summary"]["rollback_recovery_rate"]["mean"]),
            "max_length_retention": float(minimal["summary"]["retention_after_phase2"]["mean"]),
            "max_length_brain_alignment": float(minimal["summary"]["brain_alignment_score"]["mean"]),
            "max_length_control_compaction": float(minimal["summary"]["control_compaction"]["mean"]),
        },
        "hypotheses": {
            "H1_minimal_beats_unified_on_bridge_score": bool(minimal["control_bridge_score"] > unified["control_bridge_score"]),
            "H2_minimal_improves_brain_alignment": bool(minimal["summary"]["brain_alignment_score"]["mean"] > unified["summary"]["brain_alignment_score"]["mean"]),
            "H3_minimal_keeps_recovery_near_unified": bool(minimal["summary"]["rollback_recovery_rate"]["mean"] >= unified["summary"]["rollback_recovery_rate"]["mean"] - 0.03),
        },
        "project_readout": {
            "summary": "这一版把真实多步统一控制流形继续压缩成更小的最小控制桥，让共享基底、关系协议与门控调制更多落到两个低维坐标里，同时保持失败回退、恢复探测和真实任务评价外部化。",
            "next_question": "如果最小控制桥在真实多步任务上还能保持恢复率和保留率，下一步就该把这套更小结构接回真实工具接口，并继续对齐脑侧候选约束。",
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
