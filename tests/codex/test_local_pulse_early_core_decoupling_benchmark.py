#!/usr/bin/env python
"""
Compare the current coupled local replay network against an early-core decoupled
variant that keeps more causal mass in sensory/memory during concept encoding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from test_local_pulse_phase_conditioned_causal_atlas import (
    PHASE_WINDOWS,
    PhaseAwareLocalPulseNetwork,
    evaluate_accuracy,
    evaluate_phase_drop_matrix,
    summarize_phase,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    PATTERNS,
    REGION_SIZES,
    REGIONS,
    YES_SLICE,
    NO_SLICE,
    calibrate_readout,
    evaluate_encoding_separation,
    evaluate_recovery,
    evaluate_regional_specialization,
    evaluate_same_family,
    evaluate_transfer_stability,
    family_of,
    heterogeneous_params,
    mean,
    sigmoid,
    train_network,
)


def decoupled_params():
    params = heterogeneous_params()
    params[0].tonic_bias += 0.03
    params[1].leak += 0.03
    params[1].tonic_bias += 0.02
    params[2].threshold += 0.10
    params[2].feedback_gain = max(0.04, params[2].feedback_gain - 0.08)
    params[2].tonic_bias = max(0.02, params[2].tonic_bias - 0.03)
    params[3].threshold += 0.04
    return params


class EarlyCoreDecoupledReplayNetwork(PhaseAwareLocalPulseNetwork):
    def run_episode(
        self,
        concept_a: str,
        concept_b: str,
        train: bool,
        noise: float,
        lesion_region: int | None = None,
        lesion_prob: float = 0.0,
        lesion_window: Tuple[int, int] | None = None,
    ) -> Dict[str, object]:
        membranes = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
        refr = [np.zeros(size, dtype=np.int32) for size in REGION_SIZES]
        spikes_prev = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
        spike_trace = [[] for _ in REGION_SIZES]
        state_trace = [[] for _ in REGION_SIZES]
        memory_snapshot = None
        target_same = int(family_of(concept_a) == family_of(concept_b))
        concept_a_memory_template = np.tanh((PATTERNS[concept_a] @ self.input_proj) @ self.forward[0]).astype(np.float32)
        concept_b_memory_template = np.tanh((PATTERNS[concept_b] @ self.input_proj) @ self.forward[0]).astype(np.float32)
        lesion_lo, lesion_hi = lesion_window if lesion_window is not None else (-1, -1)

        for t in range(10):
            ext = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
            if t in (0, 1):
                sensory_drive = 1.30 * (PATTERNS[concept_a] @ self.input_proj)
                ext[0] += sensory_drive
                ext[1] += 0.24 * concept_a_memory_template
            if t == 2:
                ext[1] += 0.12 * concept_a_memory_template
            if t in (3, 4):
                sensory_drive = 1.18 * (PATTERNS[concept_b] @ self.input_proj)
                ext[0] += sensory_drive
                ext[1] += 0.10 * concept_b_memory_template
            if t >= 5:
                half = REGION_SIZES[2] // 2
                ext[3][YES_SLICE] += 0.32 * float(np.mean(spikes_prev[2][:half]))
                ext[3][NO_SLICE] += 0.32 * float(np.mean(spikes_prev[2][half:]))
            if train and t in (6, 7):
                ext[-1] += 0.72 * (self.motor_teacher if target_same else self.motor_teacher_no)
            if memory_snapshot is not None and t in (4, 5):
                distance = float(np.mean(np.abs(memory_snapshot - concept_b_memory_template)))
                match_signal = max(0.0, 1.0 - 1.35 * distance)
                mismatch_signal = min(1.0, 1.35 * distance)
                ext[1] += 0.18 * memory_snapshot
                half = REGION_SIZES[2] // 2
                ext[2][:half] += 0.52 * match_signal
                ext[2][half:] += 0.52 * mismatch_signal
            if self.use_replay and t in (8, 9) and memory_snapshot is not None:
                replay_gain = 0.44 if train else 0.28
                ext[1] += replay_gain * memory_snapshot
                ext[2][: REGION_SIZES[2] // 2] += 0.18 * replay_gain

            spikes_curr = []
            for ridx, size in enumerate(REGION_SIZES):
                params = self.region_params[ridx]
                local_drive = spikes_prev[ridx] @ self.intra[ridx]
                if ridx > 0:
                    forward_gain = 1.0
                    if ridx == 2 and t <= 2:
                        forward_gain = 0.12
                    local_drive += forward_gain * (spikes_prev[ridx - 1] @ self.forward[ridx - 1])
                if ridx < len(REGION_SIZES) - 1:
                    feedback_gain = params.feedback_gain
                    if ridx == 1 and t <= 2:
                        feedback_gain *= 0.35
                    local_drive += feedback_gain * (spikes_prev[ridx + 1] @ self.backward[ridx])
                local_drive = np.tanh(local_drive).astype(np.float32)
                inhib = params.inhibition * float(np.mean(spikes_prev[ridx]))
                noisy = self.rng.normal(scale=noise, size=size).astype(np.float32)
                tonic = params.tonic_bias + self.tonic_jitter[ridx]
                threshold = params.threshold + self.threshold_jitter[ridx]
                membranes[ridx] = (
                    params.leak * membranes[ridx]
                    + local_drive.astype(np.float32)
                    + ext[ridx]
                    + tonic
                    + noisy
                    - inhib
                ).astype(np.float32)
                can_fire = refr[ridx] <= 0
                spike = ((membranes[ridx] > threshold) & can_fire).astype(np.float32)
                if lesion_region is not None and ridx == lesion_region and lesion_lo <= t <= lesion_hi:
                    lesion_mask = (self.rng.random(size) > lesion_prob).astype(np.float32)
                    spike *= lesion_mask
                membranes[ridx] *= (1.0 - spike)
                refr[ridx] = np.maximum(0, refr[ridx] - 1)
                refr[ridx][spike > 0] = params.refractory
                spikes_curr.append(spike)
                spike_trace[ridx].append(spike.copy())
                state_trace[ridx].append(sigmoid(membranes[ridx].copy()))

            if t == 2:
                memory_snapshot = (0.72 * state_trace[1][-1] + 0.28 * concept_a_memory_template).astype(np.float32)

            if train:
                self._local_update(spikes_prev, spikes_curr)

            spikes_prev = spikes_curr

        comparator_state = np.stack(state_trace[2], axis=0)
        motor_trace = np.stack(spike_trace[-1], axis=0)
        half = REGION_SIZES[2] // 2
        comparator_yes = float(np.mean(comparator_state[4:, :half]))
        comparator_no = float(np.mean(comparator_state[4:, half:]))
        motor_yes = float(np.mean(motor_trace[6:, YES_SLICE]))
        motor_no = float(np.mean(motor_trace[6:, NO_SLICE]))
        yes_score = 0.54 * motor_yes + 0.46 * comparator_yes
        no_score = 0.54 * motor_no + 0.46 * comparator_no
        margin = yes_score - no_score
        if self.same_if_margin_below:
            pred_same = int(margin <= self.decision_threshold)
        else:
            pred_same = int(margin >= self.decision_threshold)
        return {
            "correct": int(pred_same == target_same),
            "target_same": target_same,
            "decision_margin": float(margin),
            "pred_same": pred_same,
            "yes_score": float(yes_score),
            "no_score": float(no_score),
            "memory_trace": np.stack(spike_trace[1], axis=0),
            "memory_state_trace": np.stack(state_trace[1], axis=0),
            "comparator_trace": np.stack(spike_trace[2], axis=0),
            "comparator_state_trace": comparator_state,
            "motor_trace": motor_trace,
        }


class TraceableCoupledReplayNetwork(PhaseAwareLocalPulseNetwork):
    def run_episode(
        self,
        concept_a: str,
        concept_b: str,
        train: bool,
        noise: float,
        lesion_region: int | None = None,
        lesion_prob: float = 0.0,
        lesion_window: Tuple[int, int] | None = None,
    ) -> Dict[str, object]:
        membranes = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
        refr = [np.zeros(size, dtype=np.int32) for size in REGION_SIZES]
        spikes_prev = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
        spike_trace = [[] for _ in REGION_SIZES]
        state_trace = [[] for _ in REGION_SIZES]
        memory_snapshot = None
        target_same = int(family_of(concept_a) == family_of(concept_b))
        concept_a_memory_template = np.tanh((PATTERNS[concept_a] @ self.input_proj) @ self.forward[0]).astype(np.float32)
        lesion_lo, lesion_hi = lesion_window if lesion_window is not None else (-1, -1)

        for t in range(10):
            ext = [np.zeros(size, dtype=np.float32) for size in REGION_SIZES]
            if t in (0, 1):
                ext[0] += 1.35 * (PATTERNS[concept_a] @ self.input_proj)
            if t in (3, 4):
                ext[0] += 1.25 * (PATTERNS[concept_b] @ self.input_proj)
            if t >= 5:
                half = REGION_SIZES[2] // 2
                ext[3][YES_SLICE] += 0.34 * float(np.mean(spikes_prev[2][:half]))
                ext[3][NO_SLICE] += 0.34 * float(np.mean(spikes_prev[2][half:]))
            if train and t in (6, 7):
                ext[-1] += 0.75 * (self.motor_teacher if target_same else self.motor_teacher_no)
            if memory_snapshot is not None and t in (4, 5):
                current_memory_like = np.tanh((PATTERNS[concept_b] @ self.input_proj) @ self.forward[0]).astype(np.float32)
                distance = float(np.mean(np.abs(memory_snapshot - current_memory_like)))
                match_signal = max(0.0, 1.0 - 1.3 * distance)
                mismatch_signal = min(1.0, 1.3 * distance)
                ext[1] += 0.22 * memory_snapshot
                half = REGION_SIZES[2] // 2
                ext[2][:half] += 0.44 * match_signal
                ext[2][half:] += 0.44 * mismatch_signal
            if self.use_replay and t in (8, 9) and memory_snapshot is not None:
                replay_gain = 0.42 if train else 0.26
                ext[1] += replay_gain * memory_snapshot
                ext[2][: REGION_SIZES[2] // 2] += 0.16 * replay_gain

            spikes_curr = []
            for ridx, size in enumerate(REGION_SIZES):
                params = self.region_params[ridx]
                local_drive = spikes_prev[ridx] @ self.intra[ridx]
                if ridx > 0:
                    local_drive += spikes_prev[ridx - 1] @ self.forward[ridx - 1]
                if ridx < len(REGION_SIZES) - 1:
                    local_drive += params.feedback_gain * (spikes_prev[ridx + 1] @ self.backward[ridx])
                local_drive = np.tanh(local_drive).astype(np.float32)
                inhib = params.inhibition * float(np.mean(spikes_prev[ridx]))
                noisy = self.rng.normal(scale=noise, size=size).astype(np.float32)
                tonic = params.tonic_bias + self.tonic_jitter[ridx]
                threshold = params.threshold + self.threshold_jitter[ridx]
                membranes[ridx] = (
                    params.leak * membranes[ridx]
                    + local_drive.astype(np.float32)
                    + ext[ridx]
                    + tonic
                    + noisy
                    - inhib
                ).astype(np.float32)
                can_fire = refr[ridx] <= 0
                spike = ((membranes[ridx] > threshold) & can_fire).astype(np.float32)
                if lesion_region is not None and ridx == lesion_region and lesion_lo <= t <= lesion_hi:
                    lesion_mask = (self.rng.random(size) > lesion_prob).astype(np.float32)
                    spike *= lesion_mask
                membranes[ridx] *= (1.0 - spike)
                refr[ridx] = np.maximum(0, refr[ridx] - 1)
                refr[ridx][spike > 0] = params.refractory
                spikes_curr.append(spike)
                spike_trace[ridx].append(spike.copy())
                state_trace[ridx].append(sigmoid(membranes[ridx].copy()))

            if t == 2:
                memory_snapshot = (0.55 * state_trace[1][-1] + 0.45 * concept_a_memory_template).astype(np.float32)

            if train:
                self._local_update(spikes_prev, spikes_curr)

            spikes_prev = spikes_curr

        comparator_trace = np.stack(spike_trace[2], axis=0)
        comparator_state = np.stack(state_trace[2], axis=0)
        motor_trace = np.stack(spike_trace[-1], axis=0)
        half = REGION_SIZES[2] // 2
        comparator_yes = float(np.mean(comparator_state[4:, :half]))
        comparator_no = float(np.mean(comparator_state[4:, half:]))
        motor_yes = float(np.mean(motor_trace[6:, YES_SLICE]))
        motor_no = float(np.mean(motor_trace[6:, NO_SLICE]))
        yes_score = 0.58 * motor_yes + 0.42 * comparator_yes
        no_score = 0.58 * motor_no + 0.42 * comparator_no
        margin = yes_score - no_score
        if self.same_if_margin_below:
            pred_same = int(margin <= self.decision_threshold)
        else:
            pred_same = int(margin >= self.decision_threshold)
        return {
            "correct": int(pred_same == target_same),
            "target_same": target_same,
            "decision_margin": float(margin),
            "pred_same": pred_same,
            "yes_score": float(yes_score),
            "no_score": float(no_score),
            "memory_trace": np.stack(spike_trace[1], axis=0),
            "memory_state_trace": np.stack(state_trace[1], axis=0),
            "comparator_trace": comparator_trace,
            "comparator_state_trace": comparator_state,
            "motor_trace": motor_trace,
        }


def build_network(kind: str):
    if kind == "baseline_coupled_replay":
        network = TraceableCoupledReplayNetwork(heterogeneous_params(), use_replay=True, seed=17)
    elif kind == "decoupled_upstream_replay":
        network = EarlyCoreDecoupledReplayNetwork(decoupled_params(), use_replay=True, seed=23)
    else:
        raise KeyError(kind)
    train_network(network, epochs=28, noise=0.03)
    calibrate_readout(network, noise=0.04)
    return network


def integration_bundle(network) -> Dict[str, float]:
    same_family_success = evaluate_same_family(network, noise=0.04, repeats=4)
    recovery_rate = evaluate_recovery(network, noise=0.04, repeats=3)
    encoding_separation = evaluate_encoding_separation(network, noise=0.04, repeats=4)
    transfer_stability = evaluate_transfer_stability(network, noise=0.04, repeats=5)
    regional_specialization = evaluate_regional_specialization(network, noise=0.04, repeats=4)
    local_integration_score = float(
        0.28 * same_family_success
        + 0.24 * recovery_rate
        + 0.18 * encoding_separation
        + 0.16 * transfer_stability
        + 0.14 * regional_specialization
    )
    return {
        "same_family_success_rate": float(same_family_success),
        "lesion_recovery_rate": float(recovery_rate),
        "encoding_separation": float(encoding_separation),
        "transfer_stability": float(transfer_stability),
        "regional_specialization": float(regional_specialization),
        "local_integration_score": float(local_integration_score),
    }


def phase_bundle(network) -> Dict[str, object]:
    baseline_accuracy = evaluate_accuracy(network, noise=0.04, repeats=5)
    matrix = evaluate_phase_drop_matrix(network, noise=0.04, repeats=4, lesion_prob=0.55)
    summary = summarize_phase(matrix)
    concept_upstream_adv = summary["concept_phase"]["upstream_mass"] - summary["concept_phase"]["downstream_mass"]
    comparison_mc_adv = summary["comparison_phase"]["memory_comparator_mass"] - summary["comparison_phase"]["sensory_motor_mass"]
    return {
        "baseline_accuracy": float(baseline_accuracy),
        "phase_drop_matrix": matrix,
        "phase_summary": summary,
        "concept_phase_upstream_advantage": float(concept_upstream_adv),
        "comparison_phase_memory_comparator_advantage": float(comparison_mc_adv),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark early-core decoupling for local pulse replay networks")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/local_pulse_early_core_decoupling_benchmark_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline = build_network("baseline_coupled_replay")
    decoupled = build_network("decoupled_upstream_replay")

    baseline_integration = integration_bundle(baseline)
    decoupled_integration = integration_bundle(decoupled)
    baseline_phase = phase_bundle(baseline)
    decoupled_phase = phase_bundle(decoupled)

    upstream_gain = decoupled_phase["concept_phase_upstream_advantage"] - baseline_phase["concept_phase_upstream_advantage"]
    integration_gain = decoupled_integration["local_integration_score"] - baseline_integration["local_integration_score"]
    recovery_gain = decoupled_integration["lesion_recovery_rate"] - baseline_integration["lesion_recovery_rate"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "early_upstream_core_decoupling_without_global_controller",
        },
        "systems": {
            "baseline_coupled_replay": {
                **baseline_integration,
                **baseline_phase,
            },
            "decoupled_upstream_replay": {
                **decoupled_integration,
                **decoupled_phase,
            },
        },
        "headline_metrics": {
            "baseline_upstream_advantage": float(baseline_phase["concept_phase_upstream_advantage"]),
            "decoupled_upstream_advantage": float(decoupled_phase["concept_phase_upstream_advantage"]),
            "upstream_advantage_gain": float(upstream_gain),
            "decoupled_local_integration_score": float(decoupled_integration["local_integration_score"]),
            "integration_gain": float(integration_gain),
            "recovery_gain": float(recovery_gain),
            "distinct_top_region_count": int(decoupled_phase["phase_summary"]["distinct_top_region_count"]),
        },
        "hypotheses": {
            "H1_decoupling_increases_upstream_advantage": bool(upstream_gain > 0.050),
            "H2_decoupling_preserves_local_integration": bool(decoupled_integration["local_integration_score"] >= baseline_integration["local_integration_score"] - 0.020),
            "H3_decoupling_keeps_comparison_core_local": bool(decoupled_phase["comparison_phase_memory_comparator_advantage"] > 0.020),
            "H4_decoupling_keeps_multi_stage_local_core": bool(decoupled_phase["phase_summary"]["distinct_top_region_count"] >= 2),
        },
        "project_readout": {
            "summary": "这一版直接测试：如果强行把早期流量留在 sensory / memory，上游局部核心能不能重新站稳，同时又不把整体整合和恢复能力打塌。",
            "next_question": "如果早期解耦能成功，下一步就该把这种上游感知核与中游比较核的分层结构搬回真实模型层带，验证真实 DNN 是否也需要类似的局部阶段解耦。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
