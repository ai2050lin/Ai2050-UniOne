#!/usr/bin/env python
"""
Stabilize the comparison-phase local core after early upstream decoupling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from test_local_pulse_early_core_decoupling_benchmark import (
    EarlyCoreDecoupledReplayNetwork,
    decoupled_params,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    PATTERNS,
    REGION_SIZES,
    YES_SLICE,
    NO_SLICE,
    calibrate_readout,
    family_of,
    sigmoid,
    train_network,
)


def stabilized_params():
    params = decoupled_params()
    params[1].leak += 0.02
    params[1].feedback_gain += 0.04
    params[2].threshold = max(0.36, params[2].threshold - 0.06)
    params[2].tonic_bias += 0.03
    params[2].feedback_gain += 0.04
    params[3].threshold += 0.08
    params[3].tonic_bias = max(0.02, params[3].tonic_bias - 0.02)
    params[3].feedback_gain = max(0.04, params[3].feedback_gain - 0.02)
    return params


class MidphaseStabilizedReplayNetwork(EarlyCoreDecoupledReplayNetwork):
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
                sensory_drive = 1.28 * (PATTERNS[concept_a] @ self.input_proj)
                ext[0] += sensory_drive
                ext[1] += 0.26 * concept_a_memory_template
            if t == 2:
                ext[1] += 0.14 * concept_a_memory_template
            if t in (3, 4):
                sensory_drive = 1.15 * (PATTERNS[concept_b] @ self.input_proj)
                ext[0] += sensory_drive
                ext[1] += 0.12 * concept_b_memory_template
            if t in (4, 5) and memory_snapshot is not None:
                distance = float(np.mean(np.abs(memory_snapshot - concept_b_memory_template)))
                match_signal = max(0.0, 1.0 - 1.35 * distance)
                mismatch_signal = min(1.0, 1.35 * distance)
                half = REGION_SIZES[2] // 2
                ext[1] += 0.22 * memory_snapshot
                ext[2][:half] += 0.58 * match_signal
                ext[2][half:] += 0.58 * mismatch_signal
                ext[2] += 0.06 * memory_snapshot.mean()
            if t >= 6:
                half = REGION_SIZES[2] // 2
                ext[3][YES_SLICE] += 0.29 * float(np.mean(spikes_prev[2][:half]))
                ext[3][NO_SLICE] += 0.29 * float(np.mean(spikes_prev[2][half:]))
            if train and t in (6, 7):
                ext[-1] += 0.70 * (self.motor_teacher if target_same else self.motor_teacher_no)
            if self.use_replay and t in (8, 9) and memory_snapshot is not None:
                replay_gain = 0.46 if train else 0.30
                ext[1] += replay_gain * memory_snapshot
                ext[2][: REGION_SIZES[2] // 2] += 0.20 * replay_gain

            spikes_curr = []
            for ridx, size in enumerate(REGION_SIZES):
                params = self.region_params[ridx]
                local_drive = spikes_prev[ridx] @ self.intra[ridx]
                if ridx > 0:
                    forward_gain = 1.0
                    if ridx == 2 and t <= 2:
                        forward_gain = 0.10
                    if ridx == 3 and t <= 6:
                        forward_gain = 0.18
                    local_drive += forward_gain * (spikes_prev[ridx - 1] @ self.forward[ridx - 1])
                if ridx < len(REGION_SIZES) - 1:
                    feedback_gain = params.feedback_gain
                    if ridx == 1 and t <= 2:
                        feedback_gain *= 0.30
                    if ridx == 2 and 4 <= t <= 6:
                        feedback_gain *= 1.15
                    local_drive += feedback_gain * (spikes_prev[ridx + 1] @ self.backward[ridx])
                if ridx == 2 and 4 <= t <= 6:
                    local_drive += 0.18 * spikes_prev[ridx]
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
                memory_snapshot = (0.76 * state_trace[1][-1] + 0.24 * concept_a_memory_template).astype(np.float32)

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
        yes_score = 0.48 * motor_yes + 0.52 * comparator_yes
        no_score = 0.48 * motor_no + 0.52 * comparator_no
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


def build_network(kind: str):
    if kind == "decoupled_upstream_replay":
        network = EarlyCoreDecoupledReplayNetwork(decoupled_params(), use_replay=True, seed=23)
    elif kind == "stabilized_midphase_replay":
        network = MidphaseStabilizedReplayNetwork(stabilized_params(), use_replay=True, seed=29)
    else:
        raise KeyError(kind)
    train_network(network, epochs=28, noise=0.03)
    calibrate_readout(network, noise=0.04)
    return network


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark midphase local core stabilization")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/local_pulse_midphase_core_stabilization_benchmark_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    decoupled = build_network("decoupled_upstream_replay")
    stabilized = build_network("stabilized_midphase_replay")

    decoupled_integration = integration_bundle(decoupled)
    stabilized_integration = integration_bundle(stabilized)
    decoupled_phase = phase_bundle(decoupled)
    stabilized_phase = phase_bundle(stabilized)

    comparison_gain = (
        stabilized_phase["comparison_phase_memory_comparator_advantage"]
        - decoupled_phase["comparison_phase_memory_comparator_advantage"]
    )
    upstream_retention = (
        stabilized_phase["concept_phase_upstream_advantage"]
        - decoupled_phase["concept_phase_upstream_advantage"]
    )
    motor_overreach_reduction = (
        decoupled_phase["phase_summary"]["comparison_phase"]["rows"]["motor"]
        - stabilized_phase["phase_summary"]["comparison_phase"]["rows"]["motor"]
    )
    integration_gain = stabilized_integration["local_integration_score"] - decoupled_integration["local_integration_score"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "midphase_local_core_stabilization_without_global_controller",
        },
        "systems": {
            "decoupled_upstream_replay": {
                **decoupled_integration,
                **decoupled_phase,
            },
            "stabilized_midphase_replay": {
                **stabilized_integration,
                **stabilized_phase,
            },
        },
        "headline_metrics": {
            "decoupled_comparison_advantage": float(decoupled_phase["comparison_phase_memory_comparator_advantage"]),
            "stabilized_comparison_advantage": float(stabilized_phase["comparison_phase_memory_comparator_advantage"]),
            "comparison_advantage_gain": float(comparison_gain),
            "upstream_retention_delta": float(upstream_retention),
            "motor_overreach_reduction": float(motor_overreach_reduction),
            "stabilized_local_integration_score": float(stabilized_integration["local_integration_score"]),
            "integration_gain": float(integration_gain),
            "distinct_top_region_count": int(stabilized_phase["phase_summary"]["distinct_top_region_count"]),
        },
        "hypotheses": {
            "H1_stabilization_improves_comparison_core": bool(comparison_gain > 0.120),
            "H2_stabilization_keeps_upstream_advantage": bool(stabilized_phase["concept_phase_upstream_advantage"] >= decoupled_phase["concept_phase_upstream_advantage"] - 0.040),
            "H3_stabilization_reduces_motor_overreach": bool(motor_overreach_reduction > 0.080),
            "H4_stabilization_preserves_multi_stage_local_core": bool(stabilized_phase["phase_summary"]["distinct_top_region_count"] >= 2),
        },
        "project_readout": {
            "summary": "这一步只做一件事：在保住早期上游感知核的前提下，把 comparison_phase 的局部核心重新稳在 memory / comparator，不让它过早滑向 motor。",
            "next_question": "如果中期比较核能稳住，下一步就该把这套“早期感知核 - 中期比较核 - 晚期恢复核”的阶段接力结构搬回真实模型层带和真实任务接口。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
