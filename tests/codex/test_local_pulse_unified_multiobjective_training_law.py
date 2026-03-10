#!/usr/bin/env python
"""
Benchmark whether a unified local encoding law should be trained with a
single aggregate objective or an explicit multiobjective structure-aware
objective under region-differentiated dynamics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from test_local_pulse_early_core_decoupling_benchmark import (
    EarlyCoreDecoupledReplayNetwork,
    TraceableCoupledReplayNetwork,
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
    heterogeneous_params,
    homogeneous_params,
    sigmoid,
    train_network,
)


@dataclass(frozen=True)
class PhaseRoutingLaw:
    sensory_gain_a: float
    sensory_gain_b: float
    early_memory_gain: float
    mid_memory_gain: float
    early_comparator_forward_scale: float
    comparison_memory_scale: float
    comparison_signal_scale: float
    comparator_recurrence_scale: float
    comparator_feedback_scale: float
    motor_release_step: int
    prerelease_motor_forward_scale: float
    motor_teacher_scale: float
    replay_memory_scale: float
    replay_comparator_scale: float


LAW_CONFIGS = {
    "score_push_regional": PhaseRoutingLaw(
        sensory_gain_a=1.34,
        sensory_gain_b=1.23,
        early_memory_gain=0.10,
        mid_memory_gain=0.08,
        early_comparator_forward_scale=0.28,
        comparison_memory_scale=0.20,
        comparison_signal_scale=0.42,
        comparator_recurrence_scale=0.04,
        comparator_feedback_scale=1.00,
        motor_release_step=5,
        prerelease_motor_forward_scale=0.42,
        motor_teacher_scale=0.74,
        replay_memory_scale=0.30,
        replay_comparator_scale=0.14,
    ),
    "balanced_multiobjective": PhaseRoutingLaw(
        sensory_gain_a=1.30,
        sensory_gain_b=1.18,
        early_memory_gain=0.23,
        mid_memory_gain=0.12,
        early_comparator_forward_scale=0.10,
        comparison_memory_scale=0.24,
        comparison_signal_scale=0.54,
        comparator_recurrence_scale=0.14,
        comparator_feedback_scale=1.12,
        motor_release_step=7,
        prerelease_motor_forward_scale=0.10,
        motor_teacher_scale=0.68,
        replay_memory_scale=0.38,
        replay_comparator_scale=0.18,
    ),
    "structure_push_regional": PhaseRoutingLaw(
        sensory_gain_a=1.26,
        sensory_gain_b=1.14,
        early_memory_gain=0.28,
        mid_memory_gain=0.15,
        early_comparator_forward_scale=0.06,
        comparison_memory_scale=0.28,
        comparison_signal_scale=0.60,
        comparator_recurrence_scale=0.18,
        comparator_feedback_scale=1.18,
        motor_release_step=8,
        prerelease_motor_forward_scale=0.04,
        motor_teacher_scale=0.62,
        replay_memory_scale=0.42,
        replay_comparator_scale=0.22,
    ),
}


def normalized(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clamped = min(max(value, lo), hi)
    return (clamped - lo) / (hi - lo)


class UnifiedObjectiveShapedReplayNetwork(EarlyCoreDecoupledReplayNetwork):
    def __init__(self, region_params: Sequence[object], law: PhaseRoutingLaw, seed: int) -> None:
        super().__init__(region_params, use_replay=True, seed=seed)
        self.law = law

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
                ext[0] += self.law.sensory_gain_a * (PATTERNS[concept_a] @ self.input_proj)
                ext[1] += self.law.early_memory_gain * concept_a_memory_template
            if t == 2:
                ext[1] += self.law.mid_memory_gain * concept_a_memory_template
            if t in (3, 4):
                ext[0] += self.law.sensory_gain_b * (PATTERNS[concept_b] @ self.input_proj)
                ext[1] += 0.5 * self.law.mid_memory_gain * concept_b_memory_template
            if memory_snapshot is not None and t in (4, 5):
                distance = float(np.mean(np.abs(memory_snapshot - concept_b_memory_template)))
                match_signal = max(0.0, 1.0 - 1.35 * distance)
                mismatch_signal = min(1.0, 1.35 * distance)
                half = REGION_SIZES[2] // 2
                ext[1] += self.law.comparison_memory_scale * memory_snapshot
                ext[2][:half] += self.law.comparison_signal_scale * match_signal
                ext[2][half:] += self.law.comparison_signal_scale * mismatch_signal
            if t >= self.law.motor_release_step:
                half = REGION_SIZES[2] // 2
                ext[3][YES_SLICE] += 0.30 * float(np.mean(spikes_prev[2][:half]))
                ext[3][NO_SLICE] += 0.30 * float(np.mean(spikes_prev[2][half:]))
            if train and t in (6, 7):
                ext[-1] += self.law.motor_teacher_scale * (
                    self.motor_teacher if target_same else self.motor_teacher_no
                )
            if self.use_replay and t in (8, 9) and memory_snapshot is not None:
                ext[1] += self.law.replay_memory_scale * memory_snapshot
                ext[2][: REGION_SIZES[2] // 2] += self.law.replay_comparator_scale * self.law.replay_memory_scale

            spikes_curr = []
            for ridx, size in enumerate(REGION_SIZES):
                params = self.region_params[ridx]
                local_drive = spikes_prev[ridx] @ self.intra[ridx]
                if ridx > 0:
                    forward_gain = 1.0
                    if ridx == 2 and t <= 2:
                        forward_gain = self.law.early_comparator_forward_scale
                    if ridx == 3 and t < self.law.motor_release_step:
                        forward_gain = self.law.prerelease_motor_forward_scale
                    local_drive += forward_gain * (spikes_prev[ridx - 1] @ self.forward[ridx - 1])
                if ridx < len(REGION_SIZES) - 1:
                    feedback_gain = params.feedback_gain
                    if ridx == 1 and t <= 2:
                        feedback_gain *= 0.35
                    if ridx == 2 and 4 <= t <= 6:
                        feedback_gain *= self.law.comparator_feedback_scale
                    local_drive += feedback_gain * (spikes_prev[ridx + 1] @ self.backward[ridx])
                if ridx == 2 and 4 <= t <= 6:
                    local_drive += self.law.comparator_recurrence_scale * spikes_prev[ridx]
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
                memory_snapshot = (0.74 * state_trace[1][-1] + 0.26 * concept_a_memory_template).astype(np.float32)

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
        yes_score = 0.52 * motor_yes + 0.48 * comparator_yes
        no_score = 0.52 * motor_no + 0.48 * comparator_no
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


def objective_scores(system: Dict[str, object]) -> Dict[str, float]:
    aggregate = float(system["local_integration_score"])
    concept_adv = float(system["concept_phase_upstream_advantage"])
    comparison_adv = float(system["comparison_phase_memory_comparator_advantage"])
    recovery = float(system["lesion_recovery_rate"])
    diversity = float(system["phase_summary"]["distinct_top_region_count"])
    structure = float(
        0.26 * normalized(concept_adv, -0.12, 0.20)
        + 0.26 * normalized(comparison_adv, -0.16, 0.18)
        + 0.18 * normalized(recovery, 0.55, 0.70)
        + 0.18 * normalized(diversity, 1.0, 3.0)
        + 0.12 * normalized(aggregate, 0.58, 0.67)
    )
    multiobjective = float(
        0.80 * structure
        + 0.20 * normalized(aggregate, 0.58, 0.67)
    )
    return {
        "aggregate_objective": aggregate,
        "structure_objective": structure,
        "multiobjective_score": multiobjective,
    }


def build_systems() -> Dict[str, object]:
    systems = {
        "shared_local_replay": TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31),
        "regional_decoupled_replay": EarlyCoreDecoupledReplayNetwork(decoupled_params(), use_replay=True, seed=23),
        "score_push_regional": UnifiedObjectiveShapedReplayNetwork(heterogeneous_params(), LAW_CONFIGS["score_push_regional"], seed=41),
        "balanced_multiobjective": UnifiedObjectiveShapedReplayNetwork(heterogeneous_params(), LAW_CONFIGS["balanced_multiobjective"], seed=43),
        "structure_push_regional": UnifiedObjectiveShapedReplayNetwork(heterogeneous_params(), LAW_CONFIGS["structure_push_regional"], seed=47),
    }
    for network in systems.values():
        train_network(network, epochs=28, noise=0.03)
        calibrate_readout(network, noise=0.04)
    return systems


def evaluate_systems() -> Dict[str, Dict[str, object]]:
    rows = {}
    for name, network in build_systems().items():
        integration = integration_bundle(network)
        phase = phase_bundle(network)
        rows[name] = {
            **integration,
            **phase,
            **objective_scores({**integration, **phase}),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified multiobjective training law benchmark for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_unified_multiobjective_training_law_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = evaluate_systems()

    aggregate_best = max(systems.items(), key=lambda item: item[1]["aggregate_objective"])[0]
    multiobjective_best = max(systems.items(), key=lambda item: item[1]["multiobjective_score"])[0]
    structure_best = max(systems.items(), key=lambda item: item[1]["structure_objective"])[0]

    aggregate_best_row = systems[aggregate_best]
    multiobjective_best_row = systems[multiobjective_best]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "unified_multiobjective_training_law_without_global_controller",
        },
        "systems": systems,
        "headline_metrics": {
            "aggregate_best_system": aggregate_best,
            "multiobjective_best_system": multiobjective_best,
            "structure_best_system": structure_best,
            "aggregate_best_score": float(aggregate_best_row["aggregate_objective"]),
            "multiobjective_best_score": float(multiobjective_best_row["multiobjective_score"]),
            "multiobjective_structure_gain": float(
                multiobjective_best_row["structure_objective"] - aggregate_best_row["structure_objective"]
            ),
            "multiobjective_concept_gain": float(
                multiobjective_best_row["concept_phase_upstream_advantage"]
                - aggregate_best_row["concept_phase_upstream_advantage"]
            ),
            "multiobjective_comparison_gain": float(
                multiobjective_best_row["comparison_phase_memory_comparator_advantage"]
                - aggregate_best_row["comparison_phase_memory_comparator_advantage"]
            ),
            "multiobjective_aggregate_gap": float(
                multiobjective_best_row["aggregate_objective"] - aggregate_best_row["aggregate_objective"]
            ),
        },
        "hypotheses": {
            "H1_aggregate_objective_prefers_shared_or_score_push": bool(
                aggregate_best in {"shared_local_replay", "score_push_regional"}
            ),
            "H2_multiobjective_prefers_region_differentiated_law": bool(multiobjective_best != "shared_local_replay"),
            "H3_multiobjective_improves_structure_over_aggregate_choice": bool(
                multiobjective_best_row["structure_objective"]
                > aggregate_best_row["structure_objective"] + 0.08
            ),
            "H4_multiobjective_keeps_aggregate_cost_bounded": bool(
                multiobjective_best_row["aggregate_objective"] >= aggregate_best_row["aggregate_objective"] - 0.045
            ),
        },
        "project_readout": {
            "summary": "这一步不再只做后验挑选，而是把单目标训练律和多目标训练律都显式成型。目标是验证：如果统一编码机制真的要兼顾脑区差异和阶段局部核职责，那么训练律本身就必须是多目标的。",
            "next_question": "如果多目标训练律能稳定减少高分但错组织的假解，下一步就该把这套训练律搬到真实模型层带和真实任务闭环里。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
