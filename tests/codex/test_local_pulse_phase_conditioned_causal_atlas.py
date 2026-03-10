#!/usr/bin/env python
"""
Build a phase-conditioned causal atlas for the local pulse network.

The key question is not whether a shared structure exists, but whether the
currently causal local core changes with phase under a strictly local update
law and no explicit global controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from test_local_pulse_region_heterogeneity_benchmark import (
    CONCEPTS,
    FAMILIES,
    FAMILY_INDEX,
    PATTERNS,
    REGION_SIZES,
    REGIONS,
    YES_SLICE,
    NO_SLICE,
    LocalPulseNetwork,
    calibrate_readout,
    family_of,
    heterogeneous_params,
    mean,
    sigmoid,
    train_network,
)


PHASE_WINDOWS = {
    "concept_phase": (0, 2),
    "comparison_phase": (4, 6),
    "recovery_phase": (8, 9),
}


class PhaseAwareLocalPulseNetwork(LocalPulseNetwork):
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
        }


def all_pairs() -> List[Tuple[str, str]]:
    words = [word for family_words in CONCEPTS.values() for word in family_words]
    return [(a, b) for a in words for b in words]


def build_network(use_replay: bool, seed: int) -> PhaseAwareLocalPulseNetwork:
    network = PhaseAwareLocalPulseNetwork(heterogeneous_params(), use_replay=use_replay, seed=seed)
    train_network(network, epochs=28, noise=0.03)
    calibrate_readout(network, noise=0.04)
    return network


def evaluate_accuracy(
    network: PhaseAwareLocalPulseNetwork,
    noise: float,
    repeats: int,
    lesion_region: int | None = None,
    lesion_window: Tuple[int, int] | None = None,
    lesion_prob: float = 0.0,
) -> float:
    rows = []
    for _ in range(repeats):
        for concept_a, concept_b in all_pairs():
            out = network.run_episode(
                concept_a,
                concept_b,
                train=False,
                noise=noise,
                lesion_region=lesion_region,
                lesion_prob=lesion_prob,
                lesion_window=lesion_window,
            )
            rows.append(float(out["correct"]))
    return mean(rows)


def evaluate_phase_drop_matrix(
    network: PhaseAwareLocalPulseNetwork,
    noise: float,
    repeats: int,
    lesion_prob: float,
) -> Dict[str, Dict[str, float]]:
    baseline = evaluate_accuracy(network, noise=noise, repeats=repeats)
    matrix: Dict[str, Dict[str, float]] = {}
    for phase_name, window in PHASE_WINDOWS.items():
        rows = {}
        for ridx, region in enumerate(REGIONS):
            lesioned = evaluate_accuracy(
                network,
                noise=noise,
                repeats=repeats,
                lesion_region=ridx,
                lesion_window=window,
                lesion_prob=lesion_prob,
            )
            rows[region] = float(max(0.0, baseline - lesioned))
        matrix[phase_name] = rows
    return matrix


def summarize_phase(matrix: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    phase_summary: Dict[str, object] = {}
    top_regions = []
    for phase_name, rows in matrix.items():
        top_region, top_drop = max(rows.items(), key=lambda item: item[1])
        top_regions.append(top_region)
        phase_summary[phase_name] = {
            "top_region": top_region,
            "top_drop": float(top_drop),
            "upstream_mass": float(rows["sensory"] + rows["memory"]),
            "downstream_mass": float(rows["comparator"] + rows["motor"]),
            "memory_comparator_mass": float(rows["memory"] + rows["comparator"]),
            "sensory_motor_mass": float(rows["sensory"] + rows["motor"]),
            "rows": rows,
        }
    phase_summary["distinct_top_region_count"] = len(set(top_regions))
    phase_summary["top_region_sequence"] = top_regions
    return phase_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-conditioned causal atlas for local pulse networks")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/local_pulse_phase_conditioned_causal_atlas_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    replay_network = build_network(use_replay=True, seed=17)
    no_replay_network = build_network(use_replay=False, seed=13)

    replay_baseline = evaluate_accuracy(replay_network, noise=0.04, repeats=5)
    no_replay_baseline = evaluate_accuracy(no_replay_network, noise=0.04, repeats=5)
    replay_matrix = evaluate_phase_drop_matrix(replay_network, noise=0.04, repeats=4, lesion_prob=0.55)
    no_replay_matrix = evaluate_phase_drop_matrix(no_replay_network, noise=0.04, repeats=4, lesion_prob=0.55)

    replay_summary = summarize_phase(replay_matrix)
    no_replay_summary = summarize_phase(no_replay_matrix)

    concept_upstream_adv = replay_summary["concept_phase"]["upstream_mass"] - replay_summary["concept_phase"]["downstream_mass"]
    comparison_mc_adv = replay_summary["comparison_phase"]["memory_comparator_mass"] - replay_summary["comparison_phase"]["sensory_motor_mass"]
    recovery_mc_adv = (
        replay_summary["recovery_phase"]["memory_comparator_mass"]
        - no_replay_summary["recovery_phase"]["memory_comparator_mass"]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "phase_conditioned_local_causal_core_without_global_controller",
        },
        "systems": {
            "heterogeneous_local_replay": {
                "baseline_accuracy": float(replay_baseline),
                "decision_threshold": float(replay_network.decision_threshold),
                "same_if_margin_below": bool(replay_network.same_if_margin_below),
                "phase_drop_matrix": replay_matrix,
                "phase_summary": replay_summary,
            },
            "heterogeneous_local_stdp": {
                "baseline_accuracy": float(no_replay_baseline),
                "decision_threshold": float(no_replay_network.decision_threshold),
                "same_if_margin_below": bool(no_replay_network.same_if_margin_below),
                "phase_drop_matrix": no_replay_matrix,
                "phase_summary": no_replay_summary,
            },
        },
        "headline_metrics": {
            "baseline_accuracy": float(replay_baseline),
            "concept_phase_upstream_advantage": float(concept_upstream_adv),
            "comparison_phase_memory_comparator_advantage": float(comparison_mc_adv),
            "recovery_phase_replay_memory_comparator_advantage": float(recovery_mc_adv),
            "distinct_top_region_count": int(replay_summary["distinct_top_region_count"]),
        },
        "hypotheses": {
            "H1_concept_phase_prefers_upstream_regions": bool(concept_upstream_adv > 0.015),
            "H2_comparison_phase_prefers_memory_comparator": bool(comparison_mc_adv > 0.020),
            "H3_recovery_phase_is_replay_dependent": bool(recovery_mc_adv > 0.020),
            "H4_no_single_global_core": bool(replay_summary["distinct_top_region_count"] >= 2),
        },
        "project_readout": {
            "summary": "这一步不再问系统有没有共享结构，而是直接问：在严格局部更新的条件下，当前因果核心会不会随阶段变化。结果如果成立，说明系统级整合不是由一个全局指挥器固定完成，而是由不同阶段的局部核心依次接管。",
            "next_question": "如果 concept / comparison / recovery 三个阶段真的对应不同局部核心，下一步就该把这些局部核心接回真实任务接口和真实模型层带，看条件门控是否也遵循同样的去全局化规律。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
