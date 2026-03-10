#!/usr/bin/env python
"""
Variable-length planning chain benchmark with rollback and trainable control.

This experiment pushes the open-world planning prototypes in three directions:
1. fixed subgoal programs -> variable-length planning chains with failure rollback
2. static replay / goal-state heuristics -> trainable closed-loop controller
3. internal toy scores -> externalized evaluation focused on recovery and stability
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench
import test_open_world_grounding_action_loop_goal_state_scan as goal_scan
import test_open_world_grounding_action_loop_stateful_scan as stateful_scan
import test_open_world_long_horizon_goal_state_benchmark as long_goal


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StageSpec:
    family: str
    span: int
    anchor_idx: int


@dataclass
class EpisodeSpec:
    stages: List[StageSpec]


def fallback_best_configs() -> Dict[str, float | int | str]:
    return {
        "action_beta": 0.10,
        "correction_mix": 0.05,
        "trust_temp": 0.08,
        "reserve_target": 0.97,
        "replay_count": 1,
        "replay_mode": "weakest",
    }


def load_best_configs() -> Dict[str, float | int | str]:
    cfg = fallback_best_configs()
    stateful_path = ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_stateful_scan_20260310.json"
    long_goal_path = ROOT / "tests" / "codex_temp" / "open_world_long_horizon_goal_state_benchmark_20260310.json"
    subgoal_path = ROOT / "tests" / "codex_temp" / "open_world_subgoal_planning_benchmark_20260310.json"

    if stateful_path.exists():
        payload = json.loads(stateful_path.read_text(encoding="utf-8"))
        best = payload.get("best_config", {})
        cfg["action_beta"] = float(best.get("action_beta", cfg["action_beta"]))
        cfg["correction_mix"] = float(best.get("correction_mix", cfg["correction_mix"]))
        cfg["trust_temp"] = float(best.get("trust_temp", cfg["trust_temp"]))

    if long_goal_path.exists():
        payload = json.loads(long_goal_path.read_text(encoding="utf-8"))
        best = payload.get("best_config", {})
        cfg["reserve_target"] = float(best.get("reserve_target", cfg["reserve_target"]))
        cfg["replay_count"] = int(best.get("replay_count", cfg["replay_count"]))
        cfg["replay_mode"] = str(best.get("replay_mode", cfg["replay_mode"]))

    if subgoal_path.exists():
        payload = json.loads(subgoal_path.read_text(encoding="utf-8"))
        best = payload.get("best_config", {})
        cfg["reserve_target"] = float(best.get("reserve_target", cfg["reserve_target"]))
        cfg["replay_count"] = int(best.get("replay_count", max(cfg["replay_count"], 2)))
        cfg["replay_mode"] = str(best.get("replay_mode", cfg["replay_mode"]))

    return cfg


def build_variable_length_episodes(
    seed: int,
    concept_budget: int,
    min_program_len: int,
    max_program_len: int,
    min_stage_span: int,
    max_stage_span: int,
) -> List[EpisodeSpec]:
    rng = np.random.default_rng(seed)
    families = list(stream_bench.FAMILIES)
    episodes: List[EpisodeSpec] = []
    used = 0

    while used < concept_budget:
        program_len = int(rng.integers(min_program_len, max_program_len + 1))
        stages: List[StageSpec] = []
        prev_family = None
        for idx in range(program_len):
            choices = [family for family in families if family != prev_family] or families
            family_choice = rng.choice(np.array(choices, dtype=object))
            family = str(family_choice.item() if hasattr(family_choice, "item") else family_choice)
            span = int(rng.integers(min_stage_span, max_stage_span + 1))
            anchor_idx = max(0, idx - 1)
            stages.append(StageSpec(family=family, span=span, anchor_idx=anchor_idx))
            prev_family = family
            used += span
            if used >= concept_budget:
                break
        episodes.append(EpisodeSpec(stages=stages))
    return episodes


def snapshot_agent_state(agent) -> Dict[str, object]:
    state = {
        "family_basis": {k: v.copy() for k, v in agent.family_basis.items()},
        "family_count": dict(agent.family_count),
        "concept_offset": {k: v.copy() for k, v in agent.concept_offset.items()},
        "concept_count": dict(agent.concept_count),
        "concept_family": dict(agent.concept_family),
    }
    if hasattr(agent, "family_trust"):
        state["family_trust"] = dict(agent.family_trust)
    if hasattr(agent, "family_seen"):
        state["family_seen"] = dict(agent.family_seen)
    if hasattr(agent, "export_control_state"):
        state["control_state"] = agent.export_control_state()
    return state


def restore_agent_state(agent, state: Dict[str, object]) -> None:
    agent.family_basis = {k: v.copy() for k, v in state["family_basis"].items()}
    agent.family_count = dict(state["family_count"])
    agent.concept_offset = {k: v.copy() for k, v in state["concept_offset"].items()}
    agent.concept_count = dict(state["concept_count"])
    agent.concept_family = dict(state["concept_family"])
    if hasattr(agent, "family_trust"):
        agent.family_trust = dict(state.get("family_trust", {}))
    if hasattr(agent, "family_seen"):
        agent.family_seen = dict(state.get("family_seen", {}))
    if hasattr(agent, "import_control_state"):
        agent.import_control_state(state.get("control_state", {}))


def corrected_family(
    agent,
    rng: np.random.Generator,
    x: np.ndarray,
    concept: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
) -> str:
    return long_goal.corrected_family_from_x(
        agent=agent,
        rng=rng,
        x=x,
        concept=concept,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        margin_threshold=margin_threshold,
    )


def probe_target_stability(
    agent,
    rng: np.random.Generator,
    target_family: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    repeats: int,
) -> float:
    ok = 0
    for _ in range(repeats):
        concept, x = long_goal.sample_target_probe(
            rng=rng,
            family=target_family,
            noise=noise * 1.10,
            dropout_p=min(0.40, dropout_p + 0.03),
            missing_modality_p=min(0.45, missing_modality_p + 0.04),
            drift_scale=drift_scale * 1.35,
        )
        ok += int(
            corrected_family(
                agent,
                rng,
                x,
                concept,
                noise=noise * 1.10,
                dropout_p=min(0.40, dropout_p + 0.03),
                missing_modality_p=min(0.45, missing_modality_p + 0.04),
                drift_scale=drift_scale * 1.35,
                margin_threshold=margin_threshold,
            )
            == target_family
        )
    return float(ok / max(1, repeats))


def sample_old_pollution(
    agent,
    rng: np.random.Generator,
    current_target: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    samples_per_family: int,
) -> Tuple[float, float]:
    wrong = 0
    hijack = 0
    total = 0
    for family in stream_bench.FAMILIES:
        concepts = list(stream_bench.PHASE1[family])
        pick_count = min(samples_per_family, len(concepts))
        chosen = rng.choice(np.array(concepts, dtype=object), size=pick_count, replace=False)
        for concept_obj in np.atleast_1d(chosen):
            concept = str(concept_obj.item() if hasattr(concept_obj, "item") else concept_obj)
            x = stream_bench.sample_stream_input(
                rng,
                concept,
                noise=noise * 1.05,
                dropout_p=min(0.35, dropout_p + 0.02),
                missing_modality_p=min(0.40, missing_modality_p + 0.03),
                drift_scale=drift_scale * 1.20,
            )
            corrected = corrected_family(
                agent,
                rng,
                x,
                concept,
                noise=noise * 1.05,
                dropout_p=min(0.35, dropout_p + 0.02),
                missing_modality_p=min(0.40, missing_modality_p + 0.03),
                drift_scale=drift_scale * 1.20,
                margin_threshold=margin_threshold,
            )
            wrong += int(corrected != family)
            hijack += int(family != current_target and corrected == current_target)
            total += 1
    return float(wrong / max(1, total)), float(hijack / max(1, total))


class TrainablePlanningAgent(stateful_scan.StatefulSharedActionAgent):
    def __init__(
        self,
        family_alpha: float,
        offset_alpha: float,
        action_beta: float,
        correction_mix: float,
        trust_temp: float,
        reserve_target: float,
        replay_count: int,
        replay_mode: str,
        meta_lr: float,
        replay_lr: float,
        policy_lr: float,
    ) -> None:
        super().__init__(
            family_alpha=family_alpha,
            offset_alpha=offset_alpha,
            action_beta=action_beta,
            correction_mix=correction_mix,
            trust_temp=trust_temp,
        )
        self.reserve_target = float(reserve_target)
        self.base_replay_count = int(replay_count)
        self.meta_lr = float(meta_lr)
        self.replay_lr = float(replay_lr)
        self.policy_lr = float(policy_lr)
        self.current_target: str | None = None
        self.goal_strength: Dict[str, float] = {family: 0.0 for family in stream_bench.FAMILIES}
        self.policy_bias: Dict[str, float] = {family: 0.0 for family in stream_bench.FAMILIES}
        self.replay_logits: Dict[str, float] = {"target": 0.0, "weakest": 0.0, "hybrid": 0.0}
        self.replay_logits[replay_mode] = 0.25
        self.replay_scale = 1.0
        self.rollback_threshold = 0.72
        self.stability_ema = 0.55
        self.pollution_ema = 0.25
        self.success_ema = 0.55
        self.failure_ema = 0.20

    def set_goal_target(self, target_family: str, stage_progress: float) -> None:
        self.current_target = target_family
        for family in stream_bench.FAMILIES:
            if family == target_family:
                update = 0.60 + 0.40 * float(stage_progress)
                self.goal_strength[family] = 0.92 * self.goal_strength[family] + 0.08 * update
            else:
                self.goal_strength[family] *= 0.96

    def family_candidates(self, x: np.ndarray) -> List[Tuple[str, float]]:
        rows = []
        for name, proto in self.family_basis.items():
            base_dist = stream_bench.sq_dist(x, proto)
            trust = self.family_trust.get(name, 0.0)
            goal_bonus = self.goal_strength.get(name, 0.0)
            policy_bonus = self.policy_bias.get(name, 0.0)
            adjusted = float(base_dist - self.trust_temp * trust - 0.18 * goal_bonus - 0.12 * policy_bonus)
            rows.append((name, adjusted))
        rows.sort(key=lambda item: item[1])
        return rows

    def observe_feedback(
        self,
        target_family: str,
        predicted_family: str,
        success: bool,
        stability_signal: float,
        pollution_signal: float,
        hijack_signal: float,
    ) -> None:
        reward = 1.0 if success else -1.0
        self.success_ema = 0.94 * self.success_ema + 0.06 * max(0.0, reward)
        self.failure_ema = 0.94 * self.failure_ema + 0.06 * max(0.0, -reward)
        self.stability_ema = 0.92 * self.stability_ema + 0.08 * stability_signal
        self.pollution_ema = 0.92 * self.pollution_ema + 0.08 * pollution_signal

        self.goal_strength[target_family] = float(
            np.clip(
                self.goal_strength[target_family] + self.meta_lr * (0.55 * reward + 0.30 * stability_signal - 0.20 * pollution_signal),
                0.0,
                1.6,
            )
        )
        self.policy_bias[target_family] = float(
            np.clip(self.policy_bias[target_family] + self.policy_lr * (0.80 * reward + 0.25 * stability_signal - 0.30 * hijack_signal), -0.8, 0.8)
        )
        if predicted_family != target_family and predicted_family in self.policy_bias:
            self.policy_bias[predicted_family] = float(
                np.clip(self.policy_bias[predicted_family] - self.policy_lr * (0.45 + 0.55 * pollution_signal), -0.8, 0.8)
            )

        self.reserve_target = float(
            np.clip(
                self.reserve_target + self.meta_lr * (0.35 * pollution_signal + 0.18 * hijack_signal + 0.20 * (1.0 - stability_signal) - 0.12 * reward),
                0.90,
                0.995,
            )
        )
        self.replay_scale = float(
            np.clip(
                self.replay_scale + self.replay_lr * (0.55 * pollution_signal + 0.25 * (1.0 - stability_signal) + 0.25 * (0.0 if success else 1.0)),
                0.6,
                2.4,
            )
        )
        self.rollback_threshold = float(
            np.clip(self.rollback_threshold + self.meta_lr * (0.28 * (0.0 if success else 1.0) - 0.12 * stability_signal), 0.62, 0.88)
        )
        self.replay_logits["target"] += self.replay_lr * (max(0.0, self.reserve_target - self.family_trust.get(target_family, 0.0)) - 0.5 * pollution_signal)
        self.replay_logits["weakest"] += self.replay_lr * (pollution_signal + 0.4 * hijack_signal)
        self.replay_logits["hybrid"] += self.replay_lr * (0.5 * (1.0 - stability_signal) + 0.3 * pollution_signal)

    def choose_replay_policy(self, current_target: str) -> Tuple[str, int]:
        weakest_family = min(self.family_trust, key=lambda family: self.family_trust[family])
        weakest_gap = max(0.0, self.reserve_target - self.family_trust.get(weakest_family, 0.0))
        target_gap = max(0.0, self.reserve_target - self.family_trust.get(current_target, 0.0))

        scores = {
            "target": self.replay_logits["target"] + 1.10 * target_gap + 0.20 * self.goal_strength.get(current_target, 0.0),
            "weakest": self.replay_logits["weakest"] + 1.00 * weakest_gap + 0.35 * self.pollution_ema,
            "hybrid": self.replay_logits["hybrid"] + 0.75 * (target_gap + weakest_gap) + 0.25 * (1.0 - self.stability_ema),
        }
        mode = max(scores, key=scores.get)
        magnitude = 0.55 * target_gap + 0.45 * weakest_gap + 0.30 * self.replay_scale
        replay_count = int(np.clip(round(self.base_replay_count + magnitude), 1, 4))
        return mode, replay_count

    def should_trigger_rollback(self, step_success: bool, stability_signal: float) -> bool:
        if not step_success:
            return True
        return stability_signal < self.rollback_threshold

    def export_control_state(self) -> Dict[str, object]:
        return {
            "reserve_target": self.reserve_target,
            "base_replay_count": self.base_replay_count,
            "goal_strength": dict(self.goal_strength),
            "policy_bias": dict(self.policy_bias),
            "replay_logits": dict(self.replay_logits),
            "replay_scale": self.replay_scale,
            "rollback_threshold": self.rollback_threshold,
            "stability_ema": self.stability_ema,
            "pollution_ema": self.pollution_ema,
            "success_ema": self.success_ema,
            "failure_ema": self.failure_ema,
            "current_target": self.current_target,
        }

    def import_control_state(self, state: Dict[str, object]) -> None:
        self.reserve_target = float(state.get("reserve_target", self.reserve_target))
        self.base_replay_count = int(state.get("base_replay_count", self.base_replay_count))
        self.goal_strength = dict(state.get("goal_strength", self.goal_strength))
        self.policy_bias = dict(state.get("policy_bias", self.policy_bias))
        self.replay_logits = dict(state.get("replay_logits", self.replay_logits))
        self.replay_scale = float(state.get("replay_scale", self.replay_scale))
        self.rollback_threshold = float(state.get("rollback_threshold", self.rollback_threshold))
        self.stability_ema = float(state.get("stability_ema", self.stability_ema))
        self.pollution_ema = float(state.get("pollution_ema", self.pollution_ema))
        self.success_ema = float(state.get("success_ema", self.success_ema))
        self.failure_ema = float(state.get("failure_ema", self.failure_ema))
        self.current_target = state.get("current_target", self.current_target)


def maybe_apply_replay(
    agent: stateful_scan.StatefulSharedActionAgent,
    rng: np.random.Generator,
    current_target: str,
    old_buffers: Dict[str, List[str]],
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    reserve_target: float,
    replay_count: int,
    replay_mode: str,
) -> int:
    return long_goal.maybe_replay_policy(
        agent=agent,
        rng=rng,
        current_target=current_target,
        reserve_target=reserve_target,
        replay_count=replay_count,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        old_buffers=old_buffers,
        replay_mode=replay_mode,
    )


def attempt_recovery(
    agent,
    rng: np.random.Generator,
    current_target: str,
    old_buffers: Dict[str, List[str]],
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    fixed_reserve_target: float,
    fixed_replay_count: int,
    fixed_replay_mode: str,
    is_trainable: bool,
) -> Tuple[bool, int, float]:
    if is_trainable:
        replay_mode, replay_count = agent.choose_replay_policy(current_target)
        reserve_target = float(agent.reserve_target)
    else:
        replay_mode = fixed_replay_mode
        replay_count = fixed_replay_count
        reserve_target = fixed_reserve_target

    updates = maybe_apply_replay(
        agent=agent,
        rng=rng,
        current_target=current_target,
        old_buffers=old_buffers,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        reserve_target=reserve_target,
        replay_count=replay_count,
        replay_mode=replay_mode,
    )
    stability = probe_target_stability(
        agent=agent,
        rng=rng,
        target_family=current_target,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        margin_threshold=margin_threshold,
        repeats=4,
    )

    if (not is_trainable) or stability >= 0.75:
        return bool(stability >= 0.75), updates, stability

    updates += maybe_apply_replay(
        agent=agent,
        rng=rng,
        current_target=current_target,
        old_buffers=old_buffers,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        reserve_target=float(agent.reserve_target),
        replay_count=min(4, replay_count + 1),
        replay_mode="hybrid",
    )
    stability_retry = probe_target_stability(
        agent=agent,
        rng=rng,
        target_family=current_target,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        margin_threshold=margin_threshold,
        repeats=4,
    )
    return bool(stability_retry >= 0.72), updates, max(stability, stability_retry)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def run_variable_planner_system(
    system_name: str,
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    min_program_len: int,
    max_program_len: int,
    min_stage_span: int,
    max_stage_span: int,
    default_cfg: Dict[str, float | int | str],
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    old_buffers = goal_scan.build_old_buffers(seed)
    stream = stream_bench.build_stream(seed)
    concept_budget = sum(1 for item in stream if item["kind"] == "concept")
    episodes = build_variable_length_episodes(
        seed=seed + 11,
        concept_budget=concept_budget,
        min_program_len=min_program_len,
        max_program_len=max_program_len,
        min_stage_span=min_stage_span,
        max_stage_span=max_stage_span,
    )

    is_trainable = system_name == "trainable_planner"
    enable_rollback = system_name in {"stateful_with_rollback", "trainable_planner"}

    if is_trainable:
        agent = TrainablePlanningAgent(
            family_alpha=0.05,
            offset_alpha=0.32,
            action_beta=float(default_cfg["action_beta"]),
            correction_mix=float(default_cfg["correction_mix"]),
            trust_temp=float(default_cfg["trust_temp"]),
            reserve_target=float(default_cfg["reserve_target"]),
            replay_count=int(default_cfg["replay_count"]),
            replay_mode=str(default_cfg["replay_mode"]),
            meta_lr=0.08,
            replay_lr=0.06,
            policy_lr=0.07,
        )
    else:
        agent = stateful_scan.StatefulSharedActionAgent(
            family_alpha=0.05,
            offset_alpha=0.32,
            action_beta=float(default_cfg["action_beta"]),
            correction_mix=float(default_cfg["correction_mix"]),
            trust_temp=float(default_cfg["trust_temp"]),
        )

    episode_idx = 0
    stage_idx = 0
    stage_remaining = episodes[0].stages[0].span
    episode_snapshots: Dict[int, Dict[str, object]] = {-1: snapshot_agent_state(agent)}
    current_episode_success = True

    episode_successes = 0
    episode_total = 0
    rollback_attempts = 0
    rollback_successes = 0
    chain_completed = 0
    chain_total = sum(len(episode.stages) for episode in episodes)
    target_capture_ok = 0
    target_capture_total = 0
    step_successes = 0
    step_total = 0
    replay_updates = 0
    stability_sum = 0.0
    stability_count = 0
    pollution_wrong_sum = 0.0
    pollution_hijack_sum = 0.0
    pollution_count = 0

    for item in stream:
        if episode_idx >= len(episodes):
            break

        current_stage = episodes[episode_idx].stages[stage_idx]
        current_target = current_stage.family
        stage_progress = 1.0 - float(stage_remaining) / float(max(1, current_stage.span))
        if is_trainable:
            agent.set_goal_target(current_target, stage_progress)

        if item["kind"] != "concept":
            if enable_rollback:
                if is_trainable:
                    replay_mode, replay_count = agent.choose_replay_policy(current_target)
                    reserve_target = float(agent.reserve_target)
                else:
                    replay_mode = str(default_cfg["replay_mode"])
                    replay_count = int(default_cfg["replay_count"])
                    reserve_target = float(default_cfg["reserve_target"])
                replay_updates += maybe_apply_replay(
                    agent=agent,
                    rng=rng,
                    current_target=current_target,
                    old_buffers=old_buffers,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                    reserve_target=reserve_target,
                    replay_count=replay_count,
                    replay_mode=replay_mode,
                )
            stability = probe_target_stability(
                agent=agent,
                rng=rng,
                target_family=current_target,
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
                margin_threshold=margin_threshold,
                repeats=2,
            )
            pollution_wrong, pollution_hijack = sample_old_pollution(
                agent=agent,
                rng=rng,
                current_target=current_target,
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
                margin_threshold=margin_threshold,
                samples_per_family=1,
            )
            stability_sum += stability
            stability_count += 1
            pollution_wrong_sum += pollution_wrong
            pollution_hijack_sum += pollution_hijack
            pollution_count += 1
            if is_trainable:
                agent.observe_feedback(
                    target_family=current_target,
                    predicted_family=current_target,
                    success=stability >= 0.70,
                    stability_signal=stability,
                    pollution_signal=pollution_wrong,
                    hijack_signal=pollution_hijack,
                )
            continue

        x = stream_bench.sample_stream_input(
            rng,
            item["concept"],
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
        )
        agent.train(x, item["family"], item["concept"])

        pred_family, _pred_concept, _margin = agent.predict_with_margin(x)
        corrected = corrected_family(
            agent=agent,
            rng=rng,
            x=x,
            concept=item["concept"],
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            margin_threshold=margin_threshold,
        )
        engage = item["family"] == current_target
        step_success = (corrected == current_target) == engage
        step_successes += int(step_success)
        step_total += 1
        if engage:
            target_capture_ok += int(corrected == current_target)
            target_capture_total += 1

        stability = probe_target_stability(
            agent=agent,
            rng=rng,
            target_family=current_target,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            margin_threshold=margin_threshold,
            repeats=2,
        )
        pollution_wrong, pollution_hijack = sample_old_pollution(
            agent=agent,
            rng=rng,
            current_target=current_target,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            margin_threshold=margin_threshold,
            samples_per_family=1,
        )
        stability_sum += stability
        stability_count += 1
        pollution_wrong_sum += pollution_wrong
        pollution_hijack_sum += pollution_hijack
        pollution_count += 1

        if is_trainable:
            agent.observe_feedback(
                target_family=current_target,
                predicted_family=corrected,
                success=step_success,
                stability_signal=stability,
                pollution_signal=pollution_wrong,
                hijack_signal=pollution_hijack,
            )
            rollback_trigger = agent.should_trigger_rollback(step_success, stability)
        else:
            rollback_trigger = enable_rollback and ((not step_success) or stability < 0.66)

        if rollback_trigger and enable_rollback:
            rollback_attempts += 1
            rollback_anchor = current_stage.anchor_idx if stage_idx > 0 else -1
            restore_agent_state(agent, episode_snapshots.get(rollback_anchor, episode_snapshots[-1]))
            recovered, used_updates, recovery_stability = attempt_recovery(
                agent=agent,
                rng=rng,
                current_target=current_target,
                old_buffers=old_buffers,
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
                margin_threshold=margin_threshold,
                fixed_reserve_target=float(default_cfg["reserve_target"]),
                fixed_replay_count=int(default_cfg["replay_count"]),
                fixed_replay_mode=str(default_cfg["replay_mode"]),
                is_trainable=is_trainable,
            )
            replay_updates += used_updates
            rollback_successes += int(recovered)
            current_episode_success = current_episode_success and recovered
            stability_sum += recovery_stability
            stability_count += 1
            if is_trainable:
                agent.observe_feedback(
                    target_family=current_target,
                    predicted_family=current_target if recovered else corrected,
                    success=recovered,
                    stability_signal=recovery_stability,
                    pollution_signal=pollution_wrong,
                    hijack_signal=pollution_hijack,
                )
            if recovered:
                continue

        stage_remaining -= 1
        if stage_remaining > 0:
            if enable_rollback and not is_trainable:
                replay_updates += maybe_apply_replay(
                    agent=agent,
                    rng=rng,
                    current_target=current_target,
                    old_buffers=old_buffers,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                    reserve_target=float(default_cfg["reserve_target"]),
                    replay_count=int(default_cfg["replay_count"]),
                    replay_mode=str(default_cfg["replay_mode"]),
                )
            continue

        chain_completed += 1
        episode_snapshots[stage_idx] = snapshot_agent_state(agent)
        stage_idx += 1

        if stage_idx >= len(episodes[episode_idx].stages):
            episode_total += 1
            episode_successes += int(current_episode_success)
            episode_idx += 1
            if episode_idx >= len(episodes):
                break
            stage_idx = 0
            current_episode_success = True
            episode_snapshots = {-1: snapshot_agent_state(agent)}

        stage_remaining = episodes[episode_idx].stages[stage_idx].span

    if episode_idx < len(episodes):
        episode_total += 1
        episode_successes += int(current_episode_success)

    mean_program_length = float(np.mean([len(episode.stages) for episode in episodes]))
    mean_stage_span = float(np.mean([stage.span for episode in episodes for stage in episode.stages]))
    episode_success_rate = float(episode_successes / max(1, episode_total))
    rollback_recovery_rate = float(rollback_successes / max(1, rollback_attempts))
    chain_completion_rate = float(chain_completed / max(1, chain_total))
    target_capture_accuracy = float(target_capture_ok / max(1, target_capture_total))
    step_success_rate = float(step_successes / max(1, step_total))
    mean_pollution_wrong = float(pollution_wrong_sum / max(1, pollution_count))
    mean_pollution_hijack = float(pollution_hijack_sum / max(1, pollution_count))
    contamination_control = float(np.clip(1.0 - (0.72 * mean_pollution_wrong + 0.28 * mean_pollution_hijack), 0.0, 1.0))
    open_environment_stability = float(stability_sum / max(1, stability_count))
    variable_planning_score = float(
        (
            1.55 * episode_success_rate
            + 1.30 * rollback_recovery_rate
            + 1.20 * contamination_control
            + 1.15 * open_environment_stability
            + 1.00 * chain_completion_rate
            + 0.90 * target_capture_accuracy
            + 0.80 * step_success_rate
        )
        / 7.90
    )

    result = {
        "episode_success_rate": episode_success_rate,
        "rollback_recovery_rate": rollback_recovery_rate,
        "contamination_control": contamination_control,
        "open_environment_stability": open_environment_stability,
        "chain_completion_rate": chain_completion_rate,
        "target_capture_accuracy": target_capture_accuracy,
        "step_success_rate": step_success_rate,
        "variable_planning_score": variable_planning_score,
        "rollback_attempts": float(rollback_attempts),
        "replay_updates": float(replay_updates),
        "mean_program_length": mean_program_length,
        "mean_stage_span": mean_stage_span,
        "mean_family_trust": float(np.mean(list(agent.family_trust.values()))),
    }
    if is_trainable:
        result.update(
            {
                "final_reserve_target": float(agent.reserve_target),
                "final_replay_scale": float(agent.replay_scale),
                "final_rollback_threshold": float(agent.rollback_threshold),
                "mean_goal_strength": float(np.mean(list(agent.goal_strength.values()))),
                "mean_policy_bias": float(np.mean(list(agent.policy_bias.values()))),
            }
        )
    else:
        result.update(
            {
                "final_reserve_target": float(default_cfg["reserve_target"]),
                "final_replay_scale": 1.0,
                "final_rollback_threshold": 0.66 if enable_rollback else 0.0,
                "mean_goal_strength": 0.0,
                "mean_policy_bias": 0.0,
            }
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Variable-length planning chain benchmark with rollback and trainable control")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--min-program-len", type=int, default=2)
    ap.add_argument("--max-program-len", type=int, default=5)
    ap.add_argument("--min-stage-span", type=int, default=7)
    ap.add_argument("--max-stage-span", type=int, default=16)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_variable_planning_trainable_benchmark_20260310.json")
    args = ap.parse_args()

    default_cfg = load_best_configs()
    t0 = time.time()

    systems = {}
    for system_name in ["stateful_no_rollback", "stateful_with_rollback", "trainable_planner"]:
        rows = []
        for offset in range(int(args.num_seeds)):
            rows.append(
                run_variable_planner_system(
                    system_name=system_name,
                    seed=int(args.seed) + offset,
                    noise=float(args.noise),
                    dropout_p=float(args.dropout_p),
                    missing_modality_p=float(args.missing_modality_p),
                    drift_scale=float(args.drift_scale),
                    margin_threshold=float(args.margin_threshold),
                    min_program_len=int(args.min_program_len),
                    max_program_len=int(args.max_program_len),
                    min_stage_span=int(args.min_stage_span),
                    max_stage_span=int(args.max_stage_span),
                    default_cfg=default_cfg,
                )
            )
        systems[system_name] = summarize(rows)

    no_rollback = systems["stateful_no_rollback"]
    static_rollback = systems["stateful_with_rollback"]
    trainable = systems["trainable_planner"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "noise": float(args.noise),
            "dropout_p": float(args.dropout_p),
            "missing_modality_p": float(args.missing_modality_p),
            "drift_scale": float(args.drift_scale),
            "margin_threshold": float(args.margin_threshold),
            "min_program_len": int(args.min_program_len),
            "max_program_len": int(args.max_program_len),
            "min_stage_span": int(args.min_stage_span),
            "max_stage_span": int(args.max_stage_span),
            "source_files": [
                "open_world_grounding_action_loop_stateful_scan_20260310.json",
                "open_world_long_horizon_goal_state_benchmark_20260310.json",
                "open_world_subgoal_planning_benchmark_20260310.json",
            ],
        },
        "systems": systems,
        "gains": {
            "rollback_episode_gain": float(static_rollback["episode_success_rate"] - no_rollback["episode_success_rate"]),
            "rollback_recovery_gain": float(static_rollback["rollback_recovery_rate"] - no_rollback["rollback_recovery_rate"]),
            "trainable_episode_gain_vs_static": float(trainable["episode_success_rate"] - static_rollback["episode_success_rate"]),
            "trainable_recovery_gain_vs_static": float(trainable["rollback_recovery_rate"] - static_rollback["rollback_recovery_rate"]),
            "trainable_contamination_gain_vs_static": float(trainable["contamination_control"] - static_rollback["contamination_control"]),
            "trainable_stability_gain_vs_static": float(trainable["open_environment_stability"] - static_rollback["open_environment_stability"]),
            "trainable_variable_planning_gain_vs_static": float(trainable["variable_planning_score"] - static_rollback["variable_planning_score"]),
        },
        "headline_metrics": {
            "episode_success_rate": float(trainable["episode_success_rate"]),
            "rollback_recovery_rate": float(trainable["rollback_recovery_rate"]),
            "contamination_control": float(trainable["contamination_control"]),
            "open_environment_stability": float(trainable["open_environment_stability"]),
            "variable_planning_score": float(trainable["variable_planning_score"]),
        },
        "project_readout": {
            "summary": "这一版把固定子目标程序推进成可变长度规划链，并加入失败回退、恢复探测和可学习的目标-动作-回放闭环，评估重点外部化到 episode 成功率、回退恢复率、旧概念污染控制和开放环境稳定性。",
            "next_question": "如果 trainable planner 在这四个指标上持续领先，下一步就该把变量规划链继续压缩成统一结构，并把回退协议接回真实任务约束与脑侧约束。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
