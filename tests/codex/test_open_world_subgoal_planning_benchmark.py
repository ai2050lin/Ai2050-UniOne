#!/usr/bin/env python
"""
Subgoal-planning benchmark for the open-world grounding agent loop.

Goal:
- extend repeated family switching into short multi-stage subgoal programs
- measure episode success, subgoal completion, and transition stability
- test whether the current long-horizon goal-state policy transfers into a
  slightly more planning-like open-world loop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench
import test_open_world_grounding_action_loop as action_loop
import test_open_world_grounding_action_loop_stateful_scan as stateful_scan
import test_open_world_long_horizon_goal_state_benchmark as long_goal


ROOT = Path(__file__).resolve().parents[2]


def build_subgoal_programs(seed: int, concept_events: int, subgoal_span: int, program_len: int) -> List[str]:
    rng = np.random.default_rng(seed)
    families = list(stream_bench.FAMILIES)
    schedule: List[str] = []
    while len(schedule) < concept_events:
        program = families.copy()
        rng.shuffle(program)
        program = program[:program_len]
        for family in program:
            schedule.extend([family] * subgoal_span)
            if len(schedule) >= concept_events:
                break
    return schedule[:concept_events]


def corrected_family(agent, rng, x, concept, noise, dropout_p, missing_modality_p, drift_scale, margin_threshold) -> str:
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


def direct_step_metrics(
    agent,
    rng: np.random.Generator,
    x: np.ndarray,
    concept: str,
    item_family: str,
    target_family: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
) -> Tuple[bool, bool]:
    pred_family, _pred_concept, _margin = agent.predict_with_margin(x)
    corrected = corrected_family(agent, rng, x, concept, noise, dropout_p, missing_modality_p, drift_scale, margin_threshold)
    engage = item_family == target_family
    return (pred_family == target_family) == engage, (corrected == target_family) == engage


def run_direct_program_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    subgoal_span: int,
    program_len: int,
    transition_probe_count: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = action_loop.DirectActionAgent()
    stream = stream_bench.build_stream(seed)
    concept_events = sum(1 for item in stream if item["kind"] == "concept")
    schedule = build_subgoal_programs(seed, concept_events, subgoal_span, program_len)

    subgoal_ok = 0
    corrected_subgoal_ok = 0
    subgoal_total = 0
    transition_ok = 0
    transition_total = 0
    target_capture_ok = 0
    target_capture_total = 0
    episode_successes = 0
    episode_total = 0
    schedule_idx = 0
    event_idx_in_program = 0
    current_target = schedule[0]
    prev_target = current_target
    program_block = subgoal_span * program_len
    current_episode_success = True

    for item in stream:
        if item["kind"] != "concept":
            _ = stream_bench.sample_noise_chunk(rng, dim=24, noise_scale=noise * 1.2)
            continue

        target_family = schedule[schedule_idx]
        phase_changed = target_family != prev_target
        current_target = target_family
        prev_target = target_family

        x = stream_bench.sample_stream_input(
            rng,
            item["concept"],
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
        )
        agent.train(x, item["family"], item["concept"])

        first_ok, corrected_ok = direct_step_metrics(
            agent=agent,
            rng=rng,
            x=x,
            concept=item["concept"],
            item_family=item["family"],
            target_family=current_target,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            margin_threshold=margin_threshold,
        )
        subgoal_ok += int(first_ok)
        corrected_subgoal_ok += int(corrected_ok)
        subgoal_total += 1
        if item["family"] == current_target:
            target_capture_ok += int(corrected_ok)
            target_capture_total += 1
        current_episode_success = current_episode_success and corrected_ok

        if phase_changed:
            for _probe in range(transition_probe_count):
                probe_concept, probe_x = long_goal.sample_target_probe(
                    rng=rng,
                    family=current_target,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                )
                corrected_probe = corrected_family(
                    agent, rng, probe_x, probe_concept, noise, dropout_p, missing_modality_p, drift_scale, margin_threshold
                )
                transition_ok += int(corrected_probe == current_target)
                transition_total += 1

        schedule_idx += 1
        event_idx_in_program += 1
        if event_idx_in_program >= program_block:
            episode_total += 1
            episode_successes += int(current_episode_success)
            event_idx_in_program = 0
            current_episode_success = True

    if event_idx_in_program > 0:
        episode_total += 1
        episode_successes += int(current_episode_success)

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    subgoal_action_accuracy = float(subgoal_ok / max(1, subgoal_total))
    corrected_subgoal_action_accuracy = float(corrected_subgoal_ok / max(1, subgoal_total))
    transition_accuracy = float(transition_ok / max(1, transition_total))
    target_capture_accuracy = float(target_capture_ok / max(1, target_capture_total))
    episode_success_rate = float(episode_successes / max(1, episode_total))
    planning_loop_score = float(
        (
            1.35 * corrected_subgoal_action_accuracy
            + 1.1 * subgoal_action_accuracy
            + 1.25 * transition_accuracy
            + 1.2 * episode_success_rate
            + 0.95 * old_retention
            + 0.85 * target_capture_accuracy
        )
        / 6.7
    )
    return {
        "subgoal_action_accuracy": subgoal_action_accuracy,
        "corrected_subgoal_action_accuracy": corrected_subgoal_action_accuracy,
        "transition_accuracy": transition_accuracy,
        "episode_success_rate": episode_success_rate,
        "old_concept_retention": old_retention,
        "target_capture_accuracy": target_capture_accuracy,
        "planning_loop_score": planning_loop_score,
    }


def run_goal_program_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    subgoal_span: int,
    program_len: int,
    transition_probe_count: int,
    action_beta: float,
    correction_mix: float,
    trust_temp: float,
    reserve_target: float,
    replay_count: int,
    replay_mode: str,
    enable_replay: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = stateful_scan.StatefulSharedActionAgent(
        family_alpha=0.05,
        offset_alpha=0.32,
        action_beta=action_beta,
        correction_mix=correction_mix,
        trust_temp=trust_temp,
    )
    stream = stream_bench.build_stream(seed)
    concept_events = sum(1 for item in stream if item["kind"] == "concept")
    schedule = build_subgoal_programs(seed, concept_events, subgoal_span, program_len)
    old_buffers = long_goal.build_old_buffers(seed)

    subgoal_ok = 0
    corrected_subgoal_ok = 0
    subgoal_total = 0
    transition_ok = 0
    transition_total = 0
    target_capture_ok = 0
    target_capture_total = 0
    episode_successes = 0
    episode_total = 0
    replay_updates = 0
    schedule_idx = 0
    event_idx_in_program = 0
    current_target = schedule[0]
    prev_target = current_target
    program_block = subgoal_span * program_len
    current_episode_success = True

    for item in stream:
        if item["kind"] != "concept":
            _ = stream_bench.sample_noise_chunk(rng, dim=24, noise_scale=noise * 1.2)
            if enable_replay:
                replay_updates += long_goal.maybe_replay_policy(
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
            continue

        target_family = schedule[schedule_idx]
        phase_changed = target_family != prev_target
        current_target = target_family
        prev_target = target_family

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
        corrected = corrected_family(agent, rng, x, item["concept"], noise, dropout_p, missing_modality_p, drift_scale, margin_threshold)
        engage = item["family"] == current_target
        first_ok = (pred_family == current_target) == engage
        corrected_ok = (corrected == current_target) == engage
        subgoal_ok += int(first_ok)
        corrected_subgoal_ok += int(corrected_ok)
        subgoal_total += 1
        if engage:
            target_capture_ok += int(corrected_ok)
            target_capture_total += 1
        current_episode_success = current_episode_success and corrected_ok

        if enable_replay:
            replay_updates += long_goal.maybe_replay_policy(
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

        if phase_changed:
            if enable_replay:
                replay_updates += long_goal.maybe_replay_policy(
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
            for _probe in range(transition_probe_count):
                probe_concept, probe_x = long_goal.sample_target_probe(
                    rng=rng,
                    family=current_target,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                )
                corrected_probe = corrected_family(
                    agent, rng, probe_x, probe_concept, noise, dropout_p, missing_modality_p, drift_scale, margin_threshold
                )
                transition_ok += int(corrected_probe == current_target)
                transition_total += 1

        schedule_idx += 1
        event_idx_in_program += 1
        if event_idx_in_program >= program_block:
            episode_total += 1
            episode_successes += int(current_episode_success)
            event_idx_in_program = 0
            current_episode_success = True

    if event_idx_in_program > 0:
        episode_total += 1
        episode_successes += int(current_episode_success)

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    subgoal_action_accuracy = float(subgoal_ok / max(1, subgoal_total))
    corrected_subgoal_action_accuracy = float(corrected_subgoal_ok / max(1, subgoal_total))
    transition_accuracy = float(transition_ok / max(1, transition_total))
    target_capture_accuracy = float(target_capture_ok / max(1, target_capture_total))
    episode_success_rate = float(episode_successes / max(1, episode_total))
    planning_loop_score = float(
        (
            1.35 * corrected_subgoal_action_accuracy
            + 1.1 * subgoal_action_accuracy
            + 1.25 * transition_accuracy
            + 1.2 * episode_success_rate
            + 0.95 * old_retention
            + 0.85 * target_capture_accuracy
        )
        / 6.7
    )
    return {
        "subgoal_action_accuracy": subgoal_action_accuracy,
        "corrected_subgoal_action_accuracy": corrected_subgoal_action_accuracy,
        "transition_accuracy": transition_accuracy,
        "episode_success_rate": episode_success_rate,
        "old_concept_retention": old_retention,
        "target_capture_accuracy": target_capture_accuracy,
        "planning_loop_score": planning_loop_score,
        "mean_family_trust": float(np.mean(list(agent.family_trust.values()))),
        "replay_updates": float(replay_updates),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Subgoal-planning benchmark for the open-world grounding action loop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_subgoal_planning_benchmark_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    stateful_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_stateful_scan_20260310.json").read_text(encoding="utf-8"))
    long_goal_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_long_horizon_goal_state_benchmark_20260310.json").read_text(encoding="utf-8"))
    stateful_best = stateful_payload["best_config"]
    long_best = long_goal_payload["best_config"]

    direct_rows = []
    stateful_rows = []
    for offset in range(int(args.num_seeds)):
        direct_rows.append(
            run_direct_program_system(
                seed=int(args.seed) + offset,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
                drift_scale=float(args.drift_scale),
                margin_threshold=float(args.margin_threshold),
                subgoal_span=12,
                program_len=3,
                transition_probe_count=3,
            )
        )
        stateful_rows.append(
            run_goal_program_system(
                seed=int(args.seed) + offset,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
                drift_scale=float(args.drift_scale),
                margin_threshold=float(args.margin_threshold),
                subgoal_span=12,
                program_len=3,
                transition_probe_count=3,
                action_beta=float(stateful_best["action_beta"]),
                correction_mix=float(stateful_best["correction_mix"]),
                trust_temp=float(stateful_best["trust_temp"]),
                reserve_target=float(long_best["reserve_target"]),
                replay_count=int(long_best["replay_count"]),
                replay_mode="target",
                enable_replay=False,
            )
        )
    baselines = {
        "direct_action": summarize(direct_rows),
        "stateful_trust": summarize(stateful_rows),
    }

    rows = []
    best = None
    for subgoal_span in [10, 12, 15]:
        for program_len in [3, 4]:
            for reserve_target in [0.95, 0.97]:
                for replay_count in [1, 2]:
                    for replay_mode in ["target", "weakest", "hybrid"]:
                        seeds = []
                        for offset in range(int(args.num_seeds)):
                            seeds.append(
                                run_goal_program_system(
                                    seed=int(args.seed) + offset,
                                    noise=float(args.noise),
                                    dropout_p=float(args.dropout_p),
                                    missing_modality_p=float(args.missing_modality_p),
                                    drift_scale=float(args.drift_scale),
                                    margin_threshold=float(args.margin_threshold),
                                    subgoal_span=int(subgoal_span),
                                    program_len=int(program_len),
                                    transition_probe_count=3,
                                    action_beta=float(stateful_best["action_beta"]),
                                    correction_mix=float(stateful_best["correction_mix"]),
                                    trust_temp=float(stateful_best["trust_temp"]),
                                    reserve_target=float(reserve_target),
                                    replay_count=int(replay_count),
                                    replay_mode=str(replay_mode),
                                    enable_replay=True,
                                )
                            )
                        summary = summarize(seeds)
                        row = {
                            "subgoal_span": int(subgoal_span),
                            "program_len": int(program_len),
                            "reserve_target": float(reserve_target),
                            "replay_count": int(replay_count),
                            "replay_mode": str(replay_mode),
                            **summary,
                            "planning_gain_vs_direct": float(summary["planning_loop_score"] - baselines["direct_action"]["planning_loop_score"]),
                            "planning_gain_vs_stateful": float(summary["planning_loop_score"] - baselines["stateful_trust"]["planning_loop_score"]),
                            "episode_success_gain_vs_direct": float(summary["episode_success_rate"] - baselines["direct_action"]["episode_success_rate"]),
                            "transition_gain_vs_direct": float(summary["transition_accuracy"] - baselines["direct_action"]["transition_accuracy"]),
                        }
                        rows.append(row)
                        objective = (
                            -(row["planning_gain_vs_direct"])
                            - 0.6 * row["episode_success_gain_vs_direct"]
                            - 0.5 * row["transition_gain_vs_direct"]
                        )
                        if best is None or objective < best[0]:
                            best = (objective, row)

    assert best is not None
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "num_seeds": int(args.num_seeds),
            "config_count": len(rows),
            "source_files": [
                "open_world_grounding_action_loop_stateful_scan_20260310.json",
                "open_world_long_horizon_goal_state_benchmark_20260310.json",
            ],
        },
        "baseline_direct_action": baselines["direct_action"],
        "baseline_stateful_trust": baselines["stateful_trust"],
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一版把长期目标状态继续推进成更像规划的子目标程序，直接测量阶段过渡、目标捕获和整段 episode 成功率是否还能同时提升。",
            "next_question": "如果 subgoal program 仍然能保持正增益，下一步就该把固定 family 程序推进成可变长度、带失败回退的规划链。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
