#!/usr/bin/env python
"""
Long-horizon goal-state benchmark for the open-world grounding agent loop.

Goal:
- extend old-concept retention into true multi-step target maintenance
- stress repeated target-family phase switching under continuous stream updates
- test whether goal-state replay can survive long-horizon switching better than
  direct and stateful baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench
import test_open_world_grounding_action_loop as action_loop
import test_open_world_grounding_action_loop_goal_state_scan as goal_scan
import test_open_world_grounding_action_loop_stateful_scan as stateful_scan


ROOT = Path(__file__).resolve().parents[2]


def family_concepts(family: str) -> List[str]:
    return list(stream_bench.PHASE1[family]) + list(stream_bench.PHASE2[family])


def build_goal_schedule(seed: int, concept_events: int, phase_span: int) -> List[str]:
    rng = np.random.default_rng(seed)
    schedule: List[str] = []
    order = list(stream_bench.FAMILIES)
    while len(schedule) < concept_events:
        rng.shuffle(order)
        for family in order:
            schedule.extend([family] * phase_span)
            if len(schedule) >= concept_events:
                break
    return schedule[:concept_events]


def sample_target_probe(
    rng: np.random.Generator,
    family: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
) -> Tuple[str, np.ndarray]:
    concept = rng.choice(family_concepts(family)).item()
    x = stream_bench.sample_stream_input(
        rng,
        concept,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    return concept, x


def corrected_family_from_x(
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
    if hasattr(agent, "corrected_family"):
        corrected, _used = agent.corrected_family(x, margin_threshold=margin_threshold)
        return corrected

    pred_family, _pred_concept, margin = agent.predict_with_margin(x)
    if margin >= margin_threshold:
        return pred_family

    x_retry = stream_bench.sample_stream_input(
        rng,
        concept,
        noise=noise * 0.75,
        dropout_p=dropout_p * 0.8,
        missing_modality_p=missing_modality_p * 0.8,
        drift_scale=drift_scale * 0.6,
    )
    merged = 0.5 * (x + x_retry)
    retry_family, _retry_concept, _retry_margin = agent.predict_with_margin(merged.astype(np.float32))
    return retry_family


def build_old_buffers(seed: int) -> Dict[str, List[str]]:
    return goal_scan.build_old_buffers(seed)


def maybe_replay_target_family(
    agent: stateful_scan.StatefulSharedActionAgent,
    rng: np.random.Generator,
    target_family: str,
    reserve_target: float,
    replay_count: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    old_buffers: Dict[str, List[str]],
) -> int:
    trust = agent.family_trust.get(target_family, 0.0)
    if trust >= reserve_target:
        return 0
    goal_scan.replay_old_family(
        agent=agent,
        rng=rng,
        family=target_family,
        replay_count=replay_count,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
        old_buffers=old_buffers,
    )
    return replay_count


def maybe_replay_policy(
    agent: stateful_scan.StatefulSharedActionAgent,
    rng: np.random.Generator,
    current_target: str,
    reserve_target: float,
    replay_count: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    old_buffers: Dict[str, List[str]],
    replay_mode: str,
) -> int:
    weakest_family = min(agent.family_trust, key=lambda fam: agent.family_trust[fam])
    weakest_trust = agent.family_trust.get(weakest_family, 0.0)

    if replay_mode == "target":
        return maybe_replay_target_family(
            agent=agent,
            rng=rng,
            target_family=current_target,
            reserve_target=reserve_target,
            replay_count=replay_count,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            old_buffers=old_buffers,
        )

    if replay_mode == "weakest":
        if weakest_trust >= reserve_target:
            return 0
        goal_scan.replay_old_family(
            agent=agent,
            rng=rng,
            family=weakest_family,
            replay_count=replay_count,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            old_buffers=old_buffers,
        )
        return replay_count

    if replay_mode == "hybrid":
        count = maybe_replay_target_family(
            agent=agent,
            rng=rng,
            target_family=current_target,
            reserve_target=reserve_target,
            replay_count=replay_count,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            old_buffers=old_buffers,
        )
        if count > 0:
            return count
        if weakest_trust >= reserve_target - 0.02:
            return 0
        goal_scan.replay_old_family(
            agent=agent,
            rng=rng,
            family=weakest_family,
            replay_count=1,
            noise=noise,
            dropout_p=dropout_p,
            missing_modality_p=missing_modality_p,
            drift_scale=drift_scale,
            old_buffers=old_buffers,
        )
        return 1

    raise KeyError(f"unknown replay_mode: {replay_mode}")


def run_direct_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    phase_span: int,
    switch_probe_count: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = action_loop.DirectActionAgent()
    stream = stream_bench.build_stream(seed)
    concept_events = sum(1 for item in stream if item["kind"] == "concept")
    schedule = build_goal_schedule(seed, concept_events, phase_span)

    goal_ok = 0
    corrected_goal_ok = 0
    goal_total = 0
    target_capture_ok = 0
    target_capture_total = 0
    phase_switch_ok = 0
    phase_switch_total = 0
    schedule_idx = 0
    current_target = schedule[0]
    prev_target = current_target

    for item in stream:
        if item["kind"] != "concept":
            _ = stream_bench.sample_noise_chunk(rng, dim=24, noise_scale=noise * 1.2)
            continue

        target_family = schedule[schedule_idx]
        schedule_idx += 1
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
        corrected_family = corrected_family_from_x(
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
        desired_engage = item["family"] == current_target
        goal_ok += int((pred_family == current_target) == desired_engage)
        corrected_goal_ok += int((corrected_family == current_target) == desired_engage)
        goal_total += 1
        if desired_engage:
            target_capture_ok += int(corrected_family == current_target)
            target_capture_total += 1

        if phase_changed:
            for _probe in range(switch_probe_count):
                probe_concept, probe_x = sample_target_probe(
                    rng=rng,
                    family=current_target,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                )
                corrected_probe_family = corrected_family_from_x(
                    agent=agent,
                    rng=rng,
                    x=probe_x,
                    concept=probe_concept,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                    margin_threshold=margin_threshold,
                )
                phase_switch_ok += int(corrected_probe_family == current_target)
                phase_switch_total += 1

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    goal_action_accuracy = float(goal_ok / max(1, goal_total))
    corrected_goal_action_accuracy = float(corrected_goal_ok / max(1, goal_total))
    target_capture_accuracy = float(target_capture_ok / max(1, target_capture_total))
    phase_switch_accuracy = float(phase_switch_ok / max(1, phase_switch_total))
    long_horizon_loop_score = float(
        (
            1.4 * corrected_goal_action_accuracy
            + 1.1 * goal_action_accuracy
            + 1.2 * phase_switch_accuracy
            + 1.1 * old_retention
            + 0.9 * target_capture_accuracy
        )
        / 5.7
    )
    return {
        "goal_action_accuracy": goal_action_accuracy,
        "corrected_goal_action_accuracy": corrected_goal_action_accuracy,
        "phase_switch_accuracy": phase_switch_accuracy,
        "old_concept_retention": old_retention,
        "target_capture_accuracy": target_capture_accuracy,
        "long_horizon_loop_score": long_horizon_loop_score,
    }


def run_stateful_goal_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    phase_span: int,
    switch_probe_count: int,
    action_beta: float,
    correction_mix: float,
    trust_temp: float,
    reserve_target: float,
    replay_count: int,
    enable_replay: bool,
    replay_mode: str = "target",
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
    schedule = build_goal_schedule(seed, concept_events, phase_span)
    old_buffers = build_old_buffers(seed)

    goal_ok = 0
    corrected_goal_ok = 0
    goal_total = 0
    target_capture_ok = 0
    target_capture_total = 0
    phase_switch_ok = 0
    phase_switch_total = 0
    replay_updates = 0
    schedule_idx = 0
    current_target = schedule[0]
    prev_target = current_target

    for item in stream:
        if item["kind"] != "concept":
            _ = stream_bench.sample_noise_chunk(rng, dim=24, noise_scale=noise * 1.2)
            if enable_replay:
                replay_updates += maybe_replay_policy(
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
        schedule_idx += 1
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
        corrected_family = corrected_family_from_x(
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
        desired_engage = item["family"] == current_target
        goal_ok += int((pred_family == current_target) == desired_engage)
        corrected_goal_ok += int((corrected_family == current_target) == desired_engage)
        goal_total += 1
        if desired_engage:
            target_capture_ok += int(corrected_family == current_target)
            target_capture_total += 1

        if enable_replay:
            replay_updates += maybe_replay_policy(
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
                replay_updates += maybe_replay_policy(
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
            for _probe in range(switch_probe_count):
                probe_concept, probe_x = sample_target_probe(
                    rng=rng,
                    family=current_target,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                )
                corrected_probe_family = corrected_family_from_x(
                    agent=agent,
                    rng=rng,
                    x=probe_x,
                    concept=probe_concept,
                    noise=noise,
                    dropout_p=dropout_p,
                    missing_modality_p=missing_modality_p,
                    drift_scale=drift_scale,
                    margin_threshold=margin_threshold,
                )
                phase_switch_ok += int(corrected_probe_family == current_target)
                phase_switch_total += 1

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    goal_action_accuracy = float(goal_ok / max(1, goal_total))
    corrected_goal_action_accuracy = float(corrected_goal_ok / max(1, goal_total))
    target_capture_accuracy = float(target_capture_ok / max(1, target_capture_total))
    phase_switch_accuracy = float(phase_switch_ok / max(1, phase_switch_total))
    long_horizon_loop_score = float(
        (
            1.4 * corrected_goal_action_accuracy
            + 1.1 * goal_action_accuracy
            + 1.2 * phase_switch_accuracy
            + 1.1 * old_retention
            + 0.9 * target_capture_accuracy
        )
        / 5.7
    )
    return {
        "goal_action_accuracy": goal_action_accuracy,
        "corrected_goal_action_accuracy": corrected_goal_action_accuracy,
        "phase_switch_accuracy": phase_switch_accuracy,
        "old_concept_retention": old_retention,
        "target_capture_accuracy": target_capture_accuracy,
        "long_horizon_loop_score": long_horizon_loop_score,
        "mean_family_trust": float(np.mean(list(agent.family_trust.values()))),
        "replay_updates": float(replay_updates),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-horizon goal-state benchmark for the open-world grounding action loop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_long_horizon_goal_state_benchmark_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    stateful_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_stateful_scan_20260310.json").read_text(encoding="utf-8"))
    goal_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_goal_state_scan_20260310.json").read_text(encoding="utf-8"))
    stateful_best = stateful_payload["best_config"]
    goal_best = goal_payload["best_config"]

    baselines = {}
    direct_rows = []
    stateful_rows = []
    for offset in range(int(args.num_seeds)):
        direct_rows.append(
            run_direct_system(
                seed=int(args.seed) + offset,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
                drift_scale=float(args.drift_scale),
                margin_threshold=float(args.margin_threshold),
                phase_span=18,
                switch_probe_count=3,
            )
        )
        stateful_rows.append(
            run_stateful_goal_system(
                seed=int(args.seed) + offset,
                noise=float(args.noise),
                dropout_p=float(args.dropout_p),
                missing_modality_p=float(args.missing_modality_p),
                drift_scale=float(args.drift_scale),
                margin_threshold=float(args.margin_threshold),
                phase_span=18,
                switch_probe_count=3,
                action_beta=float(stateful_best["action_beta"]),
                correction_mix=float(stateful_best["correction_mix"]),
                trust_temp=float(stateful_best["trust_temp"]),
                reserve_target=float(goal_best["reserve_target"]),
                replay_count=int(goal_best["replay_count"]),
                enable_replay=False,
            )
        )
    baselines["direct_action"] = summarize(direct_rows)
    baselines["stateful_trust"] = summarize(stateful_rows)

    rows = []
    best = None
    for phase_span in [12, 18, 24, 30]:
        for reserve_target in [0.93, 0.95, 0.97]:
            for replay_count in [1, 2, 3, 4]:
                for replay_mode in ["target", "weakest", "hybrid"]:
                    seeds = []
                    for offset in range(int(args.num_seeds)):
                        seeds.append(
                            run_stateful_goal_system(
                                seed=int(args.seed) + offset,
                                noise=float(args.noise),
                                dropout_p=float(args.dropout_p),
                                missing_modality_p=float(args.missing_modality_p),
                                drift_scale=float(args.drift_scale),
                                margin_threshold=float(args.margin_threshold),
                                phase_span=int(phase_span),
                                switch_probe_count=3,
                                action_beta=float(stateful_best["action_beta"]),
                                correction_mix=float(stateful_best["correction_mix"]),
                                trust_temp=float(stateful_best["trust_temp"]),
                                reserve_target=float(reserve_target),
                                replay_count=int(replay_count),
                                enable_replay=True,
                                replay_mode=str(replay_mode),
                            )
                        )
                    summary = summarize(seeds)
                    row = {
                        "phase_span": int(phase_span),
                        "reserve_target": float(reserve_target),
                        "replay_count": int(replay_count),
                        "replay_mode": str(replay_mode),
                        **summary,
                        "loop_score_gain_vs_direct": float(summary["long_horizon_loop_score"] - baselines["direct_action"]["long_horizon_loop_score"]),
                        "loop_score_gain_vs_stateful": float(summary["long_horizon_loop_score"] - baselines["stateful_trust"]["long_horizon_loop_score"]),
                        "phase_switch_gain_vs_direct": float(summary["phase_switch_accuracy"] - baselines["direct_action"]["phase_switch_accuracy"]),
                        "retention_gain_vs_direct": float(summary["old_concept_retention"] - baselines["direct_action"]["old_concept_retention"]),
                    }
                    rows.append(row)
                    objective = -(row["loop_score_gain_vs_direct"]) - 0.5 * row["phase_switch_gain_vs_direct"] - 0.4 * row["retention_gain_vs_direct"]
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
                "open_world_grounding_action_loop_goal_state_scan_20260310.json",
            ],
        },
        "baseline_direct_action": baselines["direct_action"],
        "baseline_stateful_trust": baselines["stateful_trust"],
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一版把旧概念保留目标扩成真正的长期多步目标维持，加入重复 family 阶段切换和切换后 probe，直接测量开放世界代理在更长目标链上的闭环稳定性。",
            "next_question": "如果长期目标状态在 repeated family switching 上还能保持正增益，下一步就该把阶段目标推进到真正的规划子目标和更长地平线。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
