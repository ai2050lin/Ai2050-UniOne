#!/usr/bin/env python
"""
Goal-state scan for the open-world grounding action loop.

Goal:
- add a long-term retention goal state on top of the stateful action-trust agent
- periodically replay the most under-trusted old-family reserve
- test whether this can close the remaining gap against the direct baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench
import test_open_world_grounding_action_loop as action_loop
import test_open_world_grounding_action_loop_stateful_scan as stateful_scan


ROOT = Path(__file__).resolve().parents[2]


def build_old_buffers(seed: int) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    buffers: Dict[str, List[str]] = {}
    for family in stream_bench.FAMILIES:
        concepts = list(stream_bench.PHASE1[family])
        rng.shuffle(concepts)
        buffers[family] = concepts
    return buffers


def replay_old_family(
    agent: stateful_scan.StatefulSharedActionAgent,
    rng: np.random.Generator,
    family: str,
    replay_count: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    old_buffers: Dict[str, List[str]],
) -> None:
    concepts = old_buffers[family]
    for idx in range(replay_count):
        concept = concepts[idx % len(concepts)]
        x = stream_bench.sample_stream_input(
            rng,
            concept,
            noise=noise * 0.85,
            dropout_p=dropout_p * 0.9,
            missing_modality_p=missing_modality_p * 0.9,
            drift_scale=drift_scale * 0.7,
        )
        agent.train(x, family, concept)


def run_goal_state_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    action_beta: float,
    correction_mix: float,
    trust_temp: float,
    reserve_target: float,
    replay_count: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = stateful_scan.StatefulSharedActionAgent(
        family_alpha=0.05,
        offset_alpha=0.32,
        action_beta=action_beta,
        correction_mix=correction_mix,
        trust_temp=trust_temp,
    )
    old_buffers = build_old_buffers(seed)
    dim = 24
    stream = stream_bench.build_stream(seed)

    action_ok = 0
    corrected_action_ok = 0
    action_total = 0
    update_count = 0
    replay_updates = 0

    for item in stream:
        if item["kind"] == "concept":
            x = stream_bench.sample_stream_input(
                rng,
                item["concept"],
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
            )
            agent.train(x, item["family"], item["concept"])
            update_count += 1

            pred_family, _pred_concept, _margin = agent.predict_with_margin(x)
            first_ok = action_loop.action_for_family(pred_family) == action_loop.action_for_family(item["family"])
            corrected_family, _used = agent.corrected_family(x, margin_threshold=margin_threshold)
            corrected_ok = action_loop.action_for_family(corrected_family) == action_loop.action_for_family(item["family"])
            action_ok += int(first_ok)
            corrected_action_ok += int(corrected_ok)
            action_total += 1

            if item["segment"] in {"novel", "novel_noise", "revisit"}:
                weakest_family = min(agent.family_trust, key=lambda fam: agent.family_trust[fam])
                if agent.family_trust[weakest_family] < reserve_target:
                    replay_old_family(
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
                    replay_updates += replay_count
        else:
            _ = stream_bench.sample_noise_chunk(rng, dim=dim, noise_scale=noise * 1.2)
            weakest_family = min(agent.family_trust, key=lambda fam: agent.family_trust[fam])
            if agent.family_trust[weakest_family] < reserve_target:
                replay_old_family(
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
                replay_updates += replay_count

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    action_accuracy = float(action_ok / max(1, action_total))
    corrected_action_accuracy = float(corrected_action_ok / max(1, action_total))
    loop_score = float((1.3 * corrected_action_accuracy + 1.0 * action_accuracy + 1.1 * old_retention) / 3.4)

    return {
        "action_accuracy": action_accuracy,
        "corrected_action_accuracy": corrected_action_accuracy,
        "old_concept_retention": old_retention,
        "loop_score": loop_score,
        "mean_family_trust": float(np.mean(list(agent.family_trust.values()))),
        "update_count": float(update_count),
        "replay_updates": float(replay_updates),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan goal-state replay for the open-world grounding action loop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_grounding_action_loop_goal_state_scan_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_20260310.json").read_text(encoding="utf-8"))
    stateful_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_stateful_scan_20260310.json").read_text(encoding="utf-8"))
    direct = baseline_payload["systems"]["direct_action"]
    stateful_best = stateful_payload["best_config"]

    rows = []
    best = None
    for reserve_target in [0.90, 0.93, 0.95, 0.97]:
        for replay_count in [1, 2, 3, 4]:
            seeds = []
            for offset in range(int(args.num_seeds)):
                seeds.append(
                    run_goal_state_system(
                        seed=int(args.seed) + offset,
                        noise=float(args.noise),
                        dropout_p=float(args.dropout_p),
                        missing_modality_p=float(args.missing_modality_p),
                        drift_scale=float(args.drift_scale),
                        margin_threshold=float(args.margin_threshold),
                        action_beta=float(stateful_best["action_beta"]),
                        correction_mix=float(stateful_best["correction_mix"]),
                        trust_temp=float(stateful_best["trust_temp"]),
                        reserve_target=float(reserve_target),
                        replay_count=int(replay_count),
                    )
                )
            summary = summarize(seeds)
            row = {
                "reserve_target": float(reserve_target),
                "replay_count": int(replay_count),
                **summary,
                "loop_score_gain_vs_direct": float(summary["loop_score"] - direct["loop_score"]),
                "loop_score_gain_vs_stateful": float(summary["loop_score"] - stateful_best["loop_score"]),
                "retention_gain_vs_direct": float(summary["old_concept_retention"] - direct["old_concept_retention"]),
                "corrected_action_gain_vs_direct": float(summary["corrected_action_accuracy"] - direct["corrected_action_accuracy"]),
            }
            rows.append(row)
            objective = -(row["loop_score_gain_vs_direct"]) - 0.5 * row["retention_gain_vs_direct"]
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
                "open_world_grounding_action_loop_20260310.json",
                "open_world_grounding_action_loop_stateful_scan_20260310.json",
            ],
        },
        "baseline_direct_action": direct,
        "baseline_stateful_best": stateful_best,
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一版把长期目标/保留状态并进动作回路，用旧概念回放储备来测试代理断点是否主要来自长期保留状态缺失。",
            "next_question": "如果 goal-state replay 能翻正 loop_score，下一步就该把长期多步目标也并入同一状态。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
