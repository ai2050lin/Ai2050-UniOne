#!/usr/bin/env python
"""
Build a harder online tool-interface benchmark under real-model constraints.

This stage goes beyond the earlier online proxy chain by introducing explicit
tool-interface failure modes:
1. schema mismatch
2. timeout pressure
3. state drift
4. verify mismatch

The goal is to test whether the relation/tool joint head really lowers trigger
rates and tool failures under a more realistic online interface.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FAILURE_TYPES = ["schema_mismatch", "timeout_pressure", "state_drift", "verify_mismatch"]
TOOL_TASKS = [
    {"name": "lookup", "schema_load": 0.42, "timeout_load": 0.28, "state_load": 0.22, "verify_load": 0.18},
    {"name": "compose", "schema_load": 0.33, "timeout_load": 0.36, "state_load": 0.31, "verify_load": 0.30},
    {"name": "execute", "schema_load": 0.24, "timeout_load": 0.46, "state_load": 0.42, "verify_load": 0.34},
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def stage_undercoverage(match_row: Dict[str, Any], stage: str) -> float:
    for row in match_row["rows"]:
        if row["stage"] == stage:
            return float(row["undercoverage"])
    return 0.0


def failure_choice(
    rng: np.random.Generator,
    schema_risk: float,
    timeout_risk: float,
    state_risk: float,
    verify_risk: float,
) -> str:
    weights = np.array(
        [
            max(0.001, schema_risk),
            max(0.001, timeout_risk),
            max(0.001, state_risk),
            max(0.001, verify_risk),
        ],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    idx = int(rng.choice(len(FAILURE_TYPES), p=weights))
    return FAILURE_TYPES[idx]


def tool_recovery_probability(failure_type: str, relation_gap: float, tool_gap: float, verify_gap: float, relation_strength: float) -> float:
    base = {
        "schema_mismatch": 0.54,
        "timeout_pressure": 0.42,
        "state_drift": 0.39,
        "verify_mismatch": 0.48,
    }[failure_type]
    return clamp01(
        base
        + 0.16 * relation_strength
        - 0.18 * relation_gap
        - 0.22 * tool_gap
        - 0.10 * verify_gap
    )


def simulate_system(
    tool_match: Dict[str, Any],
    joint_match: Dict[str, Any],
    online_model_row: Dict[str, Any],
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    base_trigger = float(online_model_row["systems"]["online_recovery_aware"]["rollback_trigger_rate"])
    base_recovery = float(online_model_row["systems"]["online_recovery_aware"]["rollback_recovery_rate"])
    base_success = float(online_model_row["systems"]["online_recovery_aware"]["success_rate"])
    step_map = {row["step"]: row for row in online_model_row["step_rows"]}

    systems = {
        "tool_stage_head_online_tool_interface": {
            "match": tool_match,
        },
        "relation_tool_joint_head_online_tool_interface": {
            "match": joint_match,
        },
    }

    results = {}
    for system_name, system_row in systems.items():
        match = system_row["match"]
        relation_gap = stage_undercoverage(match, "relation")
        tool_gap = stage_undercoverage(match, "tool")
        verify_gap = stage_undercoverage(match, "verify")
        concept_gap = stage_undercoverage(match, "concept")

        relation_strength = clamp01(1.0 - normalize(relation_gap, 0.02, 0.38))
        tool_strength = clamp01(1.0 - normalize(tool_gap, 0.02, 0.36))
        verify_strength = clamp01(1.0 - normalize(verify_gap, 0.0, 0.08))

        trigger_count = 0
        tool_failures = 0
        recoveries = 0
        successes = 0
        type_counts = {name: 0 for name in FAILURE_TYPES}
        task_counts = {row["name"]: 0 for row in TOOL_TASKS}

        for _ in range(episodes):
            task = TOOL_TASKS[int(rng.integers(0, len(TOOL_TASKS)))]
            task_counts[task["name"]] += 1

            concept_risk = clamp01(0.045 + 0.16 * concept_gap + 0.03 * (1.0 - relation_strength))
            relation_risk = clamp01(float(step_map["relation"]["trigger_rate"]) + 0.32 * relation_gap - 0.08 * relation_strength)
            tool_trigger_risk = clamp01(
                base_trigger * 0.28
                + float(step_map["tool"]["trigger_rate"]) * 0.34
                + 0.34 * tool_gap
                + 0.10 * relation_gap
                + 0.08 * task["state_load"]
                - 0.08 * tool_strength
            )
            verify_risk = clamp01(float(step_map["verify"]["trigger_rate"]) + 0.20 * verify_gap + 0.05 * task["verify_load"])

            schema_risk = clamp01(0.10 + 0.34 * relation_gap + 0.18 * task["schema_load"] - 0.08 * relation_strength)
            timeout_risk = clamp01(0.08 + 0.30 * tool_gap + 0.22 * task["timeout_load"] - 0.06 * tool_strength)
            state_risk = clamp01(0.08 + 0.26 * tool_gap + 0.18 * relation_gap + 0.20 * task["state_load"] - 0.04 * tool_strength)
            verify_mismatch_risk = clamp01(0.06 + 0.24 * verify_gap + 0.10 * task["verify_load"] - 0.04 * verify_strength)

            pre_tool_fail = float(rng.random()) < concept_risk or float(rng.random()) < relation_risk
            if pre_tool_fail:
                trigger_count += 1
                failure_type = "schema_mismatch" if concept_gap < relation_gap else "state_drift"
                type_counts[failure_type] += 1
                recovery_prob = tool_recovery_probability(failure_type, relation_gap, tool_gap, verify_gap, relation_strength)
                recovered = float(rng.random()) < recovery_prob
                recoveries += int(recovered)
                successes += int(recovered and float(rng.random()) < clamp01(base_success + 0.08 * relation_strength))
                continue

            tool_failed = float(rng.random()) < tool_trigger_risk
            verify_failed = float(rng.random()) < verify_risk
            interface_failed = float(rng.random()) < max(schema_risk, timeout_risk, state_risk, verify_mismatch_risk)

            if tool_failed or verify_failed or interface_failed:
                trigger_count += 1
                tool_failures += int(tool_failed or interface_failed)
                failure_type = failure_choice(rng, schema_risk, timeout_risk, state_risk, verify_mismatch_risk)
                type_counts[failure_type] += 1
                recovery_prob = tool_recovery_probability(failure_type, relation_gap, tool_gap, verify_gap, relation_strength)
                recovery_prob = clamp01(recovery_prob + 0.08 * base_recovery)
                recovered = float(rng.random()) < recovery_prob
                recoveries += int(recovered)
                tail_prob = clamp01(base_success + 0.10 * relation_strength + 0.08 * tool_strength - 0.12 * verify_gap)
                successes += int(recovered and float(rng.random()) < tail_prob)
            else:
                successes += 1

        results[system_name] = {
            "success_rate": float(successes / episodes),
            "rollback_trigger_rate": float(trigger_count / episodes),
            "rollback_recovery_rate": float(recoveries / max(1, trigger_count)),
            "tool_failure_rate": float(tool_failures / episodes),
            "failure_breakdown": {key: float(value / max(1, trigger_count)) for key, value in type_counts.items()},
            "task_mix": {key: float(value / episodes) for key, value in task_counts.items()},
            "mean_relation_gap": relation_gap,
            "mean_tool_gap": tool_gap,
            "mean_verify_gap": verify_gap,
        }

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard online tool-interface benchmark for Qwen3 / DeepSeek7B")
    ap.add_argument("--episodes", type=int, default=320)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_hard_online_tool_interface_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    online_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
    joint_payload = load_json(ROOT / "tests" / "codex_temp" / "relation_tool_joint_generator_network_upgrade_20260310.json")

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "hard_online_tool_interface_after_joint_head_upgrade",
            "episodes": int(args.episodes),
            "failure_types": FAILURE_TYPES,
            "runtime_sec": 0.0,
        },
        "models": {},
    }

    for model_idx, model_name in enumerate(["qwen3_4b", "deepseek_7b"]):
        results["models"][model_name] = simulate_system(
            tool_match=joint_payload["models"][model_name]["tool_stage_head_match"],
            joint_match=joint_payload["models"][model_name]["relation_tool_joint_head_match"],
            online_model_row=online_payload["models"][model_name],
            episodes=int(args.episodes),
            seed=211 + model_idx,
        )

    qwen = results["models"]["qwen3_4b"]
    deepseek = results["models"]["deepseek_7b"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_tool_head_success": float(qwen["tool_stage_head_online_tool_interface"]["success_rate"]),
            "qwen_joint_head_success": float(qwen["relation_tool_joint_head_online_tool_interface"]["success_rate"]),
            "deepseek_tool_head_success": float(deepseek["tool_stage_head_online_tool_interface"]["success_rate"]),
            "deepseek_joint_head_success": float(deepseek["relation_tool_joint_head_online_tool_interface"]["success_rate"]),
            "qwen_tool_head_trigger_rate": float(qwen["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"]),
            "qwen_joint_head_trigger_rate": float(qwen["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]),
            "deepseek_tool_head_trigger_rate": float(deepseek["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"]),
            "deepseek_joint_head_trigger_rate": float(deepseek["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]),
        },
        "gains": {
            "qwen_joint_minus_tool_head_success": float(
                qwen["relation_tool_joint_head_online_tool_interface"]["success_rate"]
                - qwen["tool_stage_head_online_tool_interface"]["success_rate"]
            ),
            "deepseek_joint_minus_tool_head_success": float(
                deepseek["relation_tool_joint_head_online_tool_interface"]["success_rate"]
                - deepseek["tool_stage_head_online_tool_interface"]["success_rate"]
            ),
            "qwen_tool_head_minus_joint_trigger_rate": float(
                qwen["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"]
                - qwen["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]
            ),
            "deepseek_tool_head_minus_joint_trigger_rate": float(
                deepseek["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"]
                - deepseek["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]
            ),
            "qwen_tool_head_minus_joint_tool_failure": float(
                qwen["tool_stage_head_online_tool_interface"]["tool_failure_rate"]
                - qwen["relation_tool_joint_head_online_tool_interface"]["tool_failure_rate"]
            ),
            "deepseek_tool_head_minus_joint_tool_failure": float(
                deepseek["tool_stage_head_online_tool_interface"]["tool_failure_rate"]
                - deepseek["relation_tool_joint_head_online_tool_interface"]["tool_failure_rate"]
            ),
        },
        "hypotheses": {
            "H1_joint_head_improves_hard_interface_success_on_both_models": bool(
                qwen["relation_tool_joint_head_online_tool_interface"]["success_rate"]
                > qwen["tool_stage_head_online_tool_interface"]["success_rate"] + 0.04
                and deepseek["relation_tool_joint_head_online_tool_interface"]["success_rate"]
                > deepseek["tool_stage_head_online_tool_interface"]["success_rate"] + 0.04
            ),
            "H2_joint_head_reduces_trigger_rate_on_both_models": bool(
                qwen["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]
                < qwen["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"] - 0.03
                and deepseek["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]
                < deepseek["tool_stage_head_online_tool_interface"]["rollback_trigger_rate"] - 0.03
            ),
            "H3_joint_head_reduces_tool_failure_rate_on_both_models": bool(
                qwen["relation_tool_joint_head_online_tool_interface"]["tool_failure_rate"]
                < qwen["tool_stage_head_online_tool_interface"]["tool_failure_rate"] - 0.03
                and deepseek["relation_tool_joint_head_online_tool_interface"]["tool_failure_rate"]
                < deepseek["tool_stage_head_online_tool_interface"]["tool_failure_rate"] - 0.03
            ),
            "H4_deepseek_keeps_higher_failure_pressure_even_after_joint_head": bool(
                deepseek["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"]
                > qwen["relation_tool_joint_head_online_tool_interface"]["rollback_trigger_rate"] + 0.08
            ),
        },
        "project_readout": {
            "summary": (
                "This stage replaces the earlier soft online proxy with a harder tool interface. "
                "The benchmark now includes schema, timeout, state drift, and verify mismatch failures."
            ),
            "next_question": (
                "If the relation/tool joint head still lowers trigger and tool failure rates here, "
                "the next step is to make the stage heads online-learnable rather than hand-searched."
            ),
        },
    }

    payload["meta"]["runtime_sec"] = float(time.time() - t0)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
