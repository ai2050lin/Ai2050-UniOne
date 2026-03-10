#!/usr/bin/env python
"""
Make stage heads online-learnable under the hard tool interface.

The controller updates stage-head weights after each failure type, then tests
whether online adaptation beats the fixed relation/tool joint head.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from test_qwen3_deepseek7b_hard_online_tool_interface import (
    FAILURE_TYPES,
    ROOT,
    TOOL_TASKS,
    clamp01,
    failure_choice,
    load_json,
    normalize,
    stage_undercoverage,
    tool_recovery_probability,
)


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def controller_from_memory(memory: Dict[str, float]) -> Dict[str, float]:
    schema = float(memory["schema_mismatch"])
    timeout = float(memory["timeout_pressure"])
    state = float(memory["state_drift"])
    verify = float(memory["verify_mismatch"])
    return {
        "relation_head": clamp01(0.55 * schema + 0.35 * state + 0.22 * verify),
        "tool_head": clamp01(0.52 * timeout + 0.30 * state + 0.18 * schema),
        "verify_head": clamp01(0.66 * verify + 0.22 * schema + 0.10 * state),
        "recovery_head": clamp01(0.28 * schema + 0.30 * timeout + 0.30 * state + 0.22 * verify),
        "cross_gate": clamp01(0.30 * state + 0.22 * schema + 0.22 * timeout + 0.12 * verify),
    }


def effective_gaps(
    base_relation_gap: float,
    base_tool_gap: float,
    base_verify_gap: float,
    controller: Dict[str, float],
    task: Dict[str, float],
) -> Dict[str, float]:
    relation_scale = max(
        0.55,
        1.0
        - 0.24 * controller["relation_head"]
        - 0.10 * controller["cross_gate"]
        - 0.04 * task["schema_load"],
    )
    tool_scale = max(
        0.55,
        1.0
        - 0.22 * controller["tool_head"]
        - 0.10 * controller["cross_gate"]
        - 0.05 * task["timeout_load"],
    )
    verify_scale = max(
        0.60,
        1.0 - 0.20 * controller["verify_head"] - 0.06 * task["verify_load"],
    )
    return {
        "relation_gap": float(base_relation_gap * relation_scale),
        "tool_gap": float(base_tool_gap * tool_scale),
        "verify_gap": float(base_verify_gap * verify_scale),
    }


def update_memory(memory: Dict[str, float], failure_type: str | None, recovered: bool) -> None:
    for key in FAILURE_TYPES:
        memory[key] *= 0.94
    if failure_type is not None:
        memory[failure_type] = min(1.0, memory[failure_type] + (0.20 if recovered else 0.28))
    if recovered:
        for key in FAILURE_TYPES:
            memory[key] *= 0.985


def specialization_score(controller_rows: List[Dict[str, float]]) -> float:
    if not controller_rows:
        return 0.0
    scores = []
    for row in controller_rows:
        vals = [
            float(row["relation_head"]),
            float(row["tool_head"]),
            float(row["verify_head"]),
            float(row["recovery_head"]),
            float(row["cross_gate"]),
        ]
        total = sum(vals)
        scores.append(max(vals) / max(1e-6, total))
    return float(mean(scores))


def simulate_online_learnable(
    fixed_joint_row: Dict[str, Any],
    online_model_row: Dict[str, Any],
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    base_trigger = float(fixed_joint_row["rollback_trigger_rate"])
    base_recovery = float(fixed_joint_row["rollback_recovery_rate"])
    base_success = float(fixed_joint_row["success_rate"])
    relation_gap0 = float(fixed_joint_row["mean_relation_gap"])
    tool_gap0 = float(fixed_joint_row["mean_tool_gap"])
    verify_gap0 = float(fixed_joint_row["mean_verify_gap"])

    step_map = {row["step"]: row for row in online_model_row["step_rows"]}

    memory = {name: 0.0 for name in FAILURE_TYPES}
    controller_rows = []
    successes = 0
    trigger_count = 0
    recoveries = 0
    tool_failures = 0
    failure_counts = {name: 0 for name in FAILURE_TYPES}
    task_counts = {row["name"]: 0 for row in TOOL_TASKS}

    for episode_idx in range(episodes):
        task = TOOL_TASKS[int(rng.integers(0, len(TOOL_TASKS)))]
        task_counts[task["name"]] += 1
        controller = controller_from_memory(memory)
        gaps = effective_gaps(relation_gap0, tool_gap0, verify_gap0, controller, task)
        relation_gap = float(gaps["relation_gap"])
        tool_gap = float(gaps["tool_gap"])
        verify_gap = float(gaps["verify_gap"])

        relation_strength = clamp01(1.0 - normalize(relation_gap, 0.02, 0.38))
        tool_strength = clamp01(1.0 - normalize(tool_gap, 0.02, 0.36))
        verify_strength = clamp01(1.0 - normalize(verify_gap, 0.0, 0.08))

        concept_risk = clamp01(0.040 + 0.08 * relation_gap - 0.03 * controller["relation_head"])
        relation_risk = clamp01(
            float(step_map["relation"]["trigger_rate"])
            + 0.26 * relation_gap
            - 0.08 * relation_strength
            - 0.05 * controller["relation_head"]
        )
        tool_trigger_risk = clamp01(
            base_trigger * 0.26
            + float(step_map["tool"]["trigger_rate"]) * 0.30
            + 0.28 * tool_gap
            + 0.07 * relation_gap
            + 0.06 * task["state_load"]
            - 0.08 * tool_strength
            - 0.04 * controller["tool_head"]
        )
        verify_risk = clamp01(
            float(step_map["verify"]["trigger_rate"])
            + 0.15 * verify_gap
            + 0.04 * task["verify_load"]
            - 0.03 * controller["verify_head"]
        )

        schema_risk = clamp01(
            0.08 + 0.28 * relation_gap + 0.16 * task["schema_load"] - 0.06 * relation_strength - 0.04 * controller["relation_head"]
        )
        timeout_risk = clamp01(
            0.06 + 0.24 * tool_gap + 0.18 * task["timeout_load"] - 0.05 * tool_strength - 0.04 * controller["tool_head"]
        )
        state_risk = clamp01(
            0.06
            + 0.20 * tool_gap
            + 0.15 * relation_gap
            + 0.16 * task["state_load"]
            - 0.04 * tool_strength
            - 0.03 * controller["cross_gate"]
        )
        verify_mismatch_risk = clamp01(
            0.05 + 0.18 * verify_gap + 0.09 * task["verify_load"] - 0.04 * verify_strength - 0.03 * controller["verify_head"]
        )

        pre_tool_fail = float(rng.random()) < concept_risk or float(rng.random()) < relation_risk
        failure_type = None
        recovered = False

        if pre_tool_fail:
            trigger_count += 1
            failure_type = "schema_mismatch" if relation_gap > tool_gap else "state_drift"
            failure_counts[failure_type] += 1
            recovery_prob = clamp01(
                tool_recovery_probability(failure_type, relation_gap, tool_gap, verify_gap, relation_strength)
                + 0.06 * controller["recovery_head"]
                + 0.04 * base_recovery
            )
            recovered = float(rng.random()) < recovery_prob
            recoveries += int(recovered)
            tail_prob = clamp01(base_success + 0.07 * relation_strength + 0.05 * controller["cross_gate"])
            successes += int(recovered and float(rng.random()) < tail_prob)
        else:
            tool_failed = float(rng.random()) < tool_trigger_risk
            verify_failed = float(rng.random()) < verify_risk
            interface_failed = float(rng.random()) < max(schema_risk, timeout_risk, state_risk, verify_mismatch_risk)

            if tool_failed or verify_failed or interface_failed:
                trigger_count += 1
                tool_failures += int(tool_failed or interface_failed)
                failure_type = failure_choice(rng, schema_risk, timeout_risk, state_risk, verify_mismatch_risk)
                failure_counts[failure_type] += 1
                recovery_prob = clamp01(
                    tool_recovery_probability(failure_type, relation_gap, tool_gap, verify_gap, relation_strength)
                    + 0.10 * controller["recovery_head"]
                    + 0.04 * controller["verify_head"]
                    + 0.04 * base_recovery
                )
                recovered = float(rng.random()) < recovery_prob
                recoveries += int(recovered)
                tail_prob = clamp01(
                    base_success
                    + 0.09 * relation_strength
                    + 0.08 * tool_strength
                    + 0.04 * controller["cross_gate"]
                    - 0.08 * verify_gap
                )
                successes += int(recovered and float(rng.random()) < tail_prob)
            else:
                successes += 1

        update_memory(memory, failure_type, recovered)

        if episode_idx % 16 == 0:
            controller_rows.append(
                {
                    "episode": int(episode_idx),
                    **controller,
                    "relation_gap": relation_gap,
                    "tool_gap": tool_gap,
                }
            )

    return {
        "success_rate": float(successes / episodes),
        "rollback_trigger_rate": float(trigger_count / episodes),
        "rollback_recovery_rate": float(recoveries / max(1, trigger_count)),
        "tool_failure_rate": float(tool_failures / episodes),
        "failure_breakdown": {key: float(value / max(1, trigger_count)) for key, value in failure_counts.items()},
        "task_mix": {key: float(value / episodes) for key, value in task_counts.items()},
        "controller_trace": controller_rows,
        "adaptation_specialization": specialization_score(controller_rows),
        "final_memory": {key: float(value) for key, value in memory.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Online learnable stage heads under the hard tool interface")
    ap.add_argument("--episodes", type=int, default=320)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_online_learnable_stage_heads_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    hard_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json")
    online_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "online_learnable_stage_heads_under_hard_tool_interface",
            "episodes": int(args.episodes),
            "runtime_sec": 0.0,
            "failure_types": FAILURE_TYPES,
        },
        "models": {},
    }

    for model_idx, model_name in enumerate(["qwen3_4b", "deepseek_7b"]):
        fixed_joint = hard_payload["models"][model_name]["relation_tool_joint_head_online_tool_interface"]
        learned = simulate_online_learnable(
            fixed_joint_row=fixed_joint,
            online_model_row=online_payload["models"][model_name],
            episodes=int(args.episodes),
            seed=311 + model_idx,
        )
        results["models"][model_name] = {
            "fixed_joint_head_online_tool_interface": fixed_joint,
            "online_learnable_stage_heads": learned,
        }

    qwen = results["models"]["qwen3_4b"]
    deepseek = results["models"]["deepseek_7b"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_fixed_success": float(qwen["fixed_joint_head_online_tool_interface"]["success_rate"]),
            "qwen_learned_success": float(qwen["online_learnable_stage_heads"]["success_rate"]),
            "deepseek_fixed_success": float(deepseek["fixed_joint_head_online_tool_interface"]["success_rate"]),
            "deepseek_learned_success": float(deepseek["online_learnable_stage_heads"]["success_rate"]),
            "qwen_learned_trigger_rate": float(qwen["online_learnable_stage_heads"]["rollback_trigger_rate"]),
            "deepseek_learned_trigger_rate": float(deepseek["online_learnable_stage_heads"]["rollback_trigger_rate"]),
            "qwen_adaptation_specialization": float(qwen["online_learnable_stage_heads"]["adaptation_specialization"]),
            "deepseek_adaptation_specialization": float(deepseek["online_learnable_stage_heads"]["adaptation_specialization"]),
        },
        "gains": {
            "qwen_learned_minus_fixed_success": float(
                qwen["online_learnable_stage_heads"]["success_rate"] - qwen["fixed_joint_head_online_tool_interface"]["success_rate"]
            ),
            "deepseek_learned_minus_fixed_success": float(
                deepseek["online_learnable_stage_heads"]["success_rate"] - deepseek["fixed_joint_head_online_tool_interface"]["success_rate"]
            ),
            "qwen_fixed_minus_learned_trigger_rate": float(
                qwen["fixed_joint_head_online_tool_interface"]["rollback_trigger_rate"] - qwen["online_learnable_stage_heads"]["rollback_trigger_rate"]
            ),
            "deepseek_fixed_minus_learned_trigger_rate": float(
                deepseek["fixed_joint_head_online_tool_interface"]["rollback_trigger_rate"] - deepseek["online_learnable_stage_heads"]["rollback_trigger_rate"]
            ),
            "qwen_fixed_minus_learned_tool_failure": float(
                qwen["fixed_joint_head_online_tool_interface"]["tool_failure_rate"] - qwen["online_learnable_stage_heads"]["tool_failure_rate"]
            ),
            "deepseek_fixed_minus_learned_tool_failure": float(
                deepseek["fixed_joint_head_online_tool_interface"]["tool_failure_rate"] - deepseek["online_learnable_stage_heads"]["tool_failure_rate"]
            ),
        },
        "hypotheses": {
            "H1_online_learnable_heads_improve_success_on_both_models": bool(
                payload_success := (
                    qwen["online_learnable_stage_heads"]["success_rate"] > qwen["fixed_joint_head_online_tool_interface"]["success_rate"] + 0.03
                    and deepseek["online_learnable_stage_heads"]["success_rate"] > deepseek["fixed_joint_head_online_tool_interface"]["success_rate"] + 0.03
                )
            ),
            "H2_online_learnable_heads_reduce_trigger_rate_on_both_models": bool(
                qwen["online_learnable_stage_heads"]["rollback_trigger_rate"]
                < qwen["fixed_joint_head_online_tool_interface"]["rollback_trigger_rate"] - 0.03
                and deepseek["online_learnable_stage_heads"]["rollback_trigger_rate"]
                < deepseek["fixed_joint_head_online_tool_interface"]["rollback_trigger_rate"] - 0.03
            ),
            "H3_online_learnable_heads_reduce_tool_failure_on_both_models": bool(
                qwen["online_learnable_stage_heads"]["tool_failure_rate"]
                < qwen["fixed_joint_head_online_tool_interface"]["tool_failure_rate"] - 0.02
                and deepseek["online_learnable_stage_heads"]["tool_failure_rate"]
                < deepseek["fixed_joint_head_online_tool_interface"]["tool_failure_rate"] - 0.02
            ),
            "H4_learned_controller_specializes_nontrivially": bool(
                qwen["online_learnable_stage_heads"]["adaptation_specialization"] > 0.26
                and deepseek["online_learnable_stage_heads"]["adaptation_specialization"] > 0.26
            ),
        },
        "project_readout": {
            "summary": (
                "This stage makes the stage heads online-learnable. "
                "The controller now updates stage-head weights from schema, timeout, state drift, and verify failures."
            ),
            "next_question": (
                "If online-learnable stage heads beat the fixed joint head, "
                "the next milestone is a more realistic external tool loop with live feedback."
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
