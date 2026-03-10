#!/usr/bin/env python
"""
Upgrade the generator with a joint relation/tool head on top of the existing
tool-stage head, then test whether the new relation bottleneck can be reduced
without reopening the tool-stage gap.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from test_generator_network_real_layer_band_bridge import (
    ROOT,
    STAGES,
    compare_capacity_to_demand,
    load_json,
    load_spec_from_payload,
    mean,
    stage_demands,
)
from test_tool_stage_generator_network_upgrade import (
    ToolHeadUpgradeSpec,
    clamp_floor,
    hidden_means,
    non_tool_alignment,
    tool_undercoverage,
    tool_upgraded_stage_capacities,
)


@dataclass
class RelationToolJointHeadSpec:
    relation_head_gain: float
    tool_head_gain: float
    cross_gate_gain: float
    verify_guard_gain: float
    concept_guard_gain: float


def relation_undercoverage(match_row: Dict[str, object]) -> float:
    for row in match_row["rows"]:
        if row["stage"] == "relation":
            return float(row["undercoverage"])
    return 0.0


def joint_upgraded_stage_capacities(base_capacity: Dict[str, float], hidden_mean: List[float], upgrade: RelationToolJointHeadSpec) -> Dict[str, float]:
    relation_focus = max(hidden_mean[1], 0.0) + 0.70 * max(hidden_mean[2], 0.0) + 0.25 * max(hidden_mean[3], 0.0)
    tool_focus = max(hidden_mean[3], 0.0) + 0.55 * max(hidden_mean[2], 0.0)
    verify_focus = max(hidden_mean[4], 0.0) + 0.20 * max(hidden_mean[3], 0.0)
    concept_focus = max(hidden_mean[0], 0.0)

    relation_boost = (
        0.036 * upgrade.relation_head_gain
        + 0.014 * upgrade.tool_head_gain
        + 0.024 * upgrade.cross_gate_gain
        + 0.018 * relation_focus
    )
    tool_delta = (
        0.014 * upgrade.tool_head_gain
        + 0.008 * upgrade.relation_head_gain
        + 0.012 * upgrade.cross_gate_gain
        + 0.012 * tool_focus
    )
    verify_delta = 0.010 * upgrade.verify_guard_gain + 0.006 * upgrade.cross_gate_gain + 0.006 * verify_focus
    concept_delta = 0.004 * upgrade.concept_guard_gain - 0.002 * upgrade.relation_head_gain + 0.003 * concept_focus

    return {
        "concept": clamp_floor(base_capacity["concept"] + concept_delta),
        "relation": clamp_floor(base_capacity["relation"] + relation_boost),
        "tool": clamp_floor(base_capacity["tool"] + tool_delta),
        "verify": clamp_floor(base_capacity["verify"] + verify_delta),
    }


def score_candidate(
    tool_head_matches: Dict[str, Dict[str, object]],
    joint_matches: Dict[str, Dict[str, object]],
) -> float:
    relation_gain = mean(
        [
            relation_undercoverage(tool_head_matches[model]) - relation_undercoverage(joint_matches[model])
            for model in tool_head_matches
        ]
    )
    mean_gain = mean(
        [
            float(tool_head_matches[model]["mean_undercoverage"]) - float(joint_matches[model]["mean_undercoverage"])
            for model in tool_head_matches
        ]
    )
    tool_regression = mean(
        [
            max(0.0, tool_undercoverage(joint_matches[model]) - tool_undercoverage(tool_head_matches[model]) - 0.010)
            for model in tool_head_matches
        ]
    )
    non_tool_penalty = mean(
        [
            max(0.0, non_tool_alignment(joint_matches[model]) - non_tool_alignment(tool_head_matches[model]) - 0.010)
            for model in tool_head_matches
        ]
    )
    return float(4.0 * relation_gain + 2.5 * mean_gain - 4.0 * tool_regression - 2.5 * non_tool_penalty)


def candidate_grid() -> List[RelationToolJointHeadSpec]:
    rows = []
    for relation_head_gain in [0.8, 1.0, 1.2, 1.4]:
        for tool_head_gain in [0.2, 0.4, 0.6, 0.8]:
            for cross_gate_gain in [0.4, 0.6, 0.8, 1.0]:
                for verify_guard_gain in [0.2, 0.4, 0.6]:
                    for concept_guard_gain in [0.2, 0.4]:
                        rows.append(
                            RelationToolJointHeadSpec(
                                relation_head_gain=relation_head_gain,
                                tool_head_gain=tool_head_gain,
                                cross_gate_gain=cross_gate_gain,
                                verify_guard_gain=verify_guard_gain,
                                concept_guard_gain=concept_guard_gain,
                            )
                        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Upgrade generator network with a joint relation/tool head")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/relation_tool_joint_generator_network_upgrade_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    searched_payload = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_region_family_generator_network_20260310.json")
    tool_payload = load_json(ROOT / "tests" / "codex_temp" / "tool_stage_generator_network_upgrade_20260310.json")
    structure_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    online_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    searched_spec = load_spec_from_payload(searched_payload["systems"]["generator_network_eval_family"])
    hidden_mean = hidden_means(searched_spec)
    tool_upgrade_spec = ToolHeadUpgradeSpec(
        tool_head_gain=float(tool_payload["generator_profiles"]["tool_stage_head_generator_network"]["upgrade_spec"]["tool_head_gain"]),
        relation_assist_gain=float(tool_payload["generator_profiles"]["tool_stage_head_generator_network"]["upgrade_spec"]["relation_assist_gain"]),
        verify_guard_gain=float(tool_payload["generator_profiles"]["tool_stage_head_generator_network"]["upgrade_spec"]["verify_guard_gain"]),
        concept_guard_gain=float(tool_payload["generator_profiles"]["tool_stage_head_generator_network"]["upgrade_spec"]["concept_guard_gain"]),
    )
    tool_head_capacity = tool_upgraded_stage_capacities(searched_spec, tool_upgrade_spec)

    stage_demands_by_model = {
        model_name: stage_demands(structure_payload["models"][model_name], online_payload["models"][model_name])
        for model_name in ["qwen3_4b", "deepseek_7b"]
    }
    tool_head_matches = {
        model_name: compare_capacity_to_demand(tool_head_capacity, stage_demands_by_model[model_name])
        for model_name in stage_demands_by_model
    }

    trials = []
    best = None
    for trial_id, candidate in enumerate(candidate_grid()):
        joint_capacity = joint_upgraded_stage_capacities(tool_head_capacity, hidden_mean, candidate)
        joint_matches = {
            model_name: compare_capacity_to_demand(joint_capacity, stage_demands_by_model[model_name])
            for model_name in stage_demands_by_model
        }
        score = score_candidate(tool_head_matches, joint_matches)
        trial = {
            "trial_id": trial_id,
            "relation_head_gain": candidate.relation_head_gain,
            "tool_head_gain": candidate.tool_head_gain,
            "cross_gate_gain": candidate.cross_gate_gain,
            "verify_guard_gain": candidate.verify_guard_gain,
            "concept_guard_gain": candidate.concept_guard_gain,
            "score": float(score),
            "qwen_relation_undercoverage": relation_undercoverage(joint_matches["qwen3_4b"]),
            "deepseek_relation_undercoverage": relation_undercoverage(joint_matches["deepseek_7b"]),
            "qwen_tool_undercoverage": tool_undercoverage(joint_matches["qwen3_4b"]),
            "deepseek_tool_undercoverage": tool_undercoverage(joint_matches["deepseek_7b"]),
        }
        trials.append(trial)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "trial_id": trial_id,
                "upgrade_spec": candidate,
                "stage_capacity": joint_capacity,
                "matches": joint_matches,
            }

    assert best is not None
    qwen = best["matches"]["qwen3_4b"]
    deepseek = best["matches"]["deepseek_7b"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "relation_tool_joint_head_upgrade_after_tool_stage_repair",
            "trial_count": len(trials),
            "stages": STAGES,
        },
        "generator_profiles": {
            "tool_stage_head_generator_network": {"stage_capacity": tool_head_capacity},
            "relation_tool_joint_head_generator_network": {
                "stage_capacity": best["stage_capacity"],
                "upgrade_spec": {
                    "relation_head_gain": best["upgrade_spec"].relation_head_gain,
                    "tool_head_gain": best["upgrade_spec"].tool_head_gain,
                    "cross_gate_gain": best["upgrade_spec"].cross_gate_gain,
                    "verify_guard_gain": best["upgrade_spec"].verify_guard_gain,
                    "concept_guard_gain": best["upgrade_spec"].concept_guard_gain,
                },
                "best_trial_id": int(best["trial_id"]),
            },
        },
        "models": {},
        "search_summary": {
            "top_trials": sorted(trials, key=lambda row: row["score"], reverse=True)[:8],
        },
    }

    for model_name in ["qwen3_4b", "deepseek_7b"]:
        payload["models"][model_name] = {
            "stage_demand": stage_demands_by_model[model_name],
            "tool_stage_head_match": tool_head_matches[model_name],
            "relation_tool_joint_head_match": best["matches"][model_name],
        }

    payload["headline_metrics"] = {
        "best_system": "relation_tool_joint_head_generator_network",
        "qwen_relation_undercoverage_gain": float(
            relation_undercoverage(tool_head_matches["qwen3_4b"]) - relation_undercoverage(qwen)
        ),
        "deepseek_relation_undercoverage_gain": float(
            relation_undercoverage(tool_head_matches["deepseek_7b"]) - relation_undercoverage(deepseek)
        ),
        "qwen_tool_regression_after_joint_head": float(
            tool_undercoverage(qwen) - tool_undercoverage(tool_head_matches["qwen3_4b"])
        ),
        "deepseek_tool_regression_after_joint_head": float(
            tool_undercoverage(deepseek) - tool_undercoverage(tool_head_matches["deepseek_7b"])
        ),
        "qwen_mean_undercoverage_gain": float(
            float(tool_head_matches["qwen3_4b"]["mean_undercoverage"]) - float(qwen["mean_undercoverage"])
        ),
        "deepseek_mean_undercoverage_gain": float(
            float(tool_head_matches["deepseek_7b"]["mean_undercoverage"]) - float(deepseek["mean_undercoverage"])
        ),
        "qwen_worst_stage_after_joint_head": qwen["worst_stage"],
        "deepseek_worst_stage_after_joint_head": deepseek["worst_stage"],
    }
    payload["gains"] = {
        "qwen_joint_minus_tool_head_relation_gap": float(
            relation_undercoverage(tool_head_matches["qwen3_4b"]) - relation_undercoverage(qwen)
        ),
        "deepseek_joint_minus_tool_head_relation_gap": float(
            relation_undercoverage(tool_head_matches["deepseek_7b"]) - relation_undercoverage(deepseek)
        ),
        "qwen_joint_minus_tool_head_mean_undercoverage": float(
            float(tool_head_matches["qwen3_4b"]["mean_undercoverage"]) - float(qwen["mean_undercoverage"])
        ),
        "deepseek_joint_minus_tool_head_mean_undercoverage": float(
            float(tool_head_matches["deepseek_7b"]["mean_undercoverage"]) - float(deepseek["mean_undercoverage"])
        ),
    }
    payload["hypotheses"] = {
        "H1_joint_head_reduces_qwen_relation_undercoverage": bool(
            relation_undercoverage(qwen) < relation_undercoverage(tool_head_matches["qwen3_4b"]) - 0.015
        ),
        "H2_joint_head_reduces_deepseek_relation_undercoverage": bool(
            relation_undercoverage(deepseek) < relation_undercoverage(tool_head_matches["deepseek_7b"]) - 0.020
        ),
        "H3_joint_head_keeps_tool_regression_bounded": bool(
            payload["headline_metrics"]["qwen_tool_regression_after_joint_head"] <= 0.01
            and payload["headline_metrics"]["deepseek_tool_regression_after_joint_head"] <= 0.01
        ),
        "H4_joint_head_improves_mean_undercoverage_on_both_models": bool(
            payload["headline_metrics"]["qwen_mean_undercoverage_gain"] > 0.01
            and payload["headline_metrics"]["deepseek_mean_undercoverage_gain"] > 0.01
        ),
    }
    payload["project_readout"] = {
        "summary": (
            "After the tool-stage repair, promote the generator upgrade to a joint relation/tool head. "
            "The goal is to reduce the new relation bottleneck without reopening the repaired tool-stage gap."
        ),
        "next_question": (
            "If the joint head keeps the tool gap bounded while shrinking the relation gap, "
            "the next step is a harder online tool interface with relation-aware triggers."
        ),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
