#!/usr/bin/env python
"""
Upgrade the region-family generator with a dedicated tool-stage head and test
whether it closes the real online layer-band gap without introducing large
non-tool regressions.
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
    stage_capacities,
    stage_demands,
)
from test_local_pulse_region_family_generator_network import EVAL_LATENTS, GeneratorNetworkSpec, hidden_state


@dataclass
class ToolHeadUpgradeSpec:
    tool_head_gain: float
    relation_assist_gain: float
    verify_guard_gain: float
    concept_guard_gain: float


def clamp_floor(value: float, floor: float = 0.0) -> float:
    return float(max(floor, value))


def hidden_means(spec: GeneratorNetworkSpec) -> List[float]:
    rows = [hidden_state(code, spec) for code in EVAL_LATENTS]
    return [mean([row[idx] for row in rows]) for idx in range(5)]


def tool_upgraded_stage_capacities(
    base_spec: GeneratorNetworkSpec,
    upgrade: ToolHeadUpgradeSpec,
) -> Dict[str, float]:
    base = stage_capacities(base_spec)
    hidden_mean = hidden_means(base_spec)

    tool_focus = max(hidden_mean[3], 0.0) + 0.75 * max(hidden_mean[2], 0.0) + 0.35 * max(hidden_mean[4], 0.0)
    relation_focus = max(hidden_mean[1], 0.0) + 0.40 * max(hidden_mean[2], 0.0)
    verify_focus = max(hidden_mean[4], 0.0) + 0.25 * max(hidden_mean[3], 0.0)
    concept_focus = max(hidden_mean[0], 0.0)

    tool_boost = (
        0.050 * upgrade.tool_head_gain
        + 0.028 * upgrade.relation_assist_gain
        + 0.018 * upgrade.verify_guard_gain
        + 0.030 * tool_focus
        + 0.020 * base_spec.law_scale[2]
    )
    relation_boost = (
        0.010 * upgrade.tool_head_gain
        + 0.016 * upgrade.relation_assist_gain
        + 0.010 * relation_focus
    )
    verify_boost = 0.006 * upgrade.tool_head_gain + 0.015 * upgrade.verify_guard_gain + 0.008 * verify_focus
    concept_delta = 0.006 * upgrade.concept_guard_gain - 0.003 * upgrade.tool_head_gain + 0.003 * concept_focus

    return {
        "concept": clamp_floor(base["concept"] + concept_delta),
        "relation": clamp_floor(base["relation"] + relation_boost),
        "tool": clamp_floor(base["tool"] + tool_boost),
        "verify": clamp_floor(base["verify"] + verify_boost),
    }


def non_tool_alignment(match_row: Dict[str, object]) -> float:
    rows = [row for row in match_row["rows"] if row["stage"] != "tool"]
    return float(mean([float(row["alignment_gap"]) for row in rows]))


def tool_undercoverage(match_row: Dict[str, object]) -> float:
    for row in match_row["rows"]:
        if row["stage"] == "tool":
            return float(row["undercoverage"])
    return 0.0


def score_candidate(
    searched_matches: Dict[str, Dict[str, object]],
    upgraded_matches: Dict[str, Dict[str, object]],
) -> float:
    tool_gain = mean(
        [
            tool_undercoverage(searched_matches[model]) - tool_undercoverage(upgraded_matches[model])
            for model in searched_matches
        ]
    )
    mean_gain = mean(
        [
            float(searched_matches[model]["mean_undercoverage"]) - float(upgraded_matches[model]["mean_undercoverage"])
            for model in searched_matches
        ]
    )
    gap_shrink = (
        float(searched_matches["deepseek_7b"]["mean_undercoverage"])
        - float(searched_matches["qwen3_4b"]["mean_undercoverage"])
        - (
            float(upgraded_matches["deepseek_7b"]["mean_undercoverage"])
            - float(upgraded_matches["qwen3_4b"]["mean_undercoverage"])
        )
    )
    non_tool_penalty = mean(
        [
            max(
                0.0,
                non_tool_alignment(upgraded_matches[model]) - non_tool_alignment(searched_matches[model]) - 0.012,
            )
            for model in searched_matches
        ]
    )
    return float(3.0 * tool_gain + 2.0 * mean_gain + 0.7 * gap_shrink - 2.5 * non_tool_penalty)


def candidate_grid() -> List[ToolHeadUpgradeSpec]:
    rows = []
    for tool_head_gain in [0.8, 1.0, 1.2, 1.4]:
        for relation_assist_gain in [0.4, 0.6, 0.8, 1.0]:
            for verify_guard_gain in [0.2, 0.4, 0.6]:
                for concept_guard_gain in [0.2, 0.4, 0.6]:
                    rows.append(
                        ToolHeadUpgradeSpec(
                            tool_head_gain=tool_head_gain,
                            relation_assist_gain=relation_assist_gain,
                            verify_guard_gain=verify_guard_gain,
                            concept_guard_gain=concept_guard_gain,
                        )
                    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Upgrade generator network with a dedicated tool-stage head")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/tool_stage_generator_network_upgrade_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    searched_payload = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_region_family_generator_network_20260310.json")
    end_to_end_payload = load_json(
        ROOT / "tests" / "codex_temp" / "local_pulse_end_to_end_region_family_generator_network_20260310.json"
    )
    structure_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    online_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    searched_spec = load_spec_from_payload(searched_payload["systems"]["generator_network_eval_family"])
    end_to_end_spec = load_spec_from_payload(end_to_end_payload["systems"]["end_to_end_generator_eval_family"])
    searched_capacity = stage_capacities(searched_spec)
    end_to_end_capacity = stage_capacities(end_to_end_spec)

    stage_demands_by_model = {
        model_name: stage_demands(structure_payload["models"][model_name], online_payload["models"][model_name])
        for model_name in ["qwen3_4b", "deepseek_7b"]
    }
    searched_matches = {
        model_name: compare_capacity_to_demand(searched_capacity, stage_demands_by_model[model_name])
        for model_name in stage_demands_by_model
    }
    end_to_end_matches = {
        model_name: compare_capacity_to_demand(end_to_end_capacity, stage_demands_by_model[model_name])
        for model_name in stage_demands_by_model
    }

    trials = []
    best = None
    for trial_id, candidate in enumerate(candidate_grid()):
        upgraded_capacity = tool_upgraded_stage_capacities(searched_spec, candidate)
        upgraded_matches = {
            model_name: compare_capacity_to_demand(upgraded_capacity, stage_demands_by_model[model_name])
            for model_name in stage_demands_by_model
        }
        score = score_candidate(searched_matches, upgraded_matches)
        trial = {
            "trial_id": trial_id,
            "tool_head_gain": candidate.tool_head_gain,
            "relation_assist_gain": candidate.relation_assist_gain,
            "verify_guard_gain": candidate.verify_guard_gain,
            "concept_guard_gain": candidate.concept_guard_gain,
            "score": float(score),
            "qwen_tool_undercoverage": tool_undercoverage(upgraded_matches["qwen3_4b"]),
            "deepseek_tool_undercoverage": tool_undercoverage(upgraded_matches["deepseek_7b"]),
            "qwen_mean_undercoverage": float(upgraded_matches["qwen3_4b"]["mean_undercoverage"]),
            "deepseek_mean_undercoverage": float(upgraded_matches["deepseek_7b"]["mean_undercoverage"]),
        }
        trials.append(trial)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "trial_id": trial_id,
                "upgrade_spec": candidate,
                "stage_capacity": upgraded_capacity,
                "matches": upgraded_matches,
            }

    assert best is not None
    qwen = best["matches"]["qwen3_4b"]
    deepseek = best["matches"]["deepseek_7b"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "tool_stage_generator_head_upgrade_for_real_online_band_gap",
            "trial_count": len(trials),
            "stages": STAGES,
        },
        "generator_profiles": {
            "searched_generator_network": {"stage_capacity": searched_capacity},
            "end_to_end_generator_network": {"stage_capacity": end_to_end_capacity},
            "tool_stage_head_generator_network": {
                "stage_capacity": best["stage_capacity"],
                "upgrade_spec": {
                    "tool_head_gain": best["upgrade_spec"].tool_head_gain,
                    "relation_assist_gain": best["upgrade_spec"].relation_assist_gain,
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
            "searched_generator_match": searched_matches[model_name],
            "end_to_end_generator_match": end_to_end_matches[model_name],
            "tool_stage_head_match": best["matches"][model_name],
        }

    payload["headline_metrics"] = {
        "best_system": "tool_stage_head_generator_network",
        "qwen_tool_undercoverage_gain": float(
            tool_undercoverage(searched_matches["qwen3_4b"]) - tool_undercoverage(qwen)
        ),
        "deepseek_tool_undercoverage_gain": float(
            tool_undercoverage(searched_matches["deepseek_7b"]) - tool_undercoverage(deepseek)
        ),
        "qwen_mean_undercoverage_gain": float(
            float(searched_matches["qwen3_4b"]["mean_undercoverage"]) - float(qwen["mean_undercoverage"])
        ),
        "deepseek_mean_undercoverage_gain": float(
            float(searched_matches["deepseek_7b"]["mean_undercoverage"]) - float(deepseek["mean_undercoverage"])
        ),
        "deepseek_minus_qwen_undercoverage_gap_shrink": float(
            (
                float(searched_matches["deepseek_7b"]["mean_undercoverage"])
                - float(searched_matches["qwen3_4b"]["mean_undercoverage"])
            )
            - (float(deepseek["mean_undercoverage"]) - float(qwen["mean_undercoverage"]))
        ),
        "qwen_non_tool_alignment_delta": float(
            non_tool_alignment(qwen) - non_tool_alignment(searched_matches["qwen3_4b"])
        ),
        "deepseek_non_tool_alignment_delta": float(
            non_tool_alignment(deepseek) - non_tool_alignment(searched_matches["deepseek_7b"])
        ),
        "qwen_worst_stage_after_upgrade": qwen["worst_stage"],
        "deepseek_worst_stage_after_upgrade": deepseek["worst_stage"],
    }

    payload["gains"] = {
        "qwen_tool_head_minus_end_to_end_undercoverage": float(
            float(end_to_end_matches["qwen3_4b"]["mean_undercoverage"]) - float(qwen["mean_undercoverage"])
        ),
        "deepseek_tool_head_minus_end_to_end_undercoverage": float(
            float(end_to_end_matches["deepseek_7b"]["mean_undercoverage"]) - float(deepseek["mean_undercoverage"])
        ),
        "qwen_tool_head_minus_searched_undercoverage": float(
            float(searched_matches["qwen3_4b"]["mean_undercoverage"]) - float(qwen["mean_undercoverage"])
        ),
        "deepseek_tool_head_minus_searched_undercoverage": float(
            float(searched_matches["deepseek_7b"]["mean_undercoverage"]) - float(deepseek["mean_undercoverage"])
        ),
        "qwen_tool_head_minus_searched_tool_gap": float(
            tool_undercoverage(searched_matches["qwen3_4b"]) - tool_undercoverage(qwen)
        ),
        "deepseek_tool_head_minus_searched_tool_gap": float(
            tool_undercoverage(searched_matches["deepseek_7b"]) - tool_undercoverage(deepseek)
        ),
    }

    payload["hypotheses"] = {
        "H1_tool_head_reduces_qwen_tool_undercoverage": bool(
            tool_undercoverage(qwen) < tool_undercoverage(searched_matches["qwen3_4b"]) - 0.02
        ),
        "H2_tool_head_reduces_deepseek_mean_undercoverage": bool(
            float(deepseek["mean_undercoverage"])
            < float(searched_matches["deepseek_7b"]["mean_undercoverage"]) - 0.02
        ),
        "H3_tool_head_beats_end_to_end_on_both_models": bool(
            float(qwen["mean_undercoverage"]) < float(end_to_end_matches["qwen3_4b"]["mean_undercoverage"]) - 0.01
            and float(deepseek["mean_undercoverage"])
            < float(end_to_end_matches["deepseek_7b"]["mean_undercoverage"]) - 0.01
        ),
        "H4_tool_head_keeps_non_tool_drift_bounded": bool(
            payload["headline_metrics"]["qwen_non_tool_alignment_delta"] <= 0.015
            and payload["headline_metrics"]["deepseek_non_tool_alignment_delta"] <= 0.015
        ),
    }

    payload["project_readout"] = {
        "summary": (
            "Focus the generator upgrade on the real online bottleneck: the tool stage. "
            "Instead of adding generic capacity, add a dedicated tool head and test whether it directly shrinks the real layer-band gap."
        ),
        "next_question": (
            "If the dedicated tool head keeps shrinking the real layer-band gap, "
            "the next step is a harder online tool interface to test whether trigger rates also fall."
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
