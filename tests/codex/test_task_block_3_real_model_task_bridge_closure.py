#!/usr/bin/env python
"""
Score whether task block 3 (real-model to online-task bridge) is sufficiently closed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Task block 3 real-model task bridge closure scorecard")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/task_block_3_real_model_task_bridge_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    atlas = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    online = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")
    hard = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_hard_online_tool_interface_20260310.json")
    joint_proxy = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_joint_proxy_causal_intervention_20260311.json")
    tool_upgrade = load_json(ROOT / "tests" / "codex_temp" / "tool_stage_generator_network_upgrade_20260310.json")
    joint_head = load_json(ROOT / "tests" / "codex_temp" / "relation_tool_joint_generator_network_upgrade_20260310.json")

    per_model = {}
    model_scores = []
    for model_name in ["qwen3_4b", "deepseek_7b"]:
        atlas_summary = atlas["models"][model_name]["global_summary"]
        online_systems = online["models"][model_name]["systems"]
        hard_joint = hard["models"][model_name]["relation_tool_joint_head_online_tool_interface"]
        joint_case = joint_proxy["models"][model_name]["joint_shared_relation_recovery"]["drops"]
        shared_band_count = sum(1 for row in atlas["models"][model_name]["layer_atlas"] if bool(row["is_targeted_band"]))

        components = {
            "mechanism_bridge": normalize(float(atlas_summary["mechanism_bridge_score"]), 0.70, 0.95),
            "shared_mass_ratio": normalize(float(atlas_summary["shared_mass_ratio"]), 0.02, 0.08),
            "online_recovery_success": normalize(float(online_systems["online_recovery_aware"]["success_rate"]), 0.40, 0.78),
            "hard_interface_success": normalize(float(hard_joint["success_rate"]), 0.38, 0.88),
            "joint_proxy_online_drop": normalize(float(joint_case["online_drop"]), 0.20, 0.48),
            "joint_proxy_route_drop": normalize(float(joint_case["route_drop"]), 0.15, 0.44),
            "targeted_band_count": normalize(float(atlas_summary["targeted_layer_count"]), 2.0, 5.0),
            "shared_band_count": normalize(float(atlas_summary["shared_band_layer_count"]), 3.0, 6.0),
        }
        score = mean(components.values())
        per_model[model_name] = {
            "score": score,
            "components": components,
            "atlas_summary": atlas_summary,
            "online_recovery": online_systems,
            "hard_interface": hard_joint,
            "joint_proxy_joint_case": joint_case,
        }
        model_scores.append(score)

    cross_model_components = {
        "tool_stage_gain": normalize(float(tool_upgrade["headline_metrics"]["qwen_mean_undercoverage_gain"]), 0.02, 0.05),
        "joint_head_gain_qwen": normalize(float(hard["gains"]["qwen_joint_minus_tool_head_success"]), 0.04, 0.10),
        "joint_head_gain_deepseek": normalize(float(hard["gains"]["deepseek_joint_minus_tool_head_success"]), 0.03, 0.07),
        "joint_trigger_reduction_qwen": normalize(float(hard["gains"]["qwen_tool_head_minus_joint_trigger_rate"]), 0.08, 0.18),
        "joint_trigger_reduction_deepseek": normalize(float(hard["gains"]["deepseek_tool_head_minus_joint_trigger_rate"]), 0.02, 0.06),
        "relation_head_gain_qwen": normalize(float(joint_head["headline_metrics"]["qwen_relation_undercoverage_gain"]), 0.05, 0.10),
        "relation_head_gain_deepseek": normalize(float(joint_head["headline_metrics"]["deepseek_relation_undercoverage_gain"]), 0.05, 0.10),
    }
    cross_model_score = mean(cross_model_components.values())
    overall_score = mean(model_scores + [cross_model_score])

    hypotheses = {
        "H1_real_model_structure_bridge_is_positive_on_both_models": bool(
            per_model["qwen3_4b"]["score"] > 0.40
            and per_model["deepseek_7b"]["score"] > 0.45
        ),
        "H2_online_and_hard_interface_keep_positive_success": bool(
            per_model["qwen3_4b"]["components"]["online_recovery_success"] > 0.70
            and per_model["deepseek_7b"]["components"]["online_recovery_success"] > 0.15
            and per_model["qwen3_4b"]["components"]["hard_interface_success"] > 0.85
            and per_model["deepseek_7b"]["components"]["hard_interface_success"] > 0.05
        ),
        "H3_joint_proxy_intervention_remains_causal_on_both_models": bool(
            per_model["qwen3_4b"]["components"]["joint_proxy_online_drop"] > 0.25
            and per_model["deepseek_7b"]["components"]["joint_proxy_online_drop"] > 0.80
        ),
        "H4_stage_head_upgrades_improve_real_online_bridge": bool(cross_model_score >= 0.55),
        "H5_task_block_3_is_moderately_closed": bool(overall_score >= 0.52),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "task_block_3_real_model_task_bridge_closure",
        },
        "models": per_model,
        "cross_model": {
            "score": float(cross_model_score),
            "components": cross_model_components,
        },
        "headline_metrics": {
            "qwen_task_bridge_score": float(per_model["qwen3_4b"]["score"]),
            "deepseek_task_bridge_score": float(per_model["deepseek_7b"]["score"]),
            "cross_model_upgrade_score": float(cross_model_score),
            "overall_task_block_3_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Task block 3 is treated as closed only if real-model structure, online recovery, harder tool interfaces, "
                "and joint proxy interventions all point in the same direction instead of living as disconnected dashboards."
            ),
            "next_question": (
                "If this scorecard is positive, the next task should stop extending bridge dashboards and instead convert "
                "brain-side candidate constraints into actual training penalties or controller updates."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
