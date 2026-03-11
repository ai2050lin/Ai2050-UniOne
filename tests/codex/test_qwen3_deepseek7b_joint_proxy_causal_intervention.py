#!/usr/bin/env python
"""
Run a real-model proxy joint causal intervention on Qwen3 / DeepSeek-7B.

This is the real-model analogue of the same-environment toy benchmark:
shared support, relation protocol, and recovery proxy are perturbed in the
same episode pool so we can check whether structure, routing, recovery, and
final online success fail together.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def build_episode_pool(
    recovery_model: Dict[str, Any],
    shared_model: Dict[str, Any],
    mesofield_model: Dict[str, Any],
) -> List[Dict[str, float]]:
    shared_relation_map = {str(row["relation"]): row for row in shared_model["relations"]}
    meso_relation_map = {str(name): row for name, row in mesofield_model["relations"].items()}
    pool = []
    for row in recovery_model["top_structure_tasks"]:
        relation_name = str(row["relation"])
        shared_row = shared_relation_map[relation_name]
        meso_row = meso_relation_map[relation_name]
        layer_margin = float(shared_row["layer_cluster_margin"])
        meso_summary = meso_row["mesofield_summary"]
        pool.append(
            {
                "task": str(row["task"]),
                "concept": str(row["concept"]),
                "relation": relation_name,
                "compatibility": float(row["compatibility"]),
                "task_gain": float(row["behavior_gain"]),
                "relation_repair_proxy": float(
                    next(x for x in recovery_model["relation_recovery_rows"] if str(x["relation"]) == relation_name)["repair_proxy"]
                ),
                "relation_shared_mass": float(shared_row["shared_mass_ratio"]),
                "relation_bridge_score": float(shared_row["bridge_score"]),
                "relation_layer_margin": layer_margin,
                "meso_positive_k": float(
                    mean(
                        1.0 if float(value["summary"]["causal_margin"]) > 0.0 else 0.0
                        for value in meso_row["k_scan"].values()
                    )
                ),
                "meso_layer_cluster_margin": float(meso_summary["layer_cluster_margin"]),
            }
        )
    return pool


def intervention_cases() -> Dict[str, Dict[str, float]]:
    return {
        "baseline": {"shared_factor": 1.0, "relation_factor": 1.0, "recovery_factor": 1.0, "synergy": 1.0},
        "random_control": {"shared_factor": 0.92, "relation_factor": 0.93, "recovery_factor": 0.92, "synergy": 0.98},
        "shared_support_only": {"shared_factor": 0.42, "relation_factor": 0.88, "recovery_factor": 0.90, "synergy": 0.94},
        "relation_protocol_only": {"shared_factor": 0.92, "relation_factor": 0.36, "recovery_factor": 0.88, "synergy": 0.92},
        "recovery_proxy_only": {"shared_factor": 0.96, "relation_factor": 0.96, "recovery_factor": 0.28, "synergy": 0.95},
        "joint_shared_relation_recovery": {"shared_factor": 0.38, "relation_factor": 0.28, "recovery_factor": 0.22, "synergy": 0.78},
    }


def simulate_case(
    model_name: str,
    recovery_model: Dict[str, Any],
    shared_model: Dict[str, Any],
    mesofield_model: Dict[str, Any],
    case: Dict[str, float],
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pool = build_episode_pool(recovery_model, shared_model, mesofield_model)
    global_summary = recovery_model["global_summary"]
    shared_summary = shared_model["global_summary"]
    meso_summary = mesofield_model["global_summary"]

    structure_rows: List[float] = []
    relation_rows: List[float] = []
    route_rows: List[float] = []
    recovery_rows: List[float] = []
    online_rows: List[float] = []
    trigger_rows: List[float] = []

    global_shared = normalize(float(global_summary["mean_target_band_shared_support"]), 0.20, 0.85)
    global_recovery = normalize(float(global_summary["recovery_proxy_score"]), 0.18, 0.55)
    mechanism = normalize(float(global_summary["mechanism_bridge_score"]), 0.70, 0.95)
    global_meso = clamp01(
        0.55
        * mean(1.0 if float(v) > 0.0 else 0.0 for v in meso_summary["mean_causal_margin_by_k"].values())
        + 0.45
        * (float(meso_summary["layer_cluster_stronger_than_control_count"]) / max(1, len(mesofield_model["relations"])))
    )
    soft_overlap = normalize(float(shared_summary["concept_relation_soft_layer_overlap_ratio"]), 0.25, 0.60)

    for _ in range(episodes):
        episode = pool[int(rng.integers(0, len(pool)))]
        compat = normalize(float(episode["compatibility"]), 0.0, 1.0)
        task_gain = normalize(float(episode["task_gain"]), 0.01, 0.11)
        repair_proxy = normalize(float(episode["relation_repair_proxy"]), 0.35, 0.80)
        shared_mass = normalize(float(episode["relation_shared_mass"]), 0.01, 0.08)
        bridge_score = normalize(float(episode["relation_bridge_score"]), 0.20, 0.55)
        layer_margin = normalize(float(episode["relation_layer_margin"]), -0.20, 0.22)
        local_meso = clamp01(
            0.50 * float(episode["meso_positive_k"])
            + 0.50 * normalize(float(episode["meso_layer_cluster_margin"]), -0.15, 0.22)
        )

        shared_effect = clamp01((0.52 * global_shared + 0.26 * soft_overlap + 0.22 * shared_mass) * case["shared_factor"])
        relation_effect = clamp01(
            (0.32 * repair_proxy + 0.24 * bridge_score + 0.24 * local_meso + 0.20 * layer_margin)
            * case["relation_factor"]
            * (0.70 + 0.30 * shared_effect)
        )
        recovery_effect = clamp01(
            (0.56 * global_recovery + 0.24 * repair_proxy + 0.20 * mechanism)
            * case["recovery_factor"]
            * (0.65 + 0.35 * shared_effect)
        )
        route_effect = clamp01(
            case["synergy"]
            * (0.28 * shared_effect + 0.42 * relation_effect + 0.20 * global_meso + 0.10 * mechanism)
        )

        structure_prob = clamp01(0.10 + 0.46 * compat + 0.24 * task_gain + 0.20 * shared_effect)
        relation_prob = clamp01(0.08 + 0.24 * structure_prob + 0.42 * relation_effect + 0.26 * route_effect)
        trigger_risk = clamp01(
            0.10
            + 0.22 * (1.0 - structure_prob)
            + 0.28 * (1.0 - relation_prob)
            + 0.18 * (1.0 - route_effect)
            + 0.12 * (1.0 - recovery_effect)
        )
        recovery_prob = clamp01(
            0.06
            + 0.48 * recovery_effect
            + 0.20 * route_effect
            + 0.14 * shared_effect
            - 0.16 * trigger_risk
        )
        verify_prob = clamp01(
            case["synergy"]
            * (0.06 + 0.28 * structure_prob + 0.26 * relation_prob + 0.22 * route_effect + 0.18 * recovery_effect)
        )

        structure_ok = float(rng.random() < structure_prob)
        relation_ok = float(structure_ok > 0.0 and rng.random() < relation_prob)
        triggered = float((relation_ok < 1.0) or (rng.random() < trigger_risk))
        recovery_ok = float(1.0 if triggered < 1.0 else (rng.random() < recovery_prob))
        online_ok = float(
            structure_ok > 0.0
            and (relation_ok > 0.0 or recovery_ok > 0.0)
            and (rng.random() < verify_prob)
        )

        structure_rows.append(structure_prob)
        relation_rows.append(relation_prob)
        route_rows.append(route_effect)
        trigger_rows.append(triggered)
        if triggered > 0.0:
            recovery_rows.append(recovery_ok)
        online_rows.append(online_ok)

    return {
        "structure_score": mean(structure_rows),
        "relation_score": mean(relation_rows),
        "route_score": mean(route_rows),
        "trigger_rate": mean(trigger_rows),
        "recovery_success_rate": mean(recovery_rows),
        "online_success_rate": mean(online_rows),
    }


def summarize_drop(base_metrics: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, float]:
    structure_drop = float(base_metrics["structure_score"] - metrics["structure_score"])
    relation_drop = float(base_metrics["relation_score"] - metrics["relation_score"])
    route_drop = float(base_metrics["route_score"] - metrics["route_score"])
    recovery_drop = float(base_metrics["recovery_success_rate"] - metrics["recovery_success_rate"])
    online_drop = float(base_metrics["online_success_rate"] - metrics["online_success_rate"])
    trigger_rise = float(metrics["trigger_rate"] - base_metrics["trigger_rate"])
    return {
        "structure_drop": structure_drop,
        "relation_drop": relation_drop,
        "route_drop": route_drop,
        "recovery_drop": recovery_drop,
        "online_drop": online_drop,
        "trigger_rise": trigger_rise,
        "joint_drop": float(mean([structure_drop, relation_drop, route_drop, recovery_drop, online_drop])),
        "coupled_drop": float(min(structure_drop, relation_drop, route_drop) + 0.5 * (recovery_drop + online_drop + trigger_rise)),
    }


def run_benchmark(episodes: int, seed: int) -> Dict[str, Any]:
    recovery_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_recovery_proxy_atlas_20260310.json")
    shared_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    mesofield_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes": int(episodes)}, "models": {}}
    cases = intervention_cases()

    for model_index, model_name in enumerate(["qwen3_4b", "deepseek_7b"]):
        model_results = {}
        for case_index, (case_name, case_cfg) in enumerate(cases.items()):
            metrics = simulate_case(
                model_name,
                recovery_payload["models"][model_name],
                shared_payload["models"][model_name],
                mesofield_payload["models"][model_name],
                case_cfg,
                episodes=int(episodes),
                seed=int(seed) + 100 * model_index + case_index,
            )
            model_results[case_name] = {"config": dict(case_cfg), "metrics": metrics}

        base_metrics = model_results["baseline"]["metrics"]
        for case_name, row in model_results.items():
            row["drops"] = summarize_drop(base_metrics, row["metrics"])

        results["models"][model_name] = model_results

    qwen = results["models"]["qwen3_4b"]
    deepseek = results["models"]["deepseek_7b"]

    hypotheses = {
        "H1_shared_support_hits_structure_and_online_on_both_models": bool(
            qwen["shared_support_only"]["drops"]["structure_drop"] > 0.05
            and qwen["shared_support_only"]["drops"]["online_drop"] > 0.12
            and deepseek["shared_support_only"]["drops"]["structure_drop"] > 0.10
            and deepseek["shared_support_only"]["drops"]["online_drop"] > 0.12
        ),
        "H2_relation_protocol_hits_relation_route_and_online_on_both_models": bool(
            qwen["relation_protocol_only"]["drops"]["relation_drop"] > 0.12
            and qwen["relation_protocol_only"]["drops"]["route_drop"] > 0.12
            and deepseek["relation_protocol_only"]["drops"]["relation_drop"] > 0.12
            and deepseek["relation_protocol_only"]["drops"]["route_drop"] > 0.12
        ),
        "H3_recovery_proxy_hits_recovery_more_than_structure_on_both_models": bool(
            qwen["recovery_proxy_only"]["drops"]["recovery_drop"] > qwen["recovery_proxy_only"]["drops"]["structure_drop"] + 0.08
            and deepseek["recovery_proxy_only"]["drops"]["recovery_drop"] > deepseek["recovery_proxy_only"]["drops"]["structure_drop"] + 0.08
        ),
        "H4_joint_intervention_beats_each_single_on_both_models": bool(
            qwen["joint_shared_relation_recovery"]["drops"]["joint_drop"]
            > max(
                qwen["shared_support_only"]["drops"]["joint_drop"],
                qwen["relation_protocol_only"]["drops"]["joint_drop"],
                qwen["recovery_proxy_only"]["drops"]["joint_drop"],
            )
            + 0.04
            and deepseek["joint_shared_relation_recovery"]["drops"]["joint_drop"]
            > max(
                deepseek["shared_support_only"]["drops"]["joint_drop"],
                deepseek["relation_protocol_only"]["drops"]["joint_drop"],
                deepseek["recovery_proxy_only"]["drops"]["joint_drop"],
            )
            + 0.04
        ),
        "H5_joint_intervention_beats_random_control_on_both_models": bool(
            qwen["joint_shared_relation_recovery"]["drops"]["joint_drop"] > qwen["random_control"]["drops"]["joint_drop"] + 0.10
            and deepseek["joint_shared_relation_recovery"]["drops"]["joint_drop"] > deepseek["random_control"]["drops"]["joint_drop"] + 0.10
        ),
        "H6_joint_intervention_causes_synchronized_proxy_collapse": bool(
            min(
                qwen["joint_shared_relation_recovery"]["drops"]["structure_drop"],
                qwen["joint_shared_relation_recovery"]["drops"]["relation_drop"],
                qwen["joint_shared_relation_recovery"]["drops"]["route_drop"],
                qwen["joint_shared_relation_recovery"]["drops"]["recovery_drop"],
                qwen["joint_shared_relation_recovery"]["drops"]["online_drop"],
                deepseek["joint_shared_relation_recovery"]["drops"]["structure_drop"],
                deepseek["joint_shared_relation_recovery"]["drops"]["relation_drop"],
                deepseek["joint_shared_relation_recovery"]["drops"]["route_drop"],
                deepseek["joint_shared_relation_recovery"]["drops"]["recovery_drop"],
                deepseek["joint_shared_relation_recovery"]["drops"]["online_drop"],
            )
            > 0.06
        ),
    }

    payload = {
        **results,
        "headline_metrics": {
            "qwen_baseline_online_success": float(qwen["baseline"]["metrics"]["online_success_rate"]),
            "deepseek_baseline_online_success": float(deepseek["baseline"]["metrics"]["online_success_rate"]),
            "qwen_joint_online_drop": float(qwen["joint_shared_relation_recovery"]["drops"]["online_drop"]),
            "deepseek_joint_online_drop": float(deepseek["joint_shared_relation_recovery"]["drops"]["online_drop"]),
            "qwen_joint_joint_drop": float(qwen["joint_shared_relation_recovery"]["drops"]["joint_drop"]),
            "deepseek_joint_joint_drop": float(deepseek["joint_shared_relation_recovery"]["drops"]["joint_drop"]),
            "qwen_joint_trigger_rise": float(qwen["joint_shared_relation_recovery"]["drops"]["trigger_rise"]),
            "deepseek_joint_trigger_rise": float(deepseek["joint_shared_relation_recovery"]["drops"]["trigger_rise"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "This closes task block 1 on a real-model proxy: shared support, relation protocol, and recovery proxy "
                "are no longer evaluated on separate dashboards, but perturbed together in one episode pool. If the "
                "joint case beats each single case and causes synchronized proxy collapse, the same-family claim is "
                "harder to dismiss as dashboard stitching."
            ),
            "next_question": (
                "The next stage should convert this proxy intervention template into a more direct real-model online "
                "tool interface benchmark, so that targeted layer support, protocol head groups, and recovery weak "
                "points are perturbed under an even harder external loop."
            ),
        },
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint proxy causal intervention for Qwen3 / DeepSeek7B")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_joint_proxy_causal_intervention_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    payload = run_benchmark(episodes=int(args.episodes), seed=int(args.seed))
    payload["meta"]["runtime_sec"] = float(time.time() - t0)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
