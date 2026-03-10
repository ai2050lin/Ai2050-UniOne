#!/usr/bin/env python
"""
Bridge low-dimensional generator-network capacities back to real-model online
high-risk stages.

Goal:
1. summarize stage capacities implied by the searched / end-to-end generators
2. summarize stage demands implied by real-model online recovery chains
3. test whether current generator bottlenecks fall on the same stages that are
   online-high-risk in Qwen3 / DeepSeek7B
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

from test_local_pulse_region_family_generator_network import EVAL_LATENTS, GeneratorNetworkSpec, hidden_state


ROOT = Path(__file__).resolve().parents[2]
STAGES = ["concept", "relation", "tool", "verify"]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def load_spec_from_payload(system_row: Dict[str, object]) -> GeneratorNetworkSpec:
    spec = system_row["generator_spec"]
    return GeneratorNetworkSpec(
        latent_scale=[float(v) for v in spec["latent_scale"]],
        latent_mix=[float(v) for v in spec["latent_mix"]],
        latent_bias=[float(v) for v in spec["latent_bias"]],
        region_scale=[float(v) for v in spec["region_scale"]],
        law_scale=[float(v) for v in spec["law_scale"]],
    )


def stage_capacities(spec: GeneratorNetworkSpec) -> Dict[str, float]:
    hidden_rows = [hidden_state(code, spec) for code in EVAL_LATENTS]
    hidden_mean = [mean([row[idx] for row in hidden_rows]) for idx in range(5)]

    concept_raw = (
        0.28 * spec.latent_scale[0]
        + 0.14 * spec.region_scale[0]
        + 0.18 * spec.region_scale[1]
        + 0.10 * abs(spec.latent_bias[2])
        + 0.18 * max(hidden_mean[0], 0.0)
        + 0.12 * max(hidden_mean[1], 0.0)
    )
    relation_raw = (
        0.12 * spec.latent_scale[1]
        + 0.15 * abs(spec.latent_mix[1])
        + 0.20 * spec.region_scale[4]
        + 0.19 * spec.law_scale[0]
        + 0.20 * spec.law_scale[1]
        + 0.14 * max(hidden_mean[1], 0.0)
    )
    tool_raw = (
        0.10 * spec.latent_scale[3]
        + 0.12 * abs(spec.latent_mix[3])
        + 0.15 * spec.region_scale[5]
        + 0.18 * spec.law_scale[1]
        + 0.22 * spec.law_scale[2]
        + 0.13 * max(hidden_mean[3], 0.0)
    )
    verify_raw = (
        0.08 * spec.latent_scale[4]
        + 0.08 * abs(spec.latent_mix[4])
        + 0.10 * spec.region_scale[6]
        + 0.16 * spec.law_scale[3]
        + 0.20 * spec.law_scale[4]
        + 0.18 * spec.law_scale[5]
        + 0.12 * max(hidden_mean[4], 0.0)
    )

    return {
        "concept": float(concept_raw),
        "relation": float(relation_raw),
        "tool": float(tool_raw),
        "verify": float(verify_raw),
    }


def target_stage_supports(layer_atlas: List[Dict[str, object]]) -> Dict[str, float]:
    targets = [row for row in layer_atlas if row["is_targeted_band"]]
    return {
        "concept": mean([float(row["shared_support"]) for row in targets if row["support_stage"] == "concept_biased"]),
        "relation": mean([float(row["shared_support"]) for row in targets if row["support_stage"] == "relation_biased"]),
        "balanced": mean([float(row["shared_support"]) for row in targets if row["support_stage"] == "balanced"]),
        "all": mean([float(row["shared_support"]) for row in targets]),
    }


def stage_demands(
    structure_row: Dict[str, object],
    online_row: Dict[str, object],
) -> Dict[str, float]:
    summary = structure_row["global_summary"]
    systems = online_row["systems"]["online_recovery_aware"]
    step_map = {row["step"]: row for row in online_row["step_rows"]}
    target_support = target_stage_supports(structure_row["layer_atlas"])

    orientation_gap = normalize(float(summary["orientation_gap_abs"]), 0.05, 0.65)
    mechanism = normalize(float(summary["mechanism_bridge_score"]), 0.70, 0.95)
    relation_gain = normalize(float(summary["mean_behavior_gain"]), 0.02, 0.05)
    post_stability = normalize(float(systems["mean_post_recovery_stability"]), 0.45, 0.78)

    concept = (
        float(step_map["concept"]["trigger_rate"]) * (1.0 + (1.0 - float(step_map["concept"]["recovery_success_rate"])))
        + 0.12 * orientation_gap
        + 0.08 * normalize(target_support["concept"], 0.0, 0.35)
        - 0.04 * mechanism
    )
    relation = (
        float(step_map["relation"]["trigger_rate"]) * (1.0 + (1.0 - float(step_map["relation"]["recovery_success_rate"])))
        + 0.10 * orientation_gap
        + 0.10 * normalize(target_support["relation"], 0.0, 0.35)
        + 0.05 * (1.0 - relation_gain)
    )
    tool = (
        float(step_map["tool"]["trigger_rate"]) * (1.0 + (1.0 - float(step_map["tool"]["recovery_success_rate"])))
        + 0.14 * orientation_gap
        + 0.08 * normalize(target_support["all"], 0.0, 0.35)
        + 0.06 * (1.0 - post_stability)
    )
    verify = (
        float(step_map["verify"]["trigger_rate"]) * (1.0 + (1.0 - float(step_map["verify"]["recovery_success_rate"])))
        + 0.06 * orientation_gap
        + 0.06 * normalize(target_support["balanced"], 0.0, 0.35)
        + 0.04 * (1.0 - mechanism)
    )

    return {
        "concept": float(concept),
        "relation": float(relation),
        "tool": float(tool),
        "verify": float(verify),
    }


def compare_capacity_to_demand(capacity: Dict[str, float], demand: Dict[str, float]) -> Dict[str, object]:
    rows = []
    for stage in STAGES:
        cap = float(capacity[stage])
        dem = float(demand[stage])
        rows.append(
            {
                "stage": stage,
                "capacity": cap,
                "demand": dem,
                "undercoverage": float(max(0.0, dem - cap)),
                "slack": float(max(0.0, cap - dem)),
                "alignment_gap": float(abs(dem - cap)),
            }
        )
    return {
        "rows": rows,
        "mean_undercoverage": float(mean([row["undercoverage"] for row in rows])),
        "mean_alignment_gap": float(mean([row["alignment_gap"] for row in rows])),
        "worst_stage": max(rows, key=lambda row: float(row["undercoverage"]))["stage"],
        "worst_undercoverage": max(rows, key=lambda row: float(row["undercoverage"]))["undercoverage"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge generator network capacity to real-model high-risk stages")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/generator_network_real_layer_band_bridge_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    searched_payload = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_region_family_generator_network_20260310.json")
    end_to_end_payload = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_end_to_end_region_family_generator_network_20260310.json")
    structure_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_real_model_structure_atlas_20260310.json")
    online_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_online_recovery_chain_20260310.json")

    searched_spec = load_spec_from_payload(searched_payload["systems"]["generator_network_eval_family"])
    end_to_end_spec = load_spec_from_payload(end_to_end_payload["systems"]["end_to_end_generator_eval_family"])
    searched_capacity = stage_capacities(searched_spec)
    end_to_end_capacity = stage_capacities(end_to_end_spec)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "generator_network_bridge_to_real_online_high_risk_stages",
            "runtime_sec": 0.0,
        },
        "generator_profiles": {
            "searched_generator_network": {
                "stage_capacity": searched_capacity,
            },
            "end_to_end_generator_network": {
                "stage_capacity": end_to_end_capacity,
            },
        },
        "models": {},
    }

    for model_name in ["qwen3_4b", "deepseek_7b"]:
        demand = stage_demands(structure_payload["models"][model_name], online_payload["models"][model_name])
        searched_match = compare_capacity_to_demand(searched_capacity, demand)
        end_to_end_match = compare_capacity_to_demand(end_to_end_capacity, demand)
        results["models"][model_name] = {
            "stage_demand": demand,
            "searched_generator_match": searched_match,
            "end_to_end_generator_match": end_to_end_match,
        }

    qwen = results["models"]["qwen3_4b"]
    deepseek = results["models"]["deepseek_7b"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_searched_undercoverage": float(qwen["searched_generator_match"]["mean_undercoverage"]),
            "qwen_end_to_end_undercoverage": float(qwen["end_to_end_generator_match"]["mean_undercoverage"]),
            "deepseek_searched_undercoverage": float(deepseek["searched_generator_match"]["mean_undercoverage"]),
            "deepseek_end_to_end_undercoverage": float(deepseek["end_to_end_generator_match"]["mean_undercoverage"]),
            "qwen_worst_stage": qwen["searched_generator_match"]["worst_stage"],
            "deepseek_worst_stage": deepseek["searched_generator_match"]["worst_stage"],
        },
        "gains": {
            "qwen_end_to_end_minus_searched_undercoverage": float(
                qwen["end_to_end_generator_match"]["mean_undercoverage"] - qwen["searched_generator_match"]["mean_undercoverage"]
            ),
            "deepseek_end_to_end_minus_searched_undercoverage": float(
                deepseek["end_to_end_generator_match"]["mean_undercoverage"] - deepseek["searched_generator_match"]["mean_undercoverage"]
            ),
            "deepseek_minus_qwen_searched_undercoverage": float(
                deepseek["searched_generator_match"]["mean_undercoverage"] - qwen["searched_generator_match"]["mean_undercoverage"]
            ),
            "deepseek_minus_qwen_end_to_end_undercoverage": float(
                deepseek["end_to_end_generator_match"]["mean_undercoverage"] - qwen["end_to_end_generator_match"]["mean_undercoverage"]
            ),
        },
        "hypotheses": {
            "H1_deepseek_has_larger_generator_undercoverage_than_qwen": bool(
                deepseek["searched_generator_match"]["mean_undercoverage"]
                > qwen["searched_generator_match"]["mean_undercoverage"] + 0.08
            ),
            "H2_deepseek_worst_stage_is_relation_or_tool": bool(
                deepseek["searched_generator_match"]["worst_stage"] in {"relation", "tool"}
            ),
            "H3_end_to_end_generator_does_not_fix_real_band_undercoverage": bool(
                deepseek["end_to_end_generator_match"]["mean_undercoverage"]
                >= deepseek["searched_generator_match"]["mean_undercoverage"] - 0.01
            ),
            "H4_qwen_end_to_end_undercoverage_stays_bounded": bool(
                qwen["end_to_end_generator_match"]["mean_undercoverage"]
                <= qwen["searched_generator_match"]["mean_undercoverage"] + 0.03
            ),
        },
        "project_readout": {
            "summary": "这一步把当前区域化生成网络的阶段容量直接对齐到真实模型在线高风险段。结果如果成立，就说明生成网络的主要瓶颈并不是抽象存在，而是已经落到真实在线链里的具体阶段缺口上。",
            "next_question": "如果生成网络的最大缺口已经和真实在线高风险段对齐，下一步就该直接升级生成网络结构，而不是再追加新的代理链。",
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
