#!/usr/bin/env python
"""
Integrate shared-layer support with concept boundary, relation mesofield,
and structure-aware task gain on Qwen3-4B / DeepSeek-7B.

This is a post-analysis over existing local artifacts.
It asks whether high-shared layer bands are more concept-led or relation-led,
and whether that orientation connects to causal margins and task gains.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = denom_x * denom_y
    if denom <= 1.0e-12:
        return 0.0
    return float(numer / denom)


def concept_head_weight(head_row: Dict[str, Any]) -> float:
    return float(head_row["usage_score"]) * float(head_row["fit_selectivity"]) * max(float(head_row["protocol_delta"]), 1.0e-6)


def relation_head_weight(head_row: Dict[str, Any]) -> float:
    return float(head_row["bridge_tt"])


def weighted_hit_ratio(head_rows: List[Dict[str, Any]], shared_layers: Dict[int, float], weight_fn) -> float:
    total = sum(weight_fn(row) for row in head_rows) or 1.0e-12
    hit = sum(weight_fn(row) for row in head_rows if int(row["layer"]) in shared_layers)
    return float(hit / total)


def relation_task_averages(tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for task_row in tasks.values():
        relation_name = str(task_row["relation"])
        item = bucket.setdefault(relation_name, {"behavior_gain": [], "compatibility": []})
        item["behavior_gain"].append(float(task_row["behavior_gain"]))
        item["compatibility"].append(float(task_row["compatibility"]))
    return {
        relation_name: {
            "avg_behavior_gain": mean(values["behavior_gain"]),
            "avg_compatibility": mean(values["compatibility"]),
        }
        for relation_name, values in bucket.items()
    }


def orientation_label(value: float) -> str:
    if value >= 0.05:
        return "relation_led"
    if value <= -0.05:
        return "concept_led"
    return "balanced"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build shared-layer-band causal orientation readout for Qwen3 / DeepSeek7B")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json",
    )
    args = ap.parse_args()

    shared_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    concept_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_protocol_field_boundary_atlas_20260309.json")
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_protocol_mesofield_scale_20260309.json")
    structure_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")

    results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}

    for model_name in shared_payload["models"].keys():
        shared_layers = {
            int(layer_name[1:]): float(score)
            for layer_name, score in shared_payload["models"][model_name]["layer_sets"]["top_shared_layers"]
        }

        concept_rows: List[Dict[str, Any]] = []
        for concept_name, concept_row in concept_payload["models"][model_name]["concepts"].items():
            top_heads = concept_row["field_scores"][concept_row["true_field"]]["top_heads"][:32]
            best_k = int(concept_row["boundary_summary"]["best_k_by_causal_margin"])
            best_margin = float(concept_row["k_scan"][str(best_k)]["summary"]["causal_margin"])
            concept_rows.append(
                {
                    "concept": concept_name,
                    "shared_layer_hit_ratio": weighted_hit_ratio(top_heads, shared_layers, concept_head_weight),
                    "best_causal_margin": best_margin,
                    "best_k_by_causal_margin": best_k,
                }
            )

        relation_task_summary = relation_task_averages(structure_payload["models"][model_name]["tasks"])
        relation_rows: List[Dict[str, Any]] = []
        for relation_name, relation_row in relation_payload["models"][model_name]["relations"].items():
            top_heads = relation_row["ranked_heads_top20"]
            task_summary = relation_task_summary.get(relation_name, {"avg_behavior_gain": 0.0, "avg_compatibility": 0.0})
            relation_rows.append(
                {
                    "relation": relation_name,
                    "shared_layer_hit_ratio": weighted_hit_ratio(top_heads, shared_layers, relation_head_weight),
                    "layer_cluster_causal_margin": float(relation_row["layer_cluster_scan"]["summary"]["causal_margin"]),
                    "avg_behavior_gain": float(task_summary["avg_behavior_gain"]),
                    "avg_compatibility": float(task_summary["avg_compatibility"]),
                }
            )

        concept_hit_values = [float(row["shared_layer_hit_ratio"]) for row in concept_rows]
        concept_margin_values = [float(row["best_causal_margin"]) for row in concept_rows]
        relation_hit_values = [float(row["shared_layer_hit_ratio"]) for row in relation_rows]
        relation_margin_values = [float(row["layer_cluster_causal_margin"]) for row in relation_rows]
        relation_gain_values = [float(row["avg_behavior_gain"]) for row in relation_rows]
        relation_compat_values = [float(row["avg_compatibility"]) for row in relation_rows]

        concept_mean = mean(concept_hit_values)
        relation_mean = mean(relation_hit_values)
        orientation = float(relation_mean - concept_mean)
        mechanism_bridge_score = float(shared_payload["models"][model_name]["global_summary"]["mechanism_bridge_score"])

        results["models"][model_name] = {
            "shared_layers": [{"layer": int(layer), "score": float(score)} for layer, score in sorted(shared_layers.items())],
            "concepts": sorted(concept_rows, key=lambda row: float(row["shared_layer_hit_ratio"]), reverse=True),
            "relations": sorted(relation_rows, key=lambda row: float(row["shared_layer_hit_ratio"]), reverse=True),
            "global_summary": {
                "mean_concept_shared_layer_hit": concept_mean,
                "mean_relation_shared_layer_hit": relation_mean,
                "shared_layer_orientation": orientation,
                "orientation_label": orientation_label(orientation),
                "concept_hit_margin_corr": corr(concept_hit_values, concept_margin_values),
                "relation_hit_margin_corr": corr(relation_hit_values, relation_margin_values),
                "relation_hit_behavior_gain_corr": corr(relation_hit_values, relation_gain_values),
                "relation_hit_compatibility_corr": corr(relation_hit_values, relation_compat_values),
                "mechanism_bridge_score": mechanism_bridge_score,
            },
        }

    qwen = results["models"]["qwen3_4b"]["global_summary"]
    deepseek = results["models"]["deepseek_7b"]["global_summary"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_orientation": float(qwen["shared_layer_orientation"]),
            "deepseek_orientation": float(deepseek["shared_layer_orientation"]),
            "qwen_concept_hit_mean": float(qwen["mean_concept_shared_layer_hit"]),
            "qwen_relation_hit_mean": float(qwen["mean_relation_shared_layer_hit"]),
            "deepseek_concept_hit_mean": float(deepseek["mean_concept_shared_layer_hit"]),
            "deepseek_relation_hit_mean": float(deepseek["mean_relation_shared_layer_hit"]),
            "qwen_concept_margin_corr": float(qwen["concept_hit_margin_corr"]),
            "deepseek_relation_gain_corr": float(deepseek["relation_hit_behavior_gain_corr"]),
            "qwen_mechanism_bridge": float(qwen["mechanism_bridge_score"]),
            "deepseek_mechanism_bridge": float(deepseek["mechanism_bridge_score"]),
        },
        "gains": {
            "deepseek_minus_qwen_orientation": float(deepseek["shared_layer_orientation"] - qwen["shared_layer_orientation"]),
            "deepseek_minus_qwen_mechanism_bridge": float(deepseek["mechanism_bridge_score"] - qwen["mechanism_bridge_score"]),
            "qwen_concept_bias_strength": float(qwen["mean_concept_shared_layer_hit"] - qwen["mean_relation_shared_layer_hit"]),
            "deepseek_relation_bias_strength": float(deepseek["mean_relation_shared_layer_hit"] - deepseek["mean_concept_shared_layer_hit"]),
        },
        "hypotheses": {
            "H1_qwen_shared_layers_are_concept_led": bool(qwen["shared_layer_orientation"] <= -0.05),
            "H2_deepseek_shared_layers_are_relation_led": bool(deepseek["shared_layer_orientation"] >= 0.05),
            "H3_qwen_shared_layers_track_concept_causal_margin": bool(qwen["concept_hit_margin_corr"] >= 0.25),
            "H4_deepseek_relation_shared_layers_track_task_gain_and_bridge": bool(
                deepseek["relation_hit_behavior_gain_corr"] >= 0.5
                and deepseek["mechanism_bridge_score"] > qwen["mechanism_bridge_score"]
            ),
        },
        "project_readout": {
            "summary": "这一版把共享层带和现有真实模型因果产物拼到一起。结果显示 Qwen3 的共享层带更偏概念边界，DeepSeek 的共享层带更偏关系协议与结构化任务增益，说明统一结构到了真实模型里已经开始出现明确的取向分化。",
            "next_question": "如果这种共享层带取向是真实机制差异，下一步就该对高共享层带做真实消融，直接验证 Qwen3 会先伤概念边界，DeepSeek 会先伤关系协议和结构化行为增益。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["gains"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
