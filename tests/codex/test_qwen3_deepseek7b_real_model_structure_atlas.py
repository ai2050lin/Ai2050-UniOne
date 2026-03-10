#!/usr/bin/env python
"""
Build a unified real-model structure atlas over existing Qwen3-4B / DeepSeek-7B
artifacts.

The atlas externalizes one consistent evidence chain:
1. shared support layer bands
2. predicted concept-vs-relation orientation
3. targeted ablation vulnerability and orientation mismatch
4. structure-aware task gain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def stage_label(value: float, tol: float = 0.025) -> str:
    if value >= tol:
        return "relation_biased"
    if value <= -tol:
        return "concept_biased"
    return "balanced"


def build_layer_table(
    shared_model: Dict[str, Any],
    orientation_model: Dict[str, Any],
    ablation_model: Dict[str, Any],
) -> List[Dict[str, Any]]:
    concept_layers = {
        int(layer_name[1:]): float(score)
        for layer_name, score in shared_model["layer_sets"]["top_concept_layers"]
    }
    relation_layers = {
        int(layer_name[1:]): float(score)
        for layer_name, score in shared_model["layer_sets"]["top_relation_layers"]
    }
    shared_layers = {
        int(row["layer"]): float(row["score"])
        for row in orientation_model["shared_layers"]
    }
    target_layers = {
        int(row["layer"]): float(row["shared_score"])
        for row in ablation_model["target_layers"]
    }
    control_layers = {
        int(row["layer"]): float(row["shared_score"])
        for row in ablation_model["control_layers"]
    }

    all_layers = sorted(set(concept_layers) | set(relation_layers) | set(shared_layers) | set(target_layers) | set(control_layers))
    rows = []
    for layer in all_layers:
        concept_support = float(concept_layers.get(layer, 0.0))
        relation_support = float(relation_layers.get(layer, 0.0))
        shared_support = float(shared_layers.get(layer, 0.0))
        support_bias = float(relation_support - concept_support)
        rows.append(
            {
                "layer": int(layer),
                "concept_support": concept_support,
                "relation_support": relation_support,
                "shared_support": shared_support,
                "support_bias": support_bias,
                "support_stage": stage_label(support_bias),
                "is_shared_band": bool(layer in shared_layers),
                "is_targeted_band": bool(layer in target_layers),
                "is_control_band": bool(layer in control_layers),
            }
        )
    return rows


def top_name(rows: List[Dict[str, Any]], key: str, name_key: str) -> str:
    if not rows:
        return ""
    return str(max(rows, key=lambda row: float(row[key]))[name_key])


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified real-model structure atlas for Qwen3 / DeepSeek7B")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen3_deepseek7b_real_model_structure_atlas_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    shared_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_support_head_bridge_20260310.json")
    orientation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_causal_orientation_20260310.json")
    ablation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_shared_layer_band_targeted_ablation_20260310.json")
    task_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "core_constraint": "real_model_structure_atlas_from_existing_artifacts",
            "runtime_sec": 0.0,
        },
        "models": {},
    }

    for model_name in ["qwen3_4b", "deepseek_7b"]:
        shared_model = shared_payload["models"][model_name]
        orientation_model = orientation_payload["models"][model_name]
        ablation_model = ablation_payload["models"][model_name]
        task_model = task_payload["models"][model_name]

        layer_rows = build_layer_table(shared_model, orientation_model, ablation_model)
        orientation_summary = orientation_model["global_summary"]
        ablation_summary = ablation_model["global_summary"]
        shared_summary = shared_model["global_summary"]
        task_summary = task_model["global_summary"]

        predicted_orientation = float(orientation_summary["shared_layer_orientation"])
        actual_orientation = float(ablation_summary["actual_targeted_orientation"])
        concept_probe_rows = ablation_model["probe_concepts"]
        relation_probe_rows = ablation_model["probe_relations"]
        concept_target_margin = mean([float(row["target_margin"] - row["baseline_margin"]) for row in concept_probe_rows])
        relation_target_margin = mean(
            [float(row["target_peak_pair_bridge"] - row["baseline_peak_pair_bridge"]) for row in relation_probe_rows]
        )

        results["models"][model_name] = {
            "layer_atlas": layer_rows,
            "phase_atlas": {
                "top_concept_probe": top_name(orientation_model["concepts"], "shared_layer_hit_ratio", "concept"),
                "top_relation_probe": top_name(orientation_model["relations"], "shared_layer_hit_ratio", "relation"),
                "top_task": task_summary["top_gain_task"],
            },
            "global_summary": {
                "predicted_orientation": predicted_orientation,
                "predicted_orientation_label": orientation_summary["orientation_label"],
                "actual_orientation": actual_orientation,
                "actual_orientation_label": ablation_summary["actual_orientation_label"],
                "orientation_gap": float(actual_orientation - predicted_orientation),
                "orientation_gap_abs": float(abs(actual_orientation - predicted_orientation)),
                "mean_concept_shared_hit": float(orientation_summary["mean_concept_shared_layer_hit"]),
                "mean_relation_shared_hit": float(orientation_summary["mean_relation_shared_layer_hit"]),
                "concept_hit_margin_corr": float(orientation_summary["concept_hit_margin_corr"]),
                "relation_hit_behavior_gain_corr": float(orientation_summary["relation_hit_behavior_gain_corr"]),
                "mechanism_bridge_score": float(shared_summary["mechanism_bridge_score"]),
                "soft_layer_overlap": float(shared_summary["concept_relation_soft_layer_overlap_ratio"]),
                "shared_mass_ratio": float(shared_summary["mean_shared_mass_ratio"]),
                "compact_mass_gain": float(shared_summary["compact_minus_diffuse_shared_mass"]),
                "mean_behavior_gain": float(task_summary["mean_behavior_gain"]),
                "concept_gain_rank_correlation": float(task_summary["concept_gain_rank_correlation"]),
                "mean_concept_causal_margin": float(ablation_summary["mean_concept_causal_margin"]),
                "mean_relation_causal_margin": float(ablation_summary["mean_relation_causal_margin"]),
                "concept_target_delta": concept_target_margin,
                "relation_target_delta": relation_target_margin,
                "shared_band_layer_count": int(sum(1 for row in layer_rows if row["is_shared_band"])),
                "targeted_layer_count": int(sum(1 for row in layer_rows if row["is_targeted_band"])),
            },
        }

    qwen = results["models"]["qwen3_4b"]["global_summary"]
    deepseek = results["models"]["deepseek_7b"]["global_summary"]

    payload = {
        **results,
        "headline_metrics": {
            "qwen_predicted_orientation": float(qwen["predicted_orientation"]),
            "qwen_actual_orientation": float(qwen["actual_orientation"]),
            "deepseek_predicted_orientation": float(deepseek["predicted_orientation"]),
            "deepseek_actual_orientation": float(deepseek["actual_orientation"]),
            "qwen_orientation_gap_abs": float(qwen["orientation_gap_abs"]),
            "deepseek_orientation_gap_abs": float(deepseek["orientation_gap_abs"]),
            "qwen_behavior_gain": float(qwen["mean_behavior_gain"]),
            "deepseek_behavior_gain": float(deepseek["mean_behavior_gain"]),
            "qwen_mechanism_bridge": float(qwen["mechanism_bridge_score"]),
            "deepseek_mechanism_bridge": float(deepseek["mechanism_bridge_score"]),
        },
        "gains": {
            "deepseek_minus_qwen_mechanism_bridge": float(deepseek["mechanism_bridge_score"] - qwen["mechanism_bridge_score"]),
            "deepseek_minus_qwen_behavior_gain": float(deepseek["mean_behavior_gain"] - qwen["mean_behavior_gain"]),
            "deepseek_minus_qwen_soft_layer_overlap": float(deepseek["soft_layer_overlap"] - qwen["soft_layer_overlap"]),
            "deepseek_minus_qwen_orientation_gap_abs": float(deepseek["orientation_gap_abs"] - qwen["orientation_gap_abs"]),
        },
        "hypotheses": {
            "H1_predicted_atlas_recovers_qwen_concept_and_deepseek_relation_bias": bool(
                qwen["predicted_orientation"] <= -0.05 and deepseek["predicted_orientation"] >= 0.05
            ),
            "H2_actual_targeted_ablation_breaks_predicted_orientation_on_both_models": bool(
                qwen["orientation_gap_abs"] >= 0.05 and deepseek["orientation_gap_abs"] >= 0.20
            ),
            "H3_deepseek_keeps_stronger_real_model_bridge_and_overlap": bool(
                deepseek["mechanism_bridge_score"] > qwen["mechanism_bridge_score"]
                and deepseek["soft_layer_overlap"] > qwen["soft_layer_overlap"]
            ),
            "H4_structure_aware_task_gain_stays_positive_on_both_models": bool(
                qwen["mean_behavior_gain"] > 0.0 and deepseek["mean_behavior_gain"] > 0.0
            ),
        },
        "project_readout": {
            "summary": "这一步把真实模型里的共享层带、阶段取向、定向消融和结构任务收益收成一张统一 atlas。结果既保留了正证据，也保留了负证据：预测层带取向在两类模型上都能读出来，但真正做定向消融后，取向并不会自动成立，说明真实模型里还隔着一层状态依赖的门控或容量瓶颈。",
            "next_question": "如果第一版 atlas 已经把正负证据同时钉到真实模型上，下一步就该围绕这些高共享层带做更细的阶段核和恢复链映射，而不是继续只在 toy 系统里加机制。",
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
