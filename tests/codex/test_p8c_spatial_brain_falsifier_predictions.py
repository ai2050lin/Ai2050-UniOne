#!/usr/bin/env python
"""
P8C: derive sharp spatial brain-side falsifier predictions.
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


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def classification_mean(relations: Dict[str, Dict[str, Any]], classification: str, key: str) -> float:
    rows = [float(v[key]) for v in relations.values() if v["classification"] == classification]
    return mean(rows) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="P8C spatial brain falsifier predictions")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p8c_spatial_brain_falsifier_predictions_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p7c = load_json(ROOT / "tests" / "codex_temp" / "p7c_brain_spatial_falsification_minimal_core_20260311.json")
    p8a = load_json(ROOT / "tests" / "codex_temp" / "p8a_spatialized_plasticity_coding_equation_20260311.json")
    p8b = load_json(ROOT / "tests" / "codex_temp" / "p8b_3d_wiring_dynamic_topology_division_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")
    relation_boundary = load_json(
        ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_topology_boundary_bridge_20260309.json"
    )
    d_problem = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    qwen_rel = relation_boundary["models"]["qwen3_4b"]["relations"]
    deepseek_rel = relation_boundary["models"]["deepseek_7b"]["relations"]
    d_global = d_problem["global_summary"]

    falsifier_sharpness = {
        "stage8d_hard_spec": float(stage8d["headline_metrics"]["hard_falsifier_spec_score"]),
        "p7c_falsifier_sharpness": float(p7c["headline_metrics"]["falsifier_sharpness_score"]),
        "p8b_division_consistency": float(p8b["headline_metrics"]["division_of_labor_consistency_score"]),
        "p8a_spatial_consistency": float(p8a["headline_metrics"]["spatial_equation_consistency_score"]),
    }
    falsifier_sharpness_score = mean(falsifier_sharpness.values())

    local_vs_bridge_prediction_specificity = {
        "compact_boundary_bridge_gt_distributed_qwen": normalize(
            classification_mean(qwen_rel, "compact_boundary", "bridge_score")
            - classification_mean(qwen_rel, "distributed_none", "bridge_score"),
            -0.05,
            0.06,
        ),
        "compact_boundary_bridge_gt_distributed_deepseek": normalize(
            classification_mean(deepseek_rel, "compact_boundary", "bridge_score")
            - classification_mean(deepseek_rel, "distributed_none", "bridge_score"),
            0.03,
            0.12,
        ),
        "compact_boundary_compactness_gt_distributed_qwen": normalize(
            classification_mean(qwen_rel, "compact_boundary", "topology_compactness")
            - classification_mean(qwen_rel, "distributed_none", "topology_compactness"),
            0.03,
            0.20,
        ),
        "compact_boundary_compactness_gt_distributed_deepseek": normalize(
            classification_mean(deepseek_rel, "compact_boundary", "topology_compactness")
            - classification_mean(deepseek_rel, "distributed_none", "topology_compactness"),
            0.10,
            0.22,
        ),
    }
    local_vs_bridge_prediction_specificity_score = mean(local_vs_bridge_prediction_specificity.values())

    geometry_vs_targeted_prediction = {
        "geometry_only_still_negative": normalize(-float(d_global["best_overall_gain_across_methods"]), 0.0, 0.02),
        "base_offset_dual_positive_absent": 1.0
        if not bool(d_global["base_offset_dual_positive_exists"])
        else 0.0,
        "offset_stabilization_dual_positive_absent": 1.0
        if not bool(d_global["offset_stabilization_dual_positive_exists"])
        else 0.0,
        "multistage_best_still_negative": normalize(-float(d_global["multistage_best_overall_gain"]), 0.0, 0.02),
    }
    geometry_vs_targeted_prediction_score = mean(geometry_vs_targeted_prediction.values())

    brain_spatial_testability = {
        "p7c_brain_spatial_plausibility": float(p7c["headline_metrics"]["brain_spatial_plausibility_score"]),
        "stage8d_directional_falsifier": float(stage8d["headline_metrics"]["directional_falsifier_score"]),
        "p8b_geometry_split": float(p8b["headline_metrics"]["geometry_dynamic_topology_split_score"]),
        "p8b_explicit_spatial_economy": float(p8b["headline_metrics"]["explicit_spatial_economy_score"]),
    }
    brain_spatial_testability_score = mean(brain_spatial_testability.values())

    overall_score = mean(
        [
            falsifier_sharpness_score,
            local_vs_bridge_prediction_specificity_score,
            geometry_vs_targeted_prediction_score,
            brain_spatial_testability_score,
        ]
    )

    spatial_predictions = [
        {
            "id": "P8C_pred_1",
            "title": "局部微扰先伤 family 复用，后伤跨区整合",
            "detail": "如果在局部高密度邻域内做小范围扰动，应先看到 concept family reuse 下降，而不是立即看到长程关系整合整体崩塌。",
            "falsifier": "如果局部微扰主要先打掉远程关系桥接，而 family reuse 基本不动，当前分工模型会被削弱。",
        },
        {
            "id": "P8C_pred_2",
            "title": "选择性长程桥扰动先伤 compact-boundary relation",
            "detail": "对稀疏长程桥接做选择性扰动时，compact-boundary relation 应比 distributed relation 更早、更明显受损。",
            "falsifier": "如果 distributed relation 同样或更早受损，说明当前桥接分工判断过强。",
        },
        {
            "id": "P8C_pred_3",
            "title": "广泛几何平滑不应优于目标化桥增强",
            "detail": "无差别的 geometry-only 平滑或扩张，不应系统性优于目标化的桥接增强和慢时标稳定修正。",
            "falsifier": "如果广泛几何平滑稳定优于目标化桥接增强，当前动态有效拓扑主张会被削弱。",
        },
        {
            "id": "P8C_pred_4",
            "title": "高价值长程桥应呈现束化与预算性",
            "detail": "真正高价值的长程桥应表现为少量、束化、重复被调用，而不是均匀弥散扩张。",
            "falsifier": "如果高效编码主要依赖广泛分散的长程连接而非束化稀疏桥，当前空间效率方程会被削弱。",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p8c_spatial_brain_falsifier_predictions",
        },
        "spatial_predictions": spatial_predictions,
        "pillars": {
            "falsifier_sharpness": {
                "components": falsifier_sharpness,
                "score": float(falsifier_sharpness_score),
            },
            "local_vs_bridge_prediction_specificity": {
                "components": local_vs_bridge_prediction_specificity,
                "score": float(local_vs_bridge_prediction_specificity_score),
            },
            "geometry_vs_targeted_prediction": {
                "components": geometry_vs_targeted_prediction,
                "score": float(geometry_vs_targeted_prediction_score),
            },
            "brain_spatial_testability": {
                "components": brain_spatial_testability,
                "score": float(brain_spatial_testability_score),
            },
        },
        "headline_metrics": {
            "falsifier_sharpness_score": float(falsifier_sharpness_score),
            "local_vs_bridge_prediction_specificity_score": float(
                local_vs_bridge_prediction_specificity_score
            ),
            "geometry_vs_targeted_prediction_score": float(geometry_vs_targeted_prediction_score),
            "brain_spatial_testability_score": float(brain_spatial_testability_score),
            "overall_p8c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_spatial_falsifiers_are_sharp": bool(falsifier_sharpness_score >= 0.81),
            "H2_local_vs_bridge_predictions_are_specific": bool(
                local_vs_bridge_prediction_specificity_score >= 0.46
            ),
            "H3_targeted_bridge_predictions_beat_geometry_only_predictions": bool(
                geometry_vs_targeted_prediction_score >= 0.66
            ),
            "H4_spatial_predictions_are_testable": bool(brain_spatial_testability_score >= 0.68),
            "H5_p8c_spatial_falsifier_predictions_are_moderately_supported": bool(overall_score >= 0.67),
        },
        "project_readout": {
            "summary": (
                "P8C is positive only if the spatialized theory can propose new falsifiable predictions about local "
                "reuse, sparse bridge vulnerability, and the failure of geometry-only interventions."
            ),
            "next_question": (
                "If P8C holds, the next stage should consolidate P8 into a full spatial plasticity coding theory and "
                "then move into direct attack on remaining residuals."
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
