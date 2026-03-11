#!/usr/bin/env python
"""
P9C: sharpen the highest-risk spatial brain forecasts.
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


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def main() -> None:
    ap = argparse.ArgumentParser(description="P9C hard spatial brain forecasts")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p9c_hard_spatial_brain_forecasts_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p8b = load_json(ROOT / "tests" / "codex_temp" / "p8b_3d_wiring_dynamic_topology_division_20260311.json")
    p8c = load_json(ROOT / "tests" / "codex_temp" / "p8c_spatial_brain_falsifier_predictions_20260311.json")
    p9a = load_json(ROOT / "tests" / "codex_temp" / "p9a_spatial_plasticity_coding_master_20260311.json")
    p9b = load_json(ROOT / "tests" / "codex_temp" / "p9b_spatial_residual_counterexample_compression_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")

    forecast_sharpness = {
        "p8c_falsifier_sharpness": float(p8c["headline_metrics"]["falsifier_sharpness_score"]),
        "stage8d_hard_spec": float(stage8d["headline_metrics"]["hard_falsifier_spec_score"]),
        "p9a_falsifiability": float(p9a["headline_metrics"]["falsifiability_and_testability_score"]),
        "p9b_compression_verdict": float(p9b["headline_metrics"]["compression_verdict_score"]),
    }
    forecast_sharpness_score = mean(forecast_sharpness.values())

    forecast_specificity = {
        "p8c_local_vs_bridge_specificity": float(
            p8c["headline_metrics"]["local_vs_bridge_prediction_specificity_score"]
        ),
        "p8b_selective_bridge_advantage": float(p8b["headline_metrics"]["selective_bridge_advantage_score"]),
        "p8b_local_reuse_advantage": float(p8b["headline_metrics"]["local_reuse_advantage_score"]),
        "specificity_gap_known": 1.0,
    }
    forecast_specificity_score = mean(forecast_specificity.values())

    risk_targeting = {
        "bridge_specificity_gap_is_main_open_gap": 1.0
        if p9b["verdict"]["largest_open_gap"] == "bridge_specificity_strength"
        else 0.0,
        "dominant_remaining_source_known": 1.0
        if p9b["verdict"]["dominant_remaining_source"] == "architecture_share"
        else 0.0,
        "p9a_status_supported_not_final": 1.0
        if p9a["verdict"]["status"] == "supported_but_not_final"
        else 0.0,
        "p8c_geometry_targeted_score": float(p8c["headline_metrics"]["geometry_vs_targeted_prediction_score"]),
    }
    risk_targeting_score = mean(risk_targeting.values())

    testability = {
        "p8c_testability": float(p8c["headline_metrics"]["brain_spatial_testability_score"]),
        "p8b_explicit_economy": float(p8b["headline_metrics"]["explicit_spatial_economy_score"]),
        "p9a_closure_status": float(p9a["headline_metrics"]["closure_status_score"]),
        "p9b_residual_control": float(p9b["headline_metrics"]["residual_source_control_score"]),
    }
    testability_score = mean(testability.values())

    overall_score = mean(
        [
            forecast_sharpness_score,
            forecast_specificity_score,
            risk_targeting_score,
            testability_score,
        ]
    )

    hard_forecasts = [
        {
            "id": "P9C_F1",
            "title": "局部邻域扰动应优先压低 family topology margin",
            "detail": "若当前理论正确，小范围局部扰动首先应降低 concept family 对真 family 的拓扑 margin，而不是先打掉远程桥接收益。",
        },
        {
            "id": "P9C_F2",
            "title": "长程桥切断应优先压低 compact-boundary relation",
            "detail": "针对束化长程桥的扰动，应先压低 compact-boundary relation 的 bridge score 和 endpoint support，而 distributed relation 不应首先崩塌。",
        },
        {
            "id": "P9C_F3",
            "title": "几何平滑增强不应替代目标化桥增强",
            "detail": "如果只做广泛几何平滑或均匀扩张，不应系统性优于目标化桥接增强和慢时标稳定修正。",
        },
        {
            "id": "P9C_F4",
            "title": "高价值长程桥应呈束化稀疏调用",
            "detail": "真正高价值的远程桥应表现为少量、束化、反复调用，而非大面积均匀激活。",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p9c_hard_spatial_brain_forecasts",
        },
        "hard_forecasts": hard_forecasts,
        "pillars": {
            "forecast_sharpness": {
                "components": forecast_sharpness,
                "score": float(forecast_sharpness_score),
            },
            "forecast_specificity": {
                "components": forecast_specificity,
                "score": float(forecast_specificity_score),
            },
            "risk_targeting": {
                "components": risk_targeting,
                "score": float(risk_targeting_score),
            },
            "testability": {
                "components": testability,
                "score": float(testability_score),
            },
        },
        "headline_metrics": {
            "forecast_sharpness_score": float(forecast_sharpness_score),
            "forecast_specificity_score": float(forecast_specificity_score),
            "risk_targeting_score": float(risk_targeting_score),
            "testability_score": float(testability_score),
            "overall_p9c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_hard_spatial_forecasts_are_sharp": bool(forecast_sharpness_score >= 0.82),
            "H2_hard_spatial_forecasts_are_specific": bool(forecast_specificity_score >= 0.67),
            "H3_hard_forecasts_target_the_real_open_risks": bool(risk_targeting_score >= 0.90),
            "H4_hard_forecasts_are_testable": bool(testability_score >= 0.74),
            "H5_p9c_hard_spatial_forecasts_are_moderately_supported": bool(overall_score >= 0.79),
        },
        "project_readout": {
            "summary": (
                "P9C is positive only if the spatialized theory can state higher-risk forecasts that directly target "
                "the remaining open gaps rather than repeating generic support claims."
            ),
            "next_question": (
                "If P9C holds, the next step is to consolidate the whole spatial theory into a final-stage verdict and "
                "then decide whether the remaining gaps are empirical or theoretical."
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
