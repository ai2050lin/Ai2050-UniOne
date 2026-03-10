"""
Brain-side learnable ranking two-layer unified law.

Compared with the previous brain + D + real-task co-calibration:
1. the old version hand-compressed brain rows into 4 summary signals
2. this version lets brain component scores enter the ranking layer directly

Goal:
- test whether brain-side constraints can be better aligned by a learnable
  ranking layer instead of a hand-written brain decomposition
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

COMPONENT_IDS = [
    "shared_basis",
    "sparse_offset",
    "topology_basis",
    "analogy_path",
    "protocol_routing",
    "abstraction_operator",
    "multi_timescale_control",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: List[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else 0.0


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_d_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = payload["residual_gate"]["top_overall"][0]
    rows = []
    mappings = [
        ("residual_gate", payload["residual_gate"]["top_overall"][0]),
        ("bayes_consolidation", payload["bayes_consolidation"]["top_overall"][0]),
        ("learned_controller", payload["learned_controller"]["best_overall"]),
        ("two_phase", payload["two_phase_consolidation"]["best_overall"]),
        ("three_phase", payload["three_phase_consolidation"]["best_overall"]),
        ("base_offset", payload["base_offset_consolidation"]["top_overall"][0]),
        ("offset_stabilization", payload["offset_stabilization"]["top_overall"][0]),
        ("multistage", payload["multistage_stabilization"]["top_overall"][0]),
    ]
    for subdomain, row in mappings:
        rows.append(
            {
                "row_id": f"D::{subdomain}",
                "domain": "D",
                "subdomain": subdomain,
                "signal_a": float(row["novel_gain"]),
                "signal_b": float(row["retention_gain"]),
                "signal_c": float(row["dual_score"]),
                "signal_d": 0.0,
                "brain_components": [0.0] * len(COMPONENT_IDS),
                "reference_score": float(row["overall_gain"]),
            }
        )
    return rows


def build_real_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in payload["rows"]:
        rows.append(
            {
                "row_id": row["row_id"],
                "domain": "real_task",
                "subdomain": row["model_name"],
                "signal_a": float(row["baseline_score"]),
                "signal_b": float(row["ranking_score"]),
                "signal_c": float(row["reference_score"] - row["baseline_score"]),
                "signal_d": float(row["reference_score"] - row["ranking_score"]),
                "brain_components": [0.0] * len(COMPONENT_IDS),
                "reference_score": float(row["reference_score"]),
            }
        )
    return rows


def build_brain_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, row in payload["models"].items():
        comps = row["components"]
        component_values = [float(comps[key]["score"]) for key in COMPONENT_IDS]
        rows.append(
            {
                "row_id": f"brain::{model_name}",
                "domain": "brain",
                "subdomain": model_name,
                "signal_a": 0.0,
                "signal_b": 0.0,
                "signal_c": 0.0,
                "signal_d": 0.0,
                "brain_components": component_values,
                "reference_score": float(row["overall_bridge_score"]),
                "brain_meta": {
                    "model_name": model_name,
                    "dnn_reverse_score": float(row["dnn_reverse_score"]),
                    "brain_alignment_score": float(row["brain_alignment_score"]),
                    "overall_bridge_score": float(row["overall_bridge_score"]),
                    "component_scores": {key: float(comps[key]["score"]) for key in COMPONENT_IDS},
                },
            }
        )
    return rows


def ranking_features(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row["signal_a"]),
            float(row["signal_b"]),
            float(row["signal_c"]),
            float(row["signal_d"]),
            *[float(v) for v in row["brain_components"]],
            1.0 if row["domain"] == "D" else 0.0,
            1.0 if row["domain"] == "real_task" else 0.0,
            1.0 if row["domain"] == "brain" else 0.0,
        ],
        dtype=np.float64,
    )


def calibration_features(row: Dict[str, Any], ranking_score: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(ranking_score),
            float(row["signal_c"]),
            float(row["signal_d"]),
            1.0 if row["domain"] == "D" else 0.0,
            1.0 if row["domain"] == "real_task" else 0.0,
            1.0 if row["domain"] == "brain" else 0.0,
        ],
        dtype=np.float64,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    gram = x.T @ x + ridge_lambda * eye
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def fit_ranking(rows: List[Dict[str, Any]], ridge_lambda: float) -> np.ndarray:
    x = np.stack([ranking_features(row) for row in rows])
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    return fit_ridge(x, y, ridge_lambda)


def fit_calibration(rows: List[Dict[str, Any]], ranking_scores: List[float], ridge_lambda: float) -> np.ndarray:
    x = np.stack([calibration_features(row, score) for row, score in zip(rows, ranking_scores)])
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    return fit_ridge(x, y, ridge_lambda)


def predict_ranking(row: Dict[str, Any], coef: np.ndarray) -> float:
    return float(ranking_features(row) @ coef)


def predict_calibrated(row: Dict[str, Any], ranking_score: float, coef: np.ndarray) -> float:
    return float(calibration_features(row, ranking_score) @ coef)


def evaluate(rows: List[Dict[str, Any]], ranking_coef: np.ndarray, calibration_coef: np.ndarray) -> Dict[str, Any]:
    refs: List[float] = []
    ranking_preds: List[float] = []
    calibrated_preds: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}
    out_rows = []
    brain_rows = []
    for row in rows:
        ranking_score = predict_ranking(row, ranking_coef)
        calibrated_score = predict_calibrated(row, ranking_score, calibration_coef)
        ref = float(row["reference_score"])
        gap = abs(calibrated_score - ref)
        refs.append(ref)
        ranking_preds.append(ranking_score)
        calibrated_preds.append(calibrated_score)
        gaps_by_domain[row["domain"]].append(gap)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "domain": row["domain"],
                "subdomain": row["subdomain"],
                "reference_score": ref,
                "ranking_score": ranking_score,
                "calibrated_score": calibrated_score,
                "absolute_gap": gap,
            }
        )
        if row["domain"] == "brain":
            brain_rows.append(
                {
                    **row["brain_meta"],
                    "ranking_score": ranking_score,
                    "calibrated_score": calibrated_score,
                    "absolute_gap": gap,
                }
            )
    return {
        "rows": out_rows,
        "brain_breakdown": brain_rows,
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
        "score_correlation": correlation(refs, calibrated_preds),
        "ranking_correlation": correlation(refs, ranking_preds),
        "brain_mean_gap": mean(gaps_by_domain["brain"]),
        "d_mean_gap": mean(gaps_by_domain["D"]),
        "real_task_mean_gap": mean(gaps_by_domain["real_task"]),
    }


def leave_one_out(rows: List[Dict[str, Any]], ranking_lambda: float, calibration_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    gaps: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}
    out_rows = []
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        ranking_coef = fit_ranking(train_rows, ranking_lambda)
        train_ranking_scores = [predict_ranking(row, ranking_coef) for row in train_rows]
        calibration_coef = fit_calibration(train_rows, train_ranking_scores, calibration_lambda)
        held_ranking = predict_ranking(held_row, ranking_coef)
        held_pred = predict_calibrated(held_row, held_ranking, calibration_coef)
        held_ref = float(held_row["reference_score"])
        gap = abs(held_pred - held_ref)
        refs.append(held_ref)
        preds.append(held_pred)
        gaps.append(gap)
        gaps_by_domain[held_row["domain"]].append(gap)
        out_rows.append(
            {
                "row_id": held_row["row_id"],
                "domain": held_row["domain"],
                "held_out_prediction": held_pred,
                "held_out_reference": held_ref,
                "held_out_gap": gap,
            }
        )
    return {
        "rows": out_rows,
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "brain_held_out_gap": mean(gaps_by_domain["brain"]),
        "d_held_out_gap": mean(gaps_by_domain["D"]),
        "real_task_held_out_gap": mean(gaps_by_domain["real_task"]),
        "pass": bool(mean(gaps) < 0.015 and correlation(refs, preds) > 0.90),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a learnable brain-ranking two-layer unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/brain_learnable_ranking_two_layer_unified_law_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = load_json(ROOT / "tests" / "codex_temp" / "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json")
    brain_payload = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")

    rows = build_brain_rows(brain_payload) + build_d_rows(d_payload) + build_real_rows(real_payload)

    best = None
    for ranking_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        ranking_coef = fit_ranking(rows, ranking_lambda)
        ranking_scores = [predict_ranking(row, ranking_coef) for row in rows]
        for calibration_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            calibration_coef = fit_calibration(rows, ranking_scores, calibration_lambda)
            fitted = evaluate(rows, ranking_coef, calibration_coef)
            held_out = leave_one_out(rows, ranking_lambda, calibration_lambda)
            objective = held_out["mean_held_out_gap"] - 0.010 * held_out["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (
                    objective,
                    float(ranking_lambda),
                    float(calibration_lambda),
                    ranking_coef,
                    calibration_coef,
                    fitted,
                    held_out,
                )

    _, ranking_lambda, calibration_lambda, ranking_coef, calibration_coef, fitted, held_out = best

    brain_component_coefficients = {
        comp_id: float(ranking_coef[5 + idx])
        for idx, comp_id in enumerate(COMPONENT_IDS)
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "source_files": [
                "brain_d_real_cocalibrated_two_layer_unified_law_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_brain_d_real": baseline_payload["brain_d_real_cocalibrated_two_layer_law"],
        "ranking_layer": {
            "ridge_lambda": ranking_lambda,
            "features": [
                "1",
                "signal_a",
                "signal_b",
                "signal_c",
                "signal_d",
                *COMPONENT_IDS,
                "D_indicator",
                "real_task_indicator",
                "brain_indicator",
            ],
            "coefficients": [float(x) for x in ranking_coef.tolist()],
            "brain_component_coefficients": brain_component_coefficients,
            "score_correlation": float(fitted["ranking_correlation"]),
        },
        "calibration_layer": {
            "ridge_lambda": calibration_lambda,
            "features": [
                "1",
                "ranking_score",
                "signal_c",
                "signal_d",
                "D_indicator",
                "real_task_indicator",
                "brain_indicator",
            ],
            "coefficients": [float(x) for x in calibration_coef.tolist()],
        },
        "brain_learnable_ranking_two_layer_law": {
            "mean_absolute_gap": float(fitted["mean_absolute_gap"]),
            "score_correlation": float(fitted["score_correlation"]),
            "brain_mean_gap": float(fitted["brain_mean_gap"]),
            "d_mean_gap": float(fitted["d_mean_gap"]),
            "real_task_mean_gap": float(fitted["real_task_mean_gap"]),
            "held_out_mean_gap": float(held_out["mean_held_out_gap"]),
            "held_out_score_correlation": float(held_out["held_out_score_correlation"]),
            "brain_held_out_gap": float(held_out["brain_held_out_gap"]),
            "d_held_out_gap": float(held_out["d_held_out_gap"]),
            "real_task_held_out_gap": float(held_out["real_task_held_out_gap"]),
            "brain_gap_improvement": float(baseline_payload["brain_d_real_cocalibrated_two_layer_law"]["brain_mean_gap"] - fitted["brain_mean_gap"]),
            "brain_held_out_improvement": float(baseline_payload["brain_d_real_cocalibrated_two_layer_law"]["brain_held_out_gap"] - held_out["brain_held_out_gap"]),
        },
        "brain_breakdown": fitted["brain_breakdown"],
        "rows": fitted["rows"],
        "leave_one_out": held_out,
        "project_readout": {
            "summary": "脑侧候选约束现在不再依赖手工聚合，而是直接进入可学习排序层。这样更接近把脑侧约束压成统一律中的一个可学习域。",
            "current_stage": "脑侧可学习排序层双层统一律",
            "next_question": "脑侧组件系数在扩展到更多脑侧候选时还能保持稳定吗？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
