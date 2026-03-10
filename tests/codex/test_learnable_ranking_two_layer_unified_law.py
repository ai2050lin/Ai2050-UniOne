"""
Fully learnable two-layer unified law.

Layer 1: learnable ranking layer
    learn a ridge-regularized ranking score from raw unified-law factors

Layer 2: learnable calibration layer
    fit another ridge-regularized head on top of the ranking score

This is the next step after learnable calibration:
we now test whether the ranking layer should also become learnable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


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


def ranking_features(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row["base"]),
            float(row["adaptive_offset"]),
            float(row["routing"]),
            float(row["stabilization"]),
            float(row["phase_gate"]),
            float(row["effective_offset"]) - float(row["adaptive_offset"]),
            1.0 if row["source_type"] == "bridge" else 0.0,
        ],
        dtype=np.float64,
    )


def calibration_features(row: Dict[str, Any], ranking_score: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            ranking_score,
            float(row["phase_gate"]),
            float(row["effective_offset"]) - float(row["adaptive_offset"]),
            1.0 if row["source_type"] == "bridge" else 0.0,
        ],
        dtype=np.float64,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + ridge_lambda * eye, x.T @ y)


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
    refs = []
    ranking_scores = []
    calibrated_scores = []
    out_rows = []
    for row in rows:
        ranking_score = predict_ranking(row, ranking_coef)
        calibrated_score = predict_calibrated(row, ranking_score, calibration_coef)
        refs.append(float(row["reference_score"]))
        ranking_scores.append(ranking_score)
        calibrated_scores.append(calibrated_score)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "source_type": row["source_type"],
                "reference_score": float(row["reference_score"]),
                "ranking_score": ranking_score,
                "calibrated_score": calibrated_score,
                "absolute_gap": abs(calibrated_score - float(row["reference_score"])),
            }
        )
    return {
        "rows": out_rows,
        "ranking_score_correlation": correlation(refs, ranking_scores),
        "score_correlation": correlation(refs, calibrated_scores),
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
    }


def leave_one_out(rows: List[Dict[str, Any]], ranking_lambda: float, calibration_lambda: float) -> Dict[str, Any]:
    gaps = []
    rank_refs = []
    rank_preds = []
    calibrated_refs = []
    calibrated_preds = []
    out_rows = []
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        ranking_coef = fit_ranking(train_rows, ranking_lambda)
        train_ranking_scores = [predict_ranking(row, ranking_coef) for row in train_rows]
        calibration_coef = fit_calibration(train_rows, train_ranking_scores, calibration_lambda)

        held_ranking = predict_ranking(held_row, ranking_coef)
        held_calibrated = predict_calibrated(held_row, held_ranking, calibration_coef)
        held_ref = float(held_row["reference_score"])
        gap = abs(held_calibrated - held_ref)
        gaps.append(gap)
        rank_refs.append(held_ref)
        rank_preds.append(held_ranking)
        calibrated_refs.append(held_ref)
        calibrated_preds.append(held_calibrated)
        out_rows.append(
            {
                "row_id": held_row["row_id"],
                "source_type": held_row["source_type"],
                "held_out_ranking_score": held_ranking,
                "held_out_prediction": held_calibrated,
                "held_out_reference": held_ref,
                "held_out_gap": gap,
            }
        )
    return {
        "rows": out_rows,
        "mean_held_out_gap": mean(gaps),
        "held_out_ranking_correlation": correlation(rank_refs, rank_preds),
        "held_out_score_correlation": correlation(calibrated_refs, calibrated_preds),
        "pass": bool(mean(gaps) < 0.03 and correlation(calibrated_refs, calibrated_preds) > 0.9),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a learnable ranking + calibration unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/learnable_ranking_two_layer_unified_law_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    phase_gated = load_json(ROOT / "tests" / "codex_temp" / "phase_gated_unified_update_law_20260309.json")
    baseline = load_json(ROOT / "tests" / "codex_temp" / "learnable_two_layer_unified_law_20260309.json")
    rows = phase_gated["rows"]

    best = None
    ranking_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]
    calibration_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]

    for ranking_lambda in ranking_lambdas:
        ranking_coef = fit_ranking(rows, ranking_lambda)
        ranking_scores = [predict_ranking(row, ranking_coef) for row in rows]
        for calibration_lambda in calibration_lambdas:
            calibration_coef = fit_calibration(rows, ranking_scores, calibration_lambda)
            fitted = evaluate(rows, ranking_coef, calibration_coef)
            held_out = leave_one_out(rows, ranking_lambda, calibration_lambda)
            objective = held_out["mean_held_out_gap"] - 0.015 * held_out["held_out_score_correlation"]
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

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "source": "phase_gated_unified_update_law_20260309.json",
        },
        "baseline_learnable_two_layer": {
            "mean_absolute_gap": float(baseline["learnable_two_layer_law"]["mean_absolute_gap"]),
            "score_correlation": float(baseline["learnable_two_layer_law"]["score_correlation"]),
            "mean_held_out_gap": float(baseline["leave_one_out"]["mean_held_out_gap"]),
        },
        "ranking_layer": {
            "ridge_lambda": ranking_lambda,
            "features": [
                "1",
                "base",
                "adaptive_offset",
                "routing",
                "stabilization",
                "phase_gate",
                "effective_offset - adaptive_offset",
                "bridge_indicator",
            ],
            "coefficients": [float(x) for x in ranking_coef.tolist()],
            "score_correlation": fitted["ranking_score_correlation"],
        },
        "calibration_layer": {
            "ridge_lambda": calibration_lambda,
            "features": [
                "1",
                "ranking_score",
                "phase_gate",
                "effective_offset - adaptive_offset",
                "bridge_indicator",
            ],
            "coefficients": [float(x) for x in calibration_coef.tolist()],
        },
        "learnable_ranking_two_layer_law": {
            "mean_absolute_gap": fitted["mean_absolute_gap"],
            "score_correlation": fitted["score_correlation"],
            "held_out_mean_gap": held_out["mean_held_out_gap"],
            "held_out_score_correlation": held_out["held_out_score_correlation"],
            "gap_improvement_vs_baseline": float(baseline["learnable_two_layer_law"]["mean_absolute_gap"]) - fitted["mean_absolute_gap"],
            "correlation_improvement_vs_baseline": fitted["score_correlation"] - float(baseline["learnable_two_layer_law"]["score_correlation"]),
        },
        "leave_one_out": held_out,
        "rows": fitted["rows"],
        "project_readout": {
            "current_verdict": (
                "如果排序层也学出来之后，当前误差、相关性和留一法同时稳定，"
                "就说明统一律已经从“可学习标定层”推进到“可学习双层机制”。"
            ),
            "next_question": "下一步是否应该把这套可学习双层统一律直接接入真实任务收益，而不再只在桥接样本上验证？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["learnable_ranking_two_layer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
