"""
D + real-task co-calibrated two-layer unified law.

This experiment merges:
1. D-side external closure signals
2. real-task behavior gains

into one shared ranking layer + calibration layer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

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


def build_d_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "D::residual_gate",
            "domain": "D",
            "subdomain": "residual_gate",
            "signal_a": float(payload["residual_gate"]["top_overall"][0]["novel_gain"]),
            "signal_b": float(payload["residual_gate"]["top_overall"][0]["retention_gain"]),
            "signal_c": float(payload["residual_gate"]["top_overall"][0]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["residual_gate"]["top_overall"][0]["overall_gain"]),
        },
        {
            "row_id": "D::bayes_consolidation",
            "domain": "D",
            "subdomain": "bayes_consolidation",
            "signal_a": float(payload["bayes_consolidation"]["top_overall"][0]["novel_gain"]),
            "signal_b": float(payload["bayes_consolidation"]["top_overall"][0]["retention_gain"]),
            "signal_c": float(payload["bayes_consolidation"]["top_overall"][0]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["bayes_consolidation"]["top_overall"][0]["overall_gain"]),
        },
        {
            "row_id": "D::learned_controller",
            "domain": "D",
            "subdomain": "learned_controller",
            "signal_a": float(payload["learned_controller"]["best_overall"]["novel_gain"]),
            "signal_b": float(payload["learned_controller"]["best_overall"]["retention_gain"]),
            "signal_c": float(payload["learned_controller"]["best_overall"]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["learned_controller"]["best_overall"]["overall_gain"]),
        },
        {
            "row_id": "D::two_phase",
            "domain": "D",
            "subdomain": "two_phase",
            "signal_a": float(payload["two_phase_consolidation"]["best_overall"]["novel_gain"]),
            "signal_b": float(payload["two_phase_consolidation"]["best_overall"]["retention_gain"]),
            "signal_c": float(payload["two_phase_consolidation"]["best_overall"]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["two_phase_consolidation"]["best_overall"]["overall_gain"]),
        },
        {
            "row_id": "D::three_phase",
            "domain": "D",
            "subdomain": "three_phase",
            "signal_a": float(payload["three_phase_consolidation"]["best_overall"]["novel_gain"]),
            "signal_b": float(payload["three_phase_consolidation"]["best_overall"]["retention_gain"]),
            "signal_c": float(payload["three_phase_consolidation"]["best_overall"]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["three_phase_consolidation"]["best_overall"]["overall_gain"]),
        },
        {
            "row_id": "D::base_offset",
            "domain": "D",
            "subdomain": "base_offset",
            "signal_a": float(payload["base_offset_consolidation"]["top_overall"][0]["novel_gain"]),
            "signal_b": float(payload["base_offset_consolidation"]["top_overall"][0]["retention_gain"]),
            "signal_c": float(payload["base_offset_consolidation"]["top_overall"][0]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["base_offset_consolidation"]["top_overall"][0]["overall_gain"]),
        },
        {
            "row_id": "D::offset_stabilization",
            "domain": "D",
            "subdomain": "offset_stabilization",
            "signal_a": float(payload["offset_stabilization"]["top_overall"][0]["novel_gain"]),
            "signal_b": float(payload["offset_stabilization"]["top_overall"][0]["retention_gain"]),
            "signal_c": float(payload["offset_stabilization"]["top_overall"][0]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["offset_stabilization"]["top_overall"][0]["overall_gain"]),
        },
        {
            "row_id": "D::multistage",
            "domain": "D",
            "subdomain": "multistage",
            "signal_a": float(payload["multistage_stabilization"]["top_overall"][0]["novel_gain"]),
            "signal_b": float(payload["multistage_stabilization"]["top_overall"][0]["retention_gain"]),
            "signal_c": float(payload["multistage_stabilization"]["top_overall"][0]["dual_score"]),
            "signal_d": 0.0,
            "reference_score": float(payload["multistage_stabilization"]["top_overall"][0]["overall_gain"]),
        },
    ]


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
                "reference_score": float(row["reference_score"]),
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
            1.0 if row["domain"] == "D" else 0.0,
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
    ranking_preds = []
    calibrated_preds = []
    d_gaps = []
    real_gaps = []
    out_rows = []
    for row in rows:
        ranking_score = predict_ranking(row, ranking_coef)
        calibrated_score = predict_calibrated(row, ranking_score, calibration_coef)
        ref = float(row["reference_score"])
        gap = abs(calibrated_score - ref)
        refs.append(ref)
        ranking_preds.append(ranking_score)
        calibrated_preds.append(calibrated_score)
        if row["domain"] == "D":
            d_gaps.append(gap)
        else:
            real_gaps.append(gap)
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
    return {
        "rows": out_rows,
        "ranking_correlation": correlation(refs, ranking_preds),
        "score_correlation": correlation(refs, calibrated_preds),
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
        "d_mean_gap": mean(d_gaps),
        "real_task_mean_gap": mean(real_gaps),
    }


def leave_one_out(rows: List[Dict[str, Any]], ranking_lambda: float, calibration_lambda: float) -> Dict[str, Any]:
    refs = []
    preds = []
    gaps = []
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
        "pass": bool(mean(gaps) < 0.02 and correlation(refs, preds) > 0.82),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a D + real-task co-calibrated two-layer unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/d_real_task_cocalibrated_two_layer_unified_law_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")
    rows = build_d_rows(d_payload) + build_real_rows(real_payload)

    best = None
    for ranking_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        ranking_coef = fit_ranking(rows, ranking_lambda)
        ranking_scores = [predict_ranking(row, ranking_coef) for row in rows]
        for calibration_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            calibration_coef = fit_calibration(rows, ranking_scores, calibration_lambda)
            fitted = evaluate(rows, ranking_coef, calibration_coef)
            held_out = leave_one_out(rows, ranking_lambda, calibration_lambda)
            objective = held_out["mean_held_out_gap"] - 0.008 * held_out["held_out_score_correlation"]
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
            "row_count": len(rows),
        },
        "ranking_layer": {
            "ridge_lambda": ranking_lambda,
            "features": [
                "1",
                "signal_a",
                "signal_b",
                "signal_c",
                "signal_d",
                "D_indicator",
            ],
            "coefficients": [float(x) for x in ranking_coef.tolist()],
            "score_correlation": fitted["ranking_correlation"],
        },
        "calibration_layer": {
            "ridge_lambda": calibration_lambda,
            "features": [
                "1",
                "ranking_score",
                "signal_c",
                "signal_d",
                "D_indicator",
            ],
            "coefficients": [float(x) for x in calibration_coef.tolist()],
        },
        "cocalibrated_two_layer_law": {
            "mean_absolute_gap": fitted["mean_absolute_gap"],
            "score_correlation": fitted["score_correlation"],
            "d_mean_gap": fitted["d_mean_gap"],
            "real_task_mean_gap": fitted["real_task_mean_gap"],
            "held_out_mean_gap": held_out["mean_held_out_gap"],
            "held_out_score_correlation": held_out["held_out_score_correlation"],
        },
        "leave_one_out": held_out,
        "rows": fitted["rows"],
        "project_readout": {
            "current_verdict": (
                "如果 D 和真实任务能被同一套排序层 + 标定层共同解释，"
                "就说明外部闭环已经开始收敛到单一双层统一律。"
            ),
            "next_question": "下一步是否应该再把脑侧候选约束也接入这一套共标定双层统一律？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["cocalibrated_two_layer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
