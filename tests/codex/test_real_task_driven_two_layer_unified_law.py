"""
Real-task-driven two-layer unified law.

This experiment moves the unified law out of bridge-only rows and into the
concept-conditioned real-task benchmark.

Layer 1: learn a structural ranking score from concept + relation features
Layer 2: calibrate that ranking score to task behavior gain
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


def build_rows(task_payload: Dict[str, Any], relation_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name, model_block in task_payload["models"].items():
        relation_block = relation_payload["models"][model_name]["relations"]
        for task_id, task_row in model_block["tasks"].items():
            relation_name = task_row["relation"]
            relation_row = relation_block[relation_name]
            rows.append(
                {
                    "row_id": f"{model_name}::{task_id}",
                    "model_name": model_name,
                    "concept": task_row["concept"],
                    "relation": relation_name,
                    "concept_score": float(task_row["concept_score"]),
                    "compatibility": float(task_row["compatibility"]),
                    "relation_bridge_score": float(relation_row["bridge_score"]),
                    "endpoint_support_rate": float(relation_row["endpoint_support_rate"]),
                    "topology_compactness": float(relation_row["topology_compactness"]),
                    "top4_bridge_share": float(relation_row["top4_bridge_share_in_top20"]),
                    "reference_score": float(task_row["behavior_gain"]),
                }
            )
    return rows


def baseline_score(row: Dict[str, Any]) -> float:
    return float(
        0.35 * row["concept_score"]
        + 0.30 * row["compatibility"]
        + 0.20 * row["relation_bridge_score"]
        + 0.10 * row["topology_compactness"]
        + 0.05 * row["top4_bridge_share"]
    )


def ranking_features(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row["concept_score"]),
            float(row["compatibility"]),
            float(row["relation_bridge_score"]),
            float(row["endpoint_support_rate"]),
            float(row["topology_compactness"]),
            float(row["top4_bridge_share"]),
            1.0 if row["model_name"] == "deepseek_7b" else 0.0,
        ],
        dtype=np.float64,
    )


def calibration_features(row: Dict[str, Any], ranking_score: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(ranking_score),
            float(row["compatibility"]),
            float(row["concept_score"]),
            1.0 if row["model_name"] == "deepseek_7b" else 0.0,
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
    baseline_refs = []
    baseline_preds = []
    refs = []
    ranking_preds = []
    calibrated_preds = []
    out_rows = []
    for row in rows:
        raw_baseline = baseline_score(row)
        ranking_score = predict_ranking(row, ranking_coef)
        calibrated_score = predict_calibrated(row, ranking_score, calibration_coef)
        ref = float(row["reference_score"])
        baseline_refs.append(ref)
        baseline_preds.append(raw_baseline)
        refs.append(ref)
        ranking_preds.append(ranking_score)
        calibrated_preds.append(calibrated_score)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "model_name": row["model_name"],
                "concept": row["concept"],
                "relation": row["relation"],
                "reference_score": ref,
                "baseline_score": raw_baseline,
                "ranking_score": ranking_score,
                "calibrated_score": calibrated_score,
                "absolute_gap": abs(calibrated_score - ref),
            }
        )
    return {
        "rows": out_rows,
        "baseline_correlation": correlation(baseline_refs, baseline_preds),
        "ranking_correlation": correlation(refs, ranking_preds),
        "score_correlation": correlation(refs, calibrated_preds),
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
    }


def leave_one_out(rows: List[Dict[str, Any]], ranking_lambda: float, calibration_lambda: float) -> Dict[str, Any]:
    gaps = []
    refs = []
    preds = []
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
        gaps.append(gap)
        refs.append(held_ref)
        preds.append(held_pred)
        out_rows.append(
            {
                "row_id": held_row["row_id"],
                "model_name": held_row["model_name"],
                "held_out_prediction": held_pred,
                "held_out_reference": held_ref,
                "held_out_gap": gap,
            }
        )
    return {
        "rows": out_rows,
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "pass": bool(mean(gaps) < 0.02 and correlation(refs, preds) > 0.85),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a real-task-driven two-layer unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/real_task_driven_two_layer_unified_law_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    task_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_structure_task_real_bridge_20260309.json")
    relation_payload = load_json(ROOT / "tests" / "codex_temp" / "qwen3_deepseek7b_relation_behavior_bridge_20260309.json")
    rows = build_rows(task_payload, relation_payload)

    best = None
    for ranking_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        ranking_coef = fit_ranking(rows, ranking_lambda)
        ranking_scores = [predict_ranking(row, ranking_coef) for row in rows]
        for calibration_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            calibration_coef = fit_calibration(rows, ranking_scores, calibration_lambda)
            fitted = evaluate(rows, ranking_coef, calibration_coef)
            held_out = leave_one_out(rows, ranking_lambda, calibration_lambda)
            objective = held_out["mean_held_out_gap"] - 0.01 * held_out["held_out_score_correlation"]
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
            "task_rows": len(rows),
        },
        "baseline": {
            "description": "concept_score + compatibility + relation_bridge_score 的手工结构基线",
            "score_correlation": fitted["baseline_correlation"],
        },
        "ranking_layer": {
            "ridge_lambda": ranking_lambda,
            "features": [
                "1",
                "concept_score",
                "compatibility",
                "relation_bridge_score",
                "endpoint_support_rate",
                "topology_compactness",
                "top4_bridge_share",
                "deepseek_indicator",
            ],
            "coefficients": [float(x) for x in ranking_coef.tolist()],
            "score_correlation": fitted["ranking_correlation"],
        },
        "calibration_layer": {
            "ridge_lambda": calibration_lambda,
            "features": [
                "1",
                "ranking_score",
                "compatibility",
                "concept_score",
                "deepseek_indicator",
            ],
            "coefficients": [float(x) for x in calibration_coef.tolist()],
        },
        "real_task_two_layer_law": {
            "mean_absolute_gap": fitted["mean_absolute_gap"],
            "score_correlation": fitted["score_correlation"],
            "held_out_mean_gap": held_out["mean_held_out_gap"],
            "held_out_score_correlation": held_out["held_out_score_correlation"],
            "correlation_improvement_vs_baseline": fitted["score_correlation"] - fitted["baseline_correlation"],
        },
        "leave_one_out": held_out,
        "rows": fitted["rows"],
        "project_readout": {
            "current_verdict": (
                "如果双层统一律在真实任务行上也能保持高相关和低留一法误差，"
                "就说明这条路线已经开始从内部桥接走向真实行为闭环。"
            ),
            "next_question": "下一步是否应该把这套真实任务双层统一律反向接回 D，形成同一套外部驱动标定？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["real_task_two_layer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
