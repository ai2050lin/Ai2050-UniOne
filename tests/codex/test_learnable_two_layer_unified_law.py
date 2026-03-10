"""
Learnable two-layer unified law with ridge-regularized calibration.

排序层：
    使用 phase-gated 统一律分数

标定层：
    在 [phase_gate, effective_offset shift, ranking_score, bridge_indicator]
    上训练一个带 ridge 正则的小线性头

这是从手工标定层推进到可学习标定层的第一步。
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


def features(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row["phase_gate"]),
            float(row["effective_offset"]) - float(row["adaptive_offset"]),
            float(row["unified_score"]),
            1.0 if row["source_type"] == "bridge" else 0.0,
        ],
        dtype=np.float64,
    )


def fit_ridge(rows: List[Dict[str, Any]], ridge_lambda: float) -> np.ndarray:
    x = np.stack([features(row) for row in rows])
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + ridge_lambda * eye, x.T @ y)
    return coef


def predict(row: Dict[str, Any], coef: np.ndarray) -> float:
    return float(features(row) @ coef)


def evaluate(rows: List[Dict[str, Any]], coef: np.ndarray) -> Dict[str, Any]:
    out_rows = []
    refs = []
    preds = []
    for row in rows:
        calibrated_score = predict(row, coef)
        refs.append(float(row["reference_score"]))
        preds.append(calibrated_score)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "source_type": row["source_type"],
                "reference_score": float(row["reference_score"]),
                "ranking_score": float(row["unified_score"]),
                "phase_gate": float(row["phase_gate"]),
                "calibrated_score": calibrated_score,
                "absolute_gap": abs(calibrated_score - float(row["reference_score"])),
            }
        )
    return {
        "rows": out_rows,
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
        "score_correlation": correlation(refs, preds),
    }


def leave_one_out(rows: List[Dict[str, Any]], ridge_lambda: float) -> Dict[str, Any]:
    held_rows = []
    gaps = []
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        coef = fit_ridge(train_rows, ridge_lambda)
        pred = predict(held_row, coef)
        gap = abs(pred - float(held_row["reference_score"]))
        gaps.append(gap)
        held_rows.append(
            {
                "row_id": held_row["row_id"],
                "source_type": held_row["source_type"],
                "held_out_prediction": pred,
                "held_out_reference": float(held_row["reference_score"]),
                "held_out_gap": gap,
            }
        )
    return {
        "rows": held_rows,
        "mean_held_out_gap": mean(gaps),
        "pass": bool(mean(gaps) < 0.04),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a learnable two-layer unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/learnable_two_layer_unified_law_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    phase_gated = load_json(ROOT / "tests" / "codex_temp" / "phase_gated_unified_update_law_20260309.json")
    rows = phase_gated["rows"]

    best = None
    for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0, 5.0, 10.0]:
        coef = fit_ridge(rows, ridge_lambda)
        fitted = evaluate(rows, coef)
        held_out = leave_one_out(rows, ridge_lambda)
        objective = held_out["mean_held_out_gap"] - 0.02 * fitted["score_correlation"]
        if best is None or objective < best[0]:
            best = (objective, float(ridge_lambda), coef, fitted, held_out)

    _, ridge_lambda, coef, fitted, held_out = best

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "source": "phase_gated_unified_update_law_20260309.json",
        },
        "ranking_layer": {
            "mean_absolute_gap": float(phase_gated["phase_gated_law"]["mean_absolute_gap"]),
            "score_correlation": float(phase_gated["phase_gated_law"]["score_correlation"]),
        },
        "learnable_calibration_layer": {
            "ridge_lambda": ridge_lambda,
            "features": [
                "1",
                "phase_gate",
                "effective_offset - adaptive_offset",
                "ranking_score",
                "bridge_indicator",
            ],
            "coefficients": [float(x) for x in coef.tolist()],
        },
        "learnable_two_layer_law": {
            "mean_absolute_gap": fitted["mean_absolute_gap"],
            "score_correlation": fitted["score_correlation"],
            "gap_improvement_vs_ranking": float(phase_gated["phase_gated_law"]["mean_absolute_gap"]) - fitted["mean_absolute_gap"],
            "correlation_improvement_vs_ranking": fitted["score_correlation"] - float(phase_gated["phase_gated_law"]["score_correlation"]),
        },
        "leave_one_out": held_out,
        "rows": fitted["rows"],
        "project_readout": {
            "current_verdict": (
                "可学习双层统一律如果能显著改善留一法，"
                "就说明主线应从解析标定层正式切到可学习标定层。"
            ),
            "next_question": "下一步是否应该把 ranking layer 也一起学出来，而不再固定 phase-gated 排序层？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["learnable_two_layer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
