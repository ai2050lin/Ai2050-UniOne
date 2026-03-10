"""
Cross-domain closure test for the current semantic 4D confidence packet.

Goal:
- reuse the existing brain + D + real-task row construction
- compress each row into a semantic 4D confidence packet
- test whether this packet alone can support a shared ranking + calibration law
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import test_brain_learnable_ranking_two_layer_unified_law as brain_law


ROOT = Path(__file__).resolve().parents[2]

SEMANTIC_PACKET_IDS = [
    "family_mean_confidence",
    "worst_case_confidence",
    "global_trust_reserve",
    "family_concept_margin",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: List[float]) -> float:
    return float(np.mean(np.array(xs, dtype=np.float64))) if xs else 0.0


def std(xs: List[float]) -> float:
    return float(np.std(np.array(xs, dtype=np.float64))) if xs else 0.0


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def semantic_packet(row: Dict[str, Any]) -> np.ndarray:
    if row["domain"] == "brain":
        raw = [float(v) for v in row["brain_components"]]
    else:
        raw = [
            float(row["signal_a"]),
            float(row["signal_b"]),
            float(row["signal_c"]),
            float(row["signal_d"]),
        ]

    sorted_raw = sorted(raw)
    lower_half = sorted_raw[: max(1, len(sorted_raw) // 2)]
    upper_half = sorted_raw[-max(1, len(sorted_raw) // 2):]

    packet = np.array(
        [
            mean(raw),
            min(raw) if raw else 0.0,
            max(raw) - std(raw) if raw else 0.0,
            mean(upper_half) - mean(lower_half),
        ],
        dtype=np.float64,
    )
    return packet


def ranking_features(row: Dict[str, Any]) -> np.ndarray:
    packet = semantic_packet(row)
    return np.array(
        [
            1.0,
            *[float(v) for v in packet],
            1.0 if row["domain"] == "brain" else 0.0,
            1.0 if row["domain"] == "D" else 0.0,
            1.0 if row["domain"] == "real_task" else 0.0,
        ],
        dtype=np.float64,
    )


def calibration_features(row: Dict[str, Any], ranking_score: float) -> np.ndarray:
    packet = semantic_packet(row)
    return np.array(
        [
            1.0,
            float(ranking_score),
            float(packet[2]),
            float(packet[3]),
            1.0 if row["domain"] == "brain" else 0.0,
            1.0 if row["domain"] == "D" else 0.0,
            1.0 if row["domain"] == "real_task" else 0.0,
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
                "semantic_packet": {
                    semantic_id: float(value)
                    for semantic_id, value in zip(SEMANTIC_PACKET_IDS, semantic_packet(row).tolist())
                },
            }
        )
    return {
        "rows": out_rows,
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
    return {
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "brain_held_out_gap": mean(gaps_by_domain["brain"]),
        "d_held_out_gap": mean(gaps_by_domain["D"]),
        "real_task_held_out_gap": mean(gaps_by_domain["real_task"]),
        "pass": bool(mean(gaps) < 0.02 and correlation(refs, preds) > 0.90),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test semantic 4D confidence packet cross-domain closure")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/semantic_4d_confidence_cross_domain_closure_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = load_json(ROOT / "tests" / "codex_temp" / "brain_learnable_ranking_two_layer_unified_law_20260310.json")
    brain_payload = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")

    rows = brain_law.build_brain_rows(brain_payload) + brain_law.build_d_rows(d_payload) + brain_law.build_real_rows(real_payload)

    best = None
    for ranking_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        ranking_coef = fit_ranking(rows, ranking_lambda)
        ranking_scores = [predict_ranking(row, ranking_coef) for row in rows]
        for calibration_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            calibration_coef = fit_calibration(rows, ranking_scores, calibration_lambda)
            fitted = evaluate(rows, ranking_coef, calibration_coef)
            held = leave_one_out(rows, ranking_lambda, calibration_lambda)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (
                    objective,
                    float(ranking_lambda),
                    float(calibration_lambda),
                    ranking_coef,
                    calibration_coef,
                    fitted,
                    held,
                )

    assert best is not None
    _, ranking_lambda, calibration_lambda, ranking_coef, calibration_coef, fitted, held = best

    packet_coefficients = {
        semantic_id: float(ranking_coef[1 + idx]) for idx, semantic_id in enumerate(SEMANTIC_PACKET_IDS)
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "source_files": [
                "brain_learnable_ranking_two_layer_unified_law_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_brain_learnable_ranking": baseline_payload["brain_learnable_ranking_two_layer_law"],
        "semantic_packet_ids": SEMANTIC_PACKET_IDS,
        "ranking_layer": {
            "ridge_lambda": ranking_lambda,
            "features": [
                "1",
                *SEMANTIC_PACKET_IDS,
                "brain_indicator",
                "D_indicator",
                "real_task_indicator",
            ],
            "coefficients": [float(x) for x in ranking_coef.tolist()],
            "packet_coefficients": packet_coefficients,
            "score_correlation": float(fitted["ranking_correlation"]),
        },
        "calibration_layer": {
            "ridge_lambda": calibration_lambda,
            "features": [
                "1",
                "ranking_score",
                "global_trust_reserve",
                "family_concept_margin",
                "brain_indicator",
                "D_indicator",
                "real_task_indicator",
            ],
            "coefficients": [float(x) for x in calibration_coef.tolist()],
        },
        "semantic_4d_cross_domain_closure": {
            "mean_absolute_gap": float(fitted["mean_absolute_gap"]),
            "score_correlation": float(fitted["score_correlation"]),
            "brain_mean_gap": float(fitted["brain_mean_gap"]),
            "d_mean_gap": float(fitted["d_mean_gap"]),
            "real_task_mean_gap": float(fitted["real_task_mean_gap"]),
            "held_out_mean_gap": float(held["mean_held_out_gap"]),
            "held_out_score_correlation": float(held["held_out_score_correlation"]),
            "brain_held_out_gap": float(held["brain_held_out_gap"]),
            "d_held_out_gap": float(held["d_held_out_gap"]),
            "real_task_held_out_gap": float(held["real_task_held_out_gap"]),
            "gap_delta_vs_baseline": float(
                baseline_payload["brain_learnable_ranking_two_layer_law"]["held_out_mean_gap"] - held["mean_held_out_gap"]
            ),
            "corr_delta_vs_baseline": float(
                held["held_out_score_correlation"] - baseline_payload["brain_learnable_ranking_two_layer_law"]["held_out_score_correlation"]
            ),
        },
        "rows": fitted["rows"],
        "leave_one_out": held,
        "project_readout": {
            "summary": "这一步直接测试语义 4D 置信状态包能否单独支撑 brain / D / real-task 的统一跨域闭环，而不再依赖更大的组件特征集合。",
            "next_question": "如果语义 4D 状态包已经能闭环，下一步应该比较它与更大特征集合相比的稳泛化边界。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["semantic_4d_cross_domain_closure"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
