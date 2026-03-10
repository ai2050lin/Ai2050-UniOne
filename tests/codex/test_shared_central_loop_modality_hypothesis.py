"""
Test a shared central loop hypothesis for multimodal processing.

Hypothesis:
1. Different modalities are first projected into modality-specific parameter zones.
2. A shared low-rank central loop handles the fused state.
3. Readout is shared again, instead of fully modality-separate.

This is compared against:
- fully shared law
- parameterized shared law
- modality-separate oracle
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import test_parameterized_shared_modality_law as psm


ROOT = Path(__file__).resolve().parents[2]
MODALITIES = psm.MODALITIES
BASE_FEATURES = psm.BASE_FEATURES


def base_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([[float(row[key]) for key in BASE_FEATURES] for row in rows], dtype=np.float64)


def modality_indicator(modality: str) -> List[float]:
    return psm.modality_indicator(modality)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    gram = x.T @ x + ridge_lambda * eye
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def fit_central_loop_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    x_base = base_matrix(rows)
    mean_vec = np.mean(x_base, axis=0)
    centered = x_base - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(loop_rank, vt.shape[0])
    basis = vt[:rank].T
    loop_state = centered @ basis

    design_rows = []
    for row, latent in zip(rows, loop_state):
        mod = modality_indicator(row["modality"])
        interactions: List[float] = []
        for indicator in mod:
            interactions.extend([indicator * float(value) for value in latent])
        design_rows.append(
            [
                1.0,
                *[float(value) for value in latent],
                *mod,
                *interactions,
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ]
        )
    x_design = np.array(design_rows, dtype=np.float64)
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    coef = fit_ridge(x_design, y, ridge_lambda)
    return {
        "mean_vec": mean_vec,
        "basis": basis,
        "coef": coef,
        "loop_rank": rank,
        "ridge_lambda": ridge_lambda,
    }


def predict_central_loop(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = base_matrix(rows)
    centered = x_base - model["mean_vec"]
    loop_state = centered @ model["basis"]
    preds: List[float] = []
    for row, latent in zip(rows, loop_state):
        mod = modality_indicator(row["modality"])
        interactions: List[float] = []
        for indicator in mod:
            interactions.extend([indicator * float(value) for value in latent])
        design = np.array(
            [
                1.0,
                *[float(value) for value in latent],
                *mod,
                *interactions,
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ],
            dtype=np.float64,
        )
        preds.append(float(design @ model["coef"]))
    return preds


def evaluate_rows(rows: List[Dict[str, Any]], preds: List[float]) -> Dict[str, Any]:
    refs = [float(row["reference_score"]) for row in rows]
    gaps = [abs(pred - ref) for pred, ref in zip(preds, refs)]
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for row, gap in zip(rows, gaps):
        by_modality[row["modality"]].append(gap)
    return {
        "mean_absolute_gap": psm.mean(gaps),
        "score_correlation": psm.correlation(refs, preds),
        "modality_mean_gap": {modality: psm.mean(values) for modality, values in by_modality.items()},
    }


def leave_one_out_central_loop(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_central_loop_model(train_rows, loop_rank, ridge_lambda)
        pred = predict_central_loop([held_row], model)[0]
        ref = float(held_row["reference_score"])
        gap = abs(pred - ref)
        refs.append(ref)
        preds.append(pred)
        by_modality[held_row["modality"]].append(gap)
    return {
        "mean_held_out_gap": psm.mean([abs(pred - ref) for pred, ref in zip(preds, refs)]),
        "held_out_score_correlation": psm.correlation(refs, preds),
        "modality_held_out_gap": {modality: psm.mean(values) for modality, values in by_modality.items()},
    }


def search_best_central_loop(rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_central_loop_model(rows, loop_rank, ridge_lambda)
            fitted = evaluate_rows(rows, predict_central_loop(rows, model))
            held = leave_one_out_central_loop(rows, loop_rank, ridge_lambda)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Test a shared central loop modality hypothesis")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_modality_hypothesis_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    _, shared_lambda, shared_coef, shared_fit, shared_held = psm.best_shared_model(rows, psm.shared_features)
    _, param_lambda, param_coef, param_fit, param_held = psm.best_shared_model(rows, psm.parameterized_features)
    _, central_model, central_fit, central_held = search_best_central_loop(rows)

    oracle_lambda = 1e-4
    oracle_fit = psm.evaluate_separate_oracle(rows, psm.fit_separate_oracle(rows, oracle_lambda))
    oracle_held = psm.leave_one_out_separate_oracle(rows, oracle_lambda)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
            "systems": ["direct_multimodal", "shared_offset_multimodal"],
        },
        "fully_shared_law": {
            "ridge_lambda": float(shared_lambda),
            "features": ["1"] + BASE_FEATURES + ["shared_offset_indicator"],
            "coefficients": [float(x) for x in shared_coef.tolist()],
            **shared_fit,
            **shared_held,
        },
        "parameterized_shared_law": {
            "ridge_lambda": float(param_lambda),
            "features": [
                "1",
                *BASE_FEATURES,
                "visual_indicator",
                "tactile_indicator",
                "language_indicator",
                *[f"{modality}*{feature}" for modality in ["visual", "tactile", "language"] for feature in BASE_FEATURES],
                "shared_offset_indicator",
            ],
            "coefficients": [float(x) for x in param_coef.tolist()],
            **param_fit,
            **param_held,
        },
        "shared_central_loop_law": {
            "loop_rank": int(central_model["loop_rank"]),
            "ridge_lambda": float(central_model["ridge_lambda"]),
            "basis": [[float(v) for v in row] for row in central_model["basis"].tolist()],
            "mean_vec": [float(v) for v in central_model["mean_vec"].tolist()],
            "coefficients": [float(v) for v in central_model["coef"].tolist()],
            **central_fit,
            **central_held,
        },
        "modality_separate_oracle": {
            "ridge_lambda": oracle_lambda,
            **oracle_fit,
            **oracle_held,
        },
        "hypotheses": {
            "H1_central_loop_beats_fully_shared": bool(
                central_held["mean_held_out_gap"] + 1e-12 < shared_held["mean_held_out_gap"]
            ),
            "H2_central_loop_close_to_parameterized": bool(
                abs(central_held["mean_held_out_gap"] - param_held["mean_held_out_gap"]) < 0.0015
            ),
            "H3_shared_central_loop_supported": bool(
                central_held["mean_held_out_gap"] + 1e-12 < shared_held["mean_held_out_gap"]
                and abs(central_held["mean_held_out_gap"] - param_held["mean_held_out_gap"]) < 0.0015
            ),
        },
        "project_readout": {
            "summary": "如果共享中央回路低秩律能稳定优于完全共享，同时逼近参数化共享，就支持“模态专属投影 + 共享中央回路”的写法。",
            "next_question": "模态差异主要落在输入投影、中央回路内部，还是落在最终读出标定层？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
