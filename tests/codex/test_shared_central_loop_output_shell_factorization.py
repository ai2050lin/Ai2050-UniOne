"""
Factorize the winning output shell into smaller candidates:
- calibration shell
- protocol shell
- task readout shell
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import test_parameterized_shared_modality_law as psm
import test_shared_central_loop_modality_hypothesis as scl


ROOT = Path(__file__).resolve().parents[2]
MODALITIES = psm.MODALITIES
PROTO_KEYS = ["phase1_family", "novel_family", "retention_family", "overall_family"]
TASK_KEYS = ["phase1_concept", "novel_concept", "retention_concept"]


def fit_factorized_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, shell_type: str) -> Dict[str, Any]:
    x_base = scl.base_matrix(rows)
    mean_vec = np.mean(x_base, axis=0)
    centered = x_base - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(loop_rank, vt.shape[0])
    basis = vt[:rank].T
    latent = centered @ basis

    design_rows = []
    for row, z in zip(rows, latent):
        mod = psm.modality_indicator(row["modality"])
        shell_terms: List[float] = []
        if shell_type == "calibration_shell":
            shell_terms = mod
        elif shell_type == "protocol_shell":
            proto = [float(row[key]) for key in PROTO_KEYS]
            for indicator in mod:
                shell_terms.extend([indicator * value for value in proto])
        elif shell_type == "task_readout_shell":
            task = [float(row[key]) for key in TASK_KEYS]
            for indicator in mod:
                shell_terms.extend([indicator * value for value in task])
        else:
            raise ValueError(shell_type)
        design_rows.append(
            [
                1.0,
                *[float(v) for v in z],
                *shell_terms,
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ]
        )

    x_design = np.array(design_rows, dtype=np.float64)
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    coef = scl.fit_ridge(x_design, y, ridge_lambda)
    return {
        "mean_vec": mean_vec,
        "basis": basis,
        "coef": coef,
        "loop_rank": rank,
        "ridge_lambda": ridge_lambda,
        "shell_type": shell_type,
    }


def predict_factorized_model(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    centered = x_base - model["mean_vec"]
    latent = centered @ model["basis"]
    preds: List[float] = []
    for row, z in zip(rows, latent):
        mod = psm.modality_indicator(row["modality"])
        shell_terms: List[float] = []
        if model["shell_type"] == "calibration_shell":
            shell_terms = mod
        elif model["shell_type"] == "protocol_shell":
            proto = [float(row[key]) for key in PROTO_KEYS]
            for indicator in mod:
                shell_terms.extend([indicator * value for value in proto])
        elif model["shell_type"] == "task_readout_shell":
            task = [float(row[key]) for key in TASK_KEYS]
            for indicator in mod:
                shell_terms.extend([indicator * value for value in task])
        else:
            raise ValueError(model["shell_type"])
        design = np.array(
            [
                1.0,
                *[float(v) for v in z],
                *shell_terms,
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ],
            dtype=np.float64,
        )
        preds.append(float(design @ model["coef"]))
    return preds


def leave_one_out(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, shell_type: str) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_factorized_model(train_rows, loop_rank, ridge_lambda, shell_type)
        pred = predict_factorized_model([held_row], model)[0]
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


def search_best(rows: List[Dict[str, Any]], shell_type: str):
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_factorized_model(rows, loop_rank, ridge_lambda, shell_type)
            fitted = scl.evaluate_rows(rows, predict_factorized_model(rows, model))
            held = leave_one_out(rows, loop_rank, ridge_lambda, shell_type)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Factorize the output shell around a shared central loop")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_output_shell_factorization_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    results = {}
    for shell_type in ["calibration_shell", "protocol_shell", "task_readout_shell"]:
        _, model, fitted, held = search_best(rows, shell_type)
        results[shell_type] = {
            "loop_rank": int(model["loop_rank"]),
            **fitted,
            **held,
        }

    winner = min(results.items(), key=lambda item: item[1]["mean_held_out_gap"])[0]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
        },
        "factorized_shells": results,
        "winner": winner,
        "hypotheses": {
            "H1_task_beats_calibration": bool(
                results["task_readout_shell"]["mean_held_out_gap"] + 1e-12 < results["calibration_shell"]["mean_held_out_gap"]
            ),
            "H2_task_beats_protocol": bool(
                results["task_readout_shell"]["mean_held_out_gap"] + 1e-12 < results["protocol_shell"]["mean_held_out_gap"]
            ),
            "H3_output_shell_is_readout_heavy": bool(winner == "task_readout_shell"),
        },
        "project_readout": {
            "summary": "如果任务读出壳优于校准壳和协议壳，就说明模态差异主要落在最终任务读出，而不是更浅层的统一校准。",
            "next_question": "任务读出壳里，差异主要是概念读出、关系读出，还是动作读出？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
