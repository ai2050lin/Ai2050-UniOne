"""
Test whether multimodal processing is better described as:
- a shared central loop
- plus a small modality-specific shell residual

This refines the pure shared central loop hypothesis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import test_parameterized_shared_modality_law as psm
import test_shared_central_loop_modality_hypothesis as scl


ROOT = Path(__file__).resolve().parents[2]
MODALITIES = psm.MODALITIES


def fit_shell_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    x_base = scl.base_matrix(rows)
    mean_vec = np.mean(x_base, axis=0)
    centered = x_base - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(loop_rank, vt.shape[0])
    basis = vt[:rank].T
    loop_state = centered @ basis
    recon = loop_state @ basis.T
    residual = centered - recon
    residual_norm = np.linalg.norm(residual, axis=1)

    design_rows = []
    for row, latent, res_norm in zip(rows, loop_state, residual_norm):
        mod = psm.modality_indicator(row["modality"])
        shell_terms = [indicator * float(res_norm) for indicator in mod]
        design_rows.append(
            [
                1.0,
                *[float(value) for value in latent],
                *mod,
                float(res_norm),
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
    }


def predict_shell_model(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    centered = x_base - model["mean_vec"]
    loop_state = centered @ model["basis"]
    recon = loop_state @ model["basis"].T
    residual = centered - recon
    residual_norm = np.linalg.norm(residual, axis=1)
    preds: List[float] = []
    for row, latent, res_norm in zip(rows, loop_state, residual_norm):
        mod = psm.modality_indicator(row["modality"])
        shell_terms = [indicator * float(res_norm) for indicator in mod]
        design = np.array(
            [
                1.0,
                *[float(value) for value in latent],
                *mod,
                float(res_norm),
                *shell_terms,
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ],
            dtype=np.float64,
        )
        preds.append(float(design @ model["coef"]))
    return preds


def leave_one_out_shell(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_shell_model(train_rows, loop_rank, ridge_lambda)
        pred = predict_shell_model([held_row], model)[0]
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


def search_best_shell(rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_shell_model(rows, loop_rank, ridge_lambda)
            fitted = scl.evaluate_rows(rows, predict_shell_model(rows, model))
            held = leave_one_out_shell(rows, loop_rank, ridge_lambda)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Test shared central loop plus modality shell hypothesis")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_shell_hypothesis_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    _, _, _, shared_fit, shared_held = psm.best_shared_model(rows, psm.shared_features)
    _, _, _, param_fit, param_held = psm.best_shared_model(rows, psm.parameterized_features)
    _, central_model, central_fit, central_held = scl.search_best_central_loop(rows)
    _, shell_model, shell_fit, shell_held = search_best_shell(rows)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
        },
        "fully_shared_law": shared_held,
        "parameterized_shared_law": param_held,
        "shared_central_loop_law": {
            "loop_rank": int(central_model["loop_rank"]),
            **central_fit,
            **central_held,
        },
        "shared_central_loop_shell_law": {
            "loop_rank": int(shell_model["loop_rank"]),
            **shell_fit,
            **shell_held,
        },
        "hypotheses": {
            "H1_shell_improves_central_loop": bool(
                shell_held["mean_held_out_gap"] + 1e-12 < central_held["mean_held_out_gap"]
            ),
            "H2_shell_beats_fully_shared": bool(
                shell_held["mean_held_out_gap"] + 1e-12 < shared_held["mean_held_out_gap"]
            ),
            "H3_shell_close_to_parameterized": bool(
                abs(shell_held["mean_held_out_gap"] - param_held["mean_held_out_gap"]) < 0.0015
            ),
        },
        "project_readout": {
            "summary": "如果中央回路加小型模态外壳能明显优于纯中央回路，就更支持‘统一回路 + 模态壳层’而不是‘纯统一回路’。",
            "next_question": "模态差异更像输入壳层、输出壳层，还是中央回路内部的子参数区？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
