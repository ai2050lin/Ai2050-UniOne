"""
Locate where modality-specific shell differences are most likely to live:
- input shell
- internal loop shell
- output shell
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
import test_shared_central_loop_shell_hypothesis as scs


ROOT = Path(__file__).resolve().parents[2]
MODALITIES = psm.MODALITIES


def fit_input_shell_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    x_base = scl.base_matrix(rows)
    modality_means = {}
    for modality in MODALITIES:
        subset = np.array(
            [[float(row[key]) for key in psm.BASE_FEATURES] for row in rows if row["modality"] == modality],
            dtype=np.float64,
        )
        modality_means[modality] = np.mean(subset, axis=0)

    norm_rows = np.stack([x_base[idx] - modality_means[row["modality"]] for idx, row in enumerate(rows)])
    _, _, vt = np.linalg.svd(norm_rows, full_matrices=False)
    rank = min(loop_rank, vt.shape[0])
    basis = vt[:rank].T
    latent = norm_rows @ basis

    x_design = np.array(
        [
            [1.0, *[float(v) for v in z], 1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0]
            for row, z in zip(rows, latent)
        ],
        dtype=np.float64,
    )
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    coef = scl.fit_ridge(x_design, y, ridge_lambda)
    return {
        "modality_means": {k: v.tolist() for k, v in modality_means.items()},
        "basis": basis,
        "coef": coef,
        "loop_rank": rank,
        "ridge_lambda": ridge_lambda,
    }


def predict_input_shell(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    norm_rows = np.stack(
        [x_base[idx] - np.array(model["modality_means"][row["modality"]], dtype=np.float64) for idx, row in enumerate(rows)]
    )
    latent = norm_rows @ model["basis"]
    preds: List[float] = []
    for row, z in zip(rows, latent):
        design = np.array(
            [1.0, *[float(v) for v in z], 1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0],
            dtype=np.float64,
        )
        preds.append(float(design @ model["coef"]))
    return preds


def leave_one_out(rows: List[Dict[str, Any]], fit_fn, pred_fn, loop_rank: int, ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_fn(train_rows, loop_rank, ridge_lambda)
        pred = pred_fn([held_row], model)[0]
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


def search_best(rows: List[Dict[str, Any]], fit_fn, pred_fn):
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_fn(rows, loop_rank, ridge_lambda)
            fitted = scl.evaluate_rows(rows, pred_fn(rows, model))
            held = leave_one_out(rows, fit_fn, pred_fn, loop_rank, ridge_lambda)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Locate multimodal shell placement around a shared central loop")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_shell_localization_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    _, shared_lambda, _, _, shared_held = psm.best_shared_model(rows, psm.shared_features)
    _, param_lambda, _, _, param_held = psm.best_shared_model(rows, psm.parameterized_features)
    _, input_model, _, input_held = search_best(rows, fit_input_shell_model, predict_input_shell)
    _, internal_model, _, internal_held = scl.search_best_central_loop(rows)
    _, output_model, _, output_held = scs.search_best_shell(rows)

    placements = {
        "input_shell": {
            "loop_rank": int(input_model["loop_rank"]),
            **input_held,
        },
        "internal_loop_shell": {
            "loop_rank": int(internal_model["loop_rank"]),
            **internal_held,
        },
        "output_shell": {
            "loop_rank": int(output_model["loop_rank"]),
            **output_held,
        },
    }
    best_id = min(placements.items(), key=lambda item: item[1]["mean_held_out_gap"])[0]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
        },
        "baselines": {
            "fully_shared_law": {
                "ridge_lambda": float(shared_lambda),
                **shared_held,
            },
            "parameterized_shared_law": {
                "ridge_lambda": float(param_lambda),
                **param_held,
            },
        },
        "placements": placements,
        "winner": best_id,
        "hypotheses": {
            "H1_output_beats_internal": bool(
                output_held["mean_held_out_gap"] + 1e-12 < internal_held["mean_held_out_gap"]
            ),
            "H2_output_beats_input": bool(
                output_held["mean_held_out_gap"] + 1e-12 < input_held["mean_held_out_gap"]
            ),
            "H3_shell_is_not_purely_internal": bool(best_id != "internal_loop_shell"),
        },
        "project_readout": {
            "summary": "如果输出壳或输入壳明显优于内部参数壳，就说明模态差异更像围绕统一回路的壳层，而不是中央回路内部本身。",
            "next_question": "最强壳层是否还能继续细分成输入投影壳与输出标定壳两层？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
