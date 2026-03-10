"""
Factorize the basis-heavy family shell into:
- prototype position shell
- prototype boundary width shell
- prototype spacing shell
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


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float64)))


def build_shell_terms(row: Dict[str, Any], shell_type: str) -> List[float]:
    phase1_family = float(row["phase1_family"])
    phase1_concept = float(row["phase1_concept"])
    novel_family = float(row["novel_family"])
    novel_concept = float(row["novel_concept"])
    retention_family = float(row["retention_family"])
    retention_concept = float(row["retention_concept"])
    overall_family = float(row["overall_family"])
    family_mean = mean([phase1_family, novel_family, retention_family])

    if shell_type == "prototype_position_shell":
        values = [phase1_family, novel_family, retention_family, overall_family]
    elif shell_type == "prototype_boundary_width_shell":
        values = [
            max(0.0, phase1_family - phase1_concept),
            max(0.0, novel_family - novel_concept),
            max(0.0, retention_family - retention_concept),
            max(0.0, overall_family - family_mean),
        ]
    elif shell_type == "prototype_spacing_shell":
        values = [
            abs(phase1_family - novel_family),
            abs(phase1_family - retention_family),
            abs(novel_family - retention_family),
            abs(overall_family - family_mean),
        ]
    else:
        raise ValueError(shell_type)

    shell_terms: List[float] = []
    for indicator in psm.modality_indicator(row["modality"]):
        shell_terms.extend([indicator * value for value in values])
    return shell_terms


def fit_basis_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, shell_type: str) -> Dict[str, Any]:
    x_base = scl.base_matrix(rows)
    mean_vec = np.mean(x_base, axis=0)
    centered = x_base - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(loop_rank, vt.shape[0])
    basis = vt[:rank].T
    latent = centered @ basis

    design_rows = []
    for row, z in zip(rows, latent):
        design_rows.append(
            [
                1.0,
                *[float(v) for v in z],
                *build_shell_terms(row, shell_type),
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


def predict_basis_model(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    centered = x_base - model["mean_vec"]
    latent = centered @ model["basis"]
    preds: List[float] = []
    for row, z in zip(rows, latent):
        design = np.array(
            [
                1.0,
                *[float(v) for v in z],
                *build_shell_terms(row, model["shell_type"]),
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
        model = fit_basis_model(train_rows, loop_rank, ridge_lambda, shell_type)
        pred = predict_basis_model([held_row], model)[0]
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
            model = fit_basis_model(rows, loop_rank, ridge_lambda, shell_type)
            fitted = scl.evaluate_rows(rows, predict_basis_model(rows, model))
            held = leave_one_out(rows, loop_rank, ridge_lambda, shell_type)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Factorize the basis-heavy family shell")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_basis_shell_factorization_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    results = {}
    for shell_type in [
        "prototype_position_shell",
        "prototype_boundary_width_shell",
        "prototype_spacing_shell",
    ]:
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
        "factorized_basis_shells": results,
        "winner": winner,
        "hypotheses": {
            "H1_position_beats_boundary": bool(
                results["prototype_position_shell"]["mean_held_out_gap"] + 1e-12
                < results["prototype_boundary_width_shell"]["mean_held_out_gap"]
            ),
            "H2_position_beats_spacing": bool(
                results["prototype_position_shell"]["mean_held_out_gap"] + 1e-12
                < results["prototype_spacing_shell"]["mean_held_out_gap"]
            ),
            "H3_basis_shell_is_position_heavy": bool(winner == "prototype_position_shell"),
        },
        "project_readout": {
            "summary": "如果原型位置壳优于边界宽度壳和原型间距壳，就说明多模态差异更像先重写共享基底的原型中心，再向外扩展成边界与间距差异。",
            "next_question": "共享中央回路和共享基底壳之间，最小需要交换的是原型中心坐标、原型置信度，还是更抽象的家族激活状态？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
