"""
Scan the minimal dimensionality needed for the prototype confidence state.
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


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.array(values, dtype=np.float64)))


def confidence_vector(row: Dict[str, Any]) -> np.ndarray:
    phase1_family = float(row["phase1_family"])
    phase1_concept = float(row["phase1_concept"])
    novel_family = float(row["novel_family"])
    novel_concept = float(row["novel_concept"])
    retention_family = float(row["retention_family"])
    retention_concept = float(row["retention_concept"])
    overall_family = float(row["overall_family"])

    family_values = [phase1_family, novel_family, retention_family]
    family_mean = mean(family_values)
    family_std = std(family_values)
    concept_mean = mean([phase1_concept, novel_concept, retention_concept])

    return np.array(
        [
            family_mean,
            max(0.0, 1.0 - family_std),
            min(family_values),
            max(0.0, overall_family - family_std),
            max(0.0, family_mean - concept_mean),
        ],
        dtype=np.float64,
    )


def fit_model(rows: List[Dict[str, Any]], loop_rank: int, interface_dim: int, ridge_lambda: float) -> Dict[str, Any]:
    x_base = scl.base_matrix(rows)
    base_mean = np.mean(x_base, axis=0)
    base_centered = x_base - base_mean
    _, _, base_vt = np.linalg.svd(base_centered, full_matrices=False)
    loop_rank = min(loop_rank, base_vt.shape[0])
    base_basis = base_vt[:loop_rank].T
    base_latent = base_centered @ base_basis

    conf_matrix = np.stack([confidence_vector(row) for row in rows])
    conf_mean = np.mean(conf_matrix, axis=0)
    conf_centered = conf_matrix - conf_mean
    _, _, conf_vt = np.linalg.svd(conf_centered, full_matrices=False)
    interface_dim = min(interface_dim, conf_vt.shape[0])
    conf_basis = conf_vt[:interface_dim].T
    conf_latent = conf_centered @ conf_basis

    design_rows = []
    for row, z_loop, z_conf in zip(rows, base_latent, conf_latent):
        terms = [1.0, *[float(v) for v in z_loop]]
        for indicator in psm.modality_indicator(row["modality"]):
            terms.extend([indicator * float(v) for v in z_conf])
        terms.append(1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0)
        design_rows.append(terms)

    x_design = np.array(design_rows, dtype=np.float64)
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    coef = scl.fit_ridge(x_design, y, ridge_lambda)
    return {
        "base_mean": base_mean,
        "base_basis": base_basis,
        "conf_mean": conf_mean,
        "conf_basis": conf_basis,
        "coef": coef,
        "loop_rank": loop_rank,
        "interface_dim": interface_dim,
        "ridge_lambda": ridge_lambda,
    }


def predict(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    base_latent = (x_base - model["base_mean"]) @ model["base_basis"]
    conf_matrix = np.stack([confidence_vector(row) for row in rows])
    conf_latent = (conf_matrix - model["conf_mean"]) @ model["conf_basis"]

    preds: List[float] = []
    for row, z_loop, z_conf in zip(rows, base_latent, conf_latent):
        design = [1.0, *[float(v) for v in z_loop]]
        for indicator in psm.modality_indicator(row["modality"]):
            design.extend([indicator * float(v) for v in z_conf])
        design.append(1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0)
        preds.append(float(np.array(design, dtype=np.float64) @ model["coef"]))
    return preds


def leave_one_out(rows: List[Dict[str, Any]], loop_rank: int, interface_dim: int, ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_model(train_rows, loop_rank, interface_dim, ridge_lambda)
        pred = predict([held_row], model)[0]
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


def search_best(rows: List[Dict[str, Any]], interface_dim: int):
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_model(rows, loop_rank, interface_dim, ridge_lambda)
            fitted = scl.evaluate_rows(rows, predict(rows, model))
            held = leave_one_out(rows, loop_rank, interface_dim, ridge_lambda)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan the minimal dimensionality of the prototype confidence state")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_confidence_state_dimension_scan_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    results = {}
    for interface_dim in [1, 2, 3, 4, 5]:
        _, model, fitted, held = search_best(rows, interface_dim)
        results[str(interface_dim)] = {
            "loop_rank": int(model["loop_rank"]),
            "interface_dim": int(model["interface_dim"]),
            **fitted,
            **held,
        }

    winner = min(results.items(), key=lambda item: item[1]["mean_held_out_gap"])[0]
    winner_dim = int(winner)
    if winner_dim <= 2:
        summary = "最优维度落在 2 维以内，说明共享中央回路输出给原型位置壳的接口确实非常小，更像极低维置信状态而不是内容向量。"
    elif winner_dim <= 4:
        summary = "最优维度落在 4 维，说明共享中央回路输出给原型位置壳的接口仍然是小接口，但已经不是标量级别，更像一个小型置信状态包。"
    else:
        summary = "最优维度高于 4 维，说明共享中央回路输出给原型位置壳的接口虽然仍低于高维内容空间，但已经不是很小的置信状态。"
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
        },
        "confidence_dimension_scan": results,
        "winner": winner,
        "hypotheses": {
            "H1_one_dim_suffices": bool(winner == "1"),
            "H2_two_dims_or_less_suffice": bool(int(winner) <= 2),
            "H3_confidence_interface_is_low_dim": bool(int(winner) <= 3),
        },
        "project_readout": {
            "summary": summary,
            "next_question": "这个低维置信状态能否同时稳定支撑多模态统一回路、真实任务收益、D 问题和脑侧候选约束？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
