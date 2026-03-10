"""
Test whether multimodal processing is better described as:
1. one shared mechanism with modality-specific parameters
vs
2. one fully shared mechanism
vs
3. completely separate modality-specific mechanisms

This uses the continuous multimodal grounding prototype as a controlled bench.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import test_continuous_multimodal_grounding_proto as mm


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = mm.FAMILIES
ALL_GROUPS = {family: mm.PHASE1[family] + mm.PHASE2[family] for family in FAMILIES}
MODALITIES = ["full", "visual", "tactile", "language"]


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


def mask_modality(x: np.ndarray, modality: str) -> np.ndarray:
    out = x.copy()
    if modality == "full":
        return out
    if modality == "visual":
        out[8:] = 0.0
        return out
    if modality == "tactile":
        out[:8] = 0.0
        out[16:] = 0.0
        return out
    if modality == "language":
        out[:16] = 0.0
        return out
    raise ValueError(modality)


def build_model(system_name: str):
    if system_name == "direct_multimodal":
        return mm.DirectMultimodalPrototype()
    return mm.SharedOffsetMultimodalGrounder()


def modality_eval(
    model,
    concept_groups: Dict[str, List[str]],
    repeats: int,
    rng: np.random.Generator,
    modality: str,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
) -> Dict[str, float]:
    family_ok = 0
    concept_ok = 0
    total = 0
    for _ in range(repeats):
        for family, concepts in concept_groups.items():
            for concept in concepts:
                x = mm.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p)
                x = mask_modality(x, modality)
                pred_family, pred_concept = model.predict(x)
                family_ok += int(pred_family == family)
                concept_ok += int(pred_concept == concept)
                total += 1
    return {
        "family_accuracy": float(family_ok / max(1, total)),
        "concept_accuracy": float(concept_ok / max(1, total)),
    }


def run_seed(system_name: str, seed: int, noise: float, dropout_p: float, missing_modality_p: float) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    model = build_model(system_name)

    for _ in range(42):
        for family, concepts in mm.PHASE1.items():
            for concept in concepts:
                model.train(mm.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    for _ in range(3):
        for family, concepts in mm.PHASE2.items():
            for concept in concepts:
                model.train(mm.sample_multimodal_input(rng, concept, noise, dropout_p, missing_modality_p), family, concept)

    rows: List[Dict[str, Any]] = []
    for modality in MODALITIES:
        phase1 = modality_eval(model, mm.PHASE1, 18, rng, modality, noise, dropout_p, missing_modality_p)
        novel = modality_eval(model, mm.PHASE2, 18, rng, modality, noise, dropout_p, missing_modality_p)
        retention = modality_eval(model, mm.PHASE1, 18, rng, modality, noise, dropout_p, missing_modality_p)
        overall = modality_eval(model, ALL_GROUPS, 18, rng, modality, noise, dropout_p, missing_modality_p)
        reference_score = float(
            (
                0.15 * phase1["concept_accuracy"]
                + 0.30 * novel["concept_accuracy"]
                + 0.20 * retention["concept_accuracy"]
                + 0.20 * overall["concept_accuracy"]
                + 0.15 * overall["family_accuracy"]
            )
        )
        rows.append(
            {
                "row_id": f"{system_name}::{seed}::{modality}",
                "system_name": system_name,
                "seed": seed,
                "modality": modality,
                "phase1_family": phase1["family_accuracy"],
                "phase1_concept": phase1["concept_accuracy"],
                "novel_family": novel["family_accuracy"],
                "novel_concept": novel["concept_accuracy"],
                "retention_family": retention["family_accuracy"],
                "retention_concept": retention["concept_accuracy"],
                "overall_family": overall["family_accuracy"],
                "reference_score": reference_score,
            }
        )
    return rows


BASE_FEATURES = [
    "phase1_family",
    "phase1_concept",
    "novel_family",
    "novel_concept",
    "retention_family",
    "retention_concept",
    "overall_family",
]


def modality_indicator(modality: str) -> List[float]:
    return [
        1.0 if modality == "visual" else 0.0,
        1.0 if modality == "tactile" else 0.0,
        1.0 if modality == "language" else 0.0,
    ]


def shared_features(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [1.0] + [float(row[key]) for key in BASE_FEATURES] + [1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0],
        dtype=np.float64,
    )


def parameterized_features(row: Dict[str, Any]) -> np.ndarray:
    base = [float(row[key]) for key in BASE_FEATURES]
    mod = modality_indicator(row["modality"])
    interactions = []
    for indicator in mod:
        interactions.extend([indicator * x for x in base])
    return np.array(
        [1.0] + base + mod + interactions + [1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0],
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


def fit_with_feature_fn(rows: List[Dict[str, Any]], feature_fn, ridge_lambda: float) -> np.ndarray:
    x = np.stack([feature_fn(row) for row in rows])
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    return fit_ridge(x, y, ridge_lambda)


def predict_with_feature_fn(rows: List[Dict[str, Any]], feature_fn, coef: np.ndarray) -> List[float]:
    return [float(feature_fn(row) @ coef) for row in rows]


def evaluate_rows(rows: List[Dict[str, Any]], preds: List[float]) -> Dict[str, Any]:
    refs = [float(row["reference_score"]) for row in rows]
    gaps = [abs(pred - ref) for pred, ref in zip(preds, refs)]
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for row, gap in zip(rows, gaps):
        by_modality[row["modality"]].append(gap)
    return {
        "mean_absolute_gap": mean(gaps),
        "score_correlation": correlation(refs, preds),
        "modality_mean_gap": {modality: mean(values) for modality, values in by_modality.items()},
    }


def leave_one_out_shared(rows: List[Dict[str, Any]], feature_fn, ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        coef = fit_with_feature_fn(train_rows, feature_fn, ridge_lambda)
        pred = float(feature_fn(held_row) @ coef)
        ref = float(held_row["reference_score"])
        gap = abs(pred - ref)
        refs.append(ref)
        preds.append(pred)
        by_modality[held_row["modality"]].append(gap)
    return {
        "mean_held_out_gap": mean([abs(pred - ref) for pred, ref in zip(preds, refs)]),
        "held_out_score_correlation": correlation(refs, preds),
        "modality_held_out_gap": {modality: mean(values) for modality, values in by_modality.items()},
    }


def fit_separate_oracle(rows: List[Dict[str, Any]], ridge_lambda: float) -> Dict[str, np.ndarray]:
    out = {}
    for modality in MODALITIES:
        subset = [row for row in rows if row["modality"] == modality]
        out[modality] = fit_with_feature_fn(subset, shared_features, ridge_lambda)
    return out


def evaluate_separate_oracle(rows: List[Dict[str, Any]], coefs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    preds = [float(shared_features(row) @ coefs[row["modality"]]) for row in rows]
    return evaluate_rows(rows, preds)


def leave_one_out_separate_oracle(rows: List[Dict[str, Any]], ridge_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        modality = held_row["modality"]
        subset = [row for row in train_rows if row["modality"] == modality]
        coef = fit_with_feature_fn(subset, shared_features, ridge_lambda)
        pred = float(shared_features(held_row) @ coef)
        ref = float(held_row["reference_score"])
        gap = abs(pred - ref)
        refs.append(ref)
        preds.append(pred)
        by_modality[modality].append(gap)
    return {
        "mean_held_out_gap": mean([abs(pred - ref) for pred, ref in zip(preds, refs)]),
        "held_out_score_correlation": correlation(refs, preds),
        "modality_held_out_gap": {modality: mean(values) for modality, values in by_modality.items()},
    }


def best_shared_model(rows: List[Dict[str, Any]], feature_fn):
    best = None
    for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
        coef = fit_with_feature_fn(rows, feature_fn, ridge_lambda)
        fitted = evaluate_rows(rows, predict_with_feature_fn(rows, feature_fn, coef))
        held = leave_one_out_shared(rows, feature_fn, ridge_lambda)
        objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
        if best is None or objective < best[0]:
            best = (objective, ridge_lambda, coef, fitted, held)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Test a parameterized shared modality law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/parameterized_shared_modality_law_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    _, shared_lambda, shared_coef, shared_fit, shared_held = best_shared_model(rows, shared_features)
    _, param_lambda, param_coef, param_fit, param_held = best_shared_model(rows, parameterized_features)

    oracle_lambda = 1e-4
    oracle_fit = evaluate_separate_oracle(rows, fit_separate_oracle(rows, oracle_lambda))
    oracle_held = leave_one_out_separate_oracle(rows, oracle_lambda)

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
        "modality_separate_oracle": {
            "ridge_lambda": oracle_lambda,
            **oracle_fit,
            **oracle_held,
        },
        "hypotheses": {
            "H1_parameterized_beats_fully_shared": bool(
                param_held["mean_held_out_gap"] + 1e-12 < shared_held["mean_held_out_gap"]
            ),
            "H2_parameterized_close_to_oracle": bool(
                abs(param_held["mean_held_out_gap"] - oracle_held["mean_held_out_gap"]) < 0.01
            ),
            "H3_same_mechanism_different_params_is_supported": bool(
                param_held["mean_held_out_gap"] + 1e-12 < shared_held["mean_held_out_gap"]
                and abs(param_held["mean_held_out_gap"] - oracle_held["mean_held_out_gap"]) < 0.01
            ),
        },
        "project_readout": {
            "summary": "如果参数化共享律明显优于完全共享律，并逼近模态独立拟合上限，就支持“同一机制、不同参数”的写法。",
            "next_question": "模态参数差异到底主要落在输入投影层，还是落在排序层内部系数？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
