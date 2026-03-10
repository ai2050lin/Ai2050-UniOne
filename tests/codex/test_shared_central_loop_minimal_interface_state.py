"""
Test which minimal interface state best couples the shared central loop
to the position-heavy basis shell.

Candidates:
- prototype_center_state
- prototype_confidence_state
- family_activation_state
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


def build_interface_terms(row: Dict[str, Any], state_type: str) -> List[float]:
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

    if state_type == "prototype_center_state":
        values = [phase1_family, novel_family, retention_family, overall_family]
    elif state_type == "prototype_confidence_state":
        values = [
            family_mean,
            max(0.0, 1.0 - family_std),
            min(family_values),
            max(0.0, overall_family - family_std),
        ]
    elif state_type == "family_activation_state":
        values = [
            phase1_family * phase1_concept,
            novel_family * novel_concept,
            retention_family * retention_concept,
            overall_family * concept_mean,
        ]
    else:
        raise ValueError(state_type)

    shell_terms: List[float] = []
    for indicator in psm.modality_indicator(row["modality"]):
        shell_terms.extend([indicator * value for value in values])
    return shell_terms


def fit_interface_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, state_type: str) -> Dict[str, Any]:
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
                *build_interface_terms(row, state_type),
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
        "state_type": state_type,
    }


def predict_interface_model(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
    x_base = scl.base_matrix(rows)
    centered = x_base - model["mean_vec"]
    latent = centered @ model["basis"]
    preds: List[float] = []
    for row, z in zip(rows, latent):
        design = np.array(
            [
                1.0,
                *[float(v) for v in z],
                *build_interface_terms(row, model["state_type"]),
                1.0 if row["system_name"] == "shared_offset_multimodal" else 0.0,
            ],
            dtype=np.float64,
        )
        preds.append(float(design @ model["coef"]))
    return preds


def leave_one_out(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, state_type: str) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    by_modality: Dict[str, List[float]] = {modality: [] for modality in MODALITIES}
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        model = fit_interface_model(train_rows, loop_rank, ridge_lambda, state_type)
        pred = predict_interface_model([held_row], model)[0]
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


def search_best(rows: List[Dict[str, Any]], state_type: str):
    best = None
    for loop_rank in [2, 3, 4]:
        for ridge_lambda in [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]:
            model = fit_interface_model(rows, loop_rank, ridge_lambda, state_type)
            fitted = scl.evaluate_rows(rows, predict_interface_model(rows, model))
            held = leave_one_out(rows, loop_rank, ridge_lambda, state_type)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Test minimal interface states for the shared central loop")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_minimal_interface_state_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    results = {}
    for state_type in [
        "prototype_center_state",
        "prototype_confidence_state",
        "family_activation_state",
    ]:
        _, model, fitted, held = search_best(rows, state_type)
        results[state_type] = {
            "loop_rank": int(model["loop_rank"]),
            **fitted,
            **held,
        }

    winner = min(results.items(), key=lambda item: item[1]["mean_held_out_gap"])[0]
    if winner == "prototype_confidence_state":
        summary = "当前最优接口是原型置信度状态，说明共享中央回路与原型位置壳之间更像交换原型是否稳定、是否可信的低维状态，而不是直接交换原型中心坐标。"
        next_question = "原型置信度接口最小需要几维，才能稳定外推到更多模态、真实任务和脑侧约束？"
    elif winner == "prototype_center_state":
        summary = "当前最优接口是原型中心状态，说明共享中央回路与原型位置壳之间更像直接交换原型中心坐标，而不是更高阶的置信度或激活统计。"
        next_question = "原型中心接口到底需要几维状态，才能同时支撑多模态统一回路、真实任务收益和脑侧约束？"
    else:
        summary = "当前最优接口是家族激活态，说明共享中央回路与原型位置壳之间更像交换家族级激活统计，而不是更直接的原型中心或置信度。"
        next_question = "家族激活态能否继续压缩成更小的状态变量，同时保持跨模态统一律稳定？"
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "modalities": MODALITIES,
        },
        "minimal_interface_states": results,
        "winner": winner,
        "hypotheses": {
            "H1_center_beats_confidence": bool(
                results["prototype_center_state"]["mean_held_out_gap"] + 1e-12
                < results["prototype_confidence_state"]["mean_held_out_gap"]
            ),
            "H2_center_beats_activation": bool(
                results["prototype_center_state"]["mean_held_out_gap"] + 1e-12
                < results["family_activation_state"]["mean_held_out_gap"]
            ),
            "H3_minimal_interface_is_center_like": bool(winner == "prototype_center_state"),
        },
        "project_readout": {
            "summary": summary,
            "next_question": next_question,
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
