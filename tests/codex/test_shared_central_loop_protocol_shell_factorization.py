"""
Factorize the protocol-heavy output shell into smaller protocol families:
- family protocol shell
- relation protocol shell
- action/planning protocol shell

This uses the current controlled multimodal grounding bench, so
"action/planning" is a downstream proxy based on concept-level readout.
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
FAMILY_KEYS = ["phase1_family", "novel_family", "retention_family", "overall_family"]
CONCEPT_KEYS = ["phase1_concept", "novel_concept", "retention_concept"]


def build_shell_terms(row: Dict[str, Any], shell_type: str) -> List[float]:
    mod = psm.modality_indicator(row["modality"])
    if shell_type == "family_protocol_shell":
        values = [float(row[key]) for key in FAMILY_KEYS]
    elif shell_type == "relation_protocol_shell":
        values = [
            float(row["phase1_family"]) * float(row["phase1_concept"]),
            float(row["novel_family"]) * float(row["novel_concept"]),
            float(row["retention_family"]) * float(row["retention_concept"]),
            float(row["overall_family"]) * float(row["novel_concept"]),
        ]
    elif shell_type == "action_planning_protocol_shell":
        values = [float(row[key]) for key in CONCEPT_KEYS]
    else:
        raise ValueError(shell_type)

    shell_terms: List[float] = []
    for indicator in mod:
        shell_terms.extend([indicator * value for value in values])
    return shell_terms


def fit_protocol_model(rows: List[Dict[str, Any]], loop_rank: int, ridge_lambda: float, shell_type: str) -> Dict[str, Any]:
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


def predict_protocol_model(rows: List[Dict[str, Any]], model: Dict[str, Any]) -> List[float]:
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
        model = fit_protocol_model(train_rows, loop_rank, ridge_lambda, shell_type)
        pred = predict_protocol_model([held_row], model)[0]
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
            model = fit_protocol_model(rows, loop_rank, ridge_lambda, shell_type)
            fitted = scl.evaluate_rows(rows, predict_protocol_model(rows, model))
            held = leave_one_out(rows, loop_rank, ridge_lambda, shell_type)
            objective = held["mean_held_out_gap"] - 0.01 * held["held_out_score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, model, fitted, held)
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Factorize protocol-heavy output shell by relation type")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/shared_central_loop_protocol_shell_factorization_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in range(10):
        rows.extend(psm.run_seed("direct_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))
        rows.extend(psm.run_seed("shared_offset_multimodal", seed, noise=0.18, dropout_p=0.1, missing_modality_p=0.22))

    results = {}
    for shell_type in [
        "family_protocol_shell",
        "relation_protocol_shell",
        "action_planning_protocol_shell",
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
        "factorized_protocol_shells": results,
        "winner": winner,
        "hypotheses": {
            "H1_relation_beats_family": bool(
                results["relation_protocol_shell"]["mean_held_out_gap"] + 1e-12 < results["family_protocol_shell"]["mean_held_out_gap"]
            ),
            "H2_relation_beats_action": bool(
                results["relation_protocol_shell"]["mean_held_out_gap"] + 1e-12 < results["action_planning_protocol_shell"]["mean_held_out_gap"]
            ),
            "H3_protocol_shell_is_relation_heavy": bool(winner == "relation_protocol_shell"),
        },
        "project_readout": {
            "summary": "如果 relation 协议壳优于 family 和 action/planning 协议壳，就说明多模态差异主要落在关系协议的重写，而不是更粗的族级协议或更下游的动作规划代理。",
            "next_question": "relation 协议壳里，差异主要来自上下位、部分整体、因果，还是更一般的关系绑定项？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
