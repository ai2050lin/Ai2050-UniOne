"""
Sweep structured brain-side constraint expansion mixes.

Purpose:
- the first full structured expansion may over-constrain the semantic 4D + 3D law
- search lighter mixes to identify whether there is a useful brain-constraint
  expansion regime instead of assuming more constraints are always better
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import test_brain_learnable_ranking_two_layer_unified_law as brain_law
import test_semantic_4d_confidence_vector_domain_correction as vec_corr


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_structured_constraints(
    train_rows: List[Dict[str, Any]],
    topk: int,
    include_aggregate: bool,
    alpha: float,
) -> List[Dict[str, Any]]:
    brain_rows = [row for row in train_rows if row["domain"] == "brain"]
    if not brain_rows:
        return []

    constraints: List[Dict[str, Any]] = []
    component_count = len(brain_rows[0]["brain_components"])
    overall_mean = mean([float(row["reference_score"]) for row in brain_rows])

    for row in brain_rows:
        meta = row.get("brain_meta", {})
        dnn_reverse = float(meta.get("dnn_reverse_score", row["reference_score"]))
        brain_align = float(meta.get("brain_alignment_score", row["reference_score"]))
        overall = float(meta.get("overall_bridge_score", row["reference_score"]))
        comps = np.array([float(v) for v in row["brain_components"]], dtype=np.float64)
        top_indices = np.argsort(comps)[::-1][:topk]

        for idx in top_indices.tolist():
            comp_score = float(comps[idx])
            focused = np.zeros(component_count, dtype=np.float64)
            focused[idx] = comp_score
            focused[(idx + 1) % component_count] = overall
            focused[(idx + 2) % component_count] = 0.5 * (dnn_reverse + brain_align)
            constraints.append(
                {
                    "row_id": f"{row['row_id']}::focus::{idx}",
                    "domain": "brain",
                    "subdomain": f"{row['subdomain']}_focus_{idx}",
                    "signal_a": 0.0,
                    "signal_b": 0.0,
                    "signal_c": 0.0,
                    "signal_d": 0.0,
                    "brain_components": [float(v) for v in focused.tolist()],
                    "reference_score": float(alpha * comp_score + (1.0 - alpha) * overall),
                }
            )

    if include_aggregate:
        comp_matrix = np.array([row["brain_components"] for row in brain_rows], dtype=np.float64)
        comp_means = comp_matrix.mean(axis=0)
        comp_spreads = comp_matrix.max(axis=0) - comp_matrix.min(axis=0)
        top_indices = np.argsort(comp_means)[::-1][:topk]
        for idx in top_indices.tolist():
            vector = np.zeros(component_count, dtype=np.float64)
            vector[idx] = float(comp_means[idx])
            vector[(idx + 1) % component_count] = float(1.0 - comp_spreads[idx])
            vector[(idx + 2) % component_count] = overall_mean
            constraints.append(
                {
                    "row_id": f"brain::aggregate::{idx}",
                    "domain": "brain",
                    "subdomain": f"aggregate_{idx}",
                    "signal_a": 0.0,
                    "signal_b": 0.0,
                    "signal_c": 0.0,
                    "signal_d": 0.0,
                    "brain_components": [float(v) for v in vector.tolist()],
                    "reference_score": float(alpha * comp_means[idx] + (1.0 - alpha) * (1.0 - comp_spreads[idx])),
                }
            )

    return constraints


def leave_one_out(
    base_rows: List[Dict[str, Any]],
    correction_dim: int,
    ranking_lambda: float,
    calibration_lambda: float,
    topk: int,
    include_aggregate: bool,
    alpha: float,
) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    gaps: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}
    constraint_counts: List[int] = []

    for hold_idx, held_row in enumerate(base_rows):
        train_rows = [row for idx, row in enumerate(base_rows) if idx != hold_idx]
        constraints = build_structured_constraints(train_rows, topk=topk, include_aggregate=include_aggregate, alpha=alpha)
        train_aug = train_rows + constraints
        constraint_counts.append(len(constraints))

        ranking_coef = vec_corr.fit_ranking(train_aug, ranking_lambda)
        train_ranking_scores = [vec_corr.predict_ranking(row, ranking_coef) for row in train_aug]
        calibration_coef = vec_corr.fit_calibration(train_aug, train_ranking_scores, correction_dim, calibration_lambda)
        held_ranking = vec_corr.predict_ranking(held_row, ranking_coef)
        held_pred = vec_corr.predict_calibrated(held_row, held_ranking, correction_dim, calibration_coef)
        held_ref = float(held_row["reference_score"])
        gap = abs(held_pred - held_ref)
        refs.append(held_ref)
        preds.append(held_pred)
        gaps.append(gap)
        gaps_by_domain[held_row["domain"]].append(gap)

    return {
        "topk": int(topk),
        "include_aggregate": bool(include_aggregate),
        "alpha": float(alpha),
        "mean_constraint_count": mean(constraint_counts),
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "brain_held_out_gap": mean(gaps_by_domain["brain"]),
        "d_held_out_gap": mean(gaps_by_domain["D"]),
        "real_task_held_out_gap": mean(gaps_by_domain["real_task"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep structured brain constraint expansion mixes")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/semantic_4d_brain_constraint_expansion_sweep_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_confidence_vector_domain_correction_20260310.json")
    brain_payload = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")

    base_rows = brain_law.build_brain_rows(brain_payload) + brain_law.build_d_rows(d_payload) + brain_law.build_real_rows(real_payload)
    correction_dim = int(baseline_payload["best_correction_dim"])
    ranking_lambda = float(baseline_payload["ranking_layer"]["ridge_lambda"])
    calibration_lambda = float(baseline_payload["calibration_layer"]["ridge_lambda"])
    baseline = baseline_payload["semantic_4d_vector_domain_correction"]

    rows = []
    best = None
    for topk in [1, 2, 3]:
        for include_aggregate in [False, True]:
            for alpha in [0.35, 0.5, 0.65, 0.8]:
                result = leave_one_out(
                    base_rows=base_rows,
                    correction_dim=correction_dim,
                    ranking_lambda=ranking_lambda,
                    calibration_lambda=calibration_lambda,
                    topk=topk,
                    include_aggregate=include_aggregate,
                    alpha=alpha,
                )
                result["brain_gap_improvement"] = float(baseline["brain_held_out_gap"] - result["brain_held_out_gap"])
                result["mean_gap_improvement"] = float(baseline["held_out_mean_gap"] - result["mean_held_out_gap"])
                result["corr_improvement"] = float(result["held_out_score_correlation"] - baseline["held_out_score_correlation"])
                rows.append(result)

                objective = result["brain_held_out_gap"] + 0.2 * result["mean_held_out_gap"] - 0.01 * result["held_out_score_correlation"]
                if best is None or objective < best[0]:
                    best = (objective, result)

    assert best is not None
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(base_rows),
            "config_count": len(rows),
            "source_files": [
                "semantic_4d_confidence_vector_domain_correction_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_semantic_4d_vector": baseline,
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "结构化脑侧约束不是越多越好，需要找到一个轻量但有效的约束混合区，才能在不破坏 D 和真实任务的前提下压低 brain held-out gap。",
            "current_stage": "脑侧约束混合区扫描",
            "next_question": "如果轻量混合区明显优于全量扩展，下一步就应优先扩脑侧候选覆盖面，并保持低过约束。"
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
