"""
Expand brain-side candidate coverage on top of the lightweight valid mix.

Goal:
- reuse the best light structured mix found by the sweep
- add only a tiny number of candidate-coverage anchors derived from the
  train-side brain component statistics
- test whether broader brain candidate coverage can further reduce
  brain held-out gap without returning to over-constraint
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


def build_light_focus_constraints(
    train_rows: List[Dict[str, Any]],
    topk: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    brain_rows = [row for row in train_rows if row["domain"] == "brain"]
    if not brain_rows:
        return []
    component_count = len(brain_rows[0]["brain_components"])
    constraints: List[Dict[str, Any]] = []
    for row in brain_rows:
        meta = row.get("brain_meta", {})
        overall = float(meta.get("overall_bridge_score", row["reference_score"]))
        comps = np.array(row["brain_components"], dtype=np.float64)
        top_indices = np.argsort(comps)[::-1][:topk]
        for idx in top_indices.tolist():
            focused = np.zeros(component_count, dtype=np.float64)
            focused[idx] = float(comps[idx])
            focused[(idx + 1) % component_count] = overall
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
                    "reference_score": float(alpha * float(comps[idx]) + (1.0 - alpha) * overall),
                }
            )
    return constraints


def build_candidate_coverage_constraints(
    train_rows: List[Dict[str, Any]],
    coverage_mode: str,
    coverage_weight: float,
    coverage_topm: int,
) -> List[Dict[str, Any]]:
    brain_rows = [row for row in train_rows if row["domain"] == "brain"]
    if not brain_rows or coverage_mode == "none":
        return []

    component_count = len(brain_rows[0]["brain_components"])
    comp_matrix = np.array([row["brain_components"] for row in brain_rows], dtype=np.float64)
    comp_means = comp_matrix.mean(axis=0)
    comp_spreads = comp_matrix.max(axis=0) - comp_matrix.min(axis=0)
    overall_mean = mean([float(row["reference_score"]) for row in brain_rows])
    top_indices = np.argsort(comp_means)[::-1][:coverage_topm].tolist()
    constraints: List[Dict[str, Any]] = []

    if coverage_mode in ("mean_anchor", "both"):
        for idx in top_indices:
            vector = np.zeros(component_count, dtype=np.float64)
            vector[idx] = float(comp_means[idx])
            vector[(idx + 1) % component_count] = float(1.0 - comp_spreads[idx])
            vector[(idx + 2) % component_count] = overall_mean
            constraints.append(
                {
                    "row_id": f"brain::coverage_mean::{idx}",
                    "domain": "brain",
                    "subdomain": f"coverage_mean_{idx}",
                    "signal_a": 0.0,
                    "signal_b": 0.0,
                    "signal_c": 0.0,
                    "signal_d": 0.0,
                    "brain_components": [float(v) for v in vector.tolist()],
                    "reference_score": float(coverage_weight * comp_means[idx] + (1.0 - coverage_weight) * overall_mean),
                }
            )

    if coverage_mode in ("contrast_anchor", "both") and len(top_indices) >= 2:
        left, right = top_indices[0], top_indices[1]
        vector = np.zeros(component_count, dtype=np.float64)
        vector[left] = float(comp_means[left])
        vector[right] = float(comp_means[right])
        vector[(right + 1) % component_count] = float(abs(comp_means[left] - comp_means[right]))
        margin = float(comp_means[left] - comp_means[right])
        constraints.append(
            {
                "row_id": f"brain::coverage_contrast::{left}_{right}",
                "domain": "brain",
                "subdomain": f"coverage_contrast_{left}_{right}",
                "signal_a": 0.0,
                "signal_b": 0.0,
                "signal_c": 0.0,
                "signal_d": 0.0,
                "brain_components": [float(v) for v in vector.tolist()],
                "reference_score": float(coverage_weight * margin + (1.0 - coverage_weight) * overall_mean),
            }
        )

    return constraints


def leave_one_out(
    base_rows: List[Dict[str, Any]],
    correction_dim: int,
    ranking_lambda: float,
    calibration_lambda: float,
    topk: int,
    alpha: float,
    coverage_mode: str,
    coverage_weight: float,
    coverage_topm: int,
) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    gaps: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}
    focus_counts: List[int] = []
    coverage_counts: List[int] = []

    for hold_idx, held_row in enumerate(base_rows):
        train_rows = [row for idx, row in enumerate(base_rows) if idx != hold_idx]
        focus_rows = build_light_focus_constraints(train_rows, topk=topk, alpha=alpha)
        coverage_rows = build_candidate_coverage_constraints(
            train_rows,
            coverage_mode=coverage_mode,
            coverage_weight=coverage_weight,
            coverage_topm=coverage_topm,
        )
        train_aug = train_rows + focus_rows + coverage_rows
        focus_counts.append(len(focus_rows))
        coverage_counts.append(len(coverage_rows))

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
        "alpha": float(alpha),
        "coverage_mode": coverage_mode,
        "coverage_weight": float(coverage_weight),
        "coverage_topm": int(coverage_topm),
        "mean_focus_count": mean(focus_counts),
        "mean_coverage_count": mean(coverage_counts),
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "brain_held_out_gap": mean(gaps_by_domain["brain"]),
        "d_held_out_gap": mean(gaps_by_domain["D"]),
        "real_task_held_out_gap": mean(gaps_by_domain["real_task"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand brain candidate coverage on top of the lightweight valid mix")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/semantic_4d_brain_candidate_coverage_expansion_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_confidence_vector_domain_correction_20260310.json")
    sweep_payload = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_brain_constraint_expansion_sweep_20260310.json")
    brain_payload = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")

    base_rows = brain_law.build_brain_rows(brain_payload) + brain_law.build_d_rows(d_payload) + brain_law.build_real_rows(real_payload)
    correction_dim = int(baseline_payload["best_correction_dim"])
    ranking_lambda = float(baseline_payload["ranking_layer"]["ridge_lambda"])
    calibration_lambda = float(baseline_payload["calibration_layer"]["ridge_lambda"])
    baseline = baseline_payload["semantic_4d_vector_domain_correction"]
    light_best = sweep_payload["best_config"]

    rows = []
    best = None
    for coverage_mode in ["none", "mean_anchor", "contrast_anchor", "both"]:
        for coverage_weight in [0.2, 0.35, 0.5, 0.65]:
            for coverage_topm in [1, 2]:
                result = leave_one_out(
                    base_rows=base_rows,
                    correction_dim=correction_dim,
                    ranking_lambda=ranking_lambda,
                    calibration_lambda=calibration_lambda,
                    topk=int(light_best["topk"]),
                    alpha=float(light_best["alpha"]),
                    coverage_mode=coverage_mode,
                    coverage_weight=coverage_weight,
                    coverage_topm=coverage_topm,
                )
                result["brain_gap_improvement_vs_baseline"] = float(baseline["brain_held_out_gap"] - result["brain_held_out_gap"])
                result["mean_gap_improvement_vs_baseline"] = float(baseline["held_out_mean_gap"] - result["mean_held_out_gap"])
                result["brain_gap_improvement_vs_light_best"] = float(light_best["brain_held_out_gap"] - result["brain_held_out_gap"])
                result["mean_gap_improvement_vs_light_best"] = float(light_best["mean_held_out_gap"] - result["mean_held_out_gap"])
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
                "semantic_4d_brain_constraint_expansion_sweep_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_semantic_4d_vector": baseline,
        "light_mix_best": light_best,
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一步测试在轻量有效区上加入更有结构的候选覆盖约束，看看脑侧误差能否继续下降，而不重新掉入过约束。",
            "current_stage": "脑侧候选覆盖面扩展",
            "next_question": "如果轻量覆盖扩展优于当前有效区，下一步就该沿着低过约束的方向继续扩大脑侧候选覆盖面。"
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
