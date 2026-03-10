"""
Structured brain-side constraint expansion for the semantic 4D + 3D law.

Goal:
- avoid simple noise augmentation
- expand the brain-side training surface with component-focused and
  cross-model aggregate constraints derived from the existing brain bridge
- test whether the current 4D semantic skeleton + 3D vector correction
  remains stable under a thicker brain-side constraint system
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


def build_structured_brain_constraints(train_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        for idx, comp_score in enumerate(comps.tolist()):
            focused = np.zeros(component_count, dtype=np.float64)
            focused[idx] = comp_score
            focused[(idx + 1) % component_count] = overall
            focused[(idx + 2) % component_count] = dnn_reverse
            focused[(idx + 3) % component_count] = brain_align
            constraints.append(
                {
                    "row_id": f"{row['row_id']}::component_focus::{idx}",
                    "domain": "brain",
                    "subdomain": f"{row['subdomain']}_component_{idx}",
                    "signal_a": 0.0,
                    "signal_b": 0.0,
                    "signal_c": 0.0,
                    "signal_d": 0.0,
                    "brain_components": [float(v) for v in focused.tolist()],
                    "reference_score": float(0.55 * comp_score + 0.45 * overall),
                    "constraint_kind": "component_focus",
                    "source_brain_rows": [row["row_id"]],
                }
            )

    for idx in range(component_count):
        comp_values = [float(row["brain_components"][idx]) for row in brain_rows]
        spread = max(comp_values) - min(comp_values)
        vector = np.zeros(component_count, dtype=np.float64)
        vector[idx] = mean(comp_values)
        vector[(idx + 1) % component_count] = spread
        vector[(idx + 2) % component_count] = overall_mean
        vector[(idx + 3) % component_count] = float(1.0 - spread)
        constraints.append(
            {
                "row_id": f"brain::aggregate_component::{idx}",
                "domain": "brain",
                "subdomain": f"aggregate_component_{idx}",
                "signal_a": 0.0,
                "signal_b": 0.0,
                "signal_c": 0.0,
                "signal_d": 0.0,
                "brain_components": [float(v) for v in vector.tolist()],
                "reference_score": float(0.7 * mean(comp_values) + 0.3 * (1.0 - spread)),
                "constraint_kind": "component_aggregate",
                "source_brain_rows": [row["row_id"] for row in brain_rows],
            }
        )

    return constraints


def leave_one_out_with_structured_constraints(
    base_rows: List[Dict[str, Any]],
    correction_dim: int,
    ranking_lambda: float,
    calibration_lambda: float,
) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    gaps: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}
    constraint_counts: List[int] = []

    for hold_idx, held_row in enumerate(base_rows):
        train_rows = [row for idx, row in enumerate(base_rows) if idx != hold_idx]
        structured_constraints = build_structured_brain_constraints(train_rows)
        augmented_train = train_rows + structured_constraints
        constraint_counts.append(len(structured_constraints))

        ranking_coef = vec_corr.fit_ranking(augmented_train, ranking_lambda)
        train_ranking_scores = [vec_corr.predict_ranking(row, ranking_coef) for row in augmented_train]
        calibration_coef = vec_corr.fit_calibration(augmented_train, train_ranking_scores, correction_dim, calibration_lambda)
        held_ranking = vec_corr.predict_ranking(held_row, ranking_coef)
        held_pred = vec_corr.predict_calibrated(held_row, held_ranking, correction_dim, calibration_coef)
        held_ref = float(held_row["reference_score"])
        gap = abs(held_pred - held_ref)
        refs.append(held_ref)
        preds.append(held_pred)
        gaps.append(gap)
        gaps_by_domain[held_row["domain"]].append(gap)

    return {
        "mean_held_out_gap": mean(gaps),
        "held_out_score_correlation": correlation(refs, preds),
        "brain_held_out_gap": mean(gaps_by_domain["brain"]),
        "d_held_out_gap": mean(gaps_by_domain["D"]),
        "real_task_held_out_gap": mean(gaps_by_domain["real_task"]),
        "mean_constraint_count": mean(constraint_counts),
        "pass": bool(mean(gaps) < 0.01 and correlation(refs, preds) > 0.95),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand brain-side candidate constraints for the semantic 4D + 3D law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/semantic_4d_brain_constraint_expansion_20260310.json")
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
    structured_template = build_structured_brain_constraints(base_rows)

    structured_held = leave_one_out_with_structured_constraints(
        base_rows=base_rows,
        correction_dim=correction_dim,
        ranking_lambda=ranking_lambda,
        calibration_lambda=calibration_lambda,
    )

    baseline = baseline_payload["semantic_4d_vector_domain_correction"]
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(base_rows),
            "constraint_row_count_template": len(structured_template),
            "source_files": [
                "semantic_4d_confidence_vector_domain_correction_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_semantic_4d_vector": baseline,
        "brain_constraint_expansion_leave_one_out": structured_held,
        "constraint_mix": {
            "component_focus_rows": int(sum(1 for row in structured_template if row["constraint_kind"] == "component_focus")),
            "component_aggregate_rows": int(sum(1 for row in structured_template if row["constraint_kind"] == "component_aggregate")),
        },
        "sample_constraints": [
            {
                "row_id": row["row_id"],
                "subdomain": row["subdomain"],
                "reference_score": row["reference_score"],
                "constraint_kind": row["constraint_kind"],
                "source_brain_rows": row["source_brain_rows"],
            }
            for row in structured_template[:8]
        ],
        "improvement": {
            "mean_gap_improvement": float(baseline["held_out_mean_gap"] - structured_held["mean_held_out_gap"]),
            "brain_gap_improvement": float(baseline["brain_held_out_gap"] - structured_held["brain_held_out_gap"]),
            "d_gap_improvement": float(baseline["d_held_out_gap"] - structured_held["d_held_out_gap"]),
            "real_task_gap_improvement": float(baseline["real_task_held_out_gap"] - structured_held["real_task_held_out_gap"]),
            "corr_improvement": float(structured_held["held_out_score_correlation"] - baseline["held_out_score_correlation"]),
        },
        "project_readout": {
            "summary": "这一步不做噪声扩增，而是把脑侧桥接结果系统拆成组件聚焦约束和跨模型聚合约束，测试 4D 骨架 + 3D 修正是否能在更厚的脑侧候选面上继续稳定。",
            "current_stage": "脑侧候选约束系统扩展",
            "next_question": "如果系统化脑侧约束也能稳定压低 brain held-out gap，下一步就该扩张脑侧候选覆盖面，而不是继续改主骨架。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
