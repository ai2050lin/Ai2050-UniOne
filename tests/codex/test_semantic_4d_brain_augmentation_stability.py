"""
Test whether the large brain held-out gap is mainly caused by thin brain samples.

Method:
- keep the semantic 4D + 3D vector correction law unchanged
- augment only the training-side brain rows with small controlled perturbations
- when holding out a brain row, exclude all augmentations derived from it
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


def augment_brain_rows(rows: List[Dict[str, Any]], seed: int = 0, copies_per_row: int = 24) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    augmented: List[Dict[str, Any]] = []
    for row in rows:
        if row["domain"] != "brain":
            continue
        base_components = np.array(row["brain_components"], dtype=np.float64)
        ref = float(row["reference_score"])
        for idx in range(copies_per_row):
            noise = rng.normal(0.0, 0.025, size=base_components.shape[0])
            perturbed = np.clip(base_components + noise, 0.0, 1.0)
            ref_noise = float(rng.normal(0.0, 0.006))
            augmented.append(
                {
                    "row_id": f"{row['row_id']}::aug::{idx}",
                    "domain": "brain",
                    "subdomain": f"{row['subdomain']}_aug",
                    "signal_a": 0.0,
                    "signal_b": 0.0,
                    "signal_c": 0.0,
                    "signal_d": 0.0,
                    "brain_components": [float(v) for v in perturbed.tolist()],
                    "reference_score": float(np.clip(ref + ref_noise, 0.0, 1.0)),
                    "source_brain_row": row["row_id"],
                }
            )
    return augmented


def leave_one_out_with_augmented_brain(rows: List[Dict[str, Any]], correction_dim: int, ranking_lambda: float, calibration_lambda: float) -> Dict[str, Any]:
    refs: List[float] = []
    preds: List[float] = []
    gaps: List[float] = []
    gaps_by_domain: Dict[str, List[float]] = {"brain": [], "D": [], "real_task": []}

    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        brain_aug = augment_brain_rows(train_rows, seed=hold_idx)
        if held_row["domain"] == "brain":
            brain_aug = [row for row in brain_aug if row["source_brain_row"] != held_row["row_id"]]
        augmented_train = train_rows + brain_aug

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
        "pass": bool(mean(gaps) < 0.02 and correlation(refs, preds) > 0.90),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test whether brain sample thinness causes the semantic 4D + 3D law to fail on brain held-out")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/semantic_4d_brain_augmentation_stability_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = load_json(ROOT / "tests" / "codex_temp" / "semantic_4d_confidence_vector_domain_correction_20260310.json")
    brain_payload = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    real_payload = load_json(ROOT / "tests" / "codex_temp" / "real_task_driven_two_layer_unified_law_20260310.json")

    rows = brain_law.build_brain_rows(brain_payload) + brain_law.build_d_rows(d_payload) + brain_law.build_real_rows(real_payload)
    correction_dim = int(baseline_payload["best_correction_dim"])
    ranking_lambda = float(baseline_payload["ranking_layer"]["ridge_lambda"])
    calibration_lambda = float(baseline_payload["calibration_layer"]["ridge_lambda"])

    baseline = baseline_payload["semantic_4d_vector_domain_correction"]
    augmented_held = leave_one_out_with_augmented_brain(rows, correction_dim, ranking_lambda, calibration_lambda)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "row_count": len(rows),
            "copies_per_brain_row": 24,
            "source_files": [
                "semantic_4d_confidence_vector_domain_correction_20260310.json",
                "dnn_brain_puzzle_bridge_20260308.json",
                "d_problem_atlas_summary_20260309.json",
                "real_task_driven_two_layer_unified_law_20260310.json",
            ],
        },
        "baseline_semantic_4d_vector": baseline,
        "brain_augmented_leave_one_out": augmented_held,
        "improvement": {
            "mean_gap_improvement": float(baseline["held_out_mean_gap"] - augmented_held["mean_held_out_gap"]),
            "brain_gap_improvement": float(baseline["brain_held_out_gap"] - augmented_held["brain_held_out_gap"]),
            "d_gap_improvement": float(baseline["d_held_out_gap"] - augmented_held["d_held_out_gap"]),
            "real_task_gap_improvement": float(baseline["real_task_held_out_gap"] - augmented_held["real_task_held_out_gap"]),
            "corr_improvement": float(augmented_held["held_out_score_correlation"] - baseline["held_out_score_correlation"]),
        },
        "project_readout": {
            "summary": "如果脑样本扩增后 brain_held_out_gap 明显下降，就说明前一轮脑侧泛化问题主要来自脑样本过薄，而不是 4D + 3D 结构本身错误。",
            "next_question": "如果脑侧确实是样本薄弱问题，下一步就该系统补脑侧候选约束，而不是继续改主骨架。",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
