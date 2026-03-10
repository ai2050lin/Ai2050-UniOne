"""
Search a low-degree unified update law on top of the four-factor compression:

    base / adaptive_offset / routing / stabilization

The candidate law only modifies adaptive_offset:

    effective_offset =
        adaptive_offset
        + route_gain * routing * (1 - adaptive_offset)
        - stabilize_drag * (1 - stabilization) * adaptive_offset

and then writes a single scalar unified score:

    unified_score = mean(base, effective_offset, routing, stabilization)

This is intentionally small: the goal is to test whether a two-parameter law
already retains most of the explanatory ordering of the current bridge scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


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


def law_score(factors: Dict[str, float], route_gain: float, stabilize_drag: float) -> Tuple[float, float]:
    base = float(factors["base"])
    adaptive_offset = float(factors["adaptive_offset"])
    routing = float(factors["routing"])
    stabilization = float(factors["stabilization"])
    effective_offset = clip01(
        adaptive_offset
        + route_gain * routing * (1.0 - adaptive_offset)
        - stabilize_drag * (1.0 - stabilization) * adaptive_offset
    )
    unified_score = mean([base, effective_offset, routing, stabilization])
    return unified_score, effective_offset


def evaluate(views: List[Dict[str, Any]], route_gain: float, stabilize_drag: float) -> Dict[str, Any]:
    rows = []
    refs = []
    preds = []
    raw_preds = []
    for row in views:
        unified_score, effective_offset = law_score(row["factors"], route_gain, stabilize_drag)
        reference = float(row["reference_score"])
        raw_score = float(row["compressed_score"])
        refs.append(reference)
        preds.append(unified_score)
        raw_preds.append(raw_score)
        rows.append(
            {
                "view_id": row["view_id"],
                "reference_score": reference,
                "raw_four_factor_score": raw_score,
                "unified_score": unified_score,
                "effective_offset": effective_offset,
                "absolute_gap": abs(unified_score - reference),
                "raw_gap": abs(raw_score - reference),
            }
        )
    mae = mean([r["absolute_gap"] for r in rows])
    raw_mae = mean([r["raw_gap"] for r in rows])
    corr = correlation(refs, preds)
    raw_corr = correlation(refs, raw_preds)
    return {
        "rows": rows,
        "mean_absolute_gap": mae,
        "raw_mean_absolute_gap": raw_mae,
        "gap_improvement": raw_mae - mae,
        "score_correlation": corr,
        "raw_score_correlation": raw_corr,
        "correlation_improvement": corr - raw_corr,
    }


def choose_best_law(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = []
    for route_gain in np.arange(0.0, 2.01, 0.1):
        for stabilize_drag in np.arange(0.0, 2.01, 0.1):
            result = evaluate(views, float(route_gain), float(stabilize_drag))
            objective = result["mean_absolute_gap"] - 0.03 * result["score_correlation"]
            candidates.append((objective, float(route_gain), float(stabilize_drag), result))
    best = min(candidates, key=lambda x: x[0])
    return {
        "route_gain": best[1],
        "stabilize_drag": best[2],
        "result": best[3],
    }


def leave_one_out(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    folds = []
    held_out_errors = []
    for hold_idx, held_row in enumerate(views):
        train_views = [row for idx, row in enumerate(views) if idx != hold_idx]
        best = choose_best_law(train_views)
        pred, effective_offset = law_score(
            held_row["factors"],
            best["route_gain"],
            best["stabilize_drag"],
        )
        err = abs(pred - float(held_row["reference_score"]))
        held_out_errors.append(err)
        folds.append(
            {
                "view_id": held_row["view_id"],
                "route_gain": best["route_gain"],
                "stabilize_drag": best["stabilize_drag"],
                "held_out_prediction": pred,
                "held_out_reference": float(held_row["reference_score"]),
                "held_out_gap": err,
                "effective_offset": effective_offset,
            }
        )
    return {
        "folds": folds,
        "mean_held_out_gap": mean(held_out_errors),
        "pass": bool(mean(held_out_errors) < 0.04),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a low-degree unified update law on top of four-factor compression")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/unified_update_law_candidate_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    compression = load_json(ROOT / "tests" / "codex_temp" / "unified_structure_four_factor_compression_20260309.json")
    views = compression["views"]

    best = choose_best_law(views)
    holdout = leave_one_out(views)
    result = best["result"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "source": "unified_structure_four_factor_compression_20260309.json",
        },
        "law_form": {
            "effective_offset": "adaptive_offset + route_gain * routing * (1 - adaptive_offset) - stabilize_drag * (1 - stabilization) * adaptive_offset",
            "unified_score": "mean(base, effective_offset, routing, stabilization)",
            "intuition": [
                "routing 足够强时，adaptive_offset 更容易被真正带起来。",
                "stabilization 不足时，adaptive_offset 更容易被压回去。",
                "如果两参数律已经能逼近桥接分数，说明下一步该优先压更新律，而不是继续堆对象名词。",
            ],
        },
        "best_law": {
            "route_gain": best["route_gain"],
            "stabilize_drag": best["stabilize_drag"],
            "mean_absolute_gap": result["mean_absolute_gap"],
            "raw_mean_absolute_gap": result["raw_mean_absolute_gap"],
            "gap_improvement": result["gap_improvement"],
            "score_correlation": result["score_correlation"],
            "raw_score_correlation": result["raw_score_correlation"],
            "correlation_improvement": result["correlation_improvement"],
            "pass": bool(result["mean_absolute_gap"] < 0.04 and result["score_correlation"] > 0.75),
        },
        "view_results": result["rows"],
        "leave_one_out": holdout,
        "project_readout": {
            "current_verdict": (
                "四因子结构之上已经存在一条很小的两参数更新律候选。"
                "它主要修正 adaptive_offset，而不是重新定义全部对象。"
            ),
            "risk_note": (
                "当前只有 4 个视角点，样本很少。结果更像“低维统一律存在性信号”，"
                "还不能写成最终定律。"
            ),
            "next_question": "能否把这条两参数候选律扩到真实任务闭环，而不是只在桥接分数上成立？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
