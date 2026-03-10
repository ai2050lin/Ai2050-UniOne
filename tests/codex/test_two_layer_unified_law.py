"""
Two-layer unified law:

1. 排序层：使用当前最优的 phase-gated 统一律分数
2. 标定层：在排序层之上再拟合一个小线性标定器

这个实验用来验证：下一步是否应该正式把统一律拆成
“排序层 + 标定层”，而不是继续要求一条小公式同时做完两件事。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def features(row: Dict[str, Any]) -> List[float]:
    return [
        1.0,
        float(row["phase_gate"]),
        float(row["effective_offset"]) - float(row["adaptive_offset"]),
        float(row["unified_score"]),
    ]


def fit_calibrator(rows: List[Dict[str, Any]]) -> np.ndarray:
    x = np.array([features(row) for row in rows], dtype=np.float64)
    y = np.array([float(row["reference_score"]) for row in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def predict(row: Dict[str, Any], coef: np.ndarray) -> float:
    x = np.array(features(row), dtype=np.float64)
    return float(x @ coef)


def evaluate(rows: List[Dict[str, Any]], coef: np.ndarray) -> Dict[str, Any]:
    out_rows = []
    refs = []
    preds = []
    for row in rows:
        calibrated_score = predict(row, coef)
        refs.append(float(row["reference_score"]))
        preds.append(calibrated_score)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "source_type": row["source_type"],
                "reference_score": float(row["reference_score"]),
                "ranking_score": float(row["unified_score"]),
                "calibrated_score": calibrated_score,
                "phase_gate": float(row["phase_gate"]),
                "absolute_gap": abs(calibrated_score - float(row["reference_score"])),
            }
        )
    return {
        "rows": out_rows,
        "mean_absolute_gap": mean([row["absolute_gap"] for row in out_rows]),
        "score_correlation": correlation(refs, preds),
    }


def leave_one_out(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_rows = []
    gaps = []
    for hold_idx, held_row in enumerate(rows):
        train_rows = [row for idx, row in enumerate(rows) if idx != hold_idx]
        coef = fit_calibrator(train_rows)
        pred = predict(held_row, coef)
        gap = abs(pred - float(held_row["reference_score"]))
        gaps.append(gap)
        out_rows.append(
            {
                "row_id": held_row["row_id"],
                "source_type": held_row["source_type"],
                "held_out_prediction": pred,
                "held_out_reference": float(held_row["reference_score"]),
                "held_out_gap": gap,
            }
        )
    return {
        "rows": out_rows,
        "mean_held_out_gap": mean(gaps),
        "pass": bool(mean(gaps) < 0.06),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a two-layer unified law")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/two_layer_unified_law_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    phase_gated = load_json(ROOT / "tests" / "codex_temp" / "phase_gated_unified_update_law_20260309.json")
    rows = phase_gated["rows"]
    coef = fit_calibrator(rows)
    fitted = evaluate(rows, coef)
    held_out = leave_one_out(rows)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "source": "phase_gated_unified_update_law_20260309.json",
        },
        "ranking_layer": {
            "mean_absolute_gap": float(phase_gated["phase_gated_law"]["mean_absolute_gap"]),
            "score_correlation": float(phase_gated["phase_gated_law"]["score_correlation"]),
        },
        "calibration_layer": {
            "features": [
                "1",
                "phase_gate",
                "effective_offset - adaptive_offset",
                "ranking_score",
            ],
            "coefficients": [float(x) for x in coef.tolist()],
        },
        "two_layer_law": {
            "mean_absolute_gap": fitted["mean_absolute_gap"],
            "score_correlation": fitted["score_correlation"],
            "gap_improvement_vs_ranking": float(phase_gated["phase_gated_law"]["mean_absolute_gap"]) - fitted["mean_absolute_gap"],
            "correlation_improvement_vs_ranking": fitted["score_correlation"] - float(phase_gated["phase_gated_law"]["score_correlation"]),
        },
        "leave_one_out": held_out,
        "rows": fitted["rows"],
        "project_readout": {
            "current_verdict": (
                "如果双层统一律能显著压低误差，但留一法仍然不稳，"
                "说明“排序层 + 标定层”方向是对的，"
                "但当前标定层还只是解析过拟合原型。"
            ),
            "next_question": "下一步应该先把标定层推进到可学习版本，还是直接在真实任务上联合训练排序层和标定层？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["two_layer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["leave_one_out"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
