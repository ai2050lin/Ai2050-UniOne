"""
Fit a state-variable unified update law with an explicit calibration term.

This follows the previous conclusion:
1. fixed-coefficient law can calibrate but is weak across phases
2. phase-gated law can sort but does not calibrate well enough

So this script adds:

    z_t = tanh(alpha * (stabilization - routing) + beta * (adaptive_offset - 0.5) + bias)
    phase_gate = sigmoid(gate_temp * z_t)

and a small calibration head:

    calibrated_score =
        mean(base, effective_offset, routing, stabilization)
        + cal_shift * phase_gate
        + cal_scale * (effective_offset - adaptive_offset)

The purpose is still minimality: test whether a small state variable plus a
small calibration term can jointly improve ranking and scale alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

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


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bridge_rows(compression: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in compression["views"]:
        rows.append(
            {
                "row_id": row["view_id"],
                "source_type": "bridge",
                "base": float(row["factors"]["base"]),
                "adaptive_offset": float(row["factors"]["adaptive_offset"]),
                "routing": float(row["factors"]["routing"]),
                "stabilization": float(row["factors"]["stabilization"]),
                "reference_score": float(row["reference_score"]),
            }
        )
    return rows


def d_row(name: str, row: Dict[str, Any], base_anchor: float) -> Dict[str, Any]:
    novel_gain = float(row["novel_gain"])
    retention_gain = float(row["retention_gain"])
    overall_gain = float(row["overall_gain"])
    dual_score = float(row.get("dual_score", 0.0))
    return {
        "row_id": name,
        "source_type": "D",
        "base": base_anchor,
        "adaptive_offset": clip01(0.5 + novel_gain),
        "routing": clip01(0.5 + dual_score),
        "stabilization": clip01(0.5 + retention_gain),
        "reference_score": clip01(0.5 + overall_gain),
    }


def build_rows(compression: Dict[str, Any], d_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_anchor = float(compression["factor_summary"]["means"]["base"])
    rows = bridge_rows(compression)
    rows.extend(
        [
            d_row("residual_gate", d_payload["residual_gate"]["top_overall"][0], base_anchor),
            d_row("bayes_consolidation", d_payload["bayes_consolidation"]["top_overall"][0], base_anchor),
            d_row("learned_controller", d_payload["learned_controller"]["best_overall"], base_anchor),
            d_row("two_phase", d_payload["two_phase_consolidation"]["best_overall"], base_anchor),
            d_row("three_phase", d_payload["three_phase_consolidation"]["best_overall"], base_anchor),
            d_row("base_offset", d_payload["base_offset_consolidation"]["top_overall"][0], base_anchor),
            d_row("offset_stabilization", d_payload["offset_stabilization"]["top_overall"][0], base_anchor),
            d_row("multistage", d_payload["multistage_stabilization"]["top_overall"][0], base_anchor),
        ]
    )
    return rows


def score_row(
    row: Dict[str, Any],
    internal_route_gain: float,
    grounding_stabilize_drag: float,
    gate_temp: float,
    alpha: float,
    beta: float,
    bias: float,
    cal_shift: float,
    cal_scale: float,
) -> Dict[str, float]:
    base = float(row["base"])
    adaptive_offset = float(row["adaptive_offset"])
    routing = float(row["routing"])
    stabilization = float(row["stabilization"])

    z_state = math.tanh(alpha * (stabilization - routing) + beta * (adaptive_offset - 0.5) + bias)
    phase_gate = sigmoid(gate_temp * z_state)
    route_weight = internal_route_gain * (1.0 - phase_gate)
    stabilize_weight = grounding_stabilize_drag * phase_gate
    effective_offset = clip01(
        adaptive_offset
        + route_weight * routing * (1.0 - adaptive_offset)
        - stabilize_weight * (1.0 - stabilization) * adaptive_offset
    )
    raw_score = mean([base, effective_offset, routing, stabilization])
    calibrated_score = clip01(raw_score + cal_shift * phase_gate + cal_scale * (effective_offset - adaptive_offset))
    return {
        "z_state": z_state,
        "phase_gate": phase_gate,
        "route_weight": route_weight,
        "stabilize_weight": stabilize_weight,
        "effective_offset": effective_offset,
        "raw_score": raw_score,
        "calibrated_score": calibrated_score,
    }


def evaluate(
    rows: List[Dict[str, Any]],
    internal_route_gain: float,
    grounding_stabilize_drag: float,
    gate_temp: float,
    alpha: float,
    beta: float,
    bias: float,
    cal_shift: float,
    cal_scale: float,
) -> Dict[str, Any]:
    out_rows = []
    preds = []
    refs = []
    bridge_gaps = []
    d_gaps = []
    for row in rows:
        score = score_row(
            row,
            internal_route_gain,
            grounding_stabilize_drag,
            gate_temp,
            alpha,
            beta,
            bias,
            cal_shift,
            cal_scale,
        )
        gap = abs(score["calibrated_score"] - float(row["reference_score"]))
        item = {**row, **score, "absolute_gap": gap}
        out_rows.append(item)
        refs.append(float(row["reference_score"]))
        preds.append(float(score["calibrated_score"]))
        if row["source_type"] == "bridge":
            bridge_gaps.append(gap)
        else:
            d_gaps.append(gap)
    return {
        "rows": out_rows,
        "mean_absolute_gap": mean([r["absolute_gap"] for r in out_rows]),
        "score_correlation": correlation(refs, preds),
        "bridge_mean_gap": mean(bridge_gaps),
        "d_mean_gap": mean(d_gaps),
    }


def fit_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for internal_route_gain in [0.0, 1.0, 2.0]:
        for grounding_stabilize_drag in [0.0, 1.0, 2.0]:
            for gate_temp in [2.0, 4.0, 6.0]:
                for alpha in [1.0, 2.0, 3.0]:
                    for beta in [-1.0, 0.0, 1.0]:
                        for bias in [-0.3, 0.0, 0.3]:
                            for cal_shift in [-0.2, 0.0, 0.2]:
                                for cal_scale in [-0.2, 0.0, 0.2]:
                                    result = evaluate(
                                        rows,
                                        internal_route_gain,
                                        grounding_stabilize_drag,
                                        gate_temp,
                                        alpha,
                                        beta,
                                        bias,
                                        cal_shift,
                                        cal_scale,
                                    )
                                    objective = result["mean_absolute_gap"] - 0.03 * result["score_correlation"]
                                    if best is None or objective < best[0]:
                                        best = (
                                            objective,
                                            internal_route_gain,
                                            grounding_stabilize_drag,
                                            gate_temp,
                                            alpha,
                                            beta,
                                            bias,
                                            cal_shift,
                                            cal_scale,
                                            result,
                                        )
    return {
        "internal_route_gain": best[1],
        "grounding_stabilize_drag": best[2],
        "gate_temp": best[3],
        "alpha": best[4],
        "beta": best[5],
        "bias": best[6],
        "cal_shift": best[7],
        "cal_scale": best[8],
        "result": best[9],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a state-variable calibrated unified law over bridge + D")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/state_variable_calibrated_unified_law_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    compression = load_json(ROOT / "tests" / "codex_temp" / "unified_structure_four_factor_compression_20260309.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    phase_gated = load_json(ROOT / "tests" / "codex_temp" / "phase_gated_unified_update_law_20260309.json")
    rows = build_rows(compression, d_payload)

    best = fit_best(rows)
    result = best["result"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "dataset_size": len(rows),
        },
        "law_form": {
            "z_state": "tanh(alpha * (stabilization - routing) + beta * (adaptive_offset - 0.5) + bias)",
            "phase_gate": "sigmoid(gate_temp * z_state)",
            "effective_offset": "adaptive_offset + route_weight * routing * (1 - adaptive_offset) - stabilize_weight * (1 - stabilization) * adaptive_offset",
            "calibrated_score": "mean(base, effective_offset, routing, stabilization) + cal_shift * phase_gate + cal_scale * (effective_offset - adaptive_offset)",
        },
        "phase_gated_baseline": phase_gated["phase_gated_law"],
        "state_variable_law": {
            "internal_route_gain": best["internal_route_gain"],
            "grounding_stabilize_drag": best["grounding_stabilize_drag"],
            "gate_temp": best["gate_temp"],
            "alpha": best["alpha"],
            "beta": best["beta"],
            "bias": best["bias"],
            "cal_shift": best["cal_shift"],
            "cal_scale": best["cal_scale"],
            "mean_absolute_gap": result["mean_absolute_gap"],
            "score_correlation": result["score_correlation"],
            "bridge_mean_gap": result["bridge_mean_gap"],
            "d_mean_gap": result["d_mean_gap"],
            "gap_improvement_vs_phase_gated": phase_gated["phase_gated_law"]["mean_absolute_gap"] - result["mean_absolute_gap"],
            "correlation_improvement_vs_phase_gated": result["score_correlation"] - phase_gated["phase_gated_law"]["score_correlation"],
            "pass": bool(result["mean_absolute_gap"] < 0.08 and result["score_correlation"] > 0.9),
        },
        "rows": result["rows"],
        "project_readout": {
            "current_verdict": (
                "状态变量和标定项能否同时改善误差和相关性，决定下一步应不应该把 phase_gate 升级成真正的可学习状态。"
            ),
            "next_question": "如果状态变量化仍然只能改善一边，那么下一步要不要直接拆成‘排序律 + 标定律’双层结构？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["state_variable_law"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
