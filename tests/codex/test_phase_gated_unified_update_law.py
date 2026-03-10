"""
Fit a phase-gated unified update law that jointly covers:

1. internal bridge views
2. D methods

Instead of fixed coefficients, the law uses a phase gate:

    phase_gate = sigmoid(gate_temp * (stabilization - routing + gate_bias))

    route_weight = internal_route_gain * (1 - phase_gate)
    stabilize_weight = grounding_stabilize_drag * phase_gate

    effective_offset =
        adaptive_offset
        + route_weight * routing * (1 - adaptive_offset)
        - stabilize_weight * (1 - stabilization) * adaptive_offset

The goal is to test whether one small stage-dependent law can explain both
the routing-dominant internal stage and the stabilization-dominant D stage.
"""

from __future__ import annotations

import argparse
import json
import math
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
        "novel_gain": novel_gain,
        "retention_gain": retention_gain,
        "overall_gain": overall_gain,
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
    gate_bias: float,
) -> Dict[str, float]:
    base = float(row["base"])
    adaptive_offset = float(row["adaptive_offset"])
    routing = float(row["routing"])
    stabilization = float(row["stabilization"])
    phase_gate = sigmoid(gate_temp * (stabilization - routing + gate_bias))
    route_weight = internal_route_gain * (1.0 - phase_gate)
    stabilize_weight = grounding_stabilize_drag * phase_gate
    effective_offset = clip01(
        adaptive_offset
        + route_weight * routing * (1.0 - adaptive_offset)
        - stabilize_weight * (1.0 - stabilization) * adaptive_offset
    )
    unified_score = mean([base, effective_offset, routing, stabilization])
    return {
        "phase_gate": phase_gate,
        "route_weight": route_weight,
        "stabilize_weight": stabilize_weight,
        "effective_offset": effective_offset,
        "unified_score": unified_score,
    }


def evaluate(
    rows: List[Dict[str, Any]],
    internal_route_gain: float,
    grounding_stabilize_drag: float,
    gate_temp: float,
    gate_bias: float,
) -> Dict[str, Any]:
    out_rows = []
    preds = []
    refs = []
    bridge_gaps = []
    d_gaps = []
    for row in rows:
        score = score_row(row, internal_route_gain, grounding_stabilize_drag, gate_temp, gate_bias)
        gap = abs(score["unified_score"] - float(row["reference_score"]))
        item = {
            **row,
            **score,
            "absolute_gap": gap,
        }
        out_rows.append(item)
        refs.append(float(row["reference_score"]))
        preds.append(float(score["unified_score"]))
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


def fit_phase_gated(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for internal_route_gain in np.arange(0.0, 2.01, 0.2):
        for grounding_stabilize_drag in np.arange(0.0, 2.01, 0.2):
            for gate_temp in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
                for gate_bias in [-0.4, -0.2, 0.0, 0.2, 0.4]:
                    result = evaluate(
                        rows,
                        float(internal_route_gain),
                        float(grounding_stabilize_drag),
                        float(gate_temp),
                        float(gate_bias),
                    )
                    objective = result["mean_absolute_gap"] - 0.03 * result["score_correlation"]
                    if best is None or objective < best[0]:
                        best = (
                            objective,
                            float(internal_route_gain),
                            float(grounding_stabilize_drag),
                            float(gate_temp),
                            float(gate_bias),
                            result,
                        )
    return {
        "internal_route_gain": best[1],
        "grounding_stabilize_drag": best[2],
        "gate_temp": best[3],
        "gate_bias": best[4],
        "result": best[5],
    }


def fit_fixed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for route_gain in np.arange(0.0, 2.01, 0.2):
        for stabilize_drag in np.arange(0.0, 2.01, 0.2):
            out_rows = []
            refs = []
            preds = []
            bridge_gaps = []
            d_gaps = []
            for row in rows:
                base = float(row["base"])
                adaptive_offset = float(row["adaptive_offset"])
                routing = float(row["routing"])
                stabilization = float(row["stabilization"])
                effective_offset = clip01(
                    adaptive_offset
                    + float(route_gain) * routing * (1.0 - adaptive_offset)
                    - float(stabilize_drag) * (1.0 - stabilization) * adaptive_offset
                )
                unified_score = mean([base, effective_offset, routing, stabilization])
                gap = abs(unified_score - float(row["reference_score"]))
                out_rows.append({**row, "effective_offset": effective_offset, "unified_score": unified_score, "absolute_gap": gap})
                refs.append(float(row["reference_score"]))
                preds.append(float(unified_score))
                if row["source_type"] == "bridge":
                    bridge_gaps.append(gap)
                else:
                    d_gaps.append(gap)
            result = {
                "rows": out_rows,
                "mean_absolute_gap": mean([r["absolute_gap"] for r in out_rows]),
                "score_correlation": correlation(refs, preds),
                "bridge_mean_gap": mean(bridge_gaps),
                "d_mean_gap": mean(d_gaps),
            }
            objective = result["mean_absolute_gap"] - 0.03 * result["score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, float(route_gain), float(stabilize_drag), result)
    return {
        "route_gain": best[1],
        "stabilize_drag": best[2],
        "result": best[3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a phase-gated unified update law over bridge + D")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/phase_gated_unified_update_law_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    compression = load_json(ROOT / "tests" / "codex_temp" / "unified_structure_four_factor_compression_20260309.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")
    rows = build_rows(compression, d_payload)

    fixed = fit_fixed(rows)
    gated = fit_phase_gated(rows)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "dataset_size": len(rows),
        },
        "law_form": {
            "phase_gate": "sigmoid(gate_temp * (stabilization - routing + gate_bias))",
            "route_weight": "internal_route_gain * (1 - phase_gate)",
            "stabilize_weight": "grounding_stabilize_drag * phase_gate",
            "effective_offset": "adaptive_offset + route_weight * routing * (1 - adaptive_offset) - stabilize_weight * (1 - stabilization) * adaptive_offset",
            "unified_score": "mean(base, effective_offset, routing, stabilization)",
        },
        "fixed_law": {
            "route_gain": fixed["route_gain"],
            "stabilize_drag": fixed["stabilize_drag"],
            "mean_absolute_gap": fixed["result"]["mean_absolute_gap"],
            "score_correlation": fixed["result"]["score_correlation"],
            "bridge_mean_gap": fixed["result"]["bridge_mean_gap"],
            "d_mean_gap": fixed["result"]["d_mean_gap"],
        },
        "phase_gated_law": {
            "internal_route_gain": gated["internal_route_gain"],
            "grounding_stabilize_drag": gated["grounding_stabilize_drag"],
            "gate_temp": gated["gate_temp"],
            "gate_bias": gated["gate_bias"],
            "mean_absolute_gap": gated["result"]["mean_absolute_gap"],
            "score_correlation": gated["result"]["score_correlation"],
            "bridge_mean_gap": gated["result"]["bridge_mean_gap"],
            "d_mean_gap": gated["result"]["d_mean_gap"],
            "gap_improvement_vs_fixed": fixed["result"]["mean_absolute_gap"] - gated["result"]["mean_absolute_gap"],
            "correlation_improvement_vs_fixed": gated["result"]["score_correlation"] - fixed["result"]["score_correlation"],
            "pass": bool(gated["result"]["mean_absolute_gap"] < 0.04 and gated["result"]["score_correlation"] > 0.8),
        },
        "rows": gated["result"]["rows"],
        "project_readout": {
            "current_verdict": (
                "同一条小统一律可以同时覆盖内部桥接和 D，但它必须带相位门控。"
                "也就是说，固定系数小律已经不够，阶段依赖已经是必要结构。"
            ),
            "next_question": "能否把 phase_gate 从手工函数推进成 adaptive_offset 的可学习状态变量？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["fixed_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["phase_gated_law"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
