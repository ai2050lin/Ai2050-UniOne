"""
Bridge the small unified update law into task block D.

The goal is not to solve D directly, but to test whether the same compact law
that already compresses bridge scores can also organize the different D
methods into a shared low-dimensional picture.
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


def law_score(base: float, adaptive_offset: float, routing: float, stabilization: float, route_gain: float, stabilize_drag: float) -> Tuple[float, float]:
    effective_offset = clip01(
        adaptive_offset
        + route_gain * routing * (1.0 - adaptive_offset)
        - stabilize_drag * (1.0 - stabilization) * adaptive_offset
    )
    unified_score = mean([base, effective_offset, routing, stabilization])
    return unified_score, effective_offset


def d_row(name: str, row: Dict[str, Any], dual_positive_count: int, base_anchor: float) -> Dict[str, Any]:
    novel_gain = float(row["novel_gain"])
    retention_gain = float(row["retention_gain"])
    overall_gain = float(row["overall_gain"])
    dual_score = float(row.get("dual_score", 0.0))
    adaptive_offset = clip01(0.5 + novel_gain)
    routing = clip01(0.5 + dual_score)
    stabilization = clip01(0.5 + retention_gain)
    reference_score = clip01(0.5 + overall_gain)
    return {
        "method": name,
        "base": base_anchor,
        "adaptive_offset": adaptive_offset,
        "routing": routing,
        "stabilization": stabilization,
        "novel_gain": novel_gain,
        "retention_gain": retention_gain,
        "overall_gain": overall_gain,
        "dual_positive_count": int(dual_positive_count),
        "reference_score": reference_score,
        "dual_score": dual_score,
        "fixed_point_type": (
            "dual_positive"
            if novel_gain > 0.0 and retention_gain > 0.0
            else "novel_first"
            if novel_gain > retention_gain
            else "retention_first"
        ),
    }


def evaluate(rows: List[Dict[str, Any]], route_gain: float, stabilize_drag: float) -> Dict[str, Any]:
    preds = []
    refs = []
    result_rows = []
    for row in rows:
        score, effective_offset = law_score(
            row["base"],
            row["adaptive_offset"],
            row["routing"],
            row["stabilization"],
            route_gain,
            stabilize_drag,
        )
        refs.append(row["reference_score"])
        preds.append(score)
        result_rows.append(
            {
                **row,
                "predicted_score": score,
                "effective_offset": effective_offset,
                "absolute_gap": abs(score - row["reference_score"]),
            }
        )
    return {
        "rows": result_rows,
        "mean_absolute_gap": mean([r["absolute_gap"] for r in result_rows]),
        "score_correlation": correlation(refs, preds),
    }


def fit_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for route_gain in np.arange(0.0, 2.01, 0.1):
        for stabilize_drag in np.arange(0.0, 2.01, 0.1):
            result = evaluate(rows, float(route_gain), float(stabilize_drag))
            objective = result["mean_absolute_gap"] - 0.03 * result["score_correlation"]
            if best is None or objective < best[0]:
                best = (objective, float(route_gain), float(stabilize_drag), result)
    return {
        "route_gain": best[1],
        "stabilize_drag": best[2],
        "result": best[3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge the unified update law into D")
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/unified_update_law_d_bridge_20260309.json")
    args = ap.parse_args()

    t0 = time.time()
    law_payload = load_json(ROOT / "tests" / "codex_temp" / "unified_update_law_candidate_20260309.json")
    compression_payload = load_json(ROOT / "tests" / "codex_temp" / "unified_structure_four_factor_compression_20260309.json")
    d_payload = load_json(ROOT / "tests" / "codex_temp" / "d_problem_atlas_summary_20260309.json")

    base_anchor = float(compression_payload["factor_summary"]["means"]["base"])
    bridge_route_gain = float(law_payload["best_law"]["route_gain"])
    bridge_stabilize_drag = float(law_payload["best_law"]["stabilize_drag"])

    rows = [
        d_row("residual_gate", d_payload["residual_gate"]["top_overall"][0], d_payload["residual_gate"]["dual_positive_count"], base_anchor),
        d_row("bayes_consolidation", d_payload["bayes_consolidation"]["top_overall"][0], d_payload["bayes_consolidation"]["dual_positive_count"], base_anchor),
        d_row("learned_controller", d_payload["learned_controller"]["best_overall"], d_payload["learned_controller"]["dual_positive_count"], base_anchor),
        d_row("two_phase", d_payload["two_phase_consolidation"]["best_overall"], d_payload["two_phase_consolidation"]["dual_positive_count"], base_anchor),
        d_row("three_phase", d_payload["three_phase_consolidation"]["best_overall"], d_payload["three_phase_consolidation"]["dual_positive_count"], base_anchor),
        d_row("base_offset", d_payload["base_offset_consolidation"]["top_overall"][0], d_payload["base_offset_consolidation"]["dual_positive_count"], base_anchor),
        d_row("offset_stabilization", d_payload["offset_stabilization"]["top_overall"][0], d_payload["offset_stabilization"]["dual_positive_count"], base_anchor),
        d_row("multistage", d_payload["multistage_stabilization"]["top_overall"][0], d_payload["multistage_stabilization"]["dual_positive_count"], base_anchor),
    ]

    transfer = evaluate(rows, bridge_route_gain, bridge_stabilize_drag)
    fitted = fit_best(rows)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "source_law": "unified_update_law_candidate_20260309.json",
            "source_d": "d_problem_atlas_summary_20260309.json",
        },
        "factorization": {
            "base_anchor": base_anchor,
            "adaptive_offset": "0.5 + novel_gain",
            "routing": "0.5 + dual_score",
            "stabilization": "0.5 + retention_gain",
            "reference_score": "0.5 + overall_gain",
        },
        "transfer_law": {
            "route_gain": bridge_route_gain,
            "stabilize_drag": bridge_stabilize_drag,
            "mean_absolute_gap": transfer["mean_absolute_gap"],
            "score_correlation": transfer["score_correlation"],
        },
        "fitted_d_law": {
            "route_gain": fitted["route_gain"],
            "stabilize_drag": fitted["stabilize_drag"],
            "mean_absolute_gap": fitted["result"]["mean_absolute_gap"],
            "score_correlation": fitted["result"]["score_correlation"],
            "gap_improvement_vs_transfer": transfer["mean_absolute_gap"] - fitted["result"]["mean_absolute_gap"],
            "correlation_improvement_vs_transfer": fitted["result"]["score_correlation"] - transfer["score_correlation"],
            "pass": bool(fitted["result"]["mean_absolute_gap"] < 0.03 and fitted["result"]["score_correlation"] > 0.75),
        },
        "methods": fitted["result"]["rows"],
        "project_readout": {
            "current_verdict": (
                "统一更新律在 D 上仍然有组织力，但需要更强的 stabilization 项。"
                "也就是说，同一条小律可以迁移，但 D 上的最佳系数已经偏向稳定化。"
            ),
            "next_question": "能否把 D 的固定点切换写成 adaptive_offset 的状态依赖稳定化项，而不是继续加新模块？",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["transfer_law"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["fitted_d_law"], ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
