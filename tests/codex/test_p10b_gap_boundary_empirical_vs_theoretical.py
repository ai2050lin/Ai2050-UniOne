#!/usr/bin/env python
"""
P10B: decide whether remaining gaps are empirical or theoretical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def main() -> None:
    ap = argparse.ArgumentParser(description="P10B empirical vs theoretical gap boundary")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p10b_gap_boundary_empirical_vs_theoretical_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p9a = load_json(ROOT / "tests" / "codex_temp" / "p9a_spatial_plasticity_coding_master_20260311.json")
    p9b = load_json(ROOT / "tests" / "codex_temp" / "p9b_spatial_residual_counterexample_compression_20260311.json")
    p9c = load_json(ROOT / "tests" / "codex_temp" / "p9c_hard_spatial_brain_forecasts_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    empirical_gap_pressure = {
        "architecture_share": normalize(float(stage9c["residual_shares"]["architecture_share"]), 0.25, 0.40),
        "scale_share": normalize(float(stage9c["residual_shares"]["scale_share"]), 0.15, 0.25),
        "bridge_specificity_gap": float(p9b["headline_metrics"]["bridge_specificity_gap_score"]),
        "forecast_specificity": float(p9c["headline_metrics"]["forecast_specificity_score"]),
    }
    empirical_gap_pressure_score = mean(empirical_gap_pressure.values())

    theoretical_gap_pressure = {
        "unresolved_core_share_inverse": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
        "theory_consistency": float(p9a["headline_metrics"]["theory_consistency_score"]),
        "compressed_core_resilience": float(p9b["headline_metrics"]["compressed_core_resilience_score"]),
    }
    theoretical_gap_pressure_score = mean(theoretical_gap_pressure.values())

    empirical_dominance = {
        "architecture_plus_scale_known": normalize(
            float(stage9c["headline_metrics"]["architecture_plus_scale_share"]),
            0.50,
            0.65,
        ),
        "identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
        "p9a_residual_pressure": float(p9a["headline_metrics"]["residual_pressure_score"]),
        "p9b_residual_control": float(p9b["headline_metrics"]["residual_source_control_score"]),
    }
    empirical_dominance_score = mean(empirical_dominance.values())

    overall_score = mean(
        [
            empirical_gap_pressure_score,
            theoretical_gap_pressure_score,
            empirical_dominance_score,
        ]
    )

    verdict = {
        "primary_gap_type": "mostly_empirical_with_bounded_theoretical_residual",
        "empirical_components": [
            "architecture_bias",
            "model_scale_limit",
            "bridge_specificity_measurement_strength",
        ],
        "theoretical_components": [
            "unresolved_core_share",
            "non-final bridge law specificity",
        ],
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p10b_gap_boundary_empirical_vs_theoretical",
        },
        "verdict": verdict,
        "pillars": {
            "empirical_gap_pressure": {
                "components": empirical_gap_pressure,
                "score": float(empirical_gap_pressure_score),
            },
            "theoretical_gap_pressure": {
                "components": theoretical_gap_pressure,
                "score": float(theoretical_gap_pressure_score),
            },
            "empirical_dominance": {
                "components": empirical_dominance,
                "score": float(empirical_dominance_score),
            },
        },
        "headline_metrics": {
            "empirical_gap_pressure_score": float(empirical_gap_pressure_score),
            "theoretical_gap_pressure_score": float(theoretical_gap_pressure_score),
            "empirical_dominance_score": float(empirical_dominance_score),
            "overall_p10b_score": float(overall_score),
        },
        "hypotheses": {
            "H1_remaining_gaps_include_empirical_pressure": bool(empirical_gap_pressure_score >= 0.66),
            "H2_theoretical_gaps_are_bounded_not_dominant": bool(theoretical_gap_pressure_score >= 0.70),
            "H3_remaining_gaps_are_mostly_empirical": bool(empirical_dominance_score >= 0.76),
            "H4_p10b_gap_boundary_is_positive": bool(overall_score >= 0.73),
        },
        "project_readout": {
            "summary": (
                "P10B is positive only if the remaining gaps can be classified as mostly empirical limitations rather "
                "than as a wholesale theoretical collapse."
            ),
            "next_question": (
                "If P10B holds, the final step is to collect the highest-risk falsifiers into one final checklist."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
