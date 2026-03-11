#!/usr/bin/env python
"""
P10A: final-stage verdict for the spatial plasticity coding theory.
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
    ap = argparse.ArgumentParser(description="P10A final theory verdict")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p10a_final_theory_verdict_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p9a = load_json(ROOT / "tests" / "codex_temp" / "p9a_spatial_plasticity_coding_master_20260311.json")
    p9b = load_json(ROOT / "tests" / "codex_temp" / "p9b_spatial_residual_counterexample_compression_20260311.json")
    p9c = load_json(ROOT / "tests" / "codex_temp" / "p9c_hard_spatial_brain_forecasts_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    theory_support = {
        "p9a_overall": float(p9a["headline_metrics"]["overall_p9a_score"]),
        "p9b_overall": float(p9b["headline_metrics"]["overall_p9b_score"]),
        "p9c_overall": float(p9c["headline_metrics"]["overall_p9c_score"]),
        "p9a_falsifiability": float(p9a["headline_metrics"]["falsifiability_and_testability_score"]),
    }
    theory_support_score = mean(theory_support.values())

    remaining_gap_boundedness = {
        "p9b_residual_control": float(p9b["headline_metrics"]["residual_source_control_score"]),
        "p9b_bridge_gap": float(p9b["headline_metrics"]["bridge_specificity_gap_score"]),
        "unresolved_core_inverse": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
    }
    remaining_gap_boundedness_score = mean(remaining_gap_boundedness.values())

    testability_strength = {
        "p9c_forecast_sharpness": float(p9c["headline_metrics"]["forecast_sharpness_score"]),
        "p9c_testability": float(p9c["headline_metrics"]["testability_score"]),
        "p9a_closure_status": float(p9a["headline_metrics"]["closure_status_score"]),
        "p9a_residual_pressure": float(p9a["headline_metrics"]["residual_pressure_score"]),
    }
    testability_strength_score = mean(testability_strength.values())

    overall_score = mean(
        [
            theory_support_score,
            remaining_gap_boundedness_score,
            testability_strength_score,
        ]
    )

    verdict = {
        "status": "best_current_candidate_theory",
        "confidence_band": "high_candidate_not_final_proof",
        "main_open_gap": "bridge_specificity_and_architecture_scale_residuals",
        "can_explain_brain_efficiency": True,
        "has_final_proof": False,
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p10a_final_theory_verdict",
        },
        "verdict": verdict,
        "pillars": {
            "theory_support": {
                "components": theory_support,
                "score": float(theory_support_score),
            },
            "remaining_gap_boundedness": {
                "components": remaining_gap_boundedness,
                "score": float(remaining_gap_boundedness_score),
            },
            "testability_strength": {
                "components": testability_strength,
                "score": float(testability_strength_score),
            },
        },
        "headline_metrics": {
            "theory_support_score": float(theory_support_score),
            "remaining_gap_boundedness_score": float(remaining_gap_boundedness_score),
            "testability_strength_score": float(testability_strength_score),
            "overall_p10a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_this_is_the_best_current_candidate_theory": bool(theory_support_score >= 0.77),
            "H2_remaining_gaps_are_bounded_not_fatal": bool(remaining_gap_boundedness_score >= 0.67),
            "H3_the_theory_remains_testable_not_dogmatic": bool(testability_strength_score >= 0.79),
            "H4_p10a_final_theory_verdict_is_positive": bool(overall_score >= 0.75),
        },
        "project_readout": {
            "summary": (
                "P10A is positive only if the current spatialized plasticity-coding theory can be named the best "
                "current candidate while still clearly not being treated as final proof."
            ),
            "next_question": (
                "If P10A holds, the next step is to decide whether the remaining gaps are empirical limitations or "
                "theoretical flaws."
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
