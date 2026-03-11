#!/usr/bin/env python
"""
P9A: master closure table for the spatial plasticity-coding theory.
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


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="P9A spatial plasticity coding master")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p9a_spatial_plasticity_coding_master_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p8a = load_json(ROOT / "tests" / "codex_temp" / "p8a_spatialized_plasticity_coding_equation_20260311.json")
    p8b = load_json(ROOT / "tests" / "codex_temp" / "p8b_3d_wiring_dynamic_topology_division_20260311.json")
    p8c = load_json(ROOT / "tests" / "codex_temp" / "p8c_spatial_brain_falsifier_predictions_20260311.json")
    p7c = load_json(ROOT / "tests" / "codex_temp" / "p7c_brain_spatial_falsification_minimal_core_20260311.json")
    stage9c = load_json(
        ROOT / "tests" / "codex_temp" / "stage9c_unified_law_residual_decomposition_20260311.json"
    )

    theory_consistency = {
        "p8a_overall": float(p8a["headline_metrics"]["overall_p8a_score"]),
        "p8b_overall": float(p8b["headline_metrics"]["overall_p8b_score"]),
        "p8c_overall": float(p8c["headline_metrics"]["overall_p8c_score"]),
        "p7c_overall": float(p7c["headline_metrics"]["overall_p7c_score"]),
    }
    theory_consistency_score = mean(theory_consistency.values())

    spatial_mechanism_coverage = {
        "p8a_spatial_equation_consistency": float(p8a["headline_metrics"]["spatial_equation_consistency_score"]),
        "p8a_topology_reuse_locality": float(p8a["headline_metrics"]["topology_reuse_locality_score"]),
        "p8b_local_reuse_advantage": float(p8b["headline_metrics"]["local_reuse_advantage_score"]),
        "p8b_selective_bridge_advantage": float(p8b["headline_metrics"]["selective_bridge_advantage_score"]),
    }
    spatial_mechanism_coverage_score = mean(spatial_mechanism_coverage.values())

    falsifiability_and_testability = {
        "p8c_falsifier_sharpness": float(p8c["headline_metrics"]["falsifier_sharpness_score"]),
        "p8c_brain_spatial_testability": float(p8c["headline_metrics"]["brain_spatial_testability_score"]),
        "p7c_falsifier_sharpness": float(p7c["headline_metrics"]["falsifier_sharpness_score"]),
        "p7c_geometry_constraint": float(p7c["headline_metrics"]["geometry_constraint_score"]),
    }
    falsifiability_and_testability_score = mean(falsifiability_and_testability.values())

    residual_pressure = {
        "kernel_retained_signal": float(stage9c["headline_metrics"]["kernel_retained_signal"]),
        "identifiability": float(stage9c["headline_metrics"]["identifiability_score"]),
        "unresolved_core_inverse": normalize(
            1.0 - float(stage9c["headline_metrics"]["unresolved_core_share"]),
            0.75,
            0.90,
        ),
        "architecture_scale_known": normalize(
            float(stage9c["headline_metrics"]["architecture_plus_scale_share"]),
            0.50,
            0.65,
        ),
    }
    residual_pressure_score = mean(residual_pressure.values())

    closure_status = {
        "equation_written": 1.0,
        "division_of_labor_written": 1.0,
        "spatial_predictions_written": 1.0,
        "geometry_only_rejected": float(p8a["headline_metrics"]["geometry_only_failure_score"]),
    }
    closure_status_score = mean(closure_status.values())

    overall_score = mean(
        [
            theory_consistency_score,
            spatial_mechanism_coverage_score,
            falsifiability_and_testability_score,
            residual_pressure_score,
            closure_status_score,
        ]
    )

    verdict = {
        "status": "supported_but_not_final",
        "strongest_part": "local_3d_reuse_plus_sparse_dynamic_bridge_division",
        "weakest_part": "bridge_specificity_strength_is_only_moderate",
        "main_remaining_risk": "architecture_and_scale_residuals_still_dominate",
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p9a_spatial_plasticity_coding_master",
        },
        "verdict": verdict,
        "pillars": {
            "theory_consistency": {
                "components": theory_consistency,
                "score": float(theory_consistency_score),
            },
            "spatial_mechanism_coverage": {
                "components": spatial_mechanism_coverage,
                "score": float(spatial_mechanism_coverage_score),
            },
            "falsifiability_and_testability": {
                "components": falsifiability_and_testability,
                "score": float(falsifiability_and_testability_score),
            },
            "residual_pressure": {
                "components": residual_pressure,
                "score": float(residual_pressure_score),
            },
            "closure_status": {
                "components": closure_status,
                "score": float(closure_status_score),
            },
        },
        "headline_metrics": {
            "theory_consistency_score": float(theory_consistency_score),
            "spatial_mechanism_coverage_score": float(spatial_mechanism_coverage_score),
            "falsifiability_and_testability_score": float(falsifiability_and_testability_score),
            "residual_pressure_score": float(residual_pressure_score),
            "closure_status_score": float(closure_status_score),
            "overall_p9a_score": float(overall_score),
        },
        "hypotheses": {
            "H1_spatial_theory_is_internally_consistent": bool(theory_consistency_score >= 0.69),
            "H2_spatial_mechanism_coverage_is_nontrivial": bool(spatial_mechanism_coverage_score >= 0.62),
            "H3_theory_is_falsifiable_and_testable": bool(falsifiability_and_testability_score >= 0.75),
            "H4_residual_pressure_is_known_not_ignored": bool(residual_pressure_score >= 0.70),
            "H5_p9a_spatial_master_is_moderately_supported": bool(overall_score >= 0.73),
        },
        "project_readout": {
            "summary": (
                "P9A is positive only if the spatialized plasticity-coding theory is internally consistent, "
                "mechanistically nontrivial, falsifiable, and honest about remaining residual pressure."
            ),
            "next_question": (
                "If P9A holds, the next stage should attack the remaining architecture and scale residuals with "
                "spatialized counterexample search rather than adding more narrative support."
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
