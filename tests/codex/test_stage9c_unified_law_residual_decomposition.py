#!/usr/bin/env python
"""
Decompose current residual pressure into architecture, data, scale, and
unresolved-core shares.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 9C unified law residual decomposition")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage9c_unified_law_residual_decomposition_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage7d = load_json(ROOT / "tests" / "codex_temp" / "stage7d_coding_law_verdict_master_20260311.json")
    stage8c = load_json(ROOT / "tests" / "codex_temp" / "stage8c_cross_model_task_invariants_20260311.json")
    stage8d = load_json(ROOT / "tests" / "codex_temp" / "stage8d_brain_high_risk_falsification_20260311.json")
    stage9a = load_json(ROOT / "tests" / "codex_temp" / "stage9a_mechanism_adversarial_break_test_20260311.json")

    architecture_raw = mean(
        [
            float(stage9a["headline_metrics"]["architecture_break_pressure_score"]),
            1.0
            - float(stage8c["headline_metrics"]["model_gap_structure_score"]),
        ]
    )
    data_raw = mean(
        [
            float(stage9a["headline_metrics"]["data_break_pressure_score"]),
            abs(
                float(stage8c["model_stats"]["qwen3_4b"]["compatibility_gain_corr"])
                - float(stage8c["model_stats"]["deepseek_7b"]["compatibility_gain_corr"])
            ),
        ]
    )
    scale_raw = mean(
        [
            float(stage9a["headline_metrics"]["scale_break_pressure_score"]),
            1.0 - float(stage9a["pillars"]["kernel_survival"]["components"]["stage8c_invariants"]),
        ]
    )
    unresolved_core_raw = mean(
        [
            1.0 - float(stage7d["verdict"]["verdict_support_score"]),
            1.0 - float(stage8d["pillars"]["brain_specificity"]["score"]),
            1.0 - float(stage8d["pillars"]["directional_falsifier"]["score"]),
        ]
    )

    total = architecture_raw + data_raw + scale_raw + unresolved_core_raw
    shares = {
        "architecture_share": float(architecture_raw / total),
        "data_share": float(data_raw / total),
        "scale_share": float(scale_raw / total),
        "unresolved_core_share": float(unresolved_core_raw / total),
    }

    dominant_source = max(shares, key=shares.get)
    architecture_plus_scale = shares["architecture_share"] + shares["scale_share"]
    kernel_retained_signal = mean(
        [
            float(stage7d["verdict"]["verdict_support_score"]),
            float(stage8c["headline_metrics"]["overall_stage8c_score"]),
            float(stage8d["headline_metrics"]["overall_stage8d_score"]),
        ]
    )

    identifiability = {
        "dominant_minus_second": sorted(shares.values(), reverse=True)[0] - sorted(shares.values(), reverse=True)[1],
        "architecture_plus_scale_share": architecture_plus_scale,
        "kernel_retained_signal": kernel_retained_signal,
    }
    identifiability_score = mean(
        [
            min(1.0, identifiability["dominant_minus_second"] / 0.10),
            min(1.0, architecture_plus_scale / 0.60),
            kernel_retained_signal,
        ]
    )

    overall_score = mean(
        [
            architecture_plus_scale,
            shares["unresolved_core_share"] + kernel_retained_signal * 0.25,
            identifiability_score,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage9c_unified_law_residual_decomposition",
        },
        "residual_shares": shares,
        "dominant_source": dominant_source,
        "kernel_retained_signal": float(kernel_retained_signal),
        "identifiability": identifiability,
        "headline_metrics": {
            "architecture_plus_scale_share": float(architecture_plus_scale),
            "unresolved_core_share": float(shares["unresolved_core_share"]),
            "identifiability_score": float(identifiability_score),
            "kernel_retained_signal": float(kernel_retained_signal),
            "overall_stage9c_score": float(overall_score),
        },
        "hypotheses": {
            "H1_architecture_and_scale_dominate_more_than_data": bool(architecture_plus_scale >= 0.50),
            "H2_unresolved_core_residual_remains_nontrivial": bool(shares["unresolved_core_share"] >= 0.15),
            "H3_residual_sources_are_identifiable_not_fully_mixed": bool(identifiability_score >= 0.70),
            "H4_kernel_signal_remains_nontrivial_after_decomposition": bool(kernel_retained_signal >= 0.76),
            "H5_stage9c_residual_decomposition_is_moderately_supported": bool(overall_score >= 0.63),
        },
        "project_readout": {
            "summary": (
                "Stage 9C is positive only if the remaining residual can be split into meaningful sources rather than "
                "being treated as one undifferentiated mismatch bucket."
            ),
            "next_question": (
                "If this stage holds, the next step is to attack the dominant source directly and see whether the "
                "candidate law strengthens or collapses."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["residual_shares"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
