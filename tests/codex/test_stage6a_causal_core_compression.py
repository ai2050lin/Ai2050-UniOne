#!/usr/bin/env python
"""
Score whether the stage-5 fused law can be compressed into a smaller causal core.
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


def inv_gap_score(value: float, lo: float, hi: float) -> float:
    return float(1.0 - normalize(value, lo, hi))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6A causal core compression scorecard")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage6a_causal_core_compression_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    stage5 = load_json(ROOT / "tests" / "codex_temp" / "stage5_master_closure_20260311.json")
    four_factor = load_json(ROOT / "tests" / "codex_temp" / "unified_structure_four_factor_compression_20260309.json")
    two_param = load_json(ROOT / "tests" / "codex_temp" / "unified_update_law_candidate_20260309.json")
    shell = load_json(ROOT / "tests" / "codex_temp" / "shared_central_loop_shell_localization_20260310.json")
    interface = load_json(ROOT / "tests" / "codex_temp" / "shared_central_loop_minimal_interface_state_20260310.json")
    joint = load_json(ROOT / "tests" / "codex_temp" / "joint_causal_intervention_unified_mechanism_20260311.json")

    stage5_score = float(stage5["stage5_headline_metrics"]["overall_stage5_score"])
    joint_baseline = joint["cases"]["baseline"]["metrics"]
    joint_drop = joint["cases"]["joint_shared_gate_recovery"]["drops"]

    four_factor_score_parts = {
        "compressed_mean_gap": inv_gap_score(float(four_factor["retention"]["mean_absolute_gap"]), 0.08, 0.16),
        "compressed_corr": normalize(float(four_factor["retention"]["score_correlation"]), 0.75, 0.90),
        "two_param_gap_improvement": normalize(float(two_param["best_law"]["gap_improvement"]), 0.04, 0.10),
        "four_factor_pass": 1.0 if bool(four_factor["retention"]["compression_pass"]) else 0.0,
    }
    four_factor_score = mean(four_factor_score_parts.values())

    two_param_score_parts = {
        "best_gap": inv_gap_score(float(two_param["best_law"]["mean_absolute_gap"]), 0.02, 0.10),
        "held_out_gap": inv_gap_score(float(two_param["leave_one_out"]["mean_held_out_gap"]), 0.02, 0.08),
        "score_corr": normalize(float(two_param["best_law"]["score_correlation"]), 0.75, 0.85),
        "best_law_pass": 1.0 if bool(two_param["best_law"]["pass"]) else 0.0,
        "compression_ratio_to_core": float(1.0 - 2.0 / 11.0),
    }
    two_param_score = mean(two_param_score_parts.values())

    shell_score_parts = {
        "winner_is_output_shell": 1.0 if shell["winner"] == "output_shell" else 0.0,
        "output_shell_gap": inv_gap_score(
            float(shell["placements"]["output_shell"]["mean_held_out_gap"]),
            0.006,
            0.025,
        ),
        "output_shell_corr": normalize(
            float(shell["placements"]["output_shell"]["held_out_score_correlation"]),
            0.90,
            0.99,
        ),
        "parameterized_shared_corr": normalize(
            float(shell["baselines"]["parameterized_shared_law"]["held_out_score_correlation"]),
            0.95,
            1.0,
        ),
    }
    shell_score = mean(shell_score_parts.values())

    interface_score_parts = {
        "winner_is_confidence_state": 1.0 if interface["winner"] == "prototype_confidence_state" else 0.0,
        "confidence_gap": inv_gap_score(
            float(interface["minimal_interface_states"]["prototype_confidence_state"]["mean_held_out_gap"]),
            0.005,
            0.015,
        ),
        "confidence_corr": normalize(
            float(interface["minimal_interface_states"]["prototype_confidence_state"]["held_out_score_correlation"]),
            0.93,
            0.98,
        ),
        "activation_gap_advantage": normalize(
            float(
                interface["minimal_interface_states"]["family_activation_state"]["mean_held_out_gap"]
                - interface["minimal_interface_states"]["prototype_confidence_state"]["mean_held_out_gap"]
            ),
            0.001,
            0.003,
        ),
    }
    interface_score = mean(interface_score_parts.values())

    causal_anchor_parts = {
        "stage5_master_score": normalize(stage5_score, 0.60, 0.75),
        "baseline_online_success": normalize(float(joint_baseline["online_success_rate"]), 0.85, 0.95),
        "joint_drop": normalize(float(joint_drop["joint_drop"]), 0.50, 0.75),
        "coupled_drop": normalize(float(joint_drop["coupled_drop"]), 0.55, 0.75),
    }
    causal_anchor_score = mean(causal_anchor_parts.values())

    overall_score = mean(
        [four_factor_score, two_param_score, shell_score, interface_score, causal_anchor_score]
    )

    hypotheses = {
        "H1_four_factor_compression_retains_nontrivial_signal": bool(four_factor_score >= 0.72),
        "H2_two_param_core_candidate_is_nontrivial": bool(two_param_score >= 0.78),
        "H3_shell_localization_supports_outer_shell_core": bool(shell_score >= 0.68),
        "H4_minimal_interface_supports_confidence_state": bool(interface_score >= 0.72),
        "H5_stage6a_causal_core_compression_is_moderately_closed": bool(overall_score >= 0.74),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage6a_causal_core_compression",
        },
        "pillars": {
            "four_factor_compression": {"components": four_factor_score_parts, "score": float(four_factor_score)},
            "two_param_core_candidate": {"components": two_param_score_parts, "score": float(two_param_score)},
            "shell_localization": {"components": shell_score_parts, "score": float(shell_score)},
            "minimal_interface": {"components": interface_score_parts, "score": float(interface_score)},
            "causal_anchor": {"components": causal_anchor_parts, "score": float(causal_anchor_score)},
        },
        "headline_metrics": {
            "four_factor_score": float(four_factor_score),
            "two_param_core_score": float(two_param_score),
            "shell_localization_score": float(shell_score),
            "minimal_interface_score": float(interface_score),
            "causal_anchor_score": float(causal_anchor_score),
            "overall_stage6a_score": float(overall_score),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage 6A is positive only if the stage-5 fused law can be compressed into a smaller causal core "
                "without losing too much held-out fit, shell localization, interface simplicity, or joint causal anchoring."
            ),
            "next_question": (
                "If this stage is positive, the next step is to move from compressed scorecards to a real training loop "
                "that directly optimizes the compressed core."
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
