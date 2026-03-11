#!/usr/bin/env python
"""
P2: score a multi-timescale stabilization mechanism for the structure-feature
co-generation law under continued data wash.
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
    ap = argparse.ArgumentParser(description="P2 multi-timescale stabilization mechanism")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/p2_multitimescale_stabilization_mechanism_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    p1 = load_json(ROOT / "tests" / "codex_temp" / "p1_structure_feature_cogeneration_law_20260311.json")
    stage7a = load_json(ROOT / "tests" / "codex_temp" / "stage7a_explicit_coding_law_candidate_20260311.json")
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    stage6c = load_json(
        ROOT / "tests" / "codex_temp" / "stage6c_long_horizon_open_environment_closure_20260311.json"
    )
    brain_bridge = load_json(ROOT / "tests" / "codex_temp" / "dnn_brain_puzzle_bridge_20260308.json")

    gpt2_multi = float(brain_bridge["models"]["gpt2"]["components"]["multi_timescale_control"]["score"])
    qwen_multi = float(brain_bridge["models"]["qwen3_4b"]["components"]["multi_timescale_control"]["score"])

    multi_timescale_evidence = {
        "brain_multi_timescale_gpt2": normalize(gpt2_multi, 0.75, 0.90),
        "brain_multi_timescale_qwen": normalize(qwen_multi, 0.75, 0.90),
        "brain_weight_multi_timescale": normalize(
            float(stage7a["candidate_coding_law"]["normalized_brain_weights"]["multi_timescale_control"]),
            0.12,
            0.20,
        ),
        "p1_coupling_support": float(p1["headline_metrics"]["feature_structure_coupling_score"]),
    }
    multi_timescale_evidence_score = mean(multi_timescale_evidence.values())

    fast_slow_coupling = {
        "compressed_core": float(stage6b["headline_metrics"]["compressed_core_score"]),
        "trainable_core": float(stage6b["headline_metrics"]["trainable_core_score"]),
        "online_carryover": float(stage6b["headline_metrics"]["online_carryover_score"]),
        "p1_overall": float(p1["headline_metrics"]["overall_p1_score"]),
    }
    fast_slow_coupling_score = mean(fast_slow_coupling.values())

    long_horizon_stability = {
        "long_horizon_goal": float(stage6c["headline_metrics"]["long_horizon_goal_score"]),
        "subgoal_chain": float(stage6c["headline_metrics"]["subgoal_chain_score"]),
        "variable_planner": float(stage6c["headline_metrics"]["variable_planner_score"]),
        "training_anchor": float(stage6c["headline_metrics"]["training_anchor_score"]),
    }
    long_horizon_stability_score = mean(long_horizon_stability.values())

    recovery_stabilization = {
        "tool_failure_recovery": float(stage6c["headline_metrics"]["tool_failure_recovery_score"]),
        "qwen_recovery_gain": normalize(
            float(stage6c["pillars"]["tool_failure_recovery"]["components"]["qwen_recovery_gain"]),
            0.70,
            0.80,
        ),
        "deepseek_recovery_gain": normalize(
            float(stage6c["pillars"]["tool_failure_recovery"]["components"]["deepseek_recovery_gain"]),
            0.70,
            0.80,
        ),
        "deepseek_recovery_rate": normalize(
            float(stage6c["pillars"]["tool_failure_recovery"]["components"]["deepseek_recovery_rate"]),
            0.10,
            0.25,
        ),
    }
    recovery_stabilization_score = mean(recovery_stabilization.values())

    explicitness = {
        "stage7a_gating_support": float(stage7a["headline_metrics"]["gating_support_score"]),
        "stage7a_equation_support": float(stage7a["headline_metrics"]["equation_support_score"]),
        "p1_explicitness": float(p1["headline_metrics"]["explicitness_score"]),
        "p1_brain_plausibility": float(p1["headline_metrics"]["brain_plausibility_score"]),
    }
    explicitness_score = mean(explicitness.values())

    overall_score = mean(
        [
            multi_timescale_evidence_score,
            fast_slow_coupling_score,
            long_horizon_stability_score,
            recovery_stabilization_score,
            explicitness_score,
        ]
    )

    candidate_mechanism = {
        "equations": {
            "fast_feature_update": "f_{t+1} = f_t + eta_fast * g_t * local_match_t - decay_f * f_t",
            "mid_structure_update": "A_{t+1} = A_t + eta_mid * (1 - g_t) * coactivate(f_{t+1}) - decay_A * A_t",
            "slow_stabilizer": "m_{t+1} = m_t + eta_slow * stabilize_t * (A_{t+1} - m_t)",
            "effective_readout": "y_t = readout(f_t, A_t, m_t)",
        },
        "timescale_order": "eta_fast > eta_mid > eta_slow",
        "verbal_guess": (
            "The current best P2 guess is that continued data wash stays stable only because feature state, effective "
            "structure, and long-term stabilizer memory update at different timescales under the same gated plasticity mechanism."
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "p2_multitimescale_stabilization_mechanism",
        },
        "candidate_mechanism": candidate_mechanism,
        "pillars": {
            "multi_timescale_evidence": {
                "components": multi_timescale_evidence,
                "score": float(multi_timescale_evidence_score),
            },
            "fast_slow_coupling": {
                "components": fast_slow_coupling,
                "score": float(fast_slow_coupling_score),
            },
            "long_horizon_stability": {
                "components": long_horizon_stability,
                "score": float(long_horizon_stability_score),
            },
            "recovery_stabilization": {
                "components": recovery_stabilization,
                "score": float(recovery_stabilization_score),
            },
            "explicitness": {
                "components": explicitness,
                "score": float(explicitness_score),
            },
        },
        "headline_metrics": {
            "multi_timescale_evidence_score": float(multi_timescale_evidence_score),
            "fast_slow_coupling_score": float(fast_slow_coupling_score),
            "long_horizon_stability_score": float(long_horizon_stability_score),
            "recovery_stabilization_score": float(recovery_stabilization_score),
            "explicitness_score": float(explicitness_score),
            "overall_p2_score": float(overall_score),
        },
        "hypotheses": {
            "H1_structure_feature_cogeneration_requires_multiple_timescales": bool(
                multi_timescale_evidence_score >= 0.66
            ),
            "H2_fast_and_slow_updates_can_share_one_core_mechanism": bool(fast_slow_coupling_score >= 0.73),
            "H3_long_horizon_data_wash_does_not_destroy_the_system": bool(long_horizon_stability_score >= 0.74),
            "H4_recovery_is_part_of_stabilization_not_a_separate_patch": bool(
                recovery_stabilization_score >= 0.58
            ),
            "H5_p2_multitimescale_stabilization_is_moderately_supported": bool(overall_score >= 0.70),
        },
        "project_readout": {
            "summary": (
                "P2 is positive only if the structure-feature co-generation law remains stable under continued data "
                "wash because different parts of the update live on different timescales."
            ),
            "next_question": (
                "If P2 holds, the next step is a regional differentiation block: how do different brain areas inherit "
                "different roles from the same underlying plasticity mechanism?"
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
