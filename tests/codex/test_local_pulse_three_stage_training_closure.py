#!/usr/bin/env python
"""
Benchmark a three-stage training closure that jointly scores concept,
comparison, and recovery phases under the same unified local encoding law.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from test_local_pulse_early_core_decoupling_benchmark import (
    EarlyCoreDecoupledReplayNetwork,
    TraceableCoupledReplayNetwork,
    decoupled_params,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_recovery_phase_training_law import RECOVERY_AWARE_LAW
from test_local_pulse_region_heterogeneity_benchmark import (
    calibrate_readout,
    heterogeneous_params,
    homogeneous_params,
    train_network,
)
from test_local_pulse_stage_decomposed_training_law import STAGE_DECOMPOSED_LAW
from test_local_pulse_unified_multiobjective_training_law import (
    UnifiedObjectiveShapedReplayNetwork,
    normalized,
)


def three_stage_objectives(system: Dict[str, object]) -> Dict[str, float]:
    aggregate = float(system["local_integration_score"])
    concept_adv = float(system["concept_phase_upstream_advantage"])
    comparison_adv = float(system["comparison_phase_memory_comparator_advantage"])
    recovery_local_adv = float(
        system["phase_summary"]["recovery_phase"]["memory_comparator_mass"]
        - system["phase_summary"]["recovery_phase"]["sensory_motor_mass"]
    )
    recovery_rate = float(system["lesion_recovery_rate"])
    diversity = float(system["phase_summary"]["distinct_top_region_count"])

    concept_score = normalized(concept_adv, -0.12, 0.18)
    comparison_score = normalized(comparison_adv, -0.12, 0.18)
    recovery_local_score = normalized(recovery_local_adv, -0.16, 0.04)
    recovery_rate_score = normalized(recovery_rate, 0.58, 0.70)
    aggregate_score = normalized(aggregate, 0.60, 0.66)
    diversity_score = normalized(diversity, 1.0, 3.0)
    closure_balance = min(concept_score, comparison_score, recovery_local_score)

    three_stage_score = float(
        0.30 * closure_balance
        + 0.15 * concept_score
        + 0.15 * comparison_score
        + 0.14 * recovery_local_score
        + 0.10 * recovery_rate_score
        + 0.08 * aggregate_score
        + 0.08 * diversity_score
    )
    return {
        "aggregate_objective": aggregate,
        "concept_score": concept_score,
        "comparison_score": comparison_score,
        "recovery_local_score": recovery_local_score,
        "closure_balance_score": closure_balance,
        "three_stage_score": three_stage_score,
    }


def build_systems():
    systems = {
        "shared_local_replay": {
            "network": TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31),
            "epochs": 28,
        },
        "regional_unified_multiobjective": {
            "network": EarlyCoreDecoupledReplayNetwork(decoupled_params(), use_replay=True, seed=23),
            "epochs": 28,
        },
        "regional_stage_decomposed": {
            "network": UnifiedObjectiveShapedReplayNetwork(
                heterogeneous_params(),
                STAGE_DECOMPOSED_LAW,
                seed=59,
            ),
            "epochs": 24,
        },
        "regional_recovery_aware": {
            "network": UnifiedObjectiveShapedReplayNetwork(
                heterogeneous_params(),
                RECOVERY_AWARE_LAW,
                seed=59,
            ),
            "epochs": 24,
        },
    }
    for spec in systems.values():
        train_network(spec["network"], epochs=spec["epochs"], noise=0.03)
        calibrate_readout(spec["network"], noise=0.04)
    return systems


def evaluate_systems() -> Dict[str, Dict[str, object]]:
    rows = {}
    for name, spec in build_systems().items():
        network = spec["network"]
        integration = integration_bundle(network)
        phase = phase_bundle(network)
        rows[name] = {
            **integration,
            **phase,
            **three_stage_objectives({**integration, **phase}),
            "training_epochs": int(spec["epochs"]),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-stage training closure benchmark for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_three_stage_training_closure_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = evaluate_systems()

    aggregate_best = max(systems.items(), key=lambda item: item[1]["aggregate_objective"])[0]
    closure_best = max(systems.items(), key=lambda item: item[1]["three_stage_score"])[0]

    aggregate_row = systems[aggregate_best]
    closure_row = systems[closure_best]
    unified_row = systems["regional_unified_multiobjective"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "three_stage_training_closure_without_global_controller",
        },
        "systems": systems,
        "headline_metrics": {
            "aggregate_best_system": aggregate_best,
            "closure_best_system": closure_best,
            "aggregate_best_score": float(aggregate_row["aggregate_objective"]),
            "closure_best_score": float(closure_row["three_stage_score"]),
            "closure_vs_shared_balance_gain": float(
                closure_row["closure_balance_score"] - systems["shared_local_replay"]["closure_balance_score"]
            ),
            "closure_vs_unified_comparison_gain": float(
                closure_row["comparison_phase_memory_comparator_advantage"]
                - unified_row["comparison_phase_memory_comparator_advantage"]
            ),
            "closure_vs_unified_concept_gain": float(
                closure_row["concept_phase_upstream_advantage"] - unified_row["concept_phase_upstream_advantage"]
            ),
            "closure_vs_shared_score_gap": float(
                closure_row["aggregate_objective"] - systems["shared_local_replay"]["aggregate_objective"]
            ),
        },
        "hypotheses": {
            "H1_aggregate_objective_prefers_shared_law": bool(aggregate_best == "shared_local_replay"),
            "H2_three_stage_closure_prefers_recovery_aware_law": bool(closure_best == "regional_recovery_aware"),
            "H3_closure_law_improves_balance_over_shared": bool(
                closure_row["closure_balance_score"] > systems["shared_local_replay"]["closure_balance_score"] + 0.10
            ),
            "H4_closure_law_improves_comparison_over_unified": bool(
                closure_row["comparison_phase_memory_comparator_advantage"]
                > unified_row["comparison_phase_memory_comparator_advantage"] + 0.18
            ),
            "H5_closure_law_keeps_aggregate_cost_bounded": bool(
                closure_row["aggregate_objective"] >= systems["shared_local_replay"]["aggregate_objective"] - 0.03
            ),
        },
        "project_readout": {
            "summary": "这一步把概念、比较、恢复三段训练口径真正合成一个闭环评分，不再分别看局部胜负。目标是找出当前最接近三段同时成立的统一局部律。",
            "next_question": "如果三阶段闭环已经能稳定选出区域化统一律，下一步就该把这套闭环推进到脑区参数族学习器和真实模型层带。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
