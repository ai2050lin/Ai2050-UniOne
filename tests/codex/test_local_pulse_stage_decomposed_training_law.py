#!/usr/bin/env python
"""
Benchmark whether a stage-decomposed local training law can improve both
concept-phase upstream routing and comparison-phase local routing under a
single unified local encoding mechanism.
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
from test_local_pulse_region_heterogeneity_benchmark import (
    calibrate_readout,
    heterogeneous_params,
    homogeneous_params,
    train_network,
)
from test_local_pulse_unified_multiobjective_training_law import (
    PhaseRoutingLaw,
    UnifiedObjectiveShapedReplayNetwork,
    normalized,
)


STAGE_DECOMPOSED_LAW = PhaseRoutingLaw(
    sensory_gain_a=1.3258806922278736,
    sensory_gain_b=1.170236868906142,
    early_memory_gain=0.30493739643633216,
    mid_memory_gain=0.14635041907575214,
    early_comparator_forward_scale=0.09876170641754364,
    comparison_memory_scale=0.23119824420408258,
    comparison_signal_scale=0.5473316350849592,
    comparator_recurrence_scale=0.12158418335562568,
    comparator_feedback_scale=1.12542577382452,
    motor_release_step=7,
    prerelease_motor_forward_scale=0.04574162115540195,
    motor_teacher_scale=0.7150054100613455,
    replay_memory_scale=0.2859350346796153,
    replay_comparator_scale=0.23738196366974945,
)


def stage_objectives(system: Dict[str, object]) -> Dict[str, float]:
    aggregate = float(system["local_integration_score"])
    concept_adv = float(system["concept_phase_upstream_advantage"])
    comparison_adv = float(system["comparison_phase_memory_comparator_advantage"])
    recovery = float(system["lesion_recovery_rate"])
    diversity = float(system["phase_summary"]["distinct_top_region_count"])

    concept_score = normalized(concept_adv, -0.12, 0.18)
    comparison_score = normalized(comparison_adv, -0.12, 0.18)
    recovery_score = normalized(recovery, 0.55, 0.70)
    diversity_score = normalized(diversity, 1.0, 3.0)
    aggregate_score = normalized(aggregate, 0.58, 0.66)
    stage_balance = min(concept_score, comparison_score)

    stage_decomposed_score = float(
        0.34 * stage_balance
        + 0.20 * concept_score
        + 0.20 * comparison_score
        + 0.12 * recovery_score
        + 0.08 * diversity_score
        + 0.06 * aggregate_score
    )
    return {
        "aggregate_objective": aggregate,
        "concept_objective": float(concept_score),
        "comparison_objective": float(comparison_score),
        "stage_balance_score": float(stage_balance),
        "stage_decomposed_score": float(stage_decomposed_score),
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
    }
    for spec in systems.values():
        network = spec["network"]
        epochs = spec["epochs"]
        train_network(network, epochs=epochs, noise=0.03)
        calibrate_readout(network, noise=0.04)
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
            **stage_objectives({**integration, **phase}),
            "training_epochs": int(spec["epochs"]),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-decomposed training law benchmark for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_stage_decomposed_training_law_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = evaluate_systems()

    aggregate_best = max(systems.items(), key=lambda item: item[1]["aggregate_objective"])[0]
    stage_best = max(systems.items(), key=lambda item: item[1]["stage_decomposed_score"])[0]

    aggregate_row = systems[aggregate_best]
    stage_row = systems[stage_best]
    unified_row = systems["regional_unified_multiobjective"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage_decomposed_training_law_without_global_controller",
        },
        "systems": systems,
        "headline_metrics": {
            "aggregate_best_system": aggregate_best,
            "stage_best_system": stage_best,
            "aggregate_best_score": float(aggregate_row["aggregate_objective"]),
            "stage_best_score": float(stage_row["stage_decomposed_score"]),
            "stage_vs_unified_concept_gain": float(
                stage_row["concept_phase_upstream_advantage"] - unified_row["concept_phase_upstream_advantage"]
            ),
            "stage_vs_unified_comparison_gain": float(
                stage_row["comparison_phase_memory_comparator_advantage"]
                - unified_row["comparison_phase_memory_comparator_advantage"]
            ),
            "stage_vs_aggregate_structure_balance_gain": float(
                stage_row["stage_balance_score"] - aggregate_row["stage_balance_score"]
            ),
            "stage_vs_aggregate_score_gap": float(
                stage_row["aggregate_objective"] - aggregate_row["aggregate_objective"]
            ),
        },
        "hypotheses": {
            "H1_aggregate_objective_prefers_shared_law": bool(aggregate_best == "shared_local_replay"),
            "H2_stage_objective_prefers_stage_decomposed_law": bool(stage_best == "regional_stage_decomposed"),
            "H3_stage_law_improves_comparison_over_unified": bool(
                stage_row["comparison_phase_memory_comparator_advantage"]
                > unified_row["comparison_phase_memory_comparator_advantage"] + 0.12
            ),
            "H4_stage_law_keeps_concept_advantage_close_to_unified": bool(
                stage_row["concept_phase_upstream_advantage"]
                >= unified_row["concept_phase_upstream_advantage"] - 0.05
            ),
            "H5_stage_law_improves_balance_over_aggregate_choice": bool(
                stage_row["stage_balance_score"] > aggregate_row["stage_balance_score"] + 0.25
            ),
        },
        "project_readout": {
            "summary": "这一步把统一多目标继续拆成阶段分解训练律，单独约束概念阶段和比较阶段。目标不是只让结构分变高，而是让同一套局部编码机制在两个关键阶段都形成正确的局部核心。",
            "next_question": "如果阶段分解训练律能同时拉正概念阶段和比较阶段，下一步就该把这套训练口径搬到真实模型层带和在线任务闭环里。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
