#!/usr/bin/env python
"""
Benchmark whether an explicit recovery-phase training law can improve replay
takeover while preserving concept/comparison-stage local cores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

from test_local_pulse_early_core_decoupling_benchmark import (
    TraceableCoupledReplayNetwork,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    calibrate_readout,
    heterogeneous_params,
    homogeneous_params,
    train_network,
)
from test_local_pulse_stage_decomposed_training_law import STAGE_DECOMPOSED_LAW
from test_local_pulse_unified_multiobjective_training_law import (
    PhaseRoutingLaw,
    UnifiedObjectiveShapedReplayNetwork,
    normalized,
)


RECOVERY_AWARE_LAW = PhaseRoutingLaw(
    sensory_gain_a=1.3258806922278736,
    sensory_gain_b=1.170236868906142,
    early_memory_gain=0.30493739643633216,
    mid_memory_gain=0.14635041907575214,
    early_comparator_forward_scale=0.09876170641754364,
    comparison_memory_scale=0.32460338498747,
    comparison_signal_scale=0.6395057581438616,
    comparator_recurrence_scale=0.09007138287151784,
    comparator_feedback_scale=1.0850395063380773,
    motor_release_step=8,
    prerelease_motor_forward_scale=0.04574162115540195,
    motor_teacher_scale=0.7150054100613455,
    replay_memory_scale=0.3946798215242107,
    replay_comparator_scale=0.3124968618113984,
)


def recovery_objectives(system: Dict[str, object]) -> Dict[str, float]:
    aggregate = float(system["local_integration_score"])
    concept_adv = float(system["concept_phase_upstream_advantage"])
    comparison_adv = float(system["comparison_phase_memory_comparator_advantage"])
    recovery_rate = float(system["lesion_recovery_rate"])
    diversity = float(system["phase_summary"]["distinct_top_region_count"])
    recovery_local_adv = float(
        system["phase_summary"]["recovery_phase"]["memory_comparator_mass"]
        - system["phase_summary"]["recovery_phase"]["sensory_motor_mass"]
    )

    concept_score = normalized(concept_adv, -0.12, 0.18)
    comparison_score = normalized(comparison_adv, -0.12, 0.18)
    recovery_rate_score = normalized(recovery_rate, 0.58, 0.70)
    recovery_local_score = normalized(recovery_local_adv, -0.16, 0.02)
    aggregate_score = normalized(aggregate, 0.60, 0.66)
    diversity_score = normalized(diversity, 1.0, 3.0)

    recovery_phase_score = float(
        0.30 * recovery_local_score
        + 0.22 * recovery_rate_score
        + 0.16 * concept_score
        + 0.16 * comparison_score
        + 0.08 * diversity_score
        + 0.08 * aggregate_score
    )
    return {
        "aggregate_objective": aggregate,
        "recovery_local_advantage": recovery_local_adv,
        "recovery_phase_score": recovery_phase_score,
    }


def build_systems():
    systems = {
        "shared_local_replay": {
            "network": TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31),
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
            **recovery_objectives({**integration, **phase}),
            "training_epochs": int(spec["epochs"]),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Recovery-phase training law benchmark for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_recovery_phase_training_law_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = evaluate_systems()

    aggregate_best = max(systems.items(), key=lambda item: item[1]["aggregate_objective"])[0]
    regional_candidates = {
        name: row for name, row in systems.items() if name in {"regional_stage_decomposed", "regional_recovery_aware"}
    }
    recovery_best = max(regional_candidates.items(), key=lambda item: item[1]["recovery_phase_score"])[0]

    aggregate_row = systems[aggregate_best]
    stage_row = systems["regional_stage_decomposed"]
    recovery_row = systems["regional_recovery_aware"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "recovery_phase_training_law_without_global_controller",
        },
        "systems": systems,
        "headline_metrics": {
            "aggregate_best_system": aggregate_best,
            "regional_recovery_best_system": recovery_best,
            "aggregate_best_score": float(aggregate_row["aggregate_objective"]),
            "recovery_best_score": float(recovery_row["recovery_phase_score"]),
            "recovery_vs_stage_local_adv_gain": float(
                recovery_row["recovery_local_advantage"] - stage_row["recovery_local_advantage"]
            ),
            "recovery_vs_stage_recovery_rate_gain": float(
                recovery_row["lesion_recovery_rate"] - stage_row["lesion_recovery_rate"]
            ),
            "recovery_vs_stage_comparison_delta": float(
                recovery_row["comparison_phase_memory_comparator_advantage"]
                - stage_row["comparison_phase_memory_comparator_advantage"]
            ),
            "recovery_vs_stage_concept_delta": float(
                recovery_row["concept_phase_upstream_advantage"]
                - stage_row["concept_phase_upstream_advantage"]
            ),
            "recovery_vs_aggregate_score_gap": float(
                recovery_row["aggregate_objective"] - aggregate_row["aggregate_objective"]
            ),
            "recovery_vs_shared_recovery_score_gap": float(
                recovery_row["recovery_phase_score"] - systems["shared_local_replay"]["recovery_phase_score"]
            ),
        },
        "hypotheses": {
            "H1_aggregate_objective_prefers_shared_law": bool(aggregate_best == "shared_local_replay"),
            "H2_recovery_objective_prefers_recovery_aware_regional_law": bool(
                recovery_best == "regional_recovery_aware"
            ),
            "H3_recovery_law_improves_recovery_local_adv": bool(
                recovery_row["recovery_local_advantage"] > stage_row["recovery_local_advantage"] + 0.015
            ),
            "H4_recovery_law_improves_recovery_rate": bool(
                recovery_row["lesion_recovery_rate"] >= stage_row["lesion_recovery_rate"] + 0.008
            ),
            "H5_recovery_law_keeps_concept_and_comparison_stable": bool(
                recovery_row["concept_phase_upstream_advantage"] >= stage_row["concept_phase_upstream_advantage"] - 0.03
                and recovery_row["comparison_phase_memory_comparator_advantage"]
                >= stage_row["comparison_phase_memory_comparator_advantage"] - 0.03
            ),
        },
        "project_readout": {
            "summary": "这一步把恢复阶段从附带指标推进成显式训练口径，直接比较阶段分解版和恢复增强版。目标是验证：统一局部编码机制是否能在不打坏概念阶段和比较阶段的前提下，让 replay 在恢复阶段接管得更稳。",
            "next_question": "如果恢复阶段训练律成立，下一步就该把概念、比较、恢复三段训练口径合成一套完整的分阶段训练闭环，再接回真实模型和真实任务。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
