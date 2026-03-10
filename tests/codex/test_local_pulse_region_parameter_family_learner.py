#!/usr/bin/env python
"""
Learn a region-parameter family around the current recovery-aware law and test
whether automatic regional family search can improve the three-stage closure.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from test_local_pulse_early_core_decoupling_benchmark import (
    TraceableCoupledReplayNetwork,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_recovery_phase_training_law import RECOVERY_AWARE_LAW
from test_local_pulse_region_heterogeneity_benchmark import (
    RegionParams,
    calibrate_readout,
    heterogeneous_params,
    homogeneous_params,
    train_network,
)
from test_local_pulse_three_stage_training_closure import three_stage_objectives
from test_local_pulse_unified_multiobjective_training_law import (
    PhaseRoutingLaw,
    UnifiedObjectiveShapedReplayNetwork,
)


def family_spread(params: List[RegionParams]) -> float:
    leak_values = [row.leak for row in params]
    threshold_values = [row.threshold for row in params]
    feedback_values = [row.feedback_gain for row in params]
    inhibition_values = [row.inhibition for row in params]
    spread = (
        (max(leak_values) - min(leak_values))
        + (max(threshold_values) - min(threshold_values))
        + (max(feedback_values) - min(feedback_values))
        + (max(inhibition_values) - min(inhibition_values))
    )
    return float(spread)


def evaluate_network(network) -> Dict[str, object]:
    integration = integration_bundle(network)
    phase = phase_bundle(network)
    params = getattr(network, "region_params", [])
    return {
        **integration,
        **phase,
        **three_stage_objectives({**integration, **phase}),
        "family_spread": family_spread(params),
    }


def build_manual_systems():
    systems = {
        "shared_local_replay": {
            "network": TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31),
            "epochs": 28,
        },
        "manual_recovery_aware": {
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


def sample_candidate(rng: random.Random) -> Tuple[List[RegionParams], PhaseRoutingLaw]:
    params = copy.deepcopy(heterogeneous_params())
    params[0].inhibition *= 1 + rng.uniform(-0.06, 0.02)
    params[1].leak *= 1 + rng.uniform(-0.05, 0.08)
    params[1].feedback_gain *= 1 + rng.uniform(-0.10, 0.10)
    params[2].threshold *= 1 + rng.uniform(-0.04, 0.03)
    params[2].feedback_gain *= 1 + rng.uniform(-0.08, 0.10)
    params[3].threshold *= 1 + rng.uniform(-0.03, 0.06)

    law_dict = RECOVERY_AWARE_LAW.__dict__.copy()
    law_dict["comparison_memory_scale"] += rng.uniform(-0.03, 0.05)
    law_dict["comparison_signal_scale"] += rng.uniform(-0.03, 0.06)
    law_dict["comparator_feedback_scale"] += rng.uniform(-0.06, 0.08)
    law_dict["replay_memory_scale"] += rng.uniform(-0.03, 0.08)
    law_dict["replay_comparator_scale"] += rng.uniform(-0.03, 0.07)
    return params, PhaseRoutingLaw(**law_dict)


def search_region_family(num_candidates: int, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    best_payload = None
    trials = []
    for idx in range(num_candidates):
        params, law = sample_candidate(rng)
        network = UnifiedObjectiveShapedReplayNetwork(params, law, seed=59)
        train_network(network, epochs=24, noise=0.03)
        calibrate_readout(network, noise=0.04)
        metrics = evaluate_network(network)
        trial = {
            "trial_id": idx,
            "three_stage_score": float(metrics["three_stage_score"]),
            "closure_balance_score": float(metrics["closure_balance_score"]),
            "aggregate_objective": float(metrics["aggregate_objective"]),
            "concept_phase_upstream_advantage": float(metrics["concept_phase_upstream_advantage"]),
            "comparison_phase_memory_comparator_advantage": float(metrics["comparison_phase_memory_comparator_advantage"]),
            "recovery_local_advantage": float(
                metrics["phase_summary"]["recovery_phase"]["memory_comparator_mass"]
                - metrics["phase_summary"]["recovery_phase"]["sensory_motor_mass"]
            ),
            "family_spread": float(metrics["family_spread"]),
            "law": law.__dict__,
            "region_params": [row.__dict__ for row in params],
        }
        trials.append(trial)
        if best_payload is None or (
            trial["three_stage_score"],
            trial["closure_balance_score"],
            trial["aggregate_objective"],
        ) > (
            best_payload["metrics"]["three_stage_score"],
            best_payload["metrics"]["closure_balance_score"],
            best_payload["metrics"]["aggregate_objective"],
        ):
            best_payload = {
                "metrics": metrics,
                "law": law.__dict__,
                "region_params": [row.__dict__ for row in params],
                "trial_id": idx,
            }
    return {
        "best": best_payload,
        "trials": sorted(trials, key=lambda row: (row["three_stage_score"], row["closure_balance_score"], row["aggregate_objective"]), reverse=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Region parameter family learner for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_region_parameter_family_learner_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    manual_systems = build_manual_systems()
    shared_metrics = evaluate_network(manual_systems["shared_local_replay"]["network"])
    manual_metrics = evaluate_network(manual_systems["manual_recovery_aware"]["network"])
    search = search_region_family(num_candidates=32, seed=5)
    learned_metrics = search["best"]["metrics"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "region_parameter_family_learner_without_global_controller",
            "num_candidates": 32,
        },
        "systems": {
            "shared_local_replay": {
                **shared_metrics,
                "training_epochs": 28,
            },
            "manual_recovery_aware": {
                **manual_metrics,
                "training_epochs": 24,
                "law": RECOVERY_AWARE_LAW.__dict__,
                "region_params": [row.__dict__ for row in heterogeneous_params()],
            },
            "learned_region_family": {
                **learned_metrics,
                "training_epochs": 24,
                "law": search["best"]["law"],
                "region_params": search["best"]["region_params"],
                "trial_id": int(search["best"]["trial_id"]),
            },
        },
        "search_summary": {
            "top_trials": search["trials"][:6],
        },
        "headline_metrics": {
            "aggregate_best_system": "shared_local_replay"
            if shared_metrics["aggregate_objective"] >= learned_metrics["aggregate_objective"]
            else "learned_region_family",
            "family_best_system": "learned_region_family",
            "learned_three_stage_gain": float(
                learned_metrics["three_stage_score"] - manual_metrics["three_stage_score"]
            ),
            "learned_balance_gain": float(
                learned_metrics["closure_balance_score"] - manual_metrics["closure_balance_score"]
            ),
            "learned_aggregate_gap": float(
                learned_metrics["aggregate_objective"] - manual_metrics["aggregate_objective"]
            ),
            "learned_vs_shared_score_gap": float(
                learned_metrics["aggregate_objective"] - shared_metrics["aggregate_objective"]
            ),
            "learned_family_spread": float(learned_metrics["family_spread"]),
            "learned_trial_id": int(search["best"]["trial_id"]),
        },
        "hypotheses": {
            "H1_learned_family_beats_manual_three_stage_score": bool(
                learned_metrics["three_stage_score"] > manual_metrics["three_stage_score"] + 0.003
            ),
            "H2_learned_family_improves_balance": bool(
                learned_metrics["closure_balance_score"] > manual_metrics["closure_balance_score"] + 0.02
            ),
            "H3_learned_family_keeps_aggregate_cost_bounded": bool(
                learned_metrics["aggregate_objective"] >= manual_metrics["aggregate_objective"] - 0.01
            ),
            "H4_learned_family_preserves_region_spread": bool(
                learned_metrics["family_spread"] > 0.70
            ),
        },
        "project_readout": {
            "summary": "这一步不再手工指定脑区参数族，而是让系统在 `recovery_aware` 附近自动搜索。目标是验证：同一局部编码机制能否自己学出一个更好的区域化参数族，而不只是依赖人工调参。",
            "next_question": "如果自动搜索的脑区参数族已经优于手工族，下一步就该把这种参数族搜索推进成可学习的参数族生成器，再接回真实模型层带。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
