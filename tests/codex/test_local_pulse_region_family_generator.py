#!/usr/bin/env python
"""
Compress the learned region family into a low-dimensional generator and test
whether a small latent code can recover or surpass the current learned family.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from test_local_pulse_early_core_decoupling_benchmark import (
    TraceableCoupledReplayNetwork,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    RegionParams,
    calibrate_readout,
    homogeneous_params,
    train_network,
)
from test_local_pulse_region_parameter_family_learner import family_spread
from test_local_pulse_three_stage_training_closure import three_stage_objectives
from test_local_pulse_unified_multiobjective_training_law import (
    PhaseRoutingLaw,
    UnifiedObjectiveShapedReplayNetwork,
)


LEARNED_LAW = PhaseRoutingLaw(
    sensory_gain_a=1.3258806922278736,
    sensory_gain_b=1.170236868906142,
    early_memory_gain=0.30493739643633216,
    mid_memory_gain=0.14635041907575214,
    early_comparator_forward_scale=0.09876170641754364,
    comparison_memory_scale=0.3059723839660702,
    comparison_signal_scale=0.6631401459532512,
    comparator_recurrence_scale=0.09007138287151784,
    comparator_feedback_scale=1.1039797857234084,
    motor_release_step=8,
    prerelease_motor_forward_scale=0.04574162115540195,
    motor_teacher_scale=0.7150054100613455,
    replay_memory_scale=0.46997320655235925,
    replay_comparator_scale=0.37929655367697895,
)

LEARNED_PARAMS = [
    RegionParams(
        leak=0.18,
        threshold=0.5,
        refractory=1,
        inhibition=0.18299952725368218,
        plasticity=0.02,
        feedback_gain=0.03,
        tonic_bias=0.08,
    ),
    RegionParams(
        leak=0.9718952311742866,
        threshold=0.34,
        refractory=2,
        inhibition=0.04,
        plasticity=0.04,
        feedback_gain=0.21644048477714273,
        tonic_bias=0.1,
    ),
    RegionParams(
        leak=0.58,
        threshold=0.4067637158020569,
        refractory=1,
        inhibition=0.16,
        plasticity=0.034,
        feedback_gain=0.18641040795328984,
        tonic_bias=0.07,
    ),
    RegionParams(
        leak=0.46,
        threshold=0.4414299317010634,
        refractory=1,
        inhibition=0.08,
        plasticity=0.028,
        feedback_gain=0.09,
        tonic_bias=0.05,
    ),
]


def clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def evaluate_network(network) -> Dict[str, object]:
    integration = integration_bundle(network)
    phase = phase_bundle(network)
    return {
        **integration,
        **phase,
        **three_stage_objectives({**integration, **phase}),
        "family_spread": family_spread(getattr(network, "region_params", [])),
    }


def generated_family_from_latent(code: List[float]) -> Dict[str, object]:
    z0, z1, z2, z3, z4 = code
    params = copy.deepcopy(LEARNED_PARAMS)
    params[0].inhibition = clip(params[0].inhibition * (1.0 + 0.08 * z0), 0.14, 0.22)
    params[1].leak = clip(params[1].leak * (1.0 + 0.10 * z0), 0.88, 1.05)
    params[1].feedback_gain = clip(params[1].feedback_gain * (1.0 + 0.10 * z1), 0.18, 0.26)
    params[2].threshold = clip(params[2].threshold * (1.0 + 0.06 * z2), 0.37, 0.43)
    params[2].feedback_gain = clip(params[2].feedback_gain * (1.0 + 0.12 * z1), 0.16, 0.22)
    params[3].threshold = clip(params[3].threshold * (1.0 + 0.06 * z3), 0.42, 0.47)

    law_dict = LEARNED_LAW.__dict__.copy()
    law_dict["comparison_memory_scale"] = clip(law_dict["comparison_memory_scale"] + 0.04 * z1, 0.26, 0.38)
    law_dict["comparison_signal_scale"] = clip(law_dict["comparison_signal_scale"] + 0.05 * z2, 0.60, 0.72)
    law_dict["comparator_feedback_scale"] = clip(law_dict["comparator_feedback_scale"] + 0.05 * z1, 1.00, 1.18)
    law_dict["replay_memory_scale"] = clip(law_dict["replay_memory_scale"] + 0.06 * z4, 0.40, 0.56)
    law_dict["replay_comparator_scale"] = clip(law_dict["replay_comparator_scale"] + 0.06 * z4, 0.30, 0.46)
    law_dict["motor_release_step"] = 8
    return {
        "params": params,
        "law": PhaseRoutingLaw(**law_dict),
    }


def search_generator(num_candidates: int, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    best = None
    trials = []
    for trial_id in range(num_candidates):
        code = [rng.uniform(-1.0, 1.0) for _ in range(5)]
        generated = generated_family_from_latent(code)
        network = UnifiedObjectiveShapedReplayNetwork(generated["params"], generated["law"], seed=59)
        train_network(network, epochs=24, noise=0.03)
        calibrate_readout(network, noise=0.04)
        metrics = evaluate_network(network)
        trial = {
            "trial_id": int(trial_id),
            "latent_code": code,
            "three_stage_score": float(metrics["three_stage_score"]),
            "closure_balance_score": float(metrics["closure_balance_score"]),
            "aggregate_objective": float(metrics["aggregate_objective"]),
            "family_spread": float(metrics["family_spread"]),
        }
        trials.append(trial)
        key = (
            trial["three_stage_score"],
            trial["closure_balance_score"],
            trial["aggregate_objective"],
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "trial_id": int(trial_id),
                "latent_code": code,
                "metrics": metrics,
                "law": generated["law"].__dict__,
                "region_params": [row.__dict__ for row in generated["params"]],
            }
    return {
        "best": best,
        "trials": sorted(trials, key=lambda row: (row["three_stage_score"], row["closure_balance_score"], row["aggregate_objective"]), reverse=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-dimensional region family generator for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_region_family_generator_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    shared = TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31)
    train_network(shared, epochs=28, noise=0.03)
    calibrate_readout(shared, noise=0.04)
    shared_metrics = evaluate_network(shared)

    learned = UnifiedObjectiveShapedReplayNetwork(copy.deepcopy(LEARNED_PARAMS), LEARNED_LAW, seed=59)
    train_network(learned, epochs=24, noise=0.03)
    calibrate_readout(learned, noise=0.04)
    learned_metrics = evaluate_network(learned)

    search = search_generator(num_candidates=24, seed=11)
    generated_metrics = search["best"]["metrics"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "low_dim_region_family_generator_without_global_controller",
            "generator_dim": 5,
            "search_candidates": 24,
        },
        "systems": {
            "shared_local_replay": {
                **shared_metrics,
                "training_epochs": 28,
            },
            "learned_region_family": {
                **learned_metrics,
                "training_epochs": 24,
                "law": LEARNED_LAW.__dict__,
                "region_params": [row.__dict__ for row in LEARNED_PARAMS],
            },
            "generated_region_family": {
                **generated_metrics,
                "training_epochs": 24,
                "law": search["best"]["law"],
                "region_params": search["best"]["region_params"],
                "latent_code": search["best"]["latent_code"],
                "trial_id": int(search["best"]["trial_id"]),
            },
        },
        "search_summary": {
            "top_trials": search["trials"][:6],
        },
        "headline_metrics": {
            "generator_best_system": "generated_region_family",
            "generated_three_stage_gain": float(
                generated_metrics["three_stage_score"] - learned_metrics["three_stage_score"]
            ),
            "generated_balance_gain": float(
                generated_metrics["closure_balance_score"] - learned_metrics["closure_balance_score"]
            ),
            "generated_aggregate_gap": float(
                generated_metrics["aggregate_objective"] - learned_metrics["aggregate_objective"]
            ),
            "generated_vs_shared_score_gap": float(
                generated_metrics["aggregate_objective"] - shared_metrics["aggregate_objective"]
            ),
            "generator_dim": 5,
            "manual_control_dim_estimate": 12,
            "best_trial_id": int(search["best"]["trial_id"]),
        },
        "hypotheses": {
            "H1_generator_beats_learned_family_three_stage_score": bool(
                generated_metrics["three_stage_score"] > learned_metrics["three_stage_score"] + 0.001
            ),
            "H2_generator_improves_balance": bool(
                generated_metrics["closure_balance_score"] > learned_metrics["closure_balance_score"] + 0.02
            ),
            "H3_generator_keeps_aggregate_cost_bounded": bool(
                generated_metrics["aggregate_objective"] >= learned_metrics["aggregate_objective"] - 0.01
            ),
            "H4_generator_uses_lower_control_dim": bool(5 < 12),
        },
        "project_readout": {
            "summary": "这一步把学习到的脑区参数族继续压成低维生成器。目标不是继续暴力搜索整套参数，而是验证 5 维左右的生成坐标能否复现甚至超过当前学习族。",
            "next_question": "如果低维生成器已经能逼近或超过学习族，下一步就该把它推进成真正可训练的参数族生成网络，再接回真实模型层带。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
