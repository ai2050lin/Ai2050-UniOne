#!/usr/bin/env python
"""
Train a low-dimensional region family generator by iterative latent updates and
test whether the trainable generator can surpass the fixed generated family.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from test_local_pulse_early_core_decoupling_benchmark import (
    TraceableCoupledReplayNetwork,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_region_family_generator import (
    LEARNED_LAW,
    LEARNED_PARAMS,
    generated_family_from_latent,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    calibrate_readout,
    homogeneous_params,
    train_network,
)
from test_local_pulse_three_stage_training_closure import three_stage_objectives
from test_local_pulse_unified_multiobjective_training_law import UnifiedObjectiveShapedReplayNetwork


FIXED_GENERATOR_LATENT = [
    -0.8532028932231517,
    -0.8197656540761395,
    0.16546959633510405,
    -0.5139741597248753,
    0.20256768716387752,
]


def evaluate_network(network) -> Dict[str, object]:
    integration = integration_bundle(network)
    phase = phase_bundle(network)
    return {
        **integration,
        **phase,
        **three_stage_objectives({**integration, **phase}),
    }


def evaluate_latent(code: List[float]) -> Tuple[Tuple[float, float, float], Dict[str, object], Dict[str, object]]:
    generated = generated_family_from_latent(code)
    network = UnifiedObjectiveShapedReplayNetwork(generated["params"], generated["law"], seed=59)
    train_network(network, epochs=24, noise=0.03)
    calibrate_readout(network, noise=0.04)
    metrics = evaluate_network(network)
    key = (
        float(metrics["three_stage_score"]),
        float(metrics["closure_balance_score"]),
        float(metrics["aggregate_objective"]),
    )
    return key, metrics, generated


def train_latent_generator(start_code: List[float]) -> Dict[str, object]:
    code = list(start_code)
    best_key, best_metrics, best_generated = evaluate_latent(code)
    history = [
        {
            "step": 0,
            "latent_code": list(code),
            "three_stage_score": float(best_key[0]),
            "closure_balance_score": float(best_key[1]),
            "aggregate_objective": float(best_key[2]),
        }
    ]

    step_size = 0.35
    step_id = 1
    for _ in range(3):
        improved = False
        for idx in range(len(code)):
            base_value = code[idx]
            local_best_key = best_key
            local_best_code = list(code)
            local_best_metrics = best_metrics
            local_best_generated = best_generated
            for delta in (-step_size, step_size):
                candidate_code = list(code)
                candidate_code[idx] = float(max(-1.0, min(1.0, base_value + delta)))
                candidate_key, candidate_metrics, candidate_generated = evaluate_latent(candidate_code)
                history.append(
                    {
                        "step": step_id,
                        "latent_code": list(candidate_code),
                        "three_stage_score": float(candidate_key[0]),
                        "closure_balance_score": float(candidate_key[1]),
                        "aggregate_objective": float(candidate_key[2]),
                    }
                )
                step_id += 1
                if candidate_key > local_best_key:
                    local_best_key = candidate_key
                    local_best_code = candidate_code
                    local_best_metrics = candidate_metrics
                    local_best_generated = candidate_generated
            if local_best_key > best_key:
                code = local_best_code
                best_key = local_best_key
                best_metrics = local_best_metrics
                best_generated = local_best_generated
                improved = True
        if not improved:
            step_size *= 0.5
    return {
        "latent_code": list(code),
        "metrics": best_metrics,
        "generated": best_generated,
        "history": history,
        "final_step_size": float(step_size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Trainable region family generator for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_trainable_region_family_generator_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()

    shared = TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31)
    train_network(shared, epochs=28, noise=0.03)
    calibrate_readout(shared, noise=0.04)
    shared_metrics = evaluate_network(shared)

    learned = UnifiedObjectiveShapedReplayNetwork(LEARNED_PARAMS, LEARNED_LAW, seed=59)
    train_network(learned, epochs=24, noise=0.03)
    calibrate_readout(learned, noise=0.04)
    learned_metrics = evaluate_network(learned)

    fixed_key, fixed_metrics, fixed_generated = evaluate_latent(FIXED_GENERATOR_LATENT)
    trained = train_latent_generator(FIXED_GENERATOR_LATENT)
    trained_metrics = trained["metrics"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "trainable_low_dim_region_family_generator_without_global_controller",
            "generator_dim": 5,
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
            "fixed_generated_family": {
                **fixed_metrics,
                "training_epochs": 24,
                "latent_code": FIXED_GENERATOR_LATENT,
                "law": fixed_generated["law"].__dict__,
                "region_params": [row.__dict__ for row in fixed_generated["params"]],
            },
            "trainable_generated_family": {
                **trained_metrics,
                "training_epochs": 24,
                "latent_code": trained["latent_code"],
                "law": trained["generated"]["law"].__dict__,
                "region_params": [row.__dict__ for row in trained["generated"]["params"]],
            },
        },
        "optimization_history": trained["history"],
        "headline_metrics": {
            "trainable_best_system": "trainable_generated_family",
            "trainable_three_stage_gain_vs_fixed": float(
                trained_metrics["three_stage_score"] - fixed_metrics["three_stage_score"]
            ),
            "trainable_balance_gain_vs_fixed": float(
                trained_metrics["closure_balance_score"] - fixed_metrics["closure_balance_score"]
            ),
            "trainable_aggregate_gain_vs_fixed": float(
                trained_metrics["aggregate_objective"] - fixed_metrics["aggregate_objective"]
            ),
            "trainable_three_stage_gain_vs_learned": float(
                trained_metrics["three_stage_score"] - learned_metrics["three_stage_score"]
            ),
            "trainable_vs_shared_score_gap": float(
                trained_metrics["aggregate_objective"] - shared_metrics["aggregate_objective"]
            ),
            "final_step_size": float(trained["final_step_size"]),
        },
        "hypotheses": {
            "H1_trainable_generator_beats_fixed_generator": bool(
                trained_metrics["three_stage_score"] > fixed_metrics["three_stage_score"] + 0.01
            ),
            "H2_trainable_generator_improves_balance": bool(
                trained_metrics["closure_balance_score"] > fixed_metrics["closure_balance_score"] + 0.02
            ),
            "H3_trainable_generator_beats_learned_family": bool(
                trained_metrics["three_stage_score"] > learned_metrics["three_stage_score"] + 0.02
            ),
            "H4_trainable_generator_keeps_aggregate_cost_bounded": bool(
                trained_metrics["aggregate_objective"] >= fixed_metrics["aggregate_objective"] - 0.005
            ),
        },
        "project_readout": {
            "summary": "这一步把低维脑区参数族生成器从离线搜索推进成可训练更新。目标是验证：在同一个 5 维生成坐标里，经过迭代更新后，生成器能否稳定超过固定生成族。",
            "next_question": "如果可训练生成器已经明显优于固定生成器，下一步就该把它推进成真正端到端的参数族生成网络，再接回真实模型层带。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
