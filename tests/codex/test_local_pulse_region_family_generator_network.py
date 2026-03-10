#!/usr/bin/env python
"""
Train a small low-dimensional generator network that maps latent coordinates to
region parameter families, then test whether it beats the fixed latent mapping
on held-out latent codes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from test_local_pulse_early_core_decoupling_benchmark import (
    TraceableCoupledReplayNetwork,
    integration_bundle,
    phase_bundle,
)
from test_local_pulse_region_family_generator import (
    LEARNED_LAW,
    LEARNED_PARAMS,
    clip,
    generated_family_from_latent,
)
from test_local_pulse_region_heterogeneity_benchmark import (
    RegionParams,
    calibrate_readout,
    homogeneous_params,
    train_network,
)
from test_local_pulse_region_parameter_family_learner import family_spread
from test_local_pulse_three_stage_training_closure import three_stage_objectives
from test_local_pulse_trainable_region_family_generator import FIXED_GENERATOR_LATENT
from test_local_pulse_unified_multiobjective_training_law import (
    PhaseRoutingLaw,
    UnifiedObjectiveShapedReplayNetwork,
)


def shifted(code: List[float], delta: List[float]) -> List[float]:
    return [clip(code[idx] + delta[idx], -1.0, 1.0) for idx in range(len(code))]


TRAIN_LATENTS = [
    FIXED_GENERATOR_LATENT,
    shifted(FIXED_GENERATOR_LATENT, [0.18, -0.12, 0.08, 0.00, 0.05]),
    shifted(FIXED_GENERATOR_LATENT, [-0.10, 0.16, -0.04, 0.12, -0.08]),
]

EVAL_LATENTS = [
    shifted(FIXED_GENERATOR_LATENT, [0.08, 0.14, -0.12, -0.10, 0.10]),
    shifted(FIXED_GENERATOR_LATENT, [-0.18, -0.04, 0.14, 0.08, -0.02]),
    shifted(FIXED_GENERATOR_LATENT, [0.04, -0.18, 0.10, -0.06, 0.16]),
]


@dataclass
class GeneratorNetworkSpec:
    latent_scale: List[float]
    latent_mix: List[float]
    latent_bias: List[float]
    region_scale: List[float]
    law_scale: List[float]


def base_spec() -> GeneratorNetworkSpec:
    return GeneratorNetworkSpec(
        latent_scale=[1.0, 1.0, 1.0, 1.0, 1.0],
        latent_mix=[0.00, 0.00, 0.00, 0.00, 0.00],
        latent_bias=[0.00, 0.00, 0.00, 0.00, 0.00],
        region_scale=[0.08, 0.10, 0.10, 0.06, 0.12, 0.06, 0.04],
        law_scale=[0.04, 0.05, 0.05, 0.06, 0.06, 0.03],
    )


def hidden_state(code: List[float], spec: GeneratorNetworkSpec) -> List[float]:
    rows = []
    for idx in range(len(code)):
        raw = (
            spec.latent_scale[idx] * code[idx]
            + spec.latent_mix[idx] * code[(idx + 1) % len(code)]
            + spec.latent_bias[idx]
        )
        rows.append(math.tanh(raw))
    return rows


def generated_family_from_network(code: List[float], spec: GeneratorNetworkSpec) -> Dict[str, object]:
    hidden = hidden_state(code, spec)
    hidden_mean = sum(hidden) / len(hidden)

    params = copy.deepcopy(LEARNED_PARAMS)
    params[0].inhibition = clip(params[0].inhibition * (1.0 + spec.region_scale[0] * hidden[0]), 0.14, 0.23)
    params[0].tonic_bias = clip(params[0].tonic_bias + spec.region_scale[6] * hidden_mean, 0.06, 0.12)
    params[1].leak = clip(params[1].leak * (1.0 + spec.region_scale[1] * hidden[0]), 0.88, 1.05)
    params[1].feedback_gain = clip(
        params[1].feedback_gain * (1.0 + spec.region_scale[2] * hidden[1]),
        0.17,
        0.27,
    )
    params[2].threshold = clip(params[2].threshold * (1.0 + spec.region_scale[3] * hidden[2]), 0.37, 0.43)
    params[2].feedback_gain = clip(
        params[2].feedback_gain * (1.0 + spec.region_scale[4] * (0.6 * hidden[1] + 0.4 * hidden[2])),
        0.16,
        0.23,
    )
    params[3].threshold = clip(params[3].threshold * (1.0 + spec.region_scale[5] * hidden[3]), 0.42, 0.48)
    params[3].feedback_gain = clip(params[3].feedback_gain + 0.02 * hidden_mean, 0.07, 0.11)

    law_dict = LEARNED_LAW.__dict__.copy()
    law_dict["comparison_memory_scale"] = clip(
        law_dict["comparison_memory_scale"] + spec.law_scale[0] * hidden[1] + 0.02 * hidden_mean,
        0.25,
        0.39,
    )
    law_dict["comparison_signal_scale"] = clip(
        law_dict["comparison_signal_scale"] + spec.law_scale[1] * hidden[2],
        0.60,
        0.74,
    )
    law_dict["comparator_feedback_scale"] = clip(
        law_dict["comparator_feedback_scale"] + spec.law_scale[2] * hidden[1],
        0.98,
        1.20,
    )
    law_dict["replay_memory_scale"] = clip(
        law_dict["replay_memory_scale"] + spec.law_scale[3] * hidden[4] + spec.law_scale[5] * hidden_mean,
        0.39,
        0.58,
    )
    law_dict["replay_comparator_scale"] = clip(
        law_dict["replay_comparator_scale"] + spec.law_scale[4] * hidden[4],
        0.29,
        0.47,
    )
    law_dict["motor_release_step"] = 8
    return {
        "params": params,
        "law": PhaseRoutingLaw(**law_dict),
        "hidden": hidden,
    }


def mutate_spec(spec: GeneratorNetworkSpec, rng: random.Random) -> GeneratorNetworkSpec:
    return GeneratorNetworkSpec(
        latent_scale=[clip(value + rng.uniform(-0.30, 0.30), 0.65, 1.35) for value in spec.latent_scale],
        latent_mix=[clip(value + rng.uniform(-0.22, 0.22), -0.35, 0.35) for value in spec.latent_mix],
        latent_bias=[clip(value + rng.uniform(-0.12, 0.12), -0.20, 0.20) for value in spec.latent_bias],
        region_scale=[clip(value * (1.0 + rng.uniform(-0.55, 0.55)), 0.02, 0.20) for value in spec.region_scale],
        law_scale=[clip(value * (1.0 + rng.uniform(-0.55, 0.55)), 0.02, 0.10) for value in spec.law_scale],
    )


def evaluate_network(network) -> Dict[str, object]:
    integration = integration_bundle(network)
    phase = phase_bundle(network)
    return {
        **integration,
        **phase,
        **three_stage_objectives({**integration, **phase}),
        "family_spread": family_spread(getattr(network, "region_params", [])),
    }


def evaluate_family_codes(
    name: str,
    latents: List[List[float]],
    family_builder: Callable[[List[float]], Dict[str, object]],
) -> Dict[str, object]:
    per_latent = []
    for idx, code in enumerate(latents):
        generated = family_builder(code)
        network = UnifiedObjectiveShapedReplayNetwork(generated["params"], generated["law"], seed=59 + idx)
        train_network(network, epochs=24, noise=0.03)
        calibrate_readout(network, noise=0.04)
        metrics = evaluate_network(network)
        per_latent.append(
            {
                "latent_id": idx,
                "latent_code": code,
                "three_stage_score": float(metrics["three_stage_score"]),
                "closure_balance_score": float(metrics["closure_balance_score"]),
                "aggregate_objective": float(metrics["aggregate_objective"]),
                "family_spread": float(metrics["family_spread"]),
                "comparison_advantage": float(metrics["comparison_phase_memory_comparator_advantage"]),
                "concept_advantage": float(metrics["concept_phase_upstream_advantage"]),
                "law": generated["law"].__dict__,
                "region_params": [row.__dict__ for row in generated["params"]],
                "hidden": generated.get("hidden", []),
            }
        )

    def avg(key: str) -> float:
        return float(sum(row[key] for row in per_latent) / len(per_latent))

    return {
        "system_name": name,
        "latent_count": len(latents),
        "aggregate_objective": avg("aggregate_objective"),
        "three_stage_score": avg("three_stage_score"),
        "closure_balance_score": avg("closure_balance_score"),
        "family_spread": avg("family_spread"),
        "comparison_phase_memory_comparator_advantage": avg("comparison_advantage"),
        "concept_phase_upstream_advantage": avg("concept_advantage"),
        "per_latent": per_latent,
    }


def score_key(metrics: Dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["three_stage_score"]),
        float(metrics["closure_balance_score"]),
        float(metrics["aggregate_objective"]),
    )


def train_generator_network(num_candidates: int, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    base = base_spec()
    candidates = [{"trial_id": 0, "spec": base, "source": "base"}]
    for trial_id in range(1, num_candidates):
        candidates.append(
            {
                "trial_id": trial_id,
                "spec": mutate_spec(base, rng),
                "source": "mutated",
            }
        )

    best = None
    trials = []
    for candidate in candidates:
        train_metrics = evaluate_family_codes(
            "generator_network_train_family",
            TRAIN_LATENTS,
            lambda code, spec=candidate["spec"]: generated_family_from_network(code, spec),
        )
        eval_metrics = evaluate_family_codes(
            "generator_network_eval_family",
            EVAL_LATENTS,
            lambda code, spec=candidate["spec"]: generated_family_from_network(code, spec),
        )
        trial = {
            "trial_id": int(candidate["trial_id"]),
            "source": candidate["source"],
            "train_three_stage_score": float(train_metrics["three_stage_score"]),
            "train_closure_balance_score": float(train_metrics["closure_balance_score"]),
            "train_aggregate_objective": float(train_metrics["aggregate_objective"]),
            "eval_three_stage_score": float(eval_metrics["three_stage_score"]),
            "eval_closure_balance_score": float(eval_metrics["closure_balance_score"]),
            "eval_aggregate_objective": float(eval_metrics["aggregate_objective"]),
        }
        trials.append(trial)
        key = score_key(train_metrics)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "trial_id": int(candidate["trial_id"]),
                "spec": candidate["spec"],
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }

    return {
        "best": best,
        "trials": sorted(
            trials,
            key=lambda row: (
                row["train_three_stage_score"],
                row["train_closure_balance_score"],
                row["train_aggregate_objective"],
            ),
            reverse=True,
        ),
    }


def shared_system_metrics() -> Dict[str, object]:
    shared = TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31)
    train_network(shared, epochs=28, noise=0.03)
    calibrate_readout(shared, noise=0.04)
    metrics = evaluate_network(shared)
    metrics["training_epochs"] = 28
    return metrics


def learned_system_metrics() -> Dict[str, object]:
    learned = UnifiedObjectiveShapedReplayNetwork(copy.deepcopy(LEARNED_PARAMS), LEARNED_LAW, seed=59)
    train_network(learned, epochs=24, noise=0.03)
    calibrate_readout(learned, noise=0.04)
    metrics = evaluate_network(learned)
    metrics["training_epochs"] = 24
    metrics["law"] = LEARNED_LAW.__dict__
    metrics["region_params"] = [row.__dict__ for row in LEARNED_PARAMS]
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Generator network benchmark for local pulse region families")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_region_family_generator_network_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    shared_metrics = shared_system_metrics()
    learned_metrics = learned_system_metrics()
    fixed_eval_metrics = evaluate_family_codes(
        "fixed_region_generator_eval_family",
        EVAL_LATENTS,
        generated_family_from_latent,
    )
    network_search = train_generator_network(num_candidates=10, seed=17)
    network_train_metrics = network_search["best"]["train_metrics"]
    network_eval_metrics = network_search["best"]["eval_metrics"]
    best_spec = network_search["best"]["spec"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "low_dim_region_family_generator_network_without_global_controller",
            "latent_dim": 5,
            "hidden_dim": 5,
            "train_latent_count": len(TRAIN_LATENTS),
            "eval_latent_count": len(EVAL_LATENTS),
            "search_candidates": 10,
        },
        "systems": {
            "shared_local_replay": shared_metrics,
            "learned_region_family": learned_metrics,
            "fixed_region_generator_eval_family": fixed_eval_metrics,
            "generator_network_train_family": network_train_metrics,
            "generator_network_eval_family": {
                **network_eval_metrics,
                "generator_spec": {
                    "latent_scale": best_spec.latent_scale,
                    "latent_mix": best_spec.latent_mix,
                    "latent_bias": best_spec.latent_bias,
                    "region_scale": best_spec.region_scale,
                    "law_scale": best_spec.law_scale,
                },
                "training_epochs": 24,
                "best_trial_id": int(network_search["best"]["trial_id"]),
            },
        },
        "search_summary": {
            "top_trials": network_search["trials"][:6],
            "train_latents": TRAIN_LATENTS,
            "eval_latents": EVAL_LATENTS,
        },
        "headline_metrics": {
            "network_best_system": "generator_network_eval_family",
            "network_eval_three_stage_gain_vs_fixed": float(
                network_eval_metrics["three_stage_score"] - fixed_eval_metrics["three_stage_score"]
            ),
            "network_eval_balance_gain_vs_fixed": float(
                network_eval_metrics["closure_balance_score"] - fixed_eval_metrics["closure_balance_score"]
            ),
            "network_eval_aggregate_gap_vs_fixed": float(
                network_eval_metrics["aggregate_objective"] - fixed_eval_metrics["aggregate_objective"]
            ),
            "network_eval_three_stage_gain_vs_learned": float(
                network_eval_metrics["three_stage_score"] - learned_metrics["three_stage_score"]
            ),
            "network_vs_shared_score_gap": float(
                network_eval_metrics["aggregate_objective"] - shared_metrics["aggregate_objective"]
            ),
            "network_generalization_gap": float(
                network_train_metrics["three_stage_score"] - network_eval_metrics["three_stage_score"]
            ),
            "best_trial_id": int(network_search["best"]["trial_id"]),
        },
        "hypotheses": {
            "H1_generator_network_beats_fixed_eval_family": bool(
                network_eval_metrics["three_stage_score"] > fixed_eval_metrics["three_stage_score"] + 0.01
            ),
            "H2_generator_network_improves_eval_balance": bool(
                network_eval_metrics["closure_balance_score"] > fixed_eval_metrics["closure_balance_score"] + 0.02
            ),
            "H3_generator_network_keeps_eval_cost_bounded": bool(
                network_eval_metrics["aggregate_objective"] >= fixed_eval_metrics["aggregate_objective"] - 0.01
            ),
            "H4_generator_network_generalizes_across_latents": bool(
                network_train_metrics["three_stage_score"] - network_eval_metrics["three_stage_score"] <= 0.03
            ),
        },
        "project_readout": {
            "summary": "这一步把低维脑区参数族从单点 latent 微调推进成跨 latent 的小型生成网络。目标不再是调好一个坐标，而是检验同一生成网络能否在多个 latent 上稳定产出更好的区域化统一律。",
            "next_question": "如果生成网络在 held-out latent 上也能保持三阶段闭环优势，下一步就应该把它接到真实模型层带和在线 rollback 或 recovery 闭环。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
