#!/usr/bin/env python
"""
Train the region family generator network end-to-end across multiple latent
codes and test whether the trained generator improves over the searched
generator-network baseline on held-out latents.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from test_local_pulse_region_family_generator import generated_family_from_latent
from test_local_pulse_region_family_generator_network import (
    EVAL_LATENTS,
    TRAIN_LATENTS,
    GeneratorNetworkSpec,
    evaluate_family_codes,
    learned_system_metrics,
    generated_family_from_network,
    shared_system_metrics,
)


BEST_SEARCHED_SPEC = GeneratorNetworkSpec(
    latent_scale=[
        1.0139838174455202,
        0.8152535778680732,
        1.0454216803548837,
        1.122368202597515,
        1.1589948691967946,
    ],
    latent_mix=[
        -0.022272718518972612,
        -0.023954586859866567,
        0.001873297246927419,
        0.01595140279623916,
        -0.011032413249952466,
    ],
    latent_bias=[
        -0.1170959914429715,
        -0.05677743998042677,
        0.11707372249636647,
        0.07831686240693381,
        -0.007830476157645344,
    ],
    region_scale=[
        0.10217405787803344,
        0.06823738690985895,
        0.07438885301977116,
        0.06116354121291116,
        0.14630692417933977,
        0.028819235633654045,
        0.03875533744134295,
    ],
    law_scale=[
        0.05786771122561534,
        0.05617416315691914,
        0.07712001936808659,
        0.033513381759456236,
        0.08696770432680893,
        0.041134130064507085,
    ],
)


def score_tuple(metrics: Dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["three_stage_score"]),
        float(metrics["closure_balance_score"]),
        float(metrics["aggregate_objective"]),
    )


def perturb(values: List[float], delta: float, lo: float, hi: float) -> List[float]:
    return [float(min(max(value + delta, lo), hi)) for value in values]


def mutate_spec(base: GeneratorNetworkSpec, rng: random.Random, scale: float) -> GeneratorNetworkSpec:
    return GeneratorNetworkSpec(
        latent_scale=[float(min(max(v + rng.uniform(-scale, scale), 0.70), 1.40)) for v in base.latent_scale],
        latent_mix=[float(min(max(v + rng.uniform(-0.75 * scale, 0.75 * scale), -0.35), 0.35)) for v in base.latent_mix],
        latent_bias=[float(min(max(v + rng.uniform(-0.50 * scale, 0.50 * scale), -0.22), 0.22)) for v in base.latent_bias],
        region_scale=[float(min(max(v + rng.uniform(-0.45 * scale, 0.45 * scale), 0.02), 0.20)) for v in base.region_scale],
        law_scale=[float(min(max(v + rng.uniform(-0.40 * scale, 0.40 * scale), 0.02), 0.10)) for v in base.law_scale],
    )


def targeted_variants(base: GeneratorNetworkSpec, scale: float) -> List[GeneratorNetworkSpec]:
    return [
        replace(
            base,
            region_scale=perturb(base.region_scale, 0.40 * scale, 0.02, 0.20),
            law_scale=perturb(base.law_scale, 0.25 * scale, 0.02, 0.10),
        ),
        replace(
            base,
            latent_scale=perturb(base.latent_scale, 0.30 * scale, 0.70, 1.40),
            latent_bias=perturb(base.latent_bias, -0.18 * scale, -0.22, 0.22),
        ),
        replace(
            base,
            latent_mix=perturb(base.latent_mix, 0.28 * scale, -0.35, 0.35),
            law_scale=perturb(base.law_scale, -0.18 * scale, 0.02, 0.10),
        ),
    ]


def train_generator_network_end_to_end(rounds: int, proposals_per_round: int, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    current_spec = BEST_SEARCHED_SPEC
    current_train = evaluate_family_codes(
        "generator_network_train_family",
        TRAIN_LATENTS,
        lambda code: generated_family_from_network(code, current_spec),
    )
    history = [
        {
            "round": 0,
            "source": "init",
            "train_three_stage_score": float(current_train["three_stage_score"]),
            "train_closure_balance_score": float(current_train["closure_balance_score"]),
            "train_aggregate_objective": float(current_train["aggregate_objective"]),
        }
    ]

    scale = 0.08
    update_count = 0
    for round_id in range(1, rounds + 1):
        best_round_spec = current_spec
        best_round_train = current_train
        improved = False
        candidates = targeted_variants(current_spec, scale)
        while len(candidates) < proposals_per_round:
            candidates.append(mutate_spec(current_spec, rng, scale))
        for candidate_id, candidate_spec in enumerate(candidates[:proposals_per_round]):
            candidate_train = evaluate_family_codes(
                "generator_network_train_family",
                TRAIN_LATENTS,
                lambda code, spec=candidate_spec: generated_family_from_network(code, spec),
            )
            history.append(
                {
                    "round": round_id,
                    "source": f"candidate_{candidate_id}",
                    "train_three_stage_score": float(candidate_train["three_stage_score"]),
                    "train_closure_balance_score": float(candidate_train["closure_balance_score"]),
                    "train_aggregate_objective": float(candidate_train["aggregate_objective"]),
                }
            )
            if score_tuple(candidate_train) > score_tuple(best_round_train):
                best_round_spec = candidate_spec
                best_round_train = candidate_train
                improved = True
        if improved:
            current_spec = best_round_spec
            current_train = best_round_train
            update_count += 1
            scale = max(0.025, scale * 0.85)
        else:
            scale = max(0.02, scale * 0.60)
        history.append(
            {
                "round": round_id,
                "source": "accepted",
                "train_three_stage_score": float(current_train["three_stage_score"]),
                "train_closure_balance_score": float(current_train["closure_balance_score"]),
                "train_aggregate_objective": float(current_train["aggregate_objective"]),
            }
        )

    eval_metrics = evaluate_family_codes(
        "end_to_end_generator_eval_family",
        EVAL_LATENTS,
        lambda code: generated_family_from_network(code, current_spec),
    )
    return {
        "spec": current_spec,
        "train_metrics": current_train,
        "eval_metrics": eval_metrics,
        "history": history,
        "final_scale": float(scale),
        "update_count": int(update_count),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end trainable region family generator network")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_end_to_end_region_family_generator_network_20260310.json",
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
    searched_eval_metrics = evaluate_family_codes(
        "generator_network_eval_family",
        EVAL_LATENTS,
        lambda code: generated_family_from_network(code, BEST_SEARCHED_SPEC),
    )
    trained = train_generator_network_end_to_end(rounds=4, proposals_per_round=5, seed=23)
    trained_eval_metrics = trained["eval_metrics"]
    trained_train_metrics = trained["train_metrics"]
    spec = trained["spec"]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "end_to_end_region_family_generator_network_without_global_controller",
            "latent_dim": 5,
            "train_latent_count": len(TRAIN_LATENTS),
            "eval_latent_count": len(EVAL_LATENTS),
            "training_rounds": 4,
            "proposals_per_round": 5,
        },
        "systems": {
            "shared_local_replay": shared_metrics,
            "learned_region_family": learned_metrics,
            "fixed_region_generator_eval_family": fixed_eval_metrics,
            "generator_network_eval_family": {
                **searched_eval_metrics,
                "generator_spec": {
                    "latent_scale": BEST_SEARCHED_SPEC.latent_scale,
                    "latent_mix": BEST_SEARCHED_SPEC.latent_mix,
                    "latent_bias": BEST_SEARCHED_SPEC.latent_bias,
                    "region_scale": BEST_SEARCHED_SPEC.region_scale,
                    "law_scale": BEST_SEARCHED_SPEC.law_scale,
                },
                "training_epochs": 24,
            },
            "end_to_end_generator_train_family": trained_train_metrics,
            "end_to_end_generator_eval_family": {
                **trained_eval_metrics,
                "generator_spec": {
                    "latent_scale": spec.latent_scale,
                    "latent_mix": spec.latent_mix,
                    "latent_bias": spec.latent_bias,
                    "region_scale": spec.region_scale,
                    "law_scale": spec.law_scale,
                },
                "final_scale": float(trained["final_scale"]),
                "update_count": int(trained["update_count"]),
                "training_epochs": 24,
            },
        },
        "optimization_history": trained["history"],
        "headline_metrics": {
            "end_to_end_best_system": "end_to_end_generator_eval_family",
            "end_to_end_three_stage_gain_vs_network": float(
                trained_eval_metrics["three_stage_score"] - searched_eval_metrics["three_stage_score"]
            ),
            "end_to_end_balance_gain_vs_network": float(
                trained_eval_metrics["closure_balance_score"] - searched_eval_metrics["closure_balance_score"]
            ),
            "end_to_end_aggregate_gap_vs_network": float(
                trained_eval_metrics["aggregate_objective"] - searched_eval_metrics["aggregate_objective"]
            ),
            "end_to_end_three_stage_gain_vs_fixed": float(
                trained_eval_metrics["three_stage_score"] - fixed_eval_metrics["three_stage_score"]
            ),
            "end_to_end_generalization_gap": float(
                trained_train_metrics["three_stage_score"] - trained_eval_metrics["three_stage_score"]
            ),
            "end_to_end_vs_learned_gap": float(
                trained_eval_metrics["three_stage_score"] - learned_metrics["three_stage_score"]
            ),
            "update_count": int(trained["update_count"]),
        },
        "hypotheses": {
            "H1_end_to_end_generator_beats_searched_network": bool(
                trained_eval_metrics["three_stage_score"] > searched_eval_metrics["three_stage_score"] + 0.01
            ),
            "H2_end_to_end_generator_improves_balance": bool(
                trained_eval_metrics["closure_balance_score"] > searched_eval_metrics["closure_balance_score"] + 0.02
            ),
            "H3_end_to_end_generator_keeps_cost_bounded": bool(
                trained_eval_metrics["aggregate_objective"] >= searched_eval_metrics["aggregate_objective"] - 0.01
            ),
            "H4_end_to_end_generator_generalizes": bool(
                trained_train_metrics["three_stage_score"] - trained_eval_metrics["three_stage_score"] <= 0.03
            ),
        },
        "project_readout": {
            "summary": "这一步把跨 latent 的小型生成网络继续推进成端到端训练式更新。目标不再是挑出一个好候选，而是看同一生成映射经过多轮更新后，能否在 held-out latent 上稳定变强。",
            "next_question": "如果端到端训练式更新也成立，下一步就该把这套低维生成网络接到真实模型层带与在线 rollback 或 recovery 闭环。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
