#!/usr/bin/env python
"""
Compare score-only selection against structure-aware selection for local pulse
systems with and without region-differentiated mechanisms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

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


def build_networks():
    systems = {
        "shared_local_replay": TraceableCoupledReplayNetwork(homogeneous_params(), use_replay=True, seed=31),
        "regional_local_replay": TraceableCoupledReplayNetwork(heterogeneous_params(), use_replay=True, seed=17),
        "regional_phase_tuned_replay": EarlyCoreDecoupledReplayNetwork(decoupled_params(), use_replay=True, seed=23),
    }
    for network in systems.values():
        train_network(network, epochs=28, noise=0.03)
        calibrate_readout(network, noise=0.04)
    return systems


def normalized(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clamped = min(max(value, lo), hi)
    return (clamped - lo) / (hi - lo)


def evaluate_systems() -> Dict[str, Dict[str, object]]:
    rows = {}
    for name, network in build_networks().items():
        integration = integration_bundle(network)
        phase = phase_bundle(network)
        concept_adv = float(phase["concept_phase_upstream_advantage"])
        comparison_adv = float(phase["comparison_phase_memory_comparator_advantage"])
        diversity = float(phase["phase_summary"]["distinct_top_region_count"])
        aggregate_score = float(integration["local_integration_score"])
        structure_score = float(
            0.42 * normalized(concept_adv, -0.10, 0.20)
            + 0.28 * normalized(comparison_adv, -0.25, 0.18)
            + 0.18 * normalized(diversity, 1.0, 3.0)
            + 0.12 * normalized(aggregate_score, 0.55, 0.67)
        )
        rows[name] = {
            **integration,
            **phase,
            "aggregate_score": aggregate_score,
            "structure_score": structure_score,
        }
    return rows


def pareto_front(systems: Dict[str, Dict[str, object]]) -> List[str]:
    names = list(systems.keys())
    front = []
    for name in names:
        score_a = float(systems[name]["aggregate_score"])
        score_s = float(systems[name]["structure_score"])
        dominated = False
        for other in names:
            if other == name:
                continue
            other_a = float(systems[other]["aggregate_score"])
            other_s = float(systems[other]["structure_score"])
            if other_a >= score_a and other_s >= score_s and (other_a > score_a or other_s > score_s):
                dominated = True
                break
        if not dominated:
            front.append(name)
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description="Region-differentiated multiobjective selector for local pulse systems")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/local_pulse_region_differentiated_multiobjective_selector_20260310.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systems = evaluate_systems()
    aggregate_best = max(systems.items(), key=lambda item: item[1]["aggregate_score"])[0]
    structure_best = max(systems.items(), key=lambda item: item[1]["structure_score"])[0]
    front = pareto_front(systems)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "region_differentiated_multiobjective_selector_without_global_controller",
        },
        "systems": systems,
        "headline_metrics": {
            "aggregate_best_system": aggregate_best,
            "structure_best_system": structure_best,
            "aggregate_best_score": float(systems[aggregate_best]["aggregate_score"]),
            "structure_best_score": float(systems[structure_best]["structure_score"]),
            "pareto_front": front,
            "shared_vs_tuned_score_gap": float(
                systems["shared_local_replay"]["aggregate_score"]
                - systems["regional_phase_tuned_replay"]["aggregate_score"]
            ),
            "tuned_vs_shared_structure_gap": float(
                systems["regional_phase_tuned_replay"]["structure_score"]
                - systems["shared_local_replay"]["structure_score"]
            ),
        },
        "hypotheses": {
            "H1_score_only_prefers_shared_law": bool(aggregate_best == "shared_local_replay"),
            "H2_structure_objective_prefers_region_tuned_law": bool(
                structure_best == "regional_phase_tuned_replay"
            ),
            "H3_no_single_system_wins_both": bool(aggregate_best != structure_best),
            "H4_region_tuned_system_is_pareto_member": bool("regional_phase_tuned_replay" in front),
        },
        "project_readout": {
            "summary": "这一步直接回答训练目标该怎么选。只看总分时，系统会偏向更统一、分工更少的共享局部律；把阶段局部核和脑区差异一起纳入目标后，选择结果会明显变化。",
            "next_question": "如果共享局部律和分区局部律同时落在 Pareto 前沿上，下一步就不该再追单一分数最优，而该显式做多目标局部训练律。",
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
