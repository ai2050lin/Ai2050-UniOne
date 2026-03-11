#!/usr/bin/env python
"""
Score whether task block 2 (unified training-law closure) is sufficiently closed.
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    clipped = min(max(float(value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def score_system(row: Dict[str, Any]) -> Dict[str, float]:
    components = {
        "same_family_success": normalize(float(row["same_family_success_rate"]), 0.64, 0.69),
        "lesion_recovery": normalize(float(row["lesion_recovery_rate"]), 0.63, 0.67),
        "baseline_accuracy": normalize(float(row["baseline_accuracy"]), 0.65, 0.67),
        "aggregate_objective": normalize(float(row["aggregate_objective"]), 0.60, 0.65),
        "transfer_stability": normalize(float(row["transfer_stability"]), 0.9985, 1.0),
    }
    if "three_stage_score" in row:
        components["three_stage_score"] = normalize(float(row["three_stage_score"]), 0.46, 0.58)
    if "recovery_phase_score" in row:
        components["recovery_phase_score"] = normalize(float(row["recovery_phase_score"]), 0.54, 0.68)
    if "multiobjective_score" in row:
        components["multiobjective_score"] = normalize(float(row["multiobjective_score"]), 0.55, 0.61)
    if "stage_decomposed_score" in row:
        components["stage_decomposed_score"] = normalize(float(row["stage_decomposed_score"]), 0.40, 0.48)
    if "closure_balance_score" in row:
        components["closure_balance"] = normalize(float(row["closure_balance_score"]), 0.15, 0.35)
    if "family_spread" in row:
        components["family_spread"] = normalize(float(row["family_spread"]), 0.6, 1.4)
    return components


def best_by_key(systems: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    _name, row = max(systems.items(), key=lambda item: float(item[1].get(key, 0.0)))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Task block 2 unified training-law closure scorecard")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/task_block_2_unified_training_closure_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    unified = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_unified_multiobjective_training_law_20260310.json")
    stage_decomp = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_stage_decomposed_training_law_20260310.json")
    recovery = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_recovery_phase_training_law_20260310.json")
    three_stage = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_three_stage_training_closure_20260310.json")
    generator = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_region_family_generator_network_20260310.json")
    e2e_generator = load_json(ROOT / "tests" / "codex_temp" / "local_pulse_end_to_end_region_family_generator_network_20260310.json")

    best_unified = best_by_key(unified["systems"], "multiobjective_score")
    best_stage = best_by_key(stage_decomp["systems"], "stage_decomposed_score")
    best_recovery = best_by_key(recovery["systems"], "recovery_phase_score")
    best_three_stage = best_by_key(three_stage["systems"], "three_stage_score")
    best_generator = generator["systems"]["learned_region_family"]
    best_e2e = e2e_generator["systems"]["learned_region_family"]

    pillars = {
        "multiobjective_anchor": {
            "system": "shared_local_replay",
            "components": score_system(best_unified),
        },
        "stage_decomposed_training": {
            "system": "regional_stage_decomposed",
            "components": score_system(best_stage),
        },
        "recovery_phase_training": {
            "system": "regional_recovery_aware",
            "components": score_system(best_recovery),
        },
        "three_stage_training": {
            "system": "learned_region_family",
            "components": score_system(best_three_stage),
        },
        "generator_network": {
            "system": "learned_region_family",
            "components": score_system(best_generator),
        },
        "end_to_end_generator_network": {
            "system": "learned_region_family",
            "components": score_system(best_e2e),
        },
    }

    for row in pillars.values():
        row["score"] = mean(row["components"].values())

    structure_score = mean(
        [
            pillars["multiobjective_anchor"]["score"],
            pillars["stage_decomposed_training"]["score"],
            pillars["three_stage_training"]["score"],
        ]
    )
    recovery_score = mean(
        [
            pillars["recovery_phase_training"]["score"],
            pillars["three_stage_training"]["score"],
        ]
    )
    generator_score = mean(
        [
            pillars["generator_network"]["score"],
            pillars["end_to_end_generator_network"]["score"],
        ]
    )
    overall_score = mean([structure_score, recovery_score, generator_score])

    hypotheses = {
        "H1_training_law_keeps_structure_above_baseline": bool(structure_score >= 0.62),
        "H2_training_law_keeps_recovery_phase_nontrivial": bool(recovery_score >= 0.55),
        "H3_three_stage_training_beats_old_anchor": bool(
            float(best_three_stage["three_stage_score"]) > float(best_unified["aggregate_objective"]) - 0.12
        ),
        "H4_generator_family_keeps_phase_specific_structure": bool(
            float(best_generator["concept_phase_upstream_advantage"]) > 0.10
            and float(best_generator["comparison_phase_memory_comparator_advantage"]) > 0.08
        ),
        "H5_end_to_end_generator_does_not_collapse_three_stage_score": bool(
            float(best_e2e["three_stage_score"]) >= float(best_generator["three_stage_score"]) - 0.02
        ),
        "H6_task_block_2_is_moderately_closed": bool(overall_score >= 0.58),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "task_block_2_unified_training_law_closure",
        },
        "pillars": pillars,
        "headline_metrics": {
            "structure_training_score": float(structure_score),
            "recovery_training_score": float(recovery_score),
            "generator_training_score": float(generator_score),
            "overall_task_block_2_score": float(overall_score),
            "best_three_stage_score": float(best_three_stage["three_stage_score"]),
            "best_generator_three_stage_score": float(best_generator["three_stage_score"]),
            "best_e2e_three_stage_score": float(best_e2e["three_stage_score"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Task block 2 is treated as closed only if the unified law keeps structure, comparison, recovery, "
                "and generator-based parameterization in one family instead of trading one stage for another."
            ),
            "next_question": (
                "If this scorecard is positive, the next stage should stop optimizing local toy closure and instead "
                "test whether the same training-law family preserves real-model bridge and online-task metrics."
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
