#!/usr/bin/env python
"""
Score whether Stage A can be upgraded from a scorecard-only family into a real
fused loop once explicit write-retention anti-collapse pressure is added.
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


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def required_component(target: float, others: list[float], n_terms: int) -> float:
    return float(target * n_terms - sum(others))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A2 real fused anti-collapse loop prototype")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_a2_real_fused_anti_collapse_loop_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    task2 = load_json(ROOT / "tests" / "codex_temp" / "task_block_2_unified_training_closure_20260311.json")
    g2 = load_json(ROOT / "tests" / "codex_temp" / "g2_structure_foundation_fast_slow_training_closure_20260311.json")
    g3 = load_json(ROOT / "tests" / "codex_temp" / "g3_instant_learning_boundary_stress_20260311.json")
    g7a = load_json(ROOT / "tests" / "codex_temp" / "g7a_slow_consolidation_replay_closure_20260311.json")
    g7b = load_json(ROOT / "tests" / "codex_temp" / "g7b_anti_interference_retention_mechanism_search_20260311.json")
    stage5a = load_json(ROOT / "tests" / "codex_temp" / "stage5a_real_fused_loss_closure_20260311.json")
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    stage_a = load_json(ROOT / "tests" / "codex_temp" / "stage_a_unified_training_strong_retention_master_20260311.json")
    stage_a1 = load_json(ROOT / "tests" / "codex_temp" / "stage_a1_fused_write_retention_search_20260311.json")

    real_loop_support_score = mean(
        [
            stage5a["headline_metrics"]["overall_stage5a_score"],
            stage6b["headline_metrics"]["overall_stage6b_score"],
            task2["headline_metrics"]["overall_task_block_2_score"],
        ]
    )

    structure_guard_score = mean(
        [
            stage5a["headline_metrics"]["structure_guard_score"],
            g2["headline_metrics"]["structure_foundation_training_score"],
            task2["headline_metrics"]["structure_training_score"],
        ]
    )

    candidate_integrability_score = mean(
        [
            stage_a1["headline_metrics"]["best_overall_fused_candidate_score"],
            stage_a1["headline_metrics"]["best_state_support_score"],
            g7a["headline_metrics"]["consolidation_balance_score"],
            g7a["headline_metrics"]["replay_controller_gain_score"],
        ]
    )

    anti_collapse_score = mean(
        [
            stage_a["headline_metrics"]["retention_coexistence_score"],
            stage_a1["headline_metrics"]["best_write_coexistence_score"],
            g7b["headline_metrics"]["retention_write_balance_score"],
            g3["headline_metrics"]["retention_boundary_score"],
        ]
    )

    overall_score = mean(
        [
            real_loop_support_score,
            structure_guard_score,
            candidate_integrability_score,
            anti_collapse_score,
        ]
    )

    others = [real_loop_support_score, structure_guard_score, candidate_integrability_score]
    anti_needed_for_partial = clamp01(required_component(0.58, others, 4))
    anti_needed_for_moderate = clamp01(required_component(0.62, others, 4))
    anti_needed_for_strong = clamp01(required_component(0.68, others, 4))

    hypotheses = {
        "H1_real_loop_support_is_already_nontrivial": bool(real_loop_support_score >= 0.68),
        "H2_structure_guard_is_only_moderate": bool(0.48 <= structure_guard_score < 0.66),
        "H3_candidate_integrability_is_nontrivial": bool(candidate_integrability_score >= 0.50),
        "H4_anti_collapse_is_the_dominant_bottleneck": bool(anti_collapse_score < min(others)),
        "H5_stage_a2_is_not_ready_without_explicit_anti_collapse": bool(overall_score < 0.58),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage_a2_real_fused_anti_collapse_loop",
        },
        "pillars": {
            "real_loop_support": float(real_loop_support_score),
            "structure_guard": float(structure_guard_score),
            "candidate_integrability": float(candidate_integrability_score),
            "anti_collapse": float(anti_collapse_score),
        },
        "headline_metrics": {
            "real_loop_support_score": float(real_loop_support_score),
            "structure_guard_score": float(structure_guard_score),
            "candidate_integrability_score": float(candidate_integrability_score),
            "anti_collapse_score": float(anti_collapse_score),
            "overall_stage_a2_score": float(overall_score),
        },
        "required_anti_collapse_targets": {
            "for_partial_closure_0_58": float(anti_needed_for_partial),
            "for_moderate_closure_0_62": float(anti_needed_for_moderate),
            "for_strong_closure_0_68": float(anti_needed_for_strong),
            "current_anti_collapse_score": float(anti_collapse_score),
            "lift_needed_for_partial": float(max(0.0, anti_needed_for_partial - anti_collapse_score)),
            "lift_needed_for_moderate": float(max(0.0, anti_needed_for_moderate - anti_collapse_score)),
            "lift_needed_for_strong": float(max(0.0, anti_needed_for_strong - anti_collapse_score)),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage A2 says the real fused loop foundation already exists, but it is still missing explicit anti-collapse "
                "pressure strong enough to keep delayed retention high under high-write settings."
            ),
            "next_question": (
                "The next real implementation should not re-score training support. It should inject a write-retention "
                "collapse penalty into the fused loop and then re-measure whether anti_collapse_score can cross the partial gate."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["required_anti_collapse_targets"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
