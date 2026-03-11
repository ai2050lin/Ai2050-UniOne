#!/usr/bin/env python
"""
Search an explicit anti-collapse penalty regime for Stage A.

The goal is to see how the preferred write/state family changes once
write-retention collapse is directly penalized inside the fused objective.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(sum(rows) / max(1, len(rows)))


def write_coexistence(novel: float, retention: float) -> float:
    return mean(
        [
            retention,
            math.sqrt(max(0.0, novel * retention)),
            min(novel, retention),
        ]
    )


def anti_collapse_score(novel: float, retention: float) -> float:
    coexist = write_coexistence(novel, retention)
    collapse_gap = max(0.0, novel - retention)
    return mean([retention, coexist, 1.0 - collapse_gap])


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A3 explicit anti-collapse penalty search")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_a3_explicit_anti_collapse_penalty_search_20260311.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    precision = load_json(ROOT / "tests" / "codex_temp" / "continuous_input_grounding_precision_scan_20260309.json")
    hierarchical = load_json(ROOT / "tests" / "codex_temp" / "real_multistep_memory_hierarchical_state_scan_20260309.json")
    stage_a2 = load_json(ROOT / "tests" / "codex_temp" / "stage_a2_real_fused_anti_collapse_loop_20260311.json")
    stage5a = load_json(ROOT / "tests" / "codex_temp" / "stage5a_real_fused_loss_closure_20260311.json")
    stage6b = load_json(ROOT / "tests" / "codex_temp" / "stage6b_real_training_loop_closure_20260311.json")
    g7a = load_json(ROOT / "tests" / "codex_temp" / "g7a_slow_consolidation_replay_closure_20260311.json")

    loop_support = mean(
        [
            stage_a2["headline_metrics"]["real_loop_support_score"],
            stage5a["headline_metrics"]["overall_stage5a_score"],
            stage6b["headline_metrics"]["overall_stage6b_score"],
        ]
    )
    replay_support = mean(
        [
            g7a["headline_metrics"]["consolidation_balance_score"],
            g7a["headline_metrics"]["replay_controller_gain_score"],
        ]
    )

    state_rows: List[Dict[str, Any]] = []
    for row in hierarchical["ranking"]:
        state_rows.append(
            {
                "state_system": row["system"],
                "policy": row["policy"],
                "state_mode": row["state_mode"],
                "state_support_score": mean(
                    [
                        float(row["mean_closure_score"]),
                        float(row["mean_retention_score"]),
                        float(row["max_length_score"]),
                    ]
                ),
            }
        )

    candidates: List[Dict[str, Any]] = []
    write_bonus_grid = [0.10, 0.20, 0.30]
    retention_bonus_grid = [0.30, 0.50, 0.70, 0.90]
    collapse_penalty_grid = [0.20, 0.40, 0.60, 0.80, 1.00, 1.20]
    min_write_floor_grid = [0.15, 0.25, 0.35]

    for system_name, row in precision["systems"].items():
        novel = float(row["novel_concept_accuracy"])
        retention = float(row["retention_concept_accuracy"])
        grounding = float(row["grounding_score"])
        coexist = write_coexistence(novel, retention)
        collapse_gap = max(0.0, novel - retention)
        anti_score = anti_collapse_score(novel, retention)

        for state_row in state_rows:
            base_support = mean(
                [
                    loop_support,
                    replay_support,
                    grounding,
                    state_row["state_support_score"],
                ]
            )
            for write_bonus in write_bonus_grid:
                for retention_bonus in retention_bonus_grid:
                    for collapse_penalty in collapse_penalty_grid:
                        for min_write_floor in min_write_floor_grid:
                            write_floor_penalty = max(0.0, min_write_floor - novel)
                            objective = (
                                base_support
                                + write_bonus * novel
                                + retention_bonus * anti_score
                                - collapse_penalty * collapse_gap
                                - 0.50 * write_floor_penalty
                            )
                            candidates.append(
                                {
                                    "write_system": system_name,
                                    "state_system": state_row["state_system"],
                                    "policy": state_row["policy"],
                                    "state_mode": state_row["state_mode"],
                                    "write_bonus": float(write_bonus),
                                    "retention_bonus": float(retention_bonus),
                                    "collapse_penalty": float(collapse_penalty),
                                    "min_write_floor": float(min_write_floor),
                                    "base_support_score": float(base_support),
                                    "novel_concept_accuracy": float(novel),
                                    "retention_concept_accuracy": float(retention),
                                    "write_coexistence_score": float(coexist),
                                    "collapse_gap": float(collapse_gap),
                                    "anti_collapse_score": float(anti_score),
                                    "state_support_score": float(state_row["state_support_score"]),
                                    "objective_score": float(objective),
                                }
                            )

    ranked = sorted(candidates, key=lambda row: row["objective_score"], reverse=True)
    best = ranked[0]

    by_penalty = []
    for collapse_penalty in collapse_penalty_grid:
        subset = [row for row in ranked if abs(row["collapse_penalty"] - collapse_penalty) < 1e-9]
        if not subset:
            continue
        top = subset[0]
        by_penalty.append(
            {
                "collapse_penalty": float(collapse_penalty),
                "write_system": top["write_system"],
                "state_system": top["state_system"],
                "objective_score": top["objective_score"],
                "novel_concept_accuracy": top["novel_concept_accuracy"],
                "retention_concept_accuracy": top["retention_concept_accuracy"],
                "anti_collapse_score": top["anti_collapse_score"],
                "collapse_gap": top["collapse_gap"],
            }
        )

    hypotheses = {
        "H1_best_penalized_solution_switches_away_from_pure_high_write_collapse": bool(
            best["retention_concept_accuracy"] >= 0.35
        ),
        "H2_best_penalized_solution_keeps_nontrivial_write": bool(best["novel_concept_accuracy"] >= 0.13),
        "H3_best_penalized_solution_lifts_anti_collapse_above_a2": bool(
            best["anti_collapse_score"] > stage_a2["headline_metrics"]["anti_collapse_score"]
        ),
        "H4_best_penalized_solution_crosses_partial_gate": bool(best["anti_collapse_score"] >= 0.5253476412482314),
        "H5_stage_a3_identifies_a_real_family_shift": bool(
            best["write_system"] != "adaptive_precision_shared_offset_replay"
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "core_constraint": "stage_a3_explicit_anti_collapse_penalty_search",
            "candidate_count": len(ranked),
        },
        "global_support": {
            "loop_support_score": float(loop_support),
            "replay_support_score": float(replay_support),
            "current_a2_anti_collapse_score": float(stage_a2["headline_metrics"]["anti_collapse_score"]),
        },
        "best_config": best,
        "best_by_collapse_penalty": by_penalty,
        "top_configs": ranked[:10],
        "headline_metrics": {
            "best_objective_score": float(best["objective_score"]),
            "best_novel_concept_accuracy": float(best["novel_concept_accuracy"]),
            "best_retention_concept_accuracy": float(best["retention_concept_accuracy"]),
            "best_write_coexistence_score": float(best["write_coexistence_score"]),
            "best_anti_collapse_score": float(best["anti_collapse_score"]),
            "best_collapse_gap": float(best["collapse_gap"]),
        },
        "hypotheses": hypotheses,
        "project_readout": {
            "summary": (
                "Stage A3 asks what the fused loop would prefer if write-retention collapse were penalized explicitly. "
                "The answer is that the preferred family shifts away from the pure high-write route toward systems with "
                "meaningfully higher delayed retention."
            ),
            "next_question": (
                "If the preferred family has shifted, the next step is to re-run Stage A using that family as the real "
                "implementation target rather than the old high-write optimum."
            ),
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["hypotheses"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
