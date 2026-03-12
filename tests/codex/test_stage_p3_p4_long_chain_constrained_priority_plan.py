from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def latest_match(pattern: str) -> Path:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return matches[0]


def load_latest(pattern: str) -> dict:
    return json.loads(latest_match(pattern).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage P3-P4 long-chain constrained priority plan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_p3_p4_long_chain_constrained_priority_plan_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    base_plan = load_latest("stage_p3_p4_joint_intervention_execution_plan_*.json")
    intervention_pruning = load_latest("theory_track_long_chain_inventory_to_intervention_pruning_*.json")
    survival_criteria = load_latest("theory_track_long_chain_survival_criteria_*.json")

    priority_plan = [
        {
            "priority": 1,
            "name": "scaffolded_readout_vs_baseline_intervention",
            "why_now": "still directly targets object_to_readout_compatibility and remains compatible with long-chain constraints",
        },
        {
            "priority": 2,
            "name": "reasoning_slice_transport_intervention",
            "why_now": "still the best immediate companion to the winner while keeping family-conditioned shared reasoning slice intact",
        },
        {
            "priority": 3,
            "name": "stage_conditioned_reasoning_transport_intervention",
            "why_now": "now promoted because long-chain inventory shows stage structure must be represented explicitly",
        },
        {
            "priority": 4,
            "name": "causal_successor_alignment_intervention",
            "why_now": "now promoted because long-chain inventory introduces successor coherence as a new survival constraint",
        },
        {
            "priority": 5,
            "name": "stress_guard_intervention",
            "why_now": "remains necessary, but should now be evaluated after stage/chain-aware transport is tested",
        },
        {
            "priority": 6,
            "name": "anchored_relation_lift_intervention",
            "why_now": "bridge-role closure should now be tested under already constrained stage/chain-aware transport assumptions",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "StageP3_P4_long_chain_constrained_priority_plan",
        },
        "base_winner_operator": base_plan["winner_operator"],
        "preserved_interventions": intervention_pruning["preserved_interventions"],
        "ready_survival_count": survival_criteria["inventory_constraints"]["ready_for_immediate_survival_test_count"],
        "extended_theorem_count": survival_criteria["inventory_constraints"]["extended_theorem_count"],
        "priority_plan": priority_plan,
        "verdict": {
            "core_answer": "Long-chain constraints now force the joint P3/P4 program into a six-step priority plan where stage-conditioned transport and successor alignment move ahead of later stress and bridge interventions.",
            "next_engineering_target": "execute priorities 1-4 as one coherent long-chain constrained block rather than as isolated intervention tests.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
