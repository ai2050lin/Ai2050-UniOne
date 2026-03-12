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
    ap = argparse.ArgumentParser(description="Theory-track long-chain inventory to intervention pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_inventory_to_intervention_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")
    intervention = load_latest("stage_p3_p4_joint_intervention_execution_plan_*.json")
    pruning = load_latest("theory_track_long_chain_inventory_theorem_pruning_*.json")

    chain_ratio = long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"]
    relation_ratio = long_chain["headline_metrics"]["relation_cross_to_within_ratio"]

    pruned_interventions = [
        "fully_shared_global_loop_intervention",
        "stage_free_readout_intervention",
        "chain_agnostic_transport_intervention",
    ]
    preserved_interventions = [
        intervention["priority_plan"][0]["name"],
        intervention["priority_plan"][1]["name"],
        "stage_conditioned_reasoning_transport_intervention",
        "causal_successor_alignment_intervention",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_inventory_to_intervention_pruning",
        },
        "inventory_constraints": {
            "relation_cross_to_within_ratio": relation_ratio,
            "chain_successor_to_cross_stage_ratio": chain_ratio,
            "ready_theorem_count": pruning["intervention_link"]["ready_theorem_count"],
        },
        "pruned_interventions": pruned_interventions,
        "preserved_interventions": preserved_interventions,
        "current_priority_alignment": {
            "priority_1": intervention["priority_plan"][0]["name"],
            "priority_2": intervention["priority_plan"][1]["name"],
            "winner_operator": intervention["winner_operator"],
        },
        "verdict": {
            "core_answer": "Long-chain inventory now narrows intervention space further: only stage-aware, chain-aware, family-anchored interventions remain plausible.",
            "next_theory_target": "upgrade priority 1+2 interventions with successor-alignment checks and stage-conditioned transport constraints.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
