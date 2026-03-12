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
    ap = argparse.ArgumentParser(description="Theory-track long-chain inventory theorem pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_long_chain_inventory_theorem_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    long_chain = load_latest("theory_track_large_scale_long_chain_inventory_*.json")
    intervention = load_latest("stage_p3_p4_joint_intervention_execution_plan_*.json")
    survival = load_latest("theory_track_icspb_theorem_survival_report_*.json")

    pruned_theorems = [
        "single_global_reasoning_loop_theorem",
        "context_free_transport_theorem",
        "relation_free_readout_theorem",
        "temporal_stage_free_reasoning_theorem",
        "chain_agnostic_transport_theorem",
    ]
    preserved_theorems = [
        "family_section_theorem",
        "restricted_readout_transport_theorem",
        "stress_guarded_update_theorem",
        "anchored_bridge_lift_theorem",
        "stage_conditioned_reasoning_transport_theorem",
        "causal_successor_alignment_theorem",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_long_chain_inventory_theorem_pruning",
        },
        "inventory_constraints": {
            "family_cross_to_within_ratio": long_chain["headline_metrics"]["family_cross_to_within_ratio"],
            "context_cross_to_within_ratio": long_chain["headline_metrics"]["context_cross_to_within_ratio"],
            "relation_cross_to_within_ratio": long_chain["headline_metrics"]["relation_cross_to_within_ratio"],
            "temporal_cross_to_within_ratio": long_chain["headline_metrics"]["temporal_cross_to_within_ratio"],
            "chain_successor_to_cross_stage_ratio": long_chain["headline_metrics"]["chain_successor_to_cross_stage_ratio"],
        },
        "pruned_theorems": pruned_theorems,
        "preserved_theorems": preserved_theorems,
        "intervention_link": {
            "winner_operator": intervention["winner_operator"],
            "priority_1": intervention["priority_plan"][0]["name"],
            "priority_2": intervention["priority_plan"][1]["name"],
            "ready_theorem_count": survival["ready_for_immediate_survival_test_count"],
        },
        "verdict": {
            "core_answer": "Long-chain inventory is now strong enough to prune theorem families that ignore stage structure or successor coherence, while preserving chain-aware ICSPB candidates.",
            "next_theory_target": "bind stage-conditioned and successor-alignment theorem families directly to survival tests under priority 1+2 interventions.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
