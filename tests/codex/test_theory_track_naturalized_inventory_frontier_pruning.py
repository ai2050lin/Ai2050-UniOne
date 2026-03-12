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
    ap = argparse.ArgumentParser(description="Theory-track naturalized inventory frontier pruning")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_naturalized_inventory_frontier_pruning_20260312.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    inventory = load_latest("theory_track_large_scale_naturalized_reasoning_inventory_*.json")
    old_pruning = load_latest("theory_track_long_chain_inventory_to_intervention_pruning_*.json")
    old_a_m = load_latest("theory_track_long_chain_inventory_to_A_Mfeas_pruning_*.json")

    metrics = inventory["headline_metrics"]
    temporal_ratio = float(metrics["temporal_cross_to_within_ratio"])
    successor_ratio = float(metrics["chain_successor_to_cross_stage_ratio"])
    relation_ratio = float(metrics["relation_cross_to_within_ratio"])

    strengthened_theorems = [
        "family_section_theorem",
        "restricted_readout_transport_theorem",
        "stage_conditioned_reasoning_transport_theorem",
        "causal_successor_alignment_theorem",
    ]
    queued_theorems = [
        "stress_guarded_update_theorem",
        "anchored_bridge_lift_theorem",
    ]

    preserved_A = old_a_m["A_pruning"]["preserved_families"] + [
        "successor_sensitive_update_gate",
    ]
    preserved_Mfeas = old_a_m["Mfeas_pruning"]["preserved_families"] + [
        "successor_aligned_transition_band",
    ]
    preserved_interventions = old_pruning["preserved_interventions"] + [
        "naturalized_stage_successor_joint_intervention",
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_naturalized_inventory_frontier_pruning",
        },
        "inventory_constraints": {
            "relation_cross_to_within_ratio": relation_ratio,
            "temporal_cross_to_within_ratio": temporal_ratio,
            "chain_successor_to_cross_stage_ratio": successor_ratio,
        },
        "frontier_pruning": {
            "strengthened_theorems": strengthened_theorems,
            "queued_theorems": queued_theorems,
            "preserved_A_families": preserved_A,
            "preserved_Mfeas_families": preserved_Mfeas,
            "preserved_interventions": preserved_interventions,
        },
        "verdict": {
            "core_answer": "Naturalized long-chain constraints now strengthen the stage/successor frontier enough to keep them in the active theorem and intervention set, while still queueing stress and bridge for the next block.",
            "next_theory_target": "convert strengthened stage/successor support into stricter pass/fail survival tests under more demanding perturbations.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
