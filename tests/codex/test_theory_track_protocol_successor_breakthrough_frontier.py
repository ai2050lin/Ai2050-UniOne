from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Protocol-successor breakthrough frontier")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_protocol_successor_breakthrough_frontier_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load("stage_protocol_successor_breakthrough_block_20260313.json")
    pruned = load("theory_track_systemic_inventory_master_pruning_20260312.json")

    proj = block["breakthrough_projection"]
    thresholds = block["thresholds"]

    successor_pass = float(proj["successor_coherence"]) >= float(thresholds["successor_global_support_band"])
    protocol_pass = float(proj["protocol_calling"]) >= float(thresholds["protocol_bridge_strong_band"])
    brain_pass = float(proj["brain_side_causal_closure"]) >= float(thresholds["brain_online_execution_band"])

    strict_core = [
        "family_section_theorem",
        "restricted_readout_transport_theorem",
        "stage_conditioned_reasoning_transport_theorem",
        "causal_successor_alignment_theorem",
    ]
    if protocol_pass:
        strict_core.append("protocol_task_bridge_theorem")
    if successor_pass and protocol_pass:
        strict_core.append("protocol_successor_global_support_theorem")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_protocol_successor_breakthrough_frontier",
        },
        "projected_status": {
            "successor_global_support_pass": successor_pass,
            "protocol_bridge_strong_pass": protocol_pass,
            "brain_online_execution_pass": brain_pass,
            "strict_core_size_if_executed": len(strict_core),
        },
        "strict_core_if_executed": strict_core,
        "remaining_theorems": [
            theorem
            for theorem in pruned["master_pruning"]["preserved_theorems"]
            if theorem not in strict_core
        ],
        "verdict": {
            "core_answer": (
                "If the protocol-successor-brain block is executed as one unit, the theorem frontier should move from a local-support regime toward a protocol-successor global-support regime."
            ),
            "next_target": (
                "Execute the integrated block and then pull stress_guarded_update_theorem and anchored_bridge_lift_theorem into the next strict frontier."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
