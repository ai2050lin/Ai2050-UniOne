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
    ap = argparse.ArgumentParser(description="Current-route bottleneck viability assessment")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_current_route_bottleneck_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load("theory_track_protocol_successor_closure_block_20260313.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    succ = load("theory_track_successor_coherence_closure_diagnosis_20260312.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")

    projected_gain = float(block["closure_block_projection"]["gain_vs_current"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    online_gap = float(qd["missing_axes"][0]["gap"])
    is_closed = bool(succ["closure_status"]["is_closed"])

    route_not_dead = projected_gain > 0.02 and not is_closed
    route_currently_insufficient = successor < 0.35 or protocol < 0.75 or brain < 0.8

    blockers = [
        {
            "name": "online_natural_trace_missing",
            "severity": "highest",
            "evidence": f"successor online trace gap={online_gap:.4f}",
        },
        {
            "name": "protocol_bridge_not_yet_strong_enough",
            "severity": "highest",
            "evidence": f"protocol_calling={protocol:.4f}",
        },
        {
            "name": "brain_side_causal_execution_not_yet_online",
            "severity": "high",
            "evidence": f"brain_side_causal_closure={brain:.4f}",
        },
        {
            "name": "successor_still_local_not_global",
            "severity": "high",
            "evidence": f"successor_coherence={successor:.4f}",
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_current_route_bottleneck_assessment",
        },
        "route_status": {
            "current_route_is_fundamentally_blocked": False,
            "current_route_is_currently_insufficient": route_currently_insufficient,
            "current_route_still_has_projected_gain": route_not_dead,
            "projected_gain_if_protocol_successor_block_is_executed": projected_gain,
        },
        "main_blockers": blockers,
        "verdict": {
            "core_answer": (
                "The current route is not fundamentally impossible, but it is insufficient in its present form because it still lacks online natural traces, stronger protocol bridge support, and real brain-side causal execution."
            ),
            "next_route_requirement": (
                "Upgrade the route from artifact-led theory closure into protocol-successor-brain integrated execution; otherwise repeated local improvements will continue to saturate."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
