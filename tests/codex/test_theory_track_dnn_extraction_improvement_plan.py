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
    ap = argparse.ArgumentParser(description="Systemic improvement plan for DNN extraction")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_dnn_extraction_improvement_plan_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    frontier = load("theory_track_protocol_successor_breakthrough_frontier_20260313.json")

    missing_axes = qd["missing_axes"][:5]
    priority_blocks = systemic["priority_block"]

    plan = [
        {
            "priority": 1,
            "block": "cross_model_real_long_chain_trace_capture",
            "goal": "把 successor 从 prototype inventory 推到真实模型内部自然 trace",
            "expected_impact": ["successor_coherence", "protocol bridge", "theorem survival realism"],
        },
        {
            "priority": 2,
            "block": "protocol_bridge_transport_intervention",
            "goal": "把 concept/readout/successor 真正穿过 protocol/task bridge",
            "expected_impact": ["protocol_calling", "object_to_readout_compatibility", "reasoning continuity"],
        },
        {
            "priority": 3,
            "block": "P4_online_brain_causal_execution",
            "goal": "把 brain-side causal closure 从状态机推进到在线执行",
            "expected_impact": ["brain_side_causal_closure", "theorem survival", "intervention realism"],
        },
        {
            "priority": 4,
            "block": "stress_bridge_strict_survival",
            "goal": "把 stress/bridge theorems 拉进 strict frontier",
            "expected_impact": ["P2 closure", "B-line closure", "ICSPB strict core expansion"],
        },
    ]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_dnn_extraction_improvement_plan",
        },
        "current_missing_axes": missing_axes,
        "current_priority_blocks": priority_blocks,
        "improvement_plan": plan,
        "frontier_context": frontier["projected_status"],
        "verdict": {
            "core_answer": (
                "Improvement should not focus on adding more local probes; it should focus on upgrading extraction into a four-block system that closes long-chain traces, protocol bridge, brain-side execution, and stress/bridge survival together."
            ),
            "next_target": (
                "Treat extraction as a system design problem rather than a metric accumulation problem."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
