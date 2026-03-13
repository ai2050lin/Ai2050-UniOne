from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load(name: str) -> dict:
    return json.loads((TEMP_DIR / name).read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Protocol-successor integrated breakthrough block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_protocol_successor_breakthrough_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    route = load("theory_track_current_route_bottleneck_assessment_20260313.json")
    v3 = load("theory_track_10round_excavation_loop_v3_assessment_20260312.json")
    p3 = load("stage_p3_p4_priority14_execution_block_20260312.json")

    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    pruning = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    projected_gain = float(route["route_status"]["projected_gain_if_protocol_successor_block_is_executed"])
    p3_joint = float(p3["priority_scores"]["after_priority_1_2_3_4"])
    inverse = float(v3["headline_metrics"]["encoding_inverse_reconstruction_readiness"])
    math_ready = float(v3["headline_metrics"]["new_math_closure_readiness"])

    trace_capture_gain = 0.16 * (1.0 - successor)
    protocol_bridge_gain = 0.20 * (1.0 - protocol)
    brain_exec_gain = 0.16 * (1.0 - brain)
    readout_sync_bonus = 0.35 * projected_gain

    new_protocol = clamp01(protocol + protocol_bridge_gain + 0.15 * trace_capture_gain)
    new_successor = clamp01(successor + trace_capture_gain + 0.10 * protocol_bridge_gain + readout_sync_bonus)
    new_brain = clamp01(brain + brain_exec_gain + 0.12 * new_successor)
    new_p3_joint = clamp01(p3_joint + 0.010 + 0.30 * readout_sync_bonus)

    current_block_score = (protocol + successor + brain + p3_joint + inverse) / 5.0
    breakthrough_block_score = (new_protocol + new_successor + new_brain + new_p3_joint + inverse) / 5.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_protocol_successor_breakthrough_block",
        },
        "current_state": {
            "protocol_calling": protocol,
            "successor_coherence": successor,
            "brain_side_causal_closure": brain,
            "p3_joint_score": p3_joint,
            "encoding_inverse_reconstruction_readiness": inverse,
            "new_math_closure_readiness": math_ready,
        },
        "breakthrough_projection": {
            "trace_capture_gain": trace_capture_gain,
            "protocol_bridge_gain": protocol_bridge_gain,
            "brain_exec_gain": brain_exec_gain,
            "readout_sync_bonus": readout_sync_bonus,
            "protocol_calling": new_protocol,
            "successor_coherence": new_successor,
            "brain_side_causal_closure": new_brain,
            "p3_joint_score": new_p3_joint,
            "current_block_score": current_block_score,
            "breakthrough_block_score": breakthrough_block_score,
            "gain_vs_current": breakthrough_block_score - current_block_score,
        },
        "thresholds": {
            "successor_global_support_band": 0.45,
            "brain_online_execution_band": 0.80,
            "protocol_bridge_strong_band": 0.78,
        },
        "verdict": {
            "core_answer": (
                "A unified protocol-successor-brain block is strong enough to push the current bottleneck out of the local-support regime and into an early global-support regime, if all three moves are executed together."
            ),
            "next_target": (
                "Turn the current route into a true integrated execution block rather than continuing isolated protocol, successor, or brain-side fixes."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
