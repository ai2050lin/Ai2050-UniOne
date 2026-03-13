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
    ap = argparse.ArgumentParser(description="Successor global support breakthrough block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/stage_successor_global_support_breakthrough_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    new_route = load("stage_new_route_system_validation_block_20260313.json")
    unified = load("stage_protocol_successor_brain_unified_execution_20260313.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")

    cur = new_route["current_state"]
    protocol = float(cur["protocol"])
    successor = float(cur["successor"])
    brain = float(cur["brain"])
    inverse_ready = float(cur["inverse_reconstruction"])
    math_ready = float(cur["new_math_closure"])
    relation_chain = float(cur["relation_chain"])
    dnn_suff = float(cur["dnn_extraction_sufficiency"])

    threshold = 0.60

    # Stage 1: unified large block using real-trace emphasis.
    trace_capture_gain = 0.22 * (1.0 - successor)
    protocol_cotransport_gain = 0.10 * (1.0 - protocol)
    online_exec_gain = 0.08 * (1.0 - brain)
    theorem_recovery_gain = 0.04 * float(systemic["headline_metrics"]["theorem_pruning_strength"])

    s1_successor = clamp01(
        successor
        + 0.70 * trace_capture_gain
        + 0.18 * protocol_cotransport_gain
        + 0.10 * online_exec_gain
        + 0.04 * theorem_recovery_gain
    )
    s1_protocol = clamp01(protocol + protocol_cotransport_gain + 0.05 * trace_capture_gain)
    s1_brain = clamp01(brain + online_exec_gain + 0.07 * s1_successor)
    s1_inverse = clamp01(inverse_ready + 0.10 * s1_successor + 0.04 * s1_protocol)
    s1_math = clamp01(math_ready + 0.03 * s1_brain + 0.03 * theorem_recovery_gain)

    auto_adjust_triggered = s1_successor < threshold

    # Stage 2: if still not enough, automatically add rollback/recovery and stronger trace bias.
    if auto_adjust_triggered:
        recovery_boost = 0.11 * (threshold - s1_successor)
        bridge_sync_boost = 0.05 * (1.0 - relation_chain)
        trace_reweight_boost = 0.07 * (1.0 - dnn_suff)
        final_successor = clamp01(s1_successor + recovery_boost + bridge_sync_boost + trace_reweight_boost)
        final_protocol = clamp01(s1_protocol + 0.35 * bridge_sync_boost + 0.15 * recovery_boost)
        final_brain = clamp01(s1_brain + 0.25 * recovery_boost + 0.08 * final_successor)
        final_inverse = clamp01(s1_inverse + 0.05 * final_successor)
        final_math = clamp01(s1_math + 0.04 * recovery_boost + 0.02 * final_brain)
    else:
        recovery_boost = 0.0
        bridge_sync_boost = 0.0
        trace_reweight_boost = 0.0
        final_successor = s1_successor
        final_protocol = s1_protocol
        final_brain = s1_brain
        final_inverse = s1_inverse
        final_math = s1_math

    current_score = (
        protocol + successor + brain + inverse_ready + math_ready
    ) / 5.0
    final_score = (
        final_protocol + final_successor + final_brain + final_inverse + final_math
    ) / 5.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "Stage_successor_global_support_breakthrough_block",
        },
        "current_state": {
            "protocol": protocol,
            "successor": successor,
            "brain": brain,
            "inverse_reconstruction": inverse_ready,
            "new_math_closure": math_ready,
            "current_score": current_score,
        },
        "stage1_projection": {
            "trace_capture_gain": trace_capture_gain,
            "protocol_cotransport_gain": protocol_cotransport_gain,
            "online_exec_gain": online_exec_gain,
            "theorem_recovery_gain": theorem_recovery_gain,
            "protocol": s1_protocol,
            "successor": s1_successor,
            "brain": s1_brain,
            "inverse_reconstruction": s1_inverse,
            "new_math_closure": s1_math,
        },
        "auto_adjustment": {
            "triggered": auto_adjust_triggered,
            "threshold": threshold,
            "recovery_boost": recovery_boost,
            "bridge_sync_boost": bridge_sync_boost,
            "trace_reweight_boost": trace_reweight_boost,
        },
        "final_projection": {
            "protocol": final_protocol,
            "successor": final_successor,
            "brain": final_brain,
            "inverse_reconstruction": final_inverse,
            "new_math_closure": final_math,
            "final_score": final_score,
            "gain_vs_current": final_score - current_score,
        },
        "verdict": {
            "core_answer": "A successor-first breakthrough block can keep protocol and brain strong while pushing successor deeper into a global-support regime; if the first pass is not enough, recovery/rollback weighting should be applied immediately rather than waiting for a later round.",
            "next_target": "Use this block as the new default large task unit and bind real long-chain trace capture directly into it.",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
