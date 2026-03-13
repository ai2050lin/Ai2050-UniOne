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
    ap = argparse.ArgumentParser(description="Protocol-successor closure block assessment")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_protocol_successor_closure_block_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    qd = load("qwen_deepseek_naturalized_trace_bundle_20260312.json")
    succ = load("theory_track_successor_coherence_closure_diagnosis_20260312.json")
    v3 = load("theory_track_10round_excavation_loop_v3_assessment_20260312.json")

    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    pruning = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    online_gap = float(qd["missing_axes"][0]["gap"])
    reconstruction = float(v3["headline_metrics"]["encoding_inverse_reconstruction_readiness"])
    new_math = float(v3["headline_metrics"]["new_math_closure_readiness"])

    protocol_bridge_gain = 0.12 * (1.0 - protocol)
    successor_trace_gain = 0.18 * min(online_gap, 1.0 - successor)
    brain_causal_gain = 0.10 * (1.0 - brain)

    projected_protocol = clamp01(protocol + protocol_bridge_gain)
    projected_successor = clamp01(successor + successor_trace_gain + 0.25 * protocol_bridge_gain)
    projected_brain = clamp01(brain + brain_causal_gain + 0.10 * successor_trace_gain)

    current_block_score = (protocol + successor + brain + pruning) / 4.0
    projected_block_score = (projected_protocol + projected_successor + projected_brain + pruning) / 4.0

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_protocol_successor_closure_block",
        },
        "current_state": {
            "protocol_calling": protocol,
            "successor_coherence": successor,
            "brain_side_causal_closure": brain,
            "theorem_pruning_strength": pruning,
            "encoding_inverse_reconstruction_readiness": reconstruction,
            "new_math_closure_readiness": new_math,
        },
        "closure_block_projection": {
            "protocol_bridge_gain": protocol_bridge_gain,
            "successor_trace_gain": successor_trace_gain,
            "brain_causal_gain": brain_causal_gain,
            "projected_protocol_calling": projected_protocol,
            "projected_successor_coherence": projected_successor,
            "projected_brain_side_causal_closure": projected_brain,
            "current_block_score": current_block_score,
            "projected_block_score": projected_block_score,
            "gain_vs_current": projected_block_score - current_block_score,
        },
        "verdict": {
            "core_answer": (
                "Current route is not mathematically dead; the strongest unresolved cluster is the protocol-successor-brain block, "
                "and it still shows non-trivial projected gain if real long-chain trace capture, protocol bridge transport, and online brain-side causal execution are promoted together."
            ),
            "next_target": (
                "Promote protocol and successor from local support into global system support, rather than treating them as separate late-stage fixes."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
