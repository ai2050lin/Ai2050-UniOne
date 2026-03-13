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
    ap = argparse.ArgumentParser(description="Breakthrough route progress summary")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_breakthrough_route_progress_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load("stage_protocol_successor_breakthrough_block_20260313.json")
    frontier = load("theory_track_protocol_successor_breakthrough_frontier_20260313.json")
    overall = load("theory_track_encoding_math_progress_overall_20260313.json")

    inverse = float(block["current_state"]["encoding_inverse_reconstruction_readiness"])
    math_ready = float(block["current_state"]["new_math_closure_readiness"])
    gain = float(block["breakthrough_projection"]["gain_vs_current"])
    successor = float(block["breakthrough_projection"]["successor_coherence"])
    protocol = float(block["breakthrough_projection"]["protocol_calling"])
    brain = float(block["breakthrough_projection"]["brain_side_causal_closure"])

    encoding_progress = clamp01(0.58 * inverse + 0.14 * successor + 0.14 * protocol + 0.14 * brain + 0.20 * gain)
    math_progress = clamp01(0.68 * math_ready + 0.12 * (len(frontier["strict_core_if_executed"]) / 8.0) + 0.10 * protocol + 0.10 * successor)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_breakthrough_route_progress",
        },
        "progress": {
            "encoding_mechanism_readiness_projected": encoding_progress,
            "new_math_system_readiness_projected": math_progress,
            "strict_core_if_executed": len(frontier["strict_core_if_executed"]),
            "current_route_fundamentally_blocked": overall["progress"]["current_route_fundamentally_blocked"],
            "current_route_currently_insufficient": overall["progress"]["current_route_currently_insufficient"],
        },
        "verdict": {
            "core_answer": (
                "The current route can still break through, but only if it is upgraded into a protocol-successor-brain integrated block. Without that upgrade, local improvements will continue to saturate."
            ),
            "next_step": (
                "Execute the integrated breakthrough block first, then immediately pull stress and bridge theorems into the strict frontier."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
