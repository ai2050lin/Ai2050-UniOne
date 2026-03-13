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
    ap = argparse.ArgumentParser(description="Overall progress on encoding mechanism and new math")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_encoding_math_progress_overall_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    v3 = load("theory_track_10round_excavation_loop_v3_assessment_20260312.json")
    route = load("theory_track_current_route_bottleneck_assessment_20260313.json")
    systemic = load("stage_systemic_closure_master_block_20260312.json")
    pruning = load("theory_track_systemic_inventory_master_pruning_20260312.json")

    inverse = float(v3["headline_metrics"]["encoding_inverse_reconstruction_readiness"])
    math_ready = float(v3["headline_metrics"]["new_math_closure_readiness"])
    successor = float(systemic["headline_metrics"]["successor_coherence"])
    protocol = float(systemic["headline_metrics"]["protocol_calling"])
    brain = float(systemic["headline_metrics"]["brain_side_causal_closure"])
    theorem_strength = float(systemic["headline_metrics"]["theorem_pruning_strength"])
    preserved_theorem_count = len(pruning["master_pruning"]["preserved_theorems"])

    bottleneck_penalty = (1.0 - successor) * 0.20 + (1.0 - protocol) * 0.12 + (1.0 - brain) * 0.12
    encoding_progress = clamp01(0.62 * inverse + 0.18 * theorem_strength + 0.10 * protocol + 0.10 * brain - bottleneck_penalty)
    math_progress = clamp01(0.70 * math_ready + 0.15 * theorem_strength + 0.10 * (preserved_theorem_count / 8.0) + 0.05 * protocol - 0.08 * (1.0 - successor))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_encoding_math_progress_overall",
        },
        "progress": {
            "encoding_mechanism_readiness": encoding_progress,
            "new_math_system_readiness": math_progress,
            "strict_theorem_like_count": preserved_theorem_count,
            "current_route_fundamentally_blocked": route["route_status"]["current_route_is_fundamentally_blocked"],
            "current_route_currently_insufficient": route["route_status"]["current_route_is_currently_insufficient"],
        },
        "main_open_questions": [
            "successor 如何从 local theorem-support 变成 global system-support",
            "protocol bridge 如何让 object/readout/successor 真正贯通",
            "brain-side causal closure 如何进入在线执行而不是状态机层",
            "stress_guarded_update 与 anchored_bridge_lift 如何进入 strict survival frontier",
        ],
        "verdict": {
            "core_answer": (
                "Current progress is already in the mid-to-late stage for inverse reconstruction of the encoding mechanism, but still only in the middle-to-late stage for strict closure of the new mathematical system."
            ),
            "next_step": (
                "Concentrate on the protocol-successor-brain block first, then pull stress/bridge theorems into the strict frontier."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
