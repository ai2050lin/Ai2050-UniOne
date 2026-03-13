from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(prefix: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess cross-model real long-chain trace capture")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_cross_model_real_long_chain_trace_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load_latest("stage_cross_model_real_long_chain_trace_capture_")
    cur = block["current_state"]
    final = block["final_projection"]
    passes = block["pass_status"]
    failed: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_cross_model_real_long_chain_trace_assessment",
        },
        "current_to_final": {
            "protocol": [float(cur["protocol"]), float(final["protocol"])],
            "successor": [float(cur["successor"]), float(final["successor"])],
            "brain": [float(cur["brain"]), float(final["brain"])],
            "online_trace_validation": [float(cur["online_trace_validation"]), float(final["online_trace_validation"])],
            "theorem_survival_recovery": [float(cur["theorem_survival_recovery"]), float(final["theorem_survival_recovery"])],
        },
        "headline_metrics": {
            "capture_score": float(final["capture_score"]),
            "protocol_gain": float(final["protocol"] - cur["protocol"]),
            "successor_gain": float(final["successor"] - cur["successor"]),
            "brain_gain": float(final["brain"] - cur["brain"]),
        },
        "pass_status": {
            **passes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "failed_axes": failed,
        },
        "verdict": {
            "route_ready_for_next_protocol_brain_block": all(bool(v) for v in passes.values()),
            "main_remaining_gap": "protocol_bridge_transport_online_execution" if all(bool(v) for v in passes.values()) else "cross_model_trace_still_needs_hardening",
            "core_answer": (
                "The capture block is useful only if it materially improves successor and theorem survival together; otherwise it should be reinforced before being used as the default upstream source."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
