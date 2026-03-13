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
    ap = argparse.ArgumentParser(description="Assess P4 online brain causal execution block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_p4_online_brain_causal_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load_latest("stage_p4_online_brain_causal_execution_")
    cur = block["current_state"]
    final = block["final_projection"]
    passes = block["pass_status"]
    failed: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_P4_online_brain_causal_assessment",
        },
        "current_to_final": {
            "protocol": [float(cur["protocol"]), float(final["protocol"])],
            "successor": [float(cur["successor"]), float(final["successor"])],
            "brain": [float(cur["brain"]), float(final["brain"])],
            "online_trace_validation": [float(cur["online_trace_validation"]), float(final["online_trace_validation"])],
            "theorem_survival_recovery": [float(cur["theorem_survival_recovery"]), float(final["theorem_survival_recovery"])],
        },
        "headline_metrics": {
            "current_score": float(cur["current_score"]),
            "final_score": float(final["final_score"]),
            "brain_online_closure_score": float(final["brain_online_closure_score"]),
            "gain_vs_current": float(final["gain_vs_current"]),
        },
        "pass_status": {
            **passes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "failed_axes": failed,
        },
        "verdict": {
            "route_ready_for_global_survival_system": all(bool(v) for v in passes.values()),
            "main_remaining_gap": (
                "theorem_survival_rollback_and_recovery"
                if all(bool(v) for v in passes.values())
                else "P4_online_brain_causal_execution_still_needs_hardening"
            ),
            "core_answer": (
                "P4 is only truly online when probe execution, trace validation, theorem survival, and causal intervention are fused into the same block."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
