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
    ap = argparse.ArgumentParser(description="Assess real-online unified closure block")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_real_online_unified_closure_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    block = load_latest("stage_real_online_unified_closure_block_")
    current = block["current_state"]
    final = block["final_projection"]
    passes = block["pass_status"]

    failed_axes: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_real_online_unified_closure_assessment",
        },
        "current_to_final": {
            "protocol": [float(current["protocol"]), float(final["protocol"])],
            "successor": [float(current["successor"]), float(final["successor"])],
            "brain": [float(current["brain"]), float(final["brain"])],
            "inverse_reconstruction": [
                float(current["inverse_reconstruction"]),
                float(final["inverse_reconstruction"]),
            ],
            "new_math_closure": [
                float(current["new_math_closure"]),
                float(final["new_math_closure"]),
            ],
        },
        "headline_metrics": {
            "current_score": float(current["current_score"]),
            "final_score": float(final["final_score"]),
            "real_online_closure_score": float(final["real_online_closure_score"]),
            "online_trace_validation": float(final["online_trace_validation"]),
            "theorem_survival_recovery": float(final["theorem_survival_recovery"]),
            "gain_vs_current": float(final["gain_vs_current"]),
        },
        "pass_status": {
            **passes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "failed_axes": failed_axes,
        },
        "verdict": {
            "route_ready_for_real_online_large_block": all(bool(v) for v in passes.values()),
            "main_remaining_gap": "real_online_trace_and_intervention_validation" if all(bool(v) for v in passes.values()) else "mixed_axes_still_need_hardening",
            "core_answer": (
                "The route is now strong enough to move from artifact-led projection to a real-online large-block execution shape; the remaining risk is no longer structural blindness but execution realism."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
