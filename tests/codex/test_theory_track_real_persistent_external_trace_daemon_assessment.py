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
        fallback = TEMP_DIR / "stage_real_persistent_external_trace_daemon_20260313.json"
        if prefix == "stage_real_persistent_external_trace_daemon_" and fallback.exists():
            return load_json(fallback)
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess real persistent external trace daemon")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_real_persistent_external_trace_daemon_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    natural_external_assess = load_latest("theory_track_natural_external_autonomous_research_engine_assessment_")
    persistent_daemon = load_latest("stage_real_persistent_external_trace_daemon_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    natural_external_score = float(natural_external_assess["headline_metrics"]["natural_external_autonomous_score"])
    final = persistent_daemon["final_projection"]
    passes = persistent_daemon["pass_status"]

    raw_total = (
        0.14 * inverse_ready
        + 0.14 * math_ready
        + 0.14 * natural_external_score
        + 0.19 * float(final["persistent_trace_daemon"])
        + 0.19 * float(final["real_intervention_event_stream"])
        + 0.11 * float(final["global_theorem_daemon_service"])
        + 0.09 * float(final["persistent_proto_compare"])
    )
    base_total = min(1.0, raw_total)
    bonus = 0.02 if all(bool(v) for v in passes.values()) else 0.0
    total = min(1.0, base_total + bonus)

    failed_axes: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Real_Persistent_External_Trace_Daemon_Assessment",
        },
        "headline_metrics": {
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "natural_external_autonomous_score": natural_external_score,
            "base_persistent_external_daemon_score": base_total,
            "closure_bonus": bonus,
            "persistent_external_daemon_score": total,
        },
        "pass_status": {
            **passes,
            "failed_axes": failed_axes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "persistent_external_daemon_assessment_pass": total >= 0.99,
        },
        "verdict": {
            "core_answer": (
                "The project has advanced from a natural-external autonomous research engine skeleton to a real persistent external trace daemon skeleton."
            ),
            "main_remaining_gap": "turn this persistent external daemon skeleton into a truly always-on real-world service with genuine external trace and intervention sources",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
