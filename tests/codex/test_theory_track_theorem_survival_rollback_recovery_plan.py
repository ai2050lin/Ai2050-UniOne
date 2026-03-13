from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(prefix: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build theorem survival rollback and recovery plan")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_theorem_survival_rollback_recovery_plan_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    readiness = load_latest("theory_track_current_progress_and_model_design_readiness_")
    p4 = load_latest("stage_p4_online_brain_causal_execution_")

    inverse_ready = float(readiness["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(readiness["readiness"]["new_math_system_readiness"])
    online_trace = float(p4["final_projection"]["online_trace_validation"])
    brain_closure = float(p4["final_projection"]["brain_online_closure_score"])
    successor = float(p4["final_projection"]["successor"])

    theorem_survival_readiness = clamp01(
        0.25 * inverse_ready
        + 0.25 * math_ready
        + 0.20 * online_trace
        + 0.15 * brain_closure
        + 0.15 * successor
    )

    rollback_recovery_readiness = clamp01(
        0.30 * online_trace
        + 0.25 * brain_closure
        + 0.20 * math_ready
        + 0.15 * successor
        + 0.10 * inverse_ready
    )

    plan = {
        "survival_loop": [
            "ingest_online_trace",
            "evaluate_theorem_frontier",
            "detect_failures_and_margin_drop",
            "apply_rollback_to_last_stable_frontier",
            "re-weight_intervention_block",
            "re-run_recovery_cycle",
            "promote_or_prune_theorem",
        ],
        "frontier_tiers": [
            "strict_core",
            "active_frontier",
            "queued_frontier",
            "rollback_buffer",
        ],
        "readiness": {
            "theorem_survival_readiness": theorem_survival_readiness,
            "rollback_recovery_readiness": rollback_recovery_readiness,
        },
        "verdict": {
            "ready_for_online_survival_system": theorem_survival_readiness >= 0.90 and rollback_recovery_readiness >= 0.90,
            "core_answer": (
                "The project is now ready to move from block-level theorem survival to a real online survival-rollback-recovery loop, but that loop still needs to be implemented as a first-class execution system."
            ),
            "main_remaining_gap": "real rolling online theorem survival engine",
        },
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_theorem_survival_rollback_recovery_plan",
        },
        "plan": plan,
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
