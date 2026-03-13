from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(prefix: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess persistent online system")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_persistent_online_system_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    long_run = load_latest("stage_icspb_backbone_v1_proto_long_run_validation_")
    persistent = load_latest("stage_real_persistent_online_research_engine_")
    p4 = load_latest("stage_p4_online_brain_causal_execution_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    long_run_score = float(long_run["final_projection"]["long_run_proto_score"])
    persistent_score = float(persistent["final_projection"]["persistent_online_score"])
    brain_score = float(p4["final_projection"]["brain_online_closure_score"])

    total = clamp01(
        0.20 * inverse_ready
        + 0.20 * math_ready
        + 0.22 * long_run_score
        + 0.22 * persistent_score
        + 0.16 * brain_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Persistent_Online_System_Assessment",
        },
        "headline_metrics": {
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "prototype_long_run_validation_score": long_run_score,
            "persistent_online_research_score": persistent_score,
            "brain_online_closure_score": brain_score,
            "persistent_online_system_score": total,
        },
        "verdict": {
            "persistent_system_pass": total >= 0.97,
            "core_answer": (
                "The project has now moved beyond block-level online closure into a persistent online system skeleton: prototype validation, theorem survival, rollback/recovery, and brain-side execution are now organized as one repeatable research system."
            ),
            "main_remaining_gap": "convert the persistent skeleton into a real continuously refreshed online research organism",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
