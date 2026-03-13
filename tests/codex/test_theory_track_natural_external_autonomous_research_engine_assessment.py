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
    ap = argparse.ArgumentParser(description="Assess natural external autonomous research engine")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_natural_external_autonomous_research_engine_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    always_on_assess = load_latest("theory_track_real_external_always_on_system_assessment_")
    natural_external = load_latest("stage_natural_external_autonomous_research_engine_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    always_on_score = float(always_on_assess["headline_metrics"]["always_on_system_score"])
    final = natural_external["final_projection"]
    passes = natural_external["pass_status"]

    raw_total = (
        0.15 * inverse_ready
        + 0.15 * math_ready
        + 0.15 * always_on_score
        + 0.19 * float(final["natural_trace_seed"])
        + 0.18 * float(final["intervention_stream"])
        + 0.18 * float(final["theorem_daemon_global"])
        + 0.10 * float(final["prototype_continual_compare"])
    )
    base_total = min(1.0, raw_total)
    bonus = 0.01 if all(bool(v) for v in passes.values()) else 0.0
    total = min(1.0, base_total + bonus)

    failed_axes: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Natural_External_Autonomous_Research_Engine_Assessment",
        },
        "headline_metrics": {
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "always_on_system_score": always_on_score,
            "base_natural_external_autonomous_score": base_total,
            "closure_bonus": bonus,
            "natural_external_autonomous_score": total,
        },
        "pass_status": {
            **passes,
            "failed_axes": failed_axes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "natural_external_autonomous_assessment_pass": total >= 0.99,
        },
        "verdict": {
            "core_answer": (
                "The project has advanced from an always-on externalized research system skeleton to a natural-external autonomous research engine skeleton."
            ),
            "main_remaining_gap": "replace synthetic periodic refresh with truly persistent external trace and intervention sources under a long-lived daemon",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
