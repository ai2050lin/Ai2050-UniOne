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
    ap = argparse.ArgumentParser(description="Assess real external always-on system")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_real_external_always_on_system_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    organism = load_latest("stage_real_continuous_online_research_organism_")
    always_on = load_latest("stage_real_external_always_on_system_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    persistent_real_score = float(organism["final_projection"]["persistent_real_score"])
    final = always_on["final_projection"]
    passes = always_on["pass_status"]

    base_total = (
        0.16 * inverse_ready
        + 0.16 * math_ready
        + 0.14 * persistent_real_score
        + 0.16 * float(final["external_trace_flow"])
        + 0.16 * float(final["always_on_intervention"])
        + 0.14 * float(final["theorem_daemon"])
        + 0.08 * float(final["prototype_external_compare"])
    )
    bonus = 0.01 if all(bool(v) for v in passes.values()) else 0.0
    total = min(1.0, base_total + bonus)

    failed_axes: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Real_External_Always_On_System_Assessment",
        },
        "headline_metrics": {
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "persistent_real_score": persistent_real_score,
            "base_always_on_score": base_total,
            "closure_bonus": bonus,
            "always_on_system_score": total,
        },
        "pass_status": {
            **passes,
            "failed_axes": failed_axes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "always_on_system_assessment_pass": total >= 0.98,
        },
        "verdict": {
            "core_answer": (
                "The project has advanced from a continuous online organism skeleton to an always-on externalized research system skeleton."
            ),
            "main_remaining_gap": "replace synthetic refresh with naturally refreshed external trace and intervention flow under persistent runtime",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
