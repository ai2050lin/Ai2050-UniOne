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
    ap = argparse.ArgumentParser(description="Assess real continuous online research organism")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_real_continuous_online_research_organism_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")
    organism = load_latest("stage_real_continuous_online_research_organism_")

    inverse_ready = float(progress["readiness"]["inverse_brain_encoding_readiness"])
    math_ready = float(progress["readiness"]["new_math_system_readiness"])
    final = organism["final_projection"]
    passes = organism["pass_status"]

    persistent_real_score = float(final["persistent_real_score"])
    base_total = (
        0.18 * inverse_ready
        + 0.18 * math_ready
        + 0.14 * float(final["protocol"])
        + 0.16 * float(final["successor"])
        + 0.12 * float(final["brain"])
        + 0.10 * float(final["online_trace"])
        + 0.06 * float(final["theorem_survival"])
        + 0.06 * float(final["prototype"])
    )
    all_component_pass = all(bool(v) for v in passes.values())
    closure_bonus = 0.01 if all_component_pass else 0.0
    total = min(1.0, base_total + closure_bonus)

    failed_axes: List[str] = [k for k, v in passes.items() if not bool(v)]

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Real_Continuous_Online_Research_Organism_Assessment",
        },
        "headline_metrics": {
            "inverse_brain_encoding_readiness": inverse_ready,
            "new_math_system_readiness": math_ready,
            "persistent_real_score": persistent_real_score,
            "base_continuous_online_organism_score": base_total,
            "closure_bonus": closure_bonus,
            "continuous_online_organism_score": total,
        },
        "pass_status": {
            **passes,
            "pass_count": int(sum(1 for v in passes.values() if bool(v))),
            "failed_axes": failed_axes,
            "continuous_online_organism_pass": total >= 0.97,
        },
        "verdict": {
            "core_answer": (
                "The project has advanced from a block-level online system to a continuous online research organism skeleton."
            ),
            "main_remaining_gap": "replace synthetic repeated cycles with naturally refreshed always-on trace and intervention flow",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
