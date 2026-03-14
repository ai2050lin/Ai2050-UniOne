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
        exact = TEMP_DIR / f"{prefix}20260313.json"
        if exact.exists():
            return load_json(exact)
        raise FileNotFoundError(f"missing temp json with prefix: {prefix}")
    return load_json(matches[-1])


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess constructive parameter theory status")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_constructive_parameter_theory_assessment_20260313.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    arch = load_latest("theory_track_architecture_synthesis_theorem_block_")
    init = load_latest("theory_track_parameter_initialization_theorem_block_")
    conv = load_latest("theory_track_admissible_update_convergence_theorem_block_")
    progress = load_latest("theory_track_current_progress_and_model_design_readiness_")

    architecture = float(arch["theorem"]["score"])
    initialization = float(init["theorem"]["score"])
    convergence = float(conv["theorem"]["score"])
    model_ready = float(progress["readiness"]["model_design_readiness"])

    constructive_readiness = clamp01(
        0.35 * architecture
        + 0.25 * initialization
        + 0.25 * convergence
        + 0.15 * model_ready
    )
    deterministic_training_readiness = clamp01(
        0.25 * architecture
        + 0.30 * initialization
        + 0.30 * convergence
        + 0.15 * model_ready
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Constructive_Parameter_Theory_Assessment",
        },
        "headline_metrics": {
            "architecture_synthesis_score": architecture,
            "parameter_initialization_score": initialization,
            "admissible_update_convergence_score": convergence,
            "constructive_parameter_theory_readiness": constructive_readiness,
            "deterministic_training_readiness": deterministic_training_readiness,
        },
        "verdict": {
            "structure_and_constraint_theory_strong": architecture >= 0.90,
            "constructive_parameter_theory_closed": constructive_readiness >= 0.90 and initialization >= 0.90 and convergence >= 0.90,
            "core_answer": (
                "The project now has a strong architecture synthesis theorem, but parameter initialization and admissible update convergence remain partial; training can be made strongly constrained, but not yet fully closed-form or uniquely determined."
            ),
            "main_remaining_gap": "parameter initialization + admissible update convergence",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
