from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    start = time.time()
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    bio = load_json(TEMP / "biophysical_causal_closure_assessment.json")
    constructive = load_json(TEMP / "theory_track_constructive_parameter_theory_final_closure_20260313.json")

    replay_score = float(replay["headline_metrics"]["assessment_score"])
    bio_score = float(bio["headline_metrics"]["assessment_score"])
    stability_score = float(constructive["headline_metrics"]["online_survival_stability_score"])
    rollback_score = float(constructive["headline_metrics"]["rollback_recovery_correctness_score"])

    action_score = clamp01(
        0.34 * replay_score
        + 0.24 * bio_score
        + 0.22 * stability_score
        + 0.20 * rollback_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Admissible_Path_Action_Principle_Block",
        },
        "headline_metrics": {
            "replay_score": replay_score,
            "biophysical_score": bio_score,
            "stability_score": stability_score,
            "rollback_score": rollback_score,
            "action_score": action_score,
        },
        "verdict": {
            "overall_pass": action_score >= 0.90,
            "strict_pass": action_score >= 0.95,
            "core_answer": (
                "Replay, reasoning, online update, and recovery are better viewed as one admissible-path action problem, "
                "not as separate heuristics."
            ),
        },
    }

    out_file = TEMP / "admissible_path_action_principle_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
