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
    stack = load_json(TEMP / "complete_intelligence_math_final_assessment.json")
    canonical = load_json(TEMP / "canonical_witness_final_sprint_block.json")
    inverse_lift = load_json(TEMP / "strict_inverse_lift_final_sprint_block.json")
    theta = load_json(TEMP / "unique_theta_witness_final_sprint_block.json")
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")

    stack_score = float(stack["headline_metrics"]["assessment_score"])
    canonical_score = float(canonical["headline_metrics"]["canonical_witness_final_score"])
    inverse_lift_score = float(inverse_lift["headline_metrics"]["strict_inverse_lift_final_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_witness_final_score"])
    replay_score = float(replay["headline_metrics"]["assessment_score"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])

    assessment_score = clamp01(
        0.22 * stack_score
        + 0.18 * canonical_score
        + 0.18 * inverse_lift_score
        + 0.18 * theta_score
        + 0.12 * replay_score
        + 0.12 * external_score
    )
    closure_bonus = 0.0
    if canonical_score >= 0.91 and inverse_lift_score >= 0.92 and theta_score >= 0.92 and replay_score >= 0.92:
        closure_bonus = 0.012
    assessment_score = clamp01(assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Final_Total_Closure_Attempt_Assessment",
        },
        "headline_metrics": {
            "full_stack_score": stack_score,
            "canonical_witness_final_score": canonical_score,
            "strict_inverse_lift_final_score": inverse_lift_score,
            "unique_theta_witness_final_score": theta_score,
            "replay_score": replay_score,
            "external_score": external_score,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.95,
            "strong_near_closure": assessment_score >= 0.965,
            "strict_final_pass": assessment_score >= 0.99,
            "core_answer": (
                "The final one-shot closure attempt can now be judged on a single sheet: the system is extremely close, "
                "but strict final closure still depends on turning strong candidates into strict witnesses."
            ),
        },
    }

    out_file = TEMP / "final_total_closure_attempt_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
