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
    quotient = load_json(TEMP / "gauge_quotient_canonicalization_block.json")
    action = load_json(TEMP / "admissible_path_action_principle_block.json")
    inverse_lift = load_json(TEMP / "guit_ugmt_inverse_lift_strengthened_block.json")

    quotient_score = float(quotient["headline_metrics"]["strict_candidate_score"])
    action_score = float(action["headline_metrics"]["action_score"])
    inverse_lift_score = float(inverse_lift["headline_metrics"]["strict_inverse_lift_score"])

    assessment_score = clamp01(
        0.34 * quotient_score
        + 0.32 * action_score
        + 0.34 * inverse_lift_score
    )
    closure_bonus = 0.0
    if quotient_score >= 0.885 and action_score >= 0.95 and inverse_lift_score >= 0.91:
        closure_bonus = 0.01
    assessment_score = clamp01(assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_High_Math_Strictification_Assessment",
        },
        "headline_metrics": {
            "quotient_strict_candidate_score": quotient_score,
            "action_score": action_score,
            "inverse_lift_strict_score": inverse_lift_score,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.915,
            "strong_near_closure": assessment_score >= 0.94,
            "strict_final_pass": assessment_score >= 0.96,
            "core_answer": (
                "Higher-level mathematics now pushes the project into a stricter quotient-action-lift regime, "
                "but final closure still depends on canonical witness and external proof."
            ),
        },
    }

    out_file = TEMP / "high_math_strictification_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
