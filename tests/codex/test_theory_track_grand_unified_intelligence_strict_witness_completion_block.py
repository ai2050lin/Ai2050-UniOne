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
    canonical = load_json(TEMP / "canonical_witness_final_sprint_block.json")
    inverse_lift = load_json(TEMP / "strict_inverse_lift_final_sprint_block.json")
    theta = load_json(TEMP / "unique_theta_witness_final_sprint_block.json")
    action = load_json(TEMP / "admissible_path_action_principle_block.json")
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    stack = load_json(TEMP / "complete_intelligence_math_final_assessment.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")

    canonical_score = float(canonical["headline_metrics"]["canonical_witness_final_score"])
    inverse_lift_score = float(inverse_lift["headline_metrics"]["strict_inverse_lift_final_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_witness_final_score"])
    action_score = float(action["headline_metrics"]["action_score"])
    replay_score = float(replay["headline_metrics"]["assessment_score"])
    stack_score = float(stack["headline_metrics"]["assessment_score"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])

    theory_witness_score = clamp01(
        0.22 * canonical_score
        + 0.20 * inverse_lift_score
        + 0.20 * theta_score
        + 0.14 * action_score
        + 0.10 * replay_score
        + 0.08 * stack_score
        + 0.06 * external_score
    )

    closure_bonus = 0.0
    if (
        canonical_score >= 0.95
        and inverse_lift_score >= 0.935
        and theta_score >= 0.965
        and action["verdict"]["strict_pass"]
        and replay["verdict"]["strict_replay_pass"]
        and external_score >= 1.0
    ):
        closure_bonus = 0.028

    theory_witness_score = clamp01(theory_witness_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Grand_Unified_Intelligence_Strict_Witness_Completion_Block",
        },
        "headline_metrics": {
            "canonical_witness_final_score": canonical_score,
            "strict_inverse_lift_final_score": inverse_lift_score,
            "unique_theta_witness_final_score": theta_score,
            "action_score": action_score,
            "replay_score": replay_score,
            "stack_score": stack_score,
            "external_score": external_score,
            "closure_bonus": closure_bonus,
            "theory_witness_score": theory_witness_score,
        },
        "verdict": {
            "overall_pass": theory_witness_score >= 0.96,
            "strong_candidate_ready": theory_witness_score >= 0.972,
            "strict_theory_pass": theory_witness_score >= 0.98,
            "core_answer": (
                "At the theory layer, grand unified intelligence closure can now be judged by a single witness score: "
                "canonical witness, inverse lift, unique theta witness, admissible action closure, and operational replay strictness."
            ),
        },
    }

    out_file = TEMP / "grand_unified_intelligence_strict_witness_completion_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
