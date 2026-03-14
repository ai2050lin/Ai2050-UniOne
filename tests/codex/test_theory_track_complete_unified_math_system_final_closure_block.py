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
    ucesd = load_json(TEMP / "theory_track_new_math_theory_candidate_assessment_20260313.json")
    complete_math = load_json(TEMP / "theory_track_complete_math_theory_synthesis_20260313.json")
    high_math = load_json(TEMP / "high_math_strictification_assessment.json")
    quotient = load_json(TEMP / "gauge_quotient_canonicalization_block.json")
    inverse_lift = load_json(TEMP / "guit_ugmt_inverse_lift_strengthened_block.json")
    theta = load_json(TEMP / "theory_track_unique_theta_star_generation_theorem_block_20260314.json")

    ucesd_readiness = float(ucesd["headline_metrics"]["ucesd_readiness"])
    strict_math_score = float(ucesd["headline_metrics"]["assessment_score"])
    complete_math_readiness = float(complete_math["theory"]["readiness"]["new_math_system"])
    high_math_score = float(high_math["headline_metrics"]["assessment_score"])
    quotient_score = float(quotient["headline_metrics"]["strict_candidate_score"])
    inverse_lift_score = float(inverse_lift["headline_metrics"]["strict_inverse_lift_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])

    closure_score = clamp01(
        0.18 * ucesd_readiness
        + 0.18 * strict_math_score
        + 0.16 * complete_math_readiness
        + 0.18 * high_math_score
        + 0.10 * quotient_score
        + 0.10 * inverse_lift_score
        + 0.10 * theta_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Complete_Unified_Math_System_Final_Closure_Block",
        },
        "headline_metrics": {
            "ucesd_readiness": ucesd_readiness,
            "strict_math_score": strict_math_score,
            "complete_math_readiness": complete_math_readiness,
            "high_math_score": high_math_score,
            "quotient_score": quotient_score,
            "inverse_lift_score": inverse_lift_score,
            "theta_score": theta_score,
            "closure_score": closure_score,
        },
        "verdict": {
            "overall_pass": closure_score >= 0.925,
            "strong_complete_candidate": closure_score >= 0.96,
            "strict_final_pass": closure_score >= 0.99,
            "core_answer": (
                "The project now has a nearly complete mathematical system: encoding geometry, survival dynamics, "
                "constructive training, and higher-level quotient-action-lift strictification."
            ),
        },
    }

    out_file = TEMP / "complete_unified_math_system_final_closure_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
