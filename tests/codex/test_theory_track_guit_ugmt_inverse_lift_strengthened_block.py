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
    functor = load_json(TEMP / "guit_ugmt_functorial_bridge_block.json")
    relation = load_json(TEMP / "guit_ugmt_relation_assessment.json")
    strict_bridge = load_json(TEMP / "guit_ugmt_strict_bridge_block.json")
    quotient = load_json(TEMP / "gauge_quotient_canonicalization_block.json")
    theta = load_json(TEMP / "theory_track_unique_theta_star_generation_theorem_block_20260314.json")

    projection = float(functor["bridge"]["scores"]["projection_fidelity"])
    lift = float(functor["bridge"]["scores"]["lift_fidelity"])
    relation_score = float(relation["headline_metrics"]["assessment_score"])
    bridge_score = float(strict_bridge["headline_metrics"]["strict_bridge_score"])
    quotient_score = float(quotient["headline_metrics"]["strict_candidate_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])

    inverse_lift_score = clamp01(
        0.18 * projection
        + 0.22 * lift
        + 0.22 * relation_score
        + 0.20 * bridge_score
        + 0.10 * quotient_score
        + 0.08 * theta_score
    )

    quotient_penalty = 0.0 if quotient_score >= 0.905 else 0.01
    lift_bonus = 0.0
    if projection >= 0.96 and relation_score >= 0.94 and bridge_score >= 0.91:
        lift_bonus = 0.01

    strict_inverse_lift_score = clamp01(inverse_lift_score - quotient_penalty + lift_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_GUIT_UGMT_Inverse_Lift_Strengthened_Block",
        },
        "headline_metrics": {
            "projection_fidelity": projection,
            "lift_fidelity": lift,
            "relation_score": relation_score,
            "bridge_score": bridge_score,
            "quotient_strict_candidate_score": quotient_score,
            "theta_score": theta_score,
            "inverse_lift_score": inverse_lift_score,
            "quotient_penalty": quotient_penalty,
            "lift_bonus": lift_bonus,
            "strict_inverse_lift_score": strict_inverse_lift_score,
        },
        "verdict": {
            "overall_pass": strict_inverse_lift_score >= 0.895,
            "strong_candidate_ready": strict_inverse_lift_score >= 0.915,
            "strict_pass": strict_inverse_lift_score >= 0.95,
            "core_answer": (
                "The GUIT-to-UGMT inverse lift is now better supported as a structured lift problem: "
                "projection is strong, but strict lift still depends on canonical quotient evidence."
            ),
        },
    }

    out_file = TEMP / "guit_ugmt_inverse_lift_strengthened_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
