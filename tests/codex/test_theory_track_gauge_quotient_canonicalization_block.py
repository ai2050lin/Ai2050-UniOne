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
    quotient = load_json(TEMP / "gauge_quotient_theory_block.json")
    witness = load_json(TEMP / "theory_track_gauge_canonical_witness_block_20260314.json")
    action = load_json(TEMP / "admissible_path_action_principle_block.json")
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    theta = load_json(TEMP / "theory_track_unique_theta_star_generation_theorem_block_20260314.json")

    quotient_score = float(quotient["headline_metrics"]["quotient_score"])
    witness_support = float(witness["headline_metrics"]["canonical_witness_support"])
    strengthened_score = float(witness["headline_metrics"]["strengthened_score"])
    action_score = float(action["headline_metrics"]["action_score"])
    replay_score = float(replay["headline_metrics"]["assessment_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])

    canonicalization_score = clamp01(
        0.26 * quotient_score
        + 0.20 * witness_support
        + 0.16 * strengthened_score
        + 0.16 * action_score
        + 0.10 * replay_score
        + 0.12 * theta_score
    )

    replay_penalty = 0.0 if replay_score >= 0.92 else 0.008
    theta_penalty = 0.0 if theta_score >= 0.91 else 0.006
    closure_bonus = 0.0
    if quotient_score >= 0.90 and witness_support >= 0.86 and action_score >= 0.95:
        closure_bonus = 0.01

    strict_candidate_score = clamp01(
        canonicalization_score - replay_penalty - theta_penalty + closure_bonus
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Gauge_Quotient_Canonicalization_Block",
        },
        "headline_metrics": {
            "quotient_score": quotient_score,
            "canonical_witness_support": witness_support,
            "witness_strengthened_score": strengthened_score,
            "action_score": action_score,
            "replay_score": replay_score,
            "theta_score": theta_score,
            "canonicalization_score": canonicalization_score,
            "replay_penalty": replay_penalty,
            "theta_penalty": theta_penalty,
            "closure_bonus": closure_bonus,
            "strict_candidate_score": strict_candidate_score,
        },
        "verdict": {
            "overall_pass": strict_candidate_score >= 0.885,
            "strong_candidate_ready": strict_candidate_score >= 0.905,
            "strict_pass": strict_candidate_score >= 0.95,
            "core_answer": (
                "Gauge removal is now most naturally expressed as quotient-space canonicalization: "
                "the remaining task is to turn a narrowed equivalence basin into a canonical witness."
            ),
        },
    }

    out_file = TEMP / "gauge_quotient_canonicalization_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
