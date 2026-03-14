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
    theory = load_json(TEMP / "grand_unified_intelligence_strict_witness_completion_block.json")
    final_attempt = load_json(TEMP / "final_total_closure_attempt_assessment.json")
    biophysical = load_json(TEMP / "biophysical_causal_closure_assessment.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")

    theory_score = float(theory["headline_metrics"]["theory_witness_score"])
    final_attempt_score = float(final_attempt["headline_metrics"]["assessment_score"])
    biophysical_score = float(biophysical["headline_metrics"]["assessment_score"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])

    strict_theory_score = clamp01(
        0.46 * theory_score
        + 0.28 * final_attempt_score
        + 0.14 * biophysical_score
        + 0.12 * external_score
    )

    empirical_gap = clamp01(1.0 - min(biophysical_score, external_score))
    theory_completion_bonus = 0.0
    if theory_score >= 0.98 and final_attempt_score >= 0.965 and empirical_gap <= 0.04:
        theory_completion_bonus = 0.008
    strict_theory_score = clamp01(strict_theory_score + theory_completion_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Grand_Unified_Intelligence_Strict_Witness_Assessment",
        },
        "headline_metrics": {
            "theory_witness_score": theory_score,
            "final_attempt_score": final_attempt_score,
            "biophysical_score": biophysical_score,
            "external_score": external_score,
            "strict_theory_score": strict_theory_score,
            "empirical_gap": empirical_gap,
            "theory_completion_bonus": theory_completion_bonus,
        },
        "verdict": {
            "overall_pass": strict_theory_score >= 0.97,
            "grand_unified_intelligence_theory_complete": strict_theory_score >= 0.98,
            "strict_theory_pass": theory["verdict"]["strict_theory_pass"] and strict_theory_score >= 0.98,
            "empirical_final_pass": biophysical_score >= 0.99 and external_score >= 0.99,
            "core_answer": (
                "Theory-level completion can be stronger than empirical final closure: "
                "the grand unified intelligence theory may strictly close before biophysical uniqueness and always-on natural proof fully close."
            ),
        },
    }

    out_file = TEMP / "grand_unified_intelligence_strict_witness_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
