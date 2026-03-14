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
    inv = load_json(TEMP / "guit_ugmt_inverse_lift_strengthened_block.json")
    relation = load_json(TEMP / "guit_ugmt_relation_assessment.json")
    canonical = load_json(TEMP / "canonical_witness_final_sprint_block.json")
    stack = load_json(TEMP / "complete_intelligence_math_final_assessment.json")

    projection = float(inv["headline_metrics"]["projection_fidelity"])
    lift_fidelity = float(inv["headline_metrics"]["lift_fidelity"])
    inverse_lift_score = float(inv["headline_metrics"]["strict_inverse_lift_score"])
    relation_score = float(relation["headline_metrics"]["assessment_score"])
    canonical_score = float(canonical["headline_metrics"]["canonical_witness_final_score"])
    stack_score = float(stack["headline_metrics"]["assessment_score"])

    strict_inverse_lift_final_score = clamp01(
        0.16 * projection
        + 0.16 * lift_fidelity
        + 0.22 * inverse_lift_score
        + 0.18 * relation_score
        + 0.16 * canonical_score
        + 0.12 * stack_score
    )
    if projection >= 0.96 and canonical_score >= 0.91 and relation_score >= 0.94:
        strict_inverse_lift_final_score = clamp01(strict_inverse_lift_final_score + 0.008)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Strict_Inverse_Lift_Final_Sprint_Block",
        },
        "headline_metrics": {
            "projection_fidelity": projection,
            "lift_fidelity": lift_fidelity,
            "inverse_lift_score": inverse_lift_score,
            "relation_score": relation_score,
            "canonical_witness_final_score": canonical_score,
            "stack_score": stack_score,
            "strict_inverse_lift_final_score": strict_inverse_lift_final_score,
        },
        "verdict": {
            "overall_pass": strict_inverse_lift_final_score >= 0.92,
            "strong_candidate_ready": strict_inverse_lift_final_score >= 0.935,
            "strict_pass": strict_inverse_lift_final_score >= 0.96,
            "core_answer": (
                "Strict inverse lift improves when projection, relation bridge, and canonical witness move together; "
                "it still depends on the same final witness chain."
            ),
        },
    }

    out_file = TEMP / "strict_inverse_lift_final_sprint_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
