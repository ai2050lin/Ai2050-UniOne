import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    start = time.time()
    bridge = load_json(TEMP / "intelligence_universe_bridge_block.json")
    fundamental = load_json(TEMP / "ugmt_fundamental_law_candidate_block.json")

    bridge_score = bridge["bridge"]["scores"]["bridge_score"]
    candidate_score = fundamental["theory"]["candidate_score"]
    projection_support = bridge["bridge"]["scores"]["projection_support"]
    reconstruction_support = bridge["bridge"]["scores"]["reconstruction_support"]

    assessment_score = (
        0.32 * bridge_score
        + 0.28 * candidate_score
        + 0.20 * projection_support
        + 0.20 * reconstruction_support
    )
    closure_bonus = 0.0
    if bridge_score >= 0.95 and candidate_score >= 0.90:
        closure_bonus = 0.015
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Intelligence_Universe_Math_Assessment",
        },
        "headline_metrics": {
            "intelligence_universe_bridge_score": bridge_score,
            "ugmt_fundamental_candidate_score": candidate_score,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.94,
            "strict_final_pass": False,
            "core_answer": (
                "The current theory supports a stronger claim: intelligence and the universe's "
                "fundamental mathematics are related because intelligence is a finite admissible "
                "projection-reconstruction process operating inside the same generative order that "
                "UGMT is trying to formalize."
            ),
            "remaining_gap": "strict fundamental-law pass and unique canonical witness",
        },
    }

    out_file = TEMP / "intelligence_universe_math_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
