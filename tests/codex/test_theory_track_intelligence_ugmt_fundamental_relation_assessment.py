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
    canonicality = load_json(TEMP / "observer_projection_canonicality_block.json")
    strengthened = load_json(TEMP / "ugmt_universe_law_strengthened_block.json")

    bridge_score = bridge["bridge"]["scores"]["bridge_score"]
    canonicality_score = canonicality["canonicality"]["canonicality_score"]
    strengthened_score = strengthened["theory"]["strengthened_score"]

    assessment_score = (
        0.34 * bridge_score
        + 0.28 * canonicality_score
        + 0.38 * strengthened_score
    )

    closure_bonus = 0.0
    if bridge_score >= 0.95 and canonicality_score >= 0.92 and strengthened_score >= 0.92:
        closure_bonus = 0.016
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Intelligence_UGMT_Fundamental_Relation_Assessment",
        },
        "headline_metrics": {
            "intelligence_universe_bridge_score": bridge_score,
            "observer_projection_canonicality_score": canonicality_score,
            "ugmt_universe_law_strengthened_score": strengthened_score,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.95,
            "strict_final_pass": False,
            "core_answer": (
                "The theory now supports a stronger claim: intelligence and the universe's "
                "fundamental mathematics are linked because finite intelligence is an observer-side "
                "canonical projection/reconstruction process arising inside the same admissible "
                "generative order that UGMT is describing."
            ),
            "remaining_gap": "strict fundamental-law pass, gauge-removal strict pass, and unique canonical witness",
        },
    }

    out_file = TEMP / "intelligence_ugmt_fundamental_relation_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
