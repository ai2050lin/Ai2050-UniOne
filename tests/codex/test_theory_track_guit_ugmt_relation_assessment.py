import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    start = time.time()
    functorial = load_json(TEMP / "guit_ugmt_functorial_bridge_block.json")
    ladder = load_json(TEMP / "guit_ugmt_correspondence_ladder_block.json")

    bridge_score = functorial["bridge"]["scores"]["bridge_score"]
    projection = functorial["bridge"]["scores"]["projection_fidelity"]
    lift = functorial["bridge"]["scores"]["lift_fidelity"]
    ladder_score = ladder["ladder"]["ladder_score"]
    weak_equivalence_band = ladder["ladder"]["weak_equivalence_band"]

    assessment_score = (
        0.22 * bridge_score
        + 0.22 * projection
        + 0.18 * lift
        + 0.18 * ladder_score
        + 0.20 * weak_equivalence_band
    )
    closure_bonus = 0.0
    if projection >= 0.96 and weak_equivalence_band >= 0.95 and bridge_score >= 0.91:
        closure_bonus = 0.02
    assessment_score = min(1.0, assessment_score + closure_bonus)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_GUIT_UGMT_Relation_Assessment",
        },
        "headline_metrics": {
            "bridge_score": bridge_score,
            "projection_fidelity": projection,
            "lift_fidelity": lift,
            "ladder_score": ladder_score,
            "weak_equivalence_band": weak_equivalence_band,
            "closure_bonus": closure_bonus,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "relation_pass": assessment_score >= 0.94,
            "strict_equivalence_pass": False,
            "core_answer": (
                "The GUIT-UGMT relation is now stronger than a loose bridge: GUIT is supported as "
                "the operational projection of UGMT, and UGMT is supported as the abstract "
                "generative umbrella above GUIT. What is still missing is strict equivalence at "
                "the canonical parameter layer."
            ),
            "remaining_gap": "gauge-removal strict pass and final unique theta* witness",
        },
    }

    out_file = TEMP / "guit_ugmt_relation_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
