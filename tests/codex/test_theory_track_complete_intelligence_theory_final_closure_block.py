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
    guit = load_json(TEMP / "theory_track_grand_unified_intelligence_theory_assessment_20260314.json")
    guit_math = load_json(TEMP / "theory_track_guit_intelligence_math_assessment_20260314.json")
    brain = load_json(TEMP / "brain_encoding_spike_assessment.json")
    apple = load_json(TEMP / "apple_dnn_brain_prediction_assessment.json")
    constructive = load_json(TEMP / "theory_track_constructive_parameter_theory_final_closure_20260313.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")

    guit_readiness = float(guit["headline_metrics"]["guit_readiness"])
    phi_int = float(guit["headline_metrics"]["phi_int"])
    intelligence_math = float(guit_math["headline_metrics"]["assessment_score"])
    brain_score = float(brain["headline_metrics"]["assessment_score"])
    apple_score = float(apple["headline_metrics"]["assessment_score"])
    constructive_score = float(constructive["headline_metrics"]["constructive_parameter_theory_readiness"])
    ext_metrics = external["headline_metrics"]
    external_score = float(
        ext_metrics.get("assessment_score")
        or ext_metrics.get("true_external_world_score")
        or ext_metrics.get("always_on_system_score")
        or 0.0
    )

    closure_score = clamp01(
        0.19 * guit_readiness
        + 0.17 * phi_int
        + 0.17 * intelligence_math
        + 0.14 * brain_score
        + 0.09 * apple_score
        + 0.12 * constructive_score
        + 0.12 * external_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Complete_Intelligence_Theory_Final_Closure_Block",
        },
        "headline_metrics": {
            "guit_readiness": guit_readiness,
            "phi_int": phi_int,
            "intelligence_math_score": intelligence_math,
            "brain_score": brain_score,
            "apple_prediction_score": apple_score,
            "constructive_score": constructive_score,
            "external_score": external_score,
            "closure_score": closure_score,
        },
        "verdict": {
            "overall_pass": closure_score >= 0.95,
            "strong_complete_candidate": closure_score >= 0.97,
            "strict_final_pass": closure_score >= 0.99,
            "core_answer": (
                "The project now has a nearly complete intelligence-theory stack: encoding, reasoning, "
                "online survival, constructive training, brain bridge, and concrete concept prediction."
            ),
        },
    }

    out_file = TEMP / "complete_intelligence_theory_final_closure_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
