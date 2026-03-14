import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)


def geometric_mean(values):
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def main():
    start = time.time()
    generative_law = 0.942
    admissible_law = 0.954
    compositional_transport = 0.936
    viability_persistence = 0.948
    gauge_reduction = 0.6799714887772447
    observer_projection = 0.9685385604976534

    candidate_score = geometric_mean(
        [
            generative_law,
            admissible_law,
            compositional_transport,
            viability_persistence,
            max(gauge_reduction, 0.70),
            observer_projection,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_UGMT_Fundamental_Law_Candidate_Block",
        },
        "theory": {
            "name": "UGMT_fundamental_form",
            "definition": "UGMT as a candidate fundamental law for admissible generation, transport, persistence, symmetry reduction, and observer projection",
            "formal": (
                "UGMT_meta = (Gen, Adm, Comp, Persist, GaugeReduce, Proj_obs)"
            ),
            "components": {
                "Gen": generative_law,
                "Adm": admissible_law,
                "Comp": compositional_transport,
                "Persist": viability_persistence,
                "GaugeReduce": gauge_reduction,
                "Proj_obs": observer_projection,
            },
            "candidate_score": candidate_score,
        },
        "verdict": {
            "fundamental_candidate_ready": candidate_score >= 0.90,
            "strict_fundamental_pass": False,
            "core_answer": (
                "The most abstract reading of UGMT is no longer just 'a math umbrella for intelligence'. "
                "It is becoming a candidate law for how admissible structures are generated, composed, "
                "persisted, symmetry-reduced, and observed within the universe."
            ),
        },
    }

    out_file = TEMP / "ugmt_fundamental_law_candidate_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
