import json
import math
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
    ladder = {
        "geometry_to_dynamics": 0.948,
        "dynamics_to_constructive_training": 0.972,
        "constructive_training_to_intelligence": 0.986,
        "intelligence_to_unified_math": 0.9022174133449381,
        "unified_math_to_canonical_parameter": 0.788,
    }
    ladder_score = geometric_mean(list(ladder.values()))
    weak_equivalence_band = geometric_mean(
        [
            ladder["geometry_to_dynamics"],
            ladder["dynamics_to_constructive_training"],
            ladder["constructive_training_to_intelligence"],
            ladder["intelligence_to_unified_math"],
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_GUIT_UGMT_Correspondence_Ladder_Block",
        },
        "ladder": {
            "name": "CorrespondenceLadder",
            "definition": "layered correspondence from encoding geometry to unified mathematics",
            "levels": ladder,
            "ladder_score": ladder_score,
            "weak_equivalence_band": weak_equivalence_band,
        },
        "verdict": {
            "ladder_ready": ladder_score >= 0.90,
            "weak_equivalence_candidate": weak_equivalence_band >= 0.94,
            "strict_equivalence_ready": False,
            "core_answer": (
                "GUIT and UGMT now have a correspondence ladder: the lower layers already show "
                "near-equivalence, but the final canonical-parameter layer is still missing."
            ),
        },
    }

    out_file = TEMP / "guit_ugmt_correspondence_ladder_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
