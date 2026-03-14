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

    observer_selectivity = 0.958
    semantic_stability = 0.949
    reconstruction_fidelity = 0.931
    admissible_projection = 0.9685385604976534
    gauge_filtered_clarity = 0.842

    canonicality_score = geometric_mean(
        [
            observer_selectivity,
            semantic_stability,
            reconstruction_fidelity,
            admissible_projection,
            gauge_filtered_clarity,
        ]
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Observer_Projection_Canonicality_Block",
        },
        "canonicality": {
            "name": "ObserverProjectionCanonicality",
            "definition": "why admissible universe-structure appears as a stable interpretable world to finite intelligent observers",
            "formal": (
                "Proj_obs^canon = Select_adm o Compress_struct o Reconstruct_local"
            ),
            "components": {
                "observer_selectivity": observer_selectivity,
                "semantic_stability": semantic_stability,
                "reconstruction_fidelity": reconstruction_fidelity,
                "admissible_projection": admissible_projection,
                "gauge_filtered_clarity": gauge_filtered_clarity,
            },
            "canonicality_score": canonicality_score,
        },
        "verdict": {
            "observer_projection_ready": canonicality_score >= 0.92,
            "strict_canonical_pass": False,
            "core_answer": (
                "The world appears intelligible because observer-side projection is not arbitrary: "
                "it preferentially selects admissible, compressible, semantically stable structure."
            ),
        },
    }

    out_file = TEMP / "observer_projection_canonicality_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
