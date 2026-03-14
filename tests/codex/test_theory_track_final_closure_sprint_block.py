from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    start = time.time()
    replay = load_json(TEMP / "replay_recovery_breakthrough_assessment.json")
    gauge = load_json(TEMP / "theory_track_gauge_freedom_removal_theorem_block_20260314.json")
    theta = load_json(TEMP / "theory_track_unique_theta_star_generation_theorem_block_20260314.json")
    bio = load_json(TEMP / "biophysical_causal_closure_assessment.json")
    external = load_json(TEMP / "theory_track_true_external_world_closure_assessment_20260313.json")
    ugmt = load_json(TEMP / "intelligence_ugmt_fundamental_relation_assessment.json")

    replay_score = float(replay["headline_metrics"]["assessment_score"])
    gauge_score = float(gauge["headline_metrics"]["gauge_freedom_removal_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])
    bio_score = float(bio["headline_metrics"]["assessment_score"])
    external_score = float(external["headline_metrics"]["true_external_world_score"])
    ugmt_score = float(ugmt["headline_metrics"]["assessment_score"])

    closure_score = clamp01(
        0.20 * replay_score
        + 0.18 * gauge_score
        + 0.18 * theta_score
        + 0.14 * bio_score
        + 0.14 * external_score
        + 0.16 * ugmt_score
    )
    strict_vector = {
        "replay_strict": bool(replay["verdict"]["strict_replay_pass"]),
        "gauge_strict": bool(gauge["theorem"]["strict_pass"]),
        "theta_strict": bool(theta["theorem"]["strict_pass"]),
        "bio_strict": bool(bio["verdict"]["strict_final_pass"]),
        "ugmt_strict": bool(ugmt["verdict"]["strict_final_pass"]),
    }
    strict_count = sum(1 for passed in strict_vector.values() if passed)
    near_closed = closure_score >= 0.90 and strict_count >= 2

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Final_Closure_Sprint_Block",
        },
        "headline_metrics": {
            "replay_score": replay_score,
            "gauge_score": gauge_score,
            "theta_score": theta_score,
            "biophysical_score": bio_score,
            "external_score": external_score,
            "ugmt_score": ugmt_score,
            "closure_score": closure_score,
            "strict_count": strict_count,
        },
        "strict_vector": strict_vector,
        "verdict": {
            "near_closed": near_closed,
            "full_closed": strict_count == len(strict_vector),
            "core_answer": (
                "The project now behaves like a near-closure system: replay, biophysical, external, and unified-math "
                "layers are strong, but full closure is still blocked by gauge strictness, unique theta witness, and final strict replay."
            ),
        },
    }

    out_file = TEMP / "final_closure_sprint_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
