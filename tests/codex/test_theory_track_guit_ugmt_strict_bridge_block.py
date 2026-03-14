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
    functor = load_json(TEMP / "guit_ugmt_functorial_bridge_block.json")
    relation = load_json(TEMP / "guit_ugmt_relation_assessment.json")
    universe = load_json(TEMP / "intelligence_ugmt_fundamental_relation_assessment.json")
    gauge = load_json(TEMP / "theory_track_gauge_canonical_witness_block_20260314.json")

    bridge_score = float(functor["bridge"]["scores"]["bridge_score"])
    relation_score = float(relation["headline_metrics"]["assessment_score"])
    universe_score = float(universe["headline_metrics"]["assessment_score"])
    gauge_score = float(gauge["headline_metrics"]["strengthened_score"])

    strict_bridge_score = clamp01(
        0.30 * bridge_score
        + 0.24 * relation_score
        + 0.28 * universe_score
        + 0.18 * gauge_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_GUIT_UGMT_Strict_Bridge_Block",
        },
        "headline_metrics": {
            "functorial_bridge_score": bridge_score,
            "relation_score": relation_score,
            "universe_score": universe_score,
            "gauge_score": gauge_score,
            "strict_bridge_score": strict_bridge_score,
        },
        "verdict": {
            "overall_pass": strict_bridge_score >= 0.90,
            "strict_pass": strict_bridge_score >= 0.95,
            "core_answer": (
                "A stricter GUIT-UGMT bridge needs projection, relation, universe-law, and canonical-parameter evidence to move together."
            ),
        },
    }

    out_file = TEMP / "guit_ugmt_strict_bridge_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
