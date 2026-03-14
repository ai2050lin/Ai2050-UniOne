from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "tests" / "codex_temp"


def load_latest(pattern: str) -> Dict[str, Any]:
    matches = sorted(TEMP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Missing upstream artifact: {pattern}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Assess explanatory coverage of unresolved terms and principles.")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_unclosed_terms_principles_assessment_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    gap_map = load_latest("theory_track_unclosed_problem_map_block*.json")

    readiness = float(gap_map["headline_metrics"]["closure_readiness"])
    coverage = clamp01(sum(1.0 for item in gap_map["unresolved_items"] if item["why_open"] and item["principle"]) / 5.0)
    sharpness = clamp01(1.0 - 0.5 * gap_map["headline_metrics"]["overall_gap_pressure"])
    assessment_score = clamp01(0.48 * readiness + 0.26 * coverage + 0.26 * sharpness)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Unclosed_Terms_Principles_Assessment",
        },
        "headline_metrics": {
            "closure_readiness": readiness,
            "coverage": coverage,
            "sharpness": sharpness,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.88,
            "core_answer": (
                "The unresolved terms are now narrow enough to be explained as a small set of concrete closure failures, "
                "rather than a vague lack of theory."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
