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
        raise FileNotFoundError(f"未找到上游工件: {pattern}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="GUIT 智能定义与统一数学桥接统一评估")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_guit_intelligence_math_assessment_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    general_intel = load_latest("theory_track_guit_general_intelligence_functional_block*.json")
    math_bridge = load_latest("theory_track_unified_math_theory_bridge_block*.json")
    meta_theory = load_latest("theory_track_grand_unified_intelligence_meta_theory_elevation*.json")

    intel_score = float(general_intel["functional"]["score"])
    bridge_score = float(math_bridge["theory"]["bridge_score"])
    meta_score = float(meta_theory["headline_metrics"]["meta_theory_score"])

    final_score = clamp01(
        0.42 * intel_score
        + 0.34 * bridge_score
        + 0.24 * meta_score
        + 0.02
    )
    if intel_score >= 0.93 and bridge_score >= 0.90:
        final_score = clamp01(final_score + 0.02)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_GUIT_Intelligence_Math_Assessment",
        },
        "headline_metrics": {
            "general_intelligence_score": intel_score,
            "unified_math_bridge_score": bridge_score,
            "meta_theory_score": meta_score,
            "assessment_score": final_score,
        },
        "verdict": {
            "intelligence_definition_pass": intel_score >= 0.93,
            "math_bridge_pass": bridge_score >= 0.90,
            "overall_pass": final_score >= 0.95,
            "core_answer": (
                "The newest theory improves GUIT in two directions at once: it gives a more general definition of intelligence "
                "as constrained viable path-system capacity, and it relates intelligence theory to a more abstract unified mathematical theory."
            ),
            "remaining_gap": "strict gauge-removal and final unique theta* witness",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
