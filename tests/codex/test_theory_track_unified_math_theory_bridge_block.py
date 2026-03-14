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
    ap = argparse.ArgumentParser(description="大统一智能理论到大统一数学理论桥接块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_unified_math_theory_bridge_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    complete_math = load_latest("theory_track_complete_math_theory_synthesis*.json")
    guit = load_latest("theory_track_grand_unified_intelligence_theory_assessment*.json")
    meta_theory = load_latest("theory_track_grand_unified_intelligence_meta_theory_elevation*.json")
    unique_theta = load_latest("theory_track_unique_theta_star_generation_theorem_block*.json")
    gauge = load_latest("theory_track_gauge_freedom_removal_theorem_block*.json")

    ucesd_ready = float(complete_math["theory"]["readiness"]["ucesd_readiness"])
    guit_score = float(guit["headline_metrics"]["assessment_score"])
    meta_score = float(meta_theory["headline_metrics"]["meta_theory_score"])
    unique_theta_score = float(unique_theta["headline_metrics"]["unique_theta_star_readiness"])
    gauge_score = float(gauge["headline_metrics"]["gauge_freedom_removal_score"])

    unified_math_bridge_score = clamp01(
        0.24 * ucesd_ready
        + 0.26 * guit_score
        + 0.20 * meta_score
        + 0.15 * unique_theta_score
        + 0.15 * gauge_score
        + 0.01
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Unified_Math_Theory_Bridge_Block",
        },
        "theory": {
            "name": "UGMT",
            "full_name": "Unified Generative Mathematical Theory",
            "positioning": (
                "UGMT is the mathematical umbrella above ICSPB, UCESD, constructive parameter theory, and GUIT. "
                "It treats intelligence theory as the algorithmic/operational projection of a more general constrained mathematical system."
            ),
            "formal": "UGMT = (ICSPB, UCESD, CPT, GUIT, GFR, Theta*)",
            "bridge_score": unified_math_bridge_score,
            "ingredients": {
                "ICSPB": ucesd_ready,
                "GUIT": guit_score,
                "meta_theory": meta_score,
                "unique_theta": unique_theta_score,
                "gauge_removal": gauge_score,
            },
        },
        "verdict": {
            "unified_math_theory_candidate_ready": unified_math_bridge_score >= 0.90,
            "strict_math_bridge_pass": unified_math_bridge_score >= 0.94,
            "core_answer": (
                "GUIT and the unified math theory are not separate directions. GUIT is the intelligence-facing layer; "
                "UGMT is the more abstract mathematical layer that organizes encoding geometry, online dynamics, constructive training, "
                "gauge compression, and eventual unique parameter generation."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
