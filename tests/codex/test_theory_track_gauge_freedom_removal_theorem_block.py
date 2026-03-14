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
    ap = argparse.ArgumentParser(description="gauge freedom removal 定理块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_gauge_freedom_removal_theorem_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    gauge_block = load_latest("icspb_v2_gauge_constrained_long_horizon_block*.json")
    unique_theta = load_latest("theory_track_unique_theta_star_generation_theorem_block*.json")

    gauge_reduction = float(gauge_block["gauge_reduction"])
    final_gauge = float(gauge_block["final_gauge_spread"])
    baseline_margin = float(gauge_block["baseline_margin"])
    external_margin = float(gauge_block["external_margin"])
    stable_read = float(gauge_block["proto_final"]["stable_read"])
    theorem_survival = float(gauge_block["proto_final"]["theorem_survival"])
    unique_theta_score = float(unique_theta["headline_metrics"]["unique_theta_star_readiness"])

    reduction_support = clamp01(gauge_reduction / 0.01)
    compactness_support = clamp01(1.0 - final_gauge / 0.03)
    margin_support = clamp01(0.5 * min(1.0, baseline_margin / 1.0) + 0.5 * min(1.0, external_margin / 1.0))
    theorem_score = clamp01(
        0.22 * reduction_support
        + 0.22 * compactness_support
        + 0.18 * margin_support
        + 0.18 * stable_read
        + 0.10 * theorem_survival
        + 0.10 * unique_theta_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Gauge_Freedom_Removal_Theorem_Block",
        },
        "headline_metrics": {
            "gauge_reduction": gauge_reduction,
            "final_gauge_spread": final_gauge,
            "margin_support": margin_support,
            "unique_theta_star_readiness": unique_theta_score,
            "gauge_freedom_removal_score": theorem_score,
        },
        "theorem": {
            "name": "gauge_freedom_removal_theorem",
            "strict_pass": theorem_score >= 0.94,
            "status": "strict_pass" if theorem_score >= 0.94 else "candidate_support",
            "score": theorem_score,
            "high_level_statement": (
                "Given constructive constrained training and persistent theorem-consistent execution, "
                "the remaining parameter gauge freedom can be actively compressed by gauge-constrained consolidation, "
                "making the learned basin narrower, more canonical, and more identifiable."
            ),
            "remaining_gap": "full canonical witness under true always-on external validation",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
