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
    ap = argparse.ArgumentParser(description="Map unresolved closure gaps in the current theory stack.")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_unclosed_problem_map_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    replay = load_latest("memory_replay_assessment*.json")
    gauge = load_latest("theory_track_gauge_freedom_removal_theorem_block*.json")
    theta = load_latest("theory_track_unique_theta_star_generation_theorem_block*.json")
    bio = load_latest("biophysical_causal_closure_assessment*.json")
    universe = load_latest("intelligence_ugmt_fundamental_relation_assessment*.json")

    replay_score = float(replay["headline_metrics"]["assessment_score"])
    gauge_score = float(gauge["headline_metrics"]["gauge_freedom_removal_score"])
    theta_score = float(theta["headline_metrics"]["unique_theta_star_readiness"])
    bio_score = float(bio["headline_metrics"]["assessment_score"])
    universe_score = float(universe["headline_metrics"]["assessment_score"])

    replay_gap = 1.0 - replay_score
    gauge_gap = 1.0 - gauge_score
    theta_gap = 1.0 - theta_score
    bio_gap = 1.0 - bio_score
    universe_gap = 1.0 - universe_score

    overall_gap_pressure = clamp01(
        0.22 * replay_gap
        + 0.24 * gauge_gap
        + 0.24 * theta_gap
        + 0.16 * bio_gap
        + 0.14 * universe_gap
    )
    closure_readiness = clamp01(1.0 - overall_gap_pressure)

    unresolved_items = [
        {
            "name": "strict_gate_level_replay_closure",
            "score": replay_score,
            "gap": replay_gap,
            "why_open": "Replay can recover structure, but stable_read and guarded_write do not re-close at strict gate level.",
            "principle": "Memory replay must restore not only latent structure, but also the write/read gating regime that makes replay stable.",
        },
        {
            "name": "gauge_freedom_removal_theorem",
            "score": gauge_score,
            "gap": gauge_gap,
            "why_open": "Gauge compression has support, but no strict proof that all non-canonical degrees of freedom are removed.",
            "principle": "A stronger constructive theory needs canonicalization, not just narrower basins.",
        },
        {
            "name": "unique_theta_star_generation_theorem",
            "score": theta_score,
            "gap": theta_gap,
            "why_open": "The theory points to a sharply constrained basin, but not to a strict unique closed-form witness theta*.",
            "principle": "Deterministic training still lacks final uniqueness at the canonical parameter level.",
        },
        {
            "name": "strict_biophysical_pass",
            "score": bio_score,
            "gap": bio_gap,
            "why_open": "The architecture is biophysically compatible, but not yet a final unique biological implementation theory.",
            "principle": "Compatibility is weaker than strict organism-level uniqueness.",
        },
        {
            "name": "ugmt_strict_fundamental_pass",
            "score": universe_score,
            "gap": universe_gap,
            "why_open": "The universe-facing bridge is strong, but the final fundamental-law uniqueness has not closed.",
            "principle": "UGMT still behaves like a strong candidate umbrella, not a final unique law.",
        },
    ]
    unresolved_items.sort(key=lambda item: item["gap"], reverse=True)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Unclosed_Problem_Map_Block",
        },
        "headline_metrics": {
            "replay_score": replay_score,
            "gauge_freedom_removal_score": gauge_score,
            "unique_theta_star_readiness": theta_score,
            "biophysical_closure_score": bio_score,
            "universe_bridge_score": universe_score,
            "overall_gap_pressure": overall_gap_pressure,
            "closure_readiness": closure_readiness,
        },
        "unresolved_items": unresolved_items,
        "summary": {
            "largest_gap": unresolved_items[0]["name"],
            "core_answer": (
                "The remaining closure problems have narrowed to replay gating, gauge removal, unique theta* witness, "
                "strict biophysical uniqueness, and strict universe-law closure."
            ),
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
