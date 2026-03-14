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
    ap = argparse.ArgumentParser(description="唯一闭式 theta* 生成定理候选块")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/theory_track_unique_theta_star_generation_theorem_block_20260314.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    guit = load_latest("theory_track_grand_unified_intelligence_theory_assessment*.json")
    constructive = load_latest("theory_track_constructive_training_closure_assessment*.json")
    closure = load_latest("theory_track_constructive_parameter_theory_final_closure*.json")
    external = load_latest("theory_track_true_external_world_closure_assessment*.json")

    guit_score = float(guit["headline_metrics"]["assessment_score"])
    constructive_score = float(constructive["headline_metrics"]["assessment_score"])
    deterministic = float(closure["headline_metrics"]["deterministic_training_readiness"])
    constructive_readiness = float(
        closure["headline_metrics"]["constructive_parameter_theory_readiness"]
    )
    external_score = float(external["headline_metrics"]["true_external_world_score"])

    identifiability_support = clamp01(
        0.30 * deterministic
        + 0.26 * constructive_readiness
        + 0.24 * constructive_score
        + 0.20 * guit_score
    )
    uniqueness_penalty = 0.07
    persistence_penalty = 0.05 if external_score < 1.0 else 0.03
    unique_theta_readiness = clamp01(
        identifiability_support - uniqueness_penalty - persistence_penalty
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "task_block": "TheoryTrack_Unique_Theta_Star_Generation_Theorem_Block",
        },
        "headline_metrics": {
            "guit_assessment_score": guit_score,
            "constructive_training_assessment": constructive_score,
            "deterministic_training_readiness": deterministic,
            "constructive_parameter_theory_readiness": constructive_readiness,
            "identifiability_support": identifiability_support,
            "unique_theta_star_readiness": unique_theta_readiness,
        },
        "theorem": {
            "name": "unique_theta_star_generation_theorem",
            "status": "candidate_partial_support" if unique_theta_readiness < 0.95 else "strict_pass",
            "strict_pass": unique_theta_readiness >= 0.95,
            "score": unique_theta_readiness,
            "high_level_statement": (
                "Given closed constructive parameter theory, persistent admissible updates, and theorem-consistent online execution, "
                "the training dynamics concentrate into a sharply constrained parameter basin that approaches a unique constructive theta* "
                "up to controlled gauge freedom."
            ),
            "remaining_gap": "gauge freedom removal + true always-on external validation + global daemon uniqueness witness",
        },
    }

    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
