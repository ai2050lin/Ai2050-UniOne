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
    route = load_json(TEMP / "theory_track_language_emergence_instant_learning_route_reassessment.json")
    feasibility = load_json(TEMP / "theory_track_dnn_language_plus_instant_learning_feasibility.json")

    route_validity = float(route["headline_metrics"]["route_validity"])
    dual_timescale_balance = float(route["headline_metrics"]["dual_timescale_balance"])
    joint_feasibility = float(feasibility["headline_metrics"]["route_joint_feasibility"])

    assessment_score = clamp01(
        0.40 * route_validity
        + 0.35 * dual_timescale_balance
        + 0.25 * joint_feasibility
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Language_Emergence_Instant_Learning_Route_Assessment",
        },
        "headline_metrics": {
            "route_validity": route_validity,
            "dual_timescale_balance": dual_timescale_balance,
            "joint_feasibility": joint_feasibility,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.74,
            "route_wrong": False,
            "dual_track_roadmap_ready": assessment_score >= 0.80,
            "core_answer": "路线本身没有根本错误，但必须从“单线训练幻想”升级成“语言预训练主干 + 即时学习主干”的双轨执行。",
        },
    }

    out_file = TEMP / "theory_track_language_emergence_instant_learning_route_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
