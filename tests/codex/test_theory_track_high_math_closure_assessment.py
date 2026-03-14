from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    start = time.time()
    quotient = load_json(TEMP / "gauge_quotient_theory_block.json")
    action = load_json(TEMP / "admissible_path_action_principle_block.json")
    bridge = load_json(TEMP / "guit_ugmt_strict_bridge_block.json")

    quotient_score = float(quotient["headline_metrics"]["quotient_score"])
    action_score = float(action["headline_metrics"]["action_score"])
    bridge_score = float(bridge["headline_metrics"]["strict_bridge_score"])

    assessment_score = min(1.0, 0.34 * quotient_score + 0.32 * action_score + 0.34 * bridge_score)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_High_Math_Closure_Assessment",
        },
        "headline_metrics": {
            "quotient_score": quotient_score,
            "action_score": action_score,
            "strict_bridge_score": bridge_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.89,
            "strict_final_pass": assessment_score >= 0.95,
            "core_answer": (
                "Higher-level mathematics can now organize the remaining blockers into quotient, action, and bridge layers, "
                "but strict final closure still depends on stronger canonical witness and external proof."
            ),
        },
    }

    out_file = TEMP / "high_math_closure_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
