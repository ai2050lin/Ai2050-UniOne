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

    convergence = load_json(TEMP / "agi_chat_language_capability_convergence_block.json")
    closure = load_json(TEMP / "theory_track_agi_chat_language_training_closure_assessment.json")
    scaleup = load_json(TEMP / "theory_track_agi_chat_language_scaleup_assessment.json")
    open_domain = load_json(TEMP / "theory_track_agi_chat_open_domain_assessment.json")

    convergence_score = float(convergence["headline_metrics"]["stage_score"])
    closure_score = float(closure["headline_metrics"]["assessment_score"])
    scaleup_score = float(scaleup["headline_metrics"]["assessment_score"])
    open_domain_score = float(open_domain["headline_metrics"]["assessment_score"])

    assessment_score = clamp01(
        0.34 * convergence_score
        + 0.28 * closure_score
        + 0.22 * scaleup_score
        + 0.16 * open_domain_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Language_Capability_Convergence_Assessment",
        },
        "headline_metrics": {
            "convergence_score": convergence_score,
            "language_closure_score": closure_score,
            "language_scaleup_score": scaleup_score,
            "open_domain_score": open_domain_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.90,
            "strong_language_training_ready": assessment_score >= 0.96,
            "core_answer": "语言收敛总评统一考察新训练块、语言闭合、scaleup 结果和开放域语义表现。",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_language_capability_convergence_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
