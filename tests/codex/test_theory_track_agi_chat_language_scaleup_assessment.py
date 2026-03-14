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

    scaleup = load_json(TEMP / "agi_chat_language_scaleup_training_block.json")
    language_closure = load_json(TEMP / "theory_track_agi_chat_language_training_closure_assessment.json")
    open_domain = load_json(TEMP / "theory_track_agi_chat_open_domain_assessment.json")

    scaleup_score = float(scaleup["headline_metrics"]["stage_score"])
    closure_score = float(language_closure["headline_metrics"]["assessment_score"])
    open_domain_score = float(open_domain["headline_metrics"]["assessment_score"])

    assessment_score = clamp01(
        0.38 * scaleup_score
        + 0.34 * closure_score
        + 0.28 * open_domain_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Language_Scaleup_Assessment",
        },
        "headline_metrics": {
            "scaleup_score": scaleup_score,
            "language_closure_score": closure_score,
            "open_domain_score": open_domain_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.88,
            "strong_language_scaleup_ready": assessment_score >= 0.95,
            "core_answer": "语言 scaleup 总评同时关注训练冲刺增益、语言训练闭合和开放域语义总评。",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_language_scaleup_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
