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

    open_domain = load_json(TEMP / "agi_chat_open_domain_semantic_benchmark.json")
    long_context = load_json(TEMP / "agi_chat_long_context_semantic_benchmark.json")
    reasoning = load_json(TEMP / "agi_chat_long_reasoning_benchmark.json")
    dialogue = load_json(TEMP / "agi_chat_dialogue_consistency_benchmark.json")

    open_domain_score = float(open_domain["headline_metrics"]["open_domain_score"])
    long_context_score = float(long_context["headline_metrics"]["long_context_score"])
    reasoning_score = float(reasoning["headline_metrics"]["reasoning_score"])
    dialogue_score = float(dialogue["headline_metrics"]["consistency_score"])

    assessment_score = clamp01(
        0.26 * open_domain_score
        + 0.24 * long_context_score
        + 0.26 * reasoning_score
        + 0.24 * dialogue_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Reasoning_Consistency_Assessment",
        },
        "headline_metrics": {
            "open_domain_score": open_domain_score,
            "long_context_score": long_context_score,
            "reasoning_score": reasoning_score,
            "dialogue_score": dialogue_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.78,
            "language_reasoning_ready": assessment_score >= 0.90,
            "core_answer": "当前语言高级能力总评同时覆盖开放域语义、长上下文、长知识链推理和对话一致性。",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_reasoning_consistency_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
