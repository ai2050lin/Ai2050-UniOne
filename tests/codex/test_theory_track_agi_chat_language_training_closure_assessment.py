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

    multiturn = load_json(TEMP / "agi_chat_multiturn_language_benchmark.json")
    open_domain = load_json(TEMP / "agi_chat_open_domain_semantic_benchmark.json")
    long_context = load_json(TEMP / "agi_chat_long_context_semantic_benchmark.json")
    reasoning = load_json(TEMP / "agi_chat_long_reasoning_benchmark.json")
    multi_hop = load_json(TEMP / "agi_chat_multi_hop_reasoning_benchmark.json")
    dialogue = load_json(TEMP / "agi_chat_dialogue_consistency_benchmark.json")
    long_session = load_json(TEMP / "agi_chat_long_session_stability.json")

    multiturn_score = float(multiturn["headline_metrics"]["benchmark_score"])
    open_domain_score = float(open_domain["headline_metrics"]["open_domain_score"])
    long_context_score = float(long_context["headline_metrics"]["long_context_score"])
    reasoning_score = float(reasoning["headline_metrics"]["reasoning_score"])
    multi_hop_score = float(multi_hop["headline_metrics"]["multi_hop_score"])
    dialogue_score = float(dialogue["headline_metrics"]["consistency_score"])
    long_session_score = float(long_session["headline_metrics"]["long_session_score"])

    assessment_score = clamp01(
        0.12 * multiturn_score
        + 0.16 * open_domain_score
        + 0.14 * long_context_score
        + 0.18 * reasoning_score
        + 0.18 * multi_hop_score
        + 0.12 * dialogue_score
        + 0.10 * long_session_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Language_Training_Closure_Assessment",
        },
        "headline_metrics": {
            "multiturn_score": multiturn_score,
            "open_domain_score": open_domain_score,
            "long_context_score": long_context_score,
            "reasoning_score": reasoning_score,
            "multi_hop_score": multi_hop_score,
            "dialogue_score": dialogue_score,
            "long_session_score": long_session_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.80,
            "language_training_closure_ready": assessment_score >= 0.92,
            "core_answer": "当前语言训练闭合总评同时覆盖多轮、开放域、长上下文、长知识链、多跳推理、对话一致性和长会话稳定性。",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_language_training_closure_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
