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
    consistency = load_json(TEMP / "agi_chat_icspb_consistency_block.json")
    benchmark = load_json(TEMP / "agi_chat_multiturn_language_benchmark.json")
    stability = load_json(TEMP / "agi_chat_long_session_stability.json")
    open_domain = load_json(TEMP / "agi_chat_open_domain_semantic_benchmark.json")
    long_context = load_json(TEMP / "agi_chat_long_context_semantic_benchmark.json")

    consistency_score = float(consistency["headline_metrics"]["consistency_score"])
    benchmark_score = float(benchmark["headline_metrics"]["benchmark_score"])
    stability_score = float(stability["headline_metrics"]["long_session_score"])
    open_domain_score = float(open_domain["headline_metrics"]["open_domain_score"])
    long_context_score = float(long_context["headline_metrics"]["long_context_score"])

    assessment_score = clamp01(
        0.22 * consistency_score
        + 0.20 * benchmark_score
        + 0.20 * stability_score
        + 0.18 * open_domain_score
        + 0.20 * long_context_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Open_Domain_Assessment",
        },
        "headline_metrics": {
            "consistency_score": consistency_score,
            "benchmark_score": benchmark_score,
            "stability_score": stability_score,
            "open_domain_score": open_domain_score,
            "long_context_score": long_context_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.76,
            "open_domain_dialog_ready": (
                assessment_score >= 0.88
                and open_domain["verdict"]["open_domain_ready"]
                and long_context["verdict"]["long_context_ready"]
            ),
            "core_answer": "当前语言系统总评现在同时覆盖一致性、多轮问答、长会话、开放域语义和长上下文总结。 ",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_open_domain_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
