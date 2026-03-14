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

    consistency_score = float(consistency["headline_metrics"]["consistency_score"])
    benchmark_score = float(benchmark["headline_metrics"]["benchmark_score"])
    semantic_fit_score = float(benchmark["headline_metrics"].get("semantic_fit_score", 0.0))
    stability_score = float(stability["headline_metrics"]["long_session_score"])

    assessment_score = clamp01(
        0.30 * consistency_score + 0.22 * benchmark_score + 0.20 * semantic_fit_score + 0.28 * stability_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_AGI_Chat_Language_Assessment",
        },
        "headline_metrics": {
            "consistency_score": consistency_score,
            "benchmark_score": benchmark_score,
            "semantic_fit_score": semantic_fit_score,
            "stability_score": stability_score,
            "assessment_score": assessment_score,
        },
        "verdict": {
            "overall_pass": assessment_score >= 0.70,
            "dialog_ready": assessment_score >= 0.88 and benchmark["verdict"].get("semantic_quality_pass", False),
            "long_session_ready": benchmark["verdict"]["language_dialog_ready"] and stability["verdict"]["long_session_ready"],
            "core_answer": "Language capability is now judged jointly by model-engine consistency, real multi-turn benchmark, and long-session stability.",
        },
    }

    out_file = TEMP / "theory_track_agi_chat_language_assessment.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
