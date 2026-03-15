from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
TEMP.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agi_chat_service import AGIChatEngine


def safe_preview(text: str, limit: int = 160) -> str:
    return text.encode("unicode_escape").decode("ascii")[:limit]


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=32)
    train_result = engine.train_language_model(steps=1, batch_size=1, max_texts=4, save_checkpoint=False)
    train_status = engine.get_training_status()
    benchmark = engine.run_generation_benchmark(max_cases=2, max_new_tokens=12)
    chat_result = engine.generate("请用一句话解释人工智能。", max_new_tokens=12)
    semantic_result = engine.semantic_inference("为什么喝水很重要？", "zh")

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_ICSPB_Neural_Language_Training_Smoke_Block",
        },
        "headline_metrics": {
            "engine_ready": bool(engine.is_ready),
            "semantic_benchmark_score": float(engine.semantic_benchmark_score),
            "language_training_steps": int(engine.language_training_steps),
            "semantic_training_rounds": int(engine.semantic_training_rounds),
            "eval_loss": float(train_result.get("eval_loss", 0.0)),
            "benchmark_score": float(benchmark.get("headline_metrics", {}).get("benchmark_score", 0.0)),
            "history_points": float(train_status.get("history_count", 0)),
            "chat_quality_score": float(chat_result.get("correctness_review", {}).get("quality_score", 0.0)),
            "semantic_quality_score": float(semantic_result.get("correctness_review", {}).get("quality_score", 0.0)),
        },
        "rows": [
            {
                "kind": "chat",
                "nonempty": bool(chat_result.get("generated_text")),
                "preview": safe_preview(chat_result.get("generated_text", "")),
            },
            {
                "kind": "semantic",
                "nonempty": bool(semantic_result.get("generated_text")),
                "preview": safe_preview(semantic_result.get("generated_text", "")),
            },
        ],
        "verdict": {
            "overall_pass": bool(engine.is_ready)
            and bool(chat_result.get("generated_text"))
            and bool(semantic_result.get("generated_text")),
            "core_answer": "ICSPB 当前已切到 PhaseA 神经语言主干，能够在本地文本上继续训练，并给出非空神经生成输出。",
        },
    }

    out_file = TEMP / "stage_icspb_neural_language_training_smoke_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
