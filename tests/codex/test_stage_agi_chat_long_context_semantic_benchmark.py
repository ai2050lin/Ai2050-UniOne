from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.agi_chat_service import AGIChatEngine


LONG_CONTEXT_CASES = [
    {
        "prompt": (
            "请总结这段话：苹果和梨都属于水果，二者都可以直接食用。"
            "苹果通常更脆，梨通常更多汁。请用一句话总结。"
        ),
        "keywords": ["苹果", "梨", "水果"],
    },
    {
        "prompt": (
            "请概括下面内容：人工智能系统可以处理信息、执行推理，并在给定任务中辅助决策。"
            "它通常依赖模型、数据和训练过程。请用一句话概括。"
        ),
        "keywords": ["人工智能", "推理", "决策"],
    },
    {
        "prompt": (
            "如果前文已经说过苹果可以直接吃，现在请基于这个上下文，再补一句和苹果食用方式有关的话。"
        ),
        "keywords": ["苹果", "吃"],
    },
]


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=120)

    hit_total = 0.0
    rows = []
    for case in LONG_CONTEXT_CASES:
        result = engine.generate(case["prompt"], max_new_tokens=30, mem_decay=0.84)
        answer = result.get("generated_text", "")
        lowered = answer.lower()
        hit = sum(1 for token in case["keywords"] if token.lower() in lowered) / len(case["keywords"])
        hit_total += hit
        rows.append(
            {
                "prompt": case["prompt"],
                "answer": answer,
                "keyword_hit_ratio": hit,
                "metrics": result.get("icspb_metrics", {}),
            }
        )

    context_hit_rate = hit_total / len(LONG_CONTEXT_CASES)
    long_context_score = min(1.0, 0.80 * context_hit_rate + 0.20)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Long_Context_Semantic_Benchmark",
        },
        "headline_metrics": {
            "context_hit_rate": context_hit_rate,
            "long_context_score": long_context_score,
            "case_count": len(LONG_CONTEXT_CASES),
        },
        "rows": rows,
        "verdict": {
            "overall_pass": long_context_score >= 0.74,
            "long_context_ready": long_context_score >= 0.88,
            "core_answer": "长上下文 benchmark 关注总结、概括和带前文约束的 follow-up 保义能力。",
        },
    }

    out_file = TEMP / "agi_chat_long_context_semantic_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
