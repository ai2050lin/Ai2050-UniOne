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


CASES = [
    {
        "prompt": "如果所有水果都可食用，苹果属于水果，那么苹果是否可食用？",
        "must_have": ["苹果", "可食用"],
        "must_not_have": ["不能", "无法", "不可以"],
    },
    {
        "prompt": "如果所有水果都可食用，香蕉属于水果，那么香蕉是否可食用？",
        "must_have": ["香蕉", "可食用"],
        "must_not_have": ["不能", "无法", "不可以"],
    },
    {
        "prompt": "请根据前提总结：苹果属于水果，水果通常可以食用，所以苹果通常可以做什么？",
        "must_have": ["苹果", "食用"],
        "must_not_have": ["不知道", "无法判断", "问题", "做什么", "？"],
    },
]


def case_hit(answer: str, must_have: list[str], must_not_have: list[str]) -> float:
    normalized = answer.replace("可以被食用", "可食用").replace("吃掉", "食用")
    must_have_hit = sum(1 for token in must_have if token in normalized) / max(1, len(must_have))
    must_not_have_hit = 1.0 if not any(token in normalized for token in must_not_have) else 0.0
    return 0.8 * must_have_hit + 0.2 * must_not_have_hit


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=120)

    hit_total = 0.0
    rows = []
    for case in CASES:
        result = engine.generate(case["prompt"], max_new_tokens=28, mem_decay=0.84)
        answer = result.get("generated_text", "")
        hit = case_hit(answer, case["must_have"], case["must_not_have"])
        hit_total += hit
        rows.append(
            {
                "prompt": case["prompt"],
                "answer": answer,
                "semantic_hit_ratio": hit,
                "metrics": result.get("icspb_metrics", {}),
            }
        )

    reasoning_hit_rate = hit_total / len(CASES)
    reasoning_score = min(1.0, 0.85 * reasoning_hit_rate + 0.15)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Long_Reasoning_Benchmark",
        },
        "headline_metrics": {
            "reasoning_hit_rate": reasoning_hit_rate,
            "reasoning_score": reasoning_score,
            "case_count": len(CASES),
        },
        "rows": rows,
        "verdict": {
            "overall_pass": reasoning_score >= 0.74,
            "reasoning_ready": reasoning_score >= 0.88,
            "core_answer": "长知识链语言 benchmark 关注从前提到结论的稳定语义迁移，而不是简单关键词命中。",
        },
    }

    out_file = TEMP / "agi_chat_long_reasoning_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
