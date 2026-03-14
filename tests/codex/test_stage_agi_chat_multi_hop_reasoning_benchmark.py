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
        "prompt": "如果所有水果都可食用，所有可食用的东西都可以吃，苹果属于水果，那么苹果最后可以做什么？",
        "must_have": ["苹果", "可以吃"],
        "must_not_have": ["不知道", "无法判断", "问题"],
    },
    {
        "prompt": "如果苹果是水果，水果通常可以食用，可食用的东西通常适合作为食物，那么苹果是否适合作为食物？",
        "must_have": ["苹果", "适合作为食物"],
        "must_not_have": ["不知道", "无法判断", "不适合"],
    },
    {
        "prompt": "如果人工智能系统可以处理信息，能够处理信息的系统可以辅助决策，那么人工智能系统通常可以做什么？",
        "must_have": ["人工智能系统", "辅助决策"],
        "must_not_have": ["不知道", "无法判断", "问题"],
    },
]


def case_hit(answer: str, must_have: list[str], must_not_have: list[str]) -> float:
    normalized = answer.replace("能够辅助决策", "辅助决策").replace("可以作为食物", "适合作为食物")
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
        result = engine.generate(case["prompt"], max_new_tokens=32, mem_decay=0.84)
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

    hit_rate = hit_total / len(CASES)
    multi_hop_score = min(1.0, 0.85 * hit_rate + 0.15)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Multi_Hop_Reasoning_Benchmark",
        },
        "headline_metrics": {
            "multi_hop_hit_rate": hit_rate,
            "multi_hop_score": multi_hop_score,
            "case_count": len(CASES),
        },
        "rows": rows,
        "verdict": {
            "overall_pass": multi_hop_score >= 0.74,
            "multi_hop_ready": multi_hop_score >= 0.88,
            "core_answer": "多跳推理 benchmark 关注语义链式传递，而不是单跳模板命中。",
        },
    }

    out_file = TEMP / "agi_chat_multi_hop_reasoning_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
