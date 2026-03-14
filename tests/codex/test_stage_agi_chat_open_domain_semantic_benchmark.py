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
    {"prompt": "为什么喝水很重要？", "keywords": ["代谢", "平衡", "生命"]},
    {"prompt": "请用一句话总结人工智能是什么。", "keywords": ["人工智能", "推理", "技术系统"]},
    {"prompt": "列出苹果的两个常见特征。", "keywords": ["苹果", "可食用", "圆形"]},
    {"prompt": "请介绍橙子是什么。", "keywords": ["橙子", "水果", "汁"]},
    {"prompt": "比较香蕉和苹果的共同点。", "keywords": ["香蕉", "苹果", "水果"]},
]


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=120)

    hits = 0.0
    char_total = 0
    rows = []
    for case in CASES:
        result = engine.generate(case["prompt"], max_new_tokens=28, mem_decay=0.84)
        answer = result.get("generated_text", "")
        lowered = answer.lower()
        hit = sum(1 for token in case["keywords"] if token.lower() in lowered) / len(case["keywords"])
        hits += hit
        char_total += len(answer)
        rows.append(
            {
                "prompt": case["prompt"],
                "answer": answer,
                "keyword_hit_ratio": hit,
                "metrics": result.get("icspb_metrics", {}),
            }
        )

    semantic_hit_rate = hits / len(CASES)
    avg_chars = char_total / len(CASES)
    open_domain_score = min(1.0, 0.75 * semantic_hit_rate + 0.25 * min(1.0, avg_chars / 18.0))

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Open_Domain_Semantic_Benchmark",
        },
        "headline_metrics": {
            "semantic_hit_rate": semantic_hit_rate,
            "avg_generated_chars": avg_chars,
            "open_domain_score": open_domain_score,
            "case_count": len(CASES),
        },
        "rows": rows,
        "verdict": {
            "overall_pass": open_domain_score >= 0.70,
            "open_domain_ready": open_domain_score >= 0.85,
            "core_answer": "开放域语义 benchmark 关注跨主题概念定义、原因解释、列表化回答和跨概念比较。",
        },
    }

    out_file = TEMP / "agi_chat_open_domain_semantic_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
