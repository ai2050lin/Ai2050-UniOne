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


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=120)

    sequence = [
        "请记住，我最喜欢的水果是苹果。",
        "你还记得我最喜欢的水果吗？",
        "如果你说我最喜欢香蕉就是错的，对吗？",
    ]

    rows = []
    hit_total = 0.0
    expectations = [["苹果"], ["苹果"], ["错", "苹果"]]
    for prompt, keywords in zip(sequence, expectations):
        result = engine.generate(prompt, max_new_tokens=24, mem_decay=0.84)
        answer = result.get("generated_text", "")
        lowered = answer.lower()
        hit = sum(1 for token in keywords if token.lower() in lowered) / len(keywords)
        hit_total += hit
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "keyword_hit_ratio": hit,
                "metrics": result.get("icspb_metrics", {}),
            }
        )

    consistency_hit_rate = hit_total / len(sequence)
    consistency_score = min(1.0, 0.85 * consistency_hit_rate + 0.15)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Dialogue_Consistency_Benchmark",
        },
        "headline_metrics": {
            "consistency_hit_rate": consistency_hit_rate,
            "consistency_score": consistency_score,
            "turn_count": len(sequence),
        },
        "rows": rows,
        "verdict": {
            "overall_pass": consistency_score >= 0.74,
            "dialogue_consistency_ready": consistency_score >= 0.88,
            "core_answer": "对话一致性 benchmark 关注显式事实记忆、后续回指和矛盾避免。",
        },
    }

    out_file = TEMP / "agi_chat_dialogue_consistency_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
