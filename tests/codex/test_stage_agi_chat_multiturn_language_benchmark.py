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


PROMPTS = [
    "请用一句话解释苹果是什么。",
    "把这句话改写得更简洁：我今天很高兴，因为天气很好。",
    "比较苹果和梨的相同点与不同点。",
    "如果篮子里有两个苹果，再放进去一个，现在有几个？",
    "基于你刚才的回答，再补一句和吃苹果有关的话。",
]

EXPECTATIONS = [
    ["apple", "fruit", "苹果", "水果"],
    ["高兴", "天气", "happy", "weather", "简洁"],
    ["apple", "pear", "苹果", "梨", "same", "different", "相同", "不同"],
    ["3", "three", "三个", "个"],
    ["apple", "eat", "苹果", "吃"],
]


def main() -> None:
    start = time.time()
    engine = AGIChatEngine()
    engine.initialize(max_sentences=100)

    turns = []
    success = 0
    total_chars = 0
    semantic_hits = 0.0
    for idx, prompt in enumerate(PROMPTS):
        result = engine.generate(prompt, max_new_tokens=28, mem_decay=0.84)
        text = result.get("generated_text", "")
        ok = result.get("status") == "success" and len(text.strip()) > 0
        success += int(ok)
        total_chars += len(text)
        lowered = text.lower()
        semantic_fit = 1.0 if any(token.lower() in lowered for token in EXPECTATIONS[idx]) else 0.0
        semantic_hits += semantic_fit
        turns.append(
            {
                "prompt": prompt,
                "generated_text": text,
                "ok": ok,
                "semantic_fit": semantic_fit,
                "metrics": result.get("icspb_metrics", {}),
            }
        )

    success_ratio = success / len(PROMPTS)
    avg_chars = total_chars / len(PROMPTS)
    semantic_fit_score = semantic_hits / len(PROMPTS)
    benchmark_score = min(
        1.0,
        0.35 * success_ratio + 0.20 * min(1.0, avg_chars / 18.0) + 0.45 * semantic_fit_score,
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Multiturn_Language_Benchmark",
        },
        "headline_metrics": {
            "success_ratio": success_ratio,
            "avg_generated_chars": avg_chars,
            "semantic_fit_score": semantic_fit_score,
            "benchmark_score": benchmark_score,
            "turn_count": len(PROMPTS),
        },
        "turns": turns,
        "verdict": {
            "overall_pass": benchmark_score >= 0.55,
            "language_dialog_ready": benchmark_score >= 0.80,
            "semantic_quality_pass": semantic_fit_score >= 0.60,
            "core_answer": "语言 benchmark 现在同时检查连续性与粗粒度语义命中，覆盖解释、改写、比较、算术和短 follow-up。",
        },
    }

    out_file = TEMP / "agi_chat_multiturn_language_benchmark.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
