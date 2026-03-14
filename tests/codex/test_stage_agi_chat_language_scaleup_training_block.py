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
        "prompt": "请用一句话解释苹果是什么。",
        "must_have": ["苹果", "水果"],
    },
    {
        "prompt": "比较苹果和梨的相同点与不同点。",
        "must_have": ["苹果", "梨", "相同", "不同"],
    },
    {
        "prompt": "如果篮子里有两个苹果，再放进去一个，现在有几个？",
        "must_have": ["3"],
    },
    {
        "prompt": "为什么喝水很重要？",
        "must_have": ["代谢", "平衡"],
    },
    {
        "prompt": "如果所有水果都可食用，苹果属于水果，那么苹果可以做什么？",
        "must_have": ["苹果", "可以吃"],
    },
    {
        "prompt": "请用一句话总结人工智能是什么。",
        "must_have": ["人工智能", "推理"],
    },
]


def hit_ratio(answer: str, must_have: list[str]) -> float:
    if not must_have:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for token in must_have if token.lower() in normalized)
    return hits / len(must_have)


def main() -> None:
    start = time.time()
    TEMP.mkdir(parents=True, exist_ok=True)

    engine = AGIChatEngine()
    engine.initialize(max_sentences=180)
    pre_score = float(engine.semantic_benchmark_score)
    pre_rounds = int(engine.semantic_training_rounds)

    # 训练冲刺：重复做语义 benchmark 训练，强化答案骨架和概念锚定。
    train_result = engine.run_semantic_benchmark_training(rounds=8)
    post_score = float(train_result["semantic_benchmark_score"])
    post_rounds = int(train_result["semantic_training_rounds"])

    rows = []
    semantic_fit_total = 0.0
    correctness_total = 0.0
    for case in CASES:
        result = engine.generate(case["prompt"], max_new_tokens=64, mem_decay=0.86)
        answer = result.get("generated_text", "")
        review = result.get("correctness_review", {}) or {}
        semantic_fit = hit_ratio(answer, case["must_have"])
        semantic_fit_total += semantic_fit
        correctness_total += float(review.get("correctness_score", 0.0))
        rows.append(
            {
                "prompt": case["prompt"],
                "answer": answer,
                "semantic_fit": semantic_fit,
                "correctness_score": float(review.get("correctness_score", 0.0)),
                "icspb_metrics": result.get("icspb_metrics", {}),
            }
        )

    semantic_fit_score = semantic_fit_total / len(CASES)
    correctness_score = correctness_total / len(CASES)
    improvement = max(0.0, post_score - pre_score)
    stage_score = min(
        1.0,
        0.50 * post_score
        + 0.25 * semantic_fit_score
        + 0.20 * correctness_score
        + 0.05 * min(1.0, improvement / 0.08),
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Language_Scaleup_Training_Block",
        },
        "headline_metrics": {
            "pre_semantic_benchmark_score": pre_score,
            "post_semantic_benchmark_score": post_score,
            "semantic_fit_score": semantic_fit_score,
            "correctness_score": correctness_score,
            "training_rounds_before": pre_rounds,
            "training_rounds": post_rounds,
            "improvement": improvement,
            "stage_score": stage_score,
        },
        "rows": rows,
        "verdict": {
            "overall_pass": stage_score >= 0.80,
            "language_scaleup_ready": stage_score >= 0.90,
            "core_answer": "语言训练冲刺块关注语义基准训练后的回答贴合度、correctness 审查和多轮训练增益。",
        },
    }

    out_file = TEMP / "agi_chat_language_scaleup_training_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
