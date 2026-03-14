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
        "category": "定义解释",
        "prompt": "请用一句话解释苹果是什么。",
        "must_have": ["苹果", "水果"],
    },
    {
        "category": "比较分析",
        "prompt": "比较苹果和梨的相同点与不同点。",
        "must_have": ["苹果", "梨", "相同", "不同"],
    },
    {
        "category": "简单推理",
        "prompt": "如果篮子里有两个苹果，再放进去一个，现在有几个？",
        "must_have": ["3"],
    },
    {
        "category": "开放域说明",
        "prompt": "为什么喝水很重要？",
        "must_have": ["代谢", "平衡"],
    },
    {
        "category": "长知识链",
        "prompt": "如果所有水果都可食用，苹果属于水果，那么苹果可以做什么？",
        "must_have": ["苹果", "可以吃"],
    },
    {
        "category": "抽象概念",
        "prompt": "请用一句话总结人工智能是什么。",
        "must_have": ["人工智能", "推理"],
    },
    {
        "category": "多跳推理",
        "prompt": "如果能处理信息的系统可以辅助决策，人工智能系统能处理信息，那么人工智能系统通常可以做什么？",
        "must_have": ["人工智能系统", "辅助决策"],
    },
]


def hit_ratio(answer: str, tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for token in tokens if token.lower() in normalized)
    return hits / len(tokens)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    start = time.time()
    TEMP.mkdir(parents=True, exist_ok=True)

    engine = AGIChatEngine()
    engine.initialize(max_sentences=220)
    pre_score = float(engine.semantic_benchmark_score)
    pre_rounds = int(engine.semantic_training_rounds)

    # 语言能力收敛训练：继续强化语义基准、答案骨架和 correctness 审查。
    train_result = engine.run_semantic_benchmark_training(rounds=12)
    post_score = float(train_result["semantic_benchmark_score"])
    post_rounds = int(train_result["semantic_training_rounds"])

    semantic_fit_total = 0.0
    correctness_total = 0.0
    answer_len_total = 0.0
    rows = []

    for case in CASES:
        result = engine.generate(case["prompt"], max_new_tokens=72, mem_decay=0.86)
        answer = result.get("generated_text", "")
        review = result.get("correctness_review", {}) or {}
        semantic_fit = hit_ratio(answer, case["must_have"])
        correctness = float(review.get("correctness_score", 0.0))
        answer_len = len(answer)

        semantic_fit_total += semantic_fit
        correctness_total += correctness
        answer_len_total += answer_len

        rows.append(
            {
                "category": case["category"],
                "prompt": case["prompt"],
                "answer": answer,
                "semantic_fit": semantic_fit,
                "correctness_score": correctness,
                "answer_length": answer_len,
                "icspb_metrics": result.get("icspb_metrics", {}),
            }
        )

    semantic_fit_score = semantic_fit_total / len(CASES)
    correctness_score = correctness_total / len(CASES)
    avg_answer_len = answer_len_total / len(CASES)
    improvement = max(0.0, post_score - pre_score)

    stage_score = clamp01(
        0.34 * post_score
        + 0.28 * semantic_fit_score
        + 0.22 * correctness_score
        + 0.10 * min(1.0, avg_answer_len / 26.0)
        + 0.06 * min(1.0, improvement / 0.06)
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "Stage_AGI_Chat_Language_Capability_Convergence_Block",
        },
        "headline_metrics": {
            "pre_semantic_benchmark_score": pre_score,
            "post_semantic_benchmark_score": post_score,
            "training_rounds_before": pre_rounds,
            "training_rounds_after": post_rounds,
            "semantic_fit_score": semantic_fit_score,
            "correctness_score": correctness_score,
            "avg_answer_len": avg_answer_len,
            "improvement": improvement,
            "stage_score": stage_score,
        },
        "rows": rows,
        "verdict": {
            "overall_pass": stage_score >= 0.86,
            "language_convergence_ready": stage_score >= 0.93,
            "core_answer": "语言能力收敛训练块统一衡量语义贴合、正确性审查、回答长度和训练后收敛程度。",
        },
    }

    out_file = TEMP / "agi_chat_language_capability_convergence_block.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
