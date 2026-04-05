#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from multimodel_language_shared import free_model, load_model_bundle
from qwen3_language_shared import move_batch_to_model_device
from stage433_polysemous_noun_family_generalization import POLYSEMOUS_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage510_gemma4_polysemy_prompt_calibration_20260404"
)

PROMPT_STYLES = [
    {
        "style_id": "digit_marked",
        "answer_space": ["1", "2"],
        "build_prompt": lambda case, noun_spec: (
            f'Sentence: "{case["sentence"]}"\n'
            f'Question: In this sentence, does the marked word refer to 1 {noun_spec["sense_a_name"]} or 2 {noun_spec["sense_b_name"]}?\n'
            f'Answer with one digit only: 1 {noun_spec["sense_a_name"]} 2 {noun_spec["sense_b_name"]}\n'
            "Answer:"
        ),
        "label_to_answer": lambda noun_spec, label: "1" if label == 0 else "2",
    },
    {
        "style_id": "word_lower",
        "answer_space": None,
        "build_prompt": lambda case, noun_spec: (
            f'Sentence: "{case["sentence"]}"\n'
            f'Question: What does the word {case["target"]} refer to here?\n'
            f'Answer with one word only: {noun_spec["sense_a_name"]} or {noun_spec["sense_b_name"]}\n'
            "Answer:"
        ),
        "label_to_answer": lambda noun_spec, label: noun_spec["sense_a_name"] if label == 0 else noun_spec["sense_b_name"],
    },
    {
        "style_id": "letter_ab",
        "answer_space": ["A", "B"],
        "build_prompt": lambda case, noun_spec: (
            f'Sentence: "{case["sentence"]}"\n'
            f'Choose the correct meaning of the marked word.\n'
            f'A. {noun_spec["sense_a_name"]}\n'
            f'B. {noun_spec["sense_b_name"]}\n'
            "Answer with one capital letter only.\n"
            "Answer:"
        ),
        "label_to_answer": lambda noun_spec, label: "A" if label == 0 else "B",
    },
    {
        "style_id": "semantic_tag",
        "answer_space": ["sense_a", "sense_b"],
        "build_prompt": lambda case, noun_spec: (
            f'Sentence: "{case["sentence"]}"\n'
            f'Is the marked word used with meaning `sense_a` = {noun_spec["sense_a_name"]} or `sense_b` = {noun_spec["sense_b_name"]}?\n'
            "Answer with one tag only: sense_a or sense_b.\n"
            "Answer:"
        ),
        "label_to_answer": lambda noun_spec, label: "sense_a" if label == 0 else "sense_b",
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_cases() -> List[Tuple[Dict[str, object], Dict[str, str], int]]:
    rows: List[Tuple[Dict[str, object], Dict[str, str], int]] = []
    for noun_spec in POLYSEMOUS_CASES:
        for case in noun_spec["sense_a_cases"]:
            rows.append((noun_spec, case, 0))
        for case in noun_spec["sense_b_cases"]:
            rows.append((noun_spec, case, 1))
    return rows


def score_candidate_avg_logprob(model, tokenizer, prompt: str, candidate: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("-inf")
    encoded = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
    encoded = move_batch_to_model_device(model, encoded)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False, return_dict=True).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    count = 0
    for pos in range(len(prompt_ids), len(full_ids)):
        prev = pos - 1
        token_id = full_ids[pos]
        total += float(log_probs[prev, token_id].item())
        count += 1
    return total / max(1, count)


def score_answer_space(model, tokenizer, prompt: str, answer_space: Sequence[str]) -> Dict[str, float]:
    return {answer: score_candidate_avg_logprob(model, tokenizer, prompt, answer) for answer in answer_space}


def evaluate_style(model, tokenizer, style_spec: Dict[str, object]) -> Dict[str, object]:
    cases = build_cases()
    per_case = []
    noun_stats: Dict[str, Dict[str, float]] = {}
    for noun_spec, case, label in cases:
        prompt = style_spec["build_prompt"](case, noun_spec)
        if style_spec["answer_space"] is None:
            answer_space = [str(noun_spec["sense_a_name"]), str(noun_spec["sense_b_name"])]
        else:
            answer_space = list(style_spec["answer_space"])
        target_answer = style_spec["label_to_answer"](noun_spec, label)
        scores = score_answer_space(model, tokenizer, prompt, answer_space)
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        is_correct = prediction == target_answer
        correct_score = float(scores[target_answer])
        per_case.append(
            {
                "noun_id": noun_spec["noun_id"],
                "sentence": case["sentence"],
                "expected": target_answer,
                "predicted": prediction,
                "correct_score": correct_score,
                "is_correct": is_correct,
            }
        )
        noun_row = noun_stats.setdefault(
            str(noun_spec["noun_id"]),
            {"total": 0.0, "correct": 0.0, "score_sum": 0.0},
        )
        noun_row["total"] += 1.0
        noun_row["correct"] += 1.0 if is_correct else 0.0
        noun_row["score_sum"] += correct_score

    noun_results = {}
    for noun_id, row in noun_stats.items():
        noun_results[noun_id] = {
            "accuracy": row["correct"] / max(1.0, row["total"]),
            "mean_correct_score": row["score_sum"] / max(1.0, row["total"]),
        }
    accuracy = sum(1.0 if row["is_correct"] else 0.0 for row in per_case) / max(1, len(per_case))
    mean_correct_score = sum(float(row["correct_score"]) for row in per_case) / max(1, len(per_case))
    return {
        "style_id": style_spec["style_id"],
        "accuracy": accuracy,
        "mean_correct_score": mean_correct_score,
        "noun_results": noun_results,
        "per_case": per_case,
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 核心结论",
        summary["core_answer"],
        "",
    ]
    for row in summary["style_results"]:
        lines.extend(
            [
                f"## {row['style_id']}",
                f"- 总准确率：`{row['accuracy']:.4f}`",
                f"- 平均正确分数：`{row['mean_correct_score']:.4f}`",
                "",
            ]
        )
        for noun_id, noun_row in row["noun_results"].items():
            lines.extend(
                [
                    f"- {noun_id}: accuracy=`{noun_row['accuracy']:.4f}`, mean_correct_score=`{noun_row['mean_correct_score']:.4f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 多义词提示模板重标定")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    started = time.time()
    prefer_cuda = (not args.cpu) and torch.cuda.is_available()
    model, tokenizer = load_model_bundle("gemma4", prefer_cuda=prefer_cuda)
    try:
        style_results = [evaluate_style(model, tokenizer, style) for style in PROMPT_STYLES]
    finally:
        free_model(model)

    ranked = sorted(style_results, key=lambda row: (float(row["accuracy"]), float(row["mean_correct_score"])), reverse=True)
    best = ranked[0]
    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage510_gemma4_polysemy_prompt_calibration",
        "title": "Gemma4 多义词提示模板重标定",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "used_cuda": bool(prefer_cuda and torch.cuda.is_available()),
        "style_results": style_results,
        "best_style": {
            "style_id": best["style_id"],
            "accuracy": best["accuracy"],
            "mean_correct_score": best["mean_correct_score"],
        },
        "core_answer": (
            f"Gemma4 在当前多义词判别里最适合的提示格式是 `{best['style_id']}`，"
            "因此后续需要先按模型偏好的回答接口重标定，再讨论切换机制本身。"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
