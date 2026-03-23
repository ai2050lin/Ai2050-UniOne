#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import torch

from stage124_noun_neuron_basic_probe import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage156_apple_context_bias_shift_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE156_APPLE_CONTEXT_BIAS_SHIFT_REPORT.md"

FAMILIES = [
    "sentiment_bias",
    "expectation_bias",
    "contrast_bias",
    "repair_bias",
    "category_bias",
]
DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]
POSITIVE_WORDS = ["fresh", "ready", "saved", "served"]
NEGATIVE_WORDS = ["lost", "broken", "delayed", "ignored"]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_prompt_pair(family_name: str, difficulty: str) -> Tuple[str, str]:
    if family_name == "sentiment_bias":
        if difficulty == "easy":
            return (
                "The committee praised the apple, so after the team handles it, it becomes",
                "The committee doubted the apple, so after the team handles it, it becomes",
            )
        if difficulty == "medium":
            return (
                "Although the report mentioned the pear, the committee praised the apple, so after the team handles it, it becomes",
                "Although the report mentioned the pear, the committee doubted the apple, so after the team handles it, it becomes",
            )
        if difficulty == "hard":
            return (
                "Although the report praised the pear, the committee still praised the apple, so after the team handles it, it becomes",
                "Although the report praised the pear, the committee still doubted the apple, so after the team handles it, it becomes",
            )
        return (
            "If the archive doubted the pear but the committee praised the apple, then after the team handles it, it becomes",
            "If the archive praised the pear but the committee doubted the apple, then after the team handles it, it becomes",
        )
    if family_name == "expectation_bias":
        if difficulty == "easy":
            return (
                "The team expects the apple to matter, so after it works on it, it becomes",
                "The team expects the apple to fail, so after it works on it, it becomes",
            )
        if difficulty == "medium":
            return (
                "Because the plan expects the apple to matter, after the team works on it, it becomes",
                "Because the plan expects the apple to fail, after the team works on it, it becomes",
            )
        if difficulty == "hard":
            return (
                "Although the plan highlights the pear, it expects the apple to matter, so after the team works on it, it becomes",
                "Although the plan highlights the pear, it expects the apple to fail, so after the team works on it, it becomes",
            )
        return (
            "If the plan expects the pear to matter but still expects the apple to matter, after the team works on it, it becomes",
            "If the plan expects the pear to matter but expects the apple to fail, after the team works on it, it becomes",
        )
    if family_name == "contrast_bias":
        if difficulty == "easy":
            return (
                "The note favored the apple, not the pear, so after the team handles it, it becomes",
                "The note favored the pear, not the apple, so after the team handles it, it becomes",
            )
        if difficulty == "medium":
            return (
                "Because the note favored the apple and not the pear, after the team handles it, it becomes",
                "Because the note favored the pear and not the apple, after the team handles it, it becomes",
            )
        if difficulty == "hard":
            return (
                "Although the archive praised the pear, the note favored the apple, so after the team handles it, it becomes",
                "Although the archive praised the pear, the note still rejected the apple, so after the team handles it, it becomes",
            )
        return (
            "If the archive favored the pear but the note favored the apple, after the team handles it, it becomes",
            "If the archive favored the pear and the note rejected the apple, after the team handles it, it becomes",
        )
    if family_name == "repair_bias":
        if difficulty == "easy":
            return (
                "The archive first doubted the apple, but later praised it, so after the team handles it, it becomes",
                "The archive first praised the apple, but later doubted it, so after the team handles it, it becomes",
            )
        if difficulty == "medium":
            return (
                "Because the archive first doubted the apple but later praised it, after the team handles it, it becomes",
                "Because the archive first praised the apple but later doubted it, after the team handles it, it becomes",
            )
        if difficulty == "hard":
            return (
                "Although the archive first pushed the pear, it later praised the apple, so after the team handles it, it becomes",
                "Although the archive first praised the pear, it later doubted the apple, so after the team handles it, it becomes",
            )
        return (
            "If the archive first doubted the apple but later repaired its view and praised it, after the team handles it, it becomes",
            "If the archive first praised the apple but later reversed itself and doubted it, after the team handles it, it becomes",
        )
    if difficulty == "easy":
        return (
            "The apple is the better fruit example than the pear, so after the team handles it, it becomes",
            "The pear is the better fruit example than the apple, so after the team handles it, it becomes",
        )
    if difficulty == "medium":
        return (
            "Because the apple is the better fruit example than the pear, after the team handles it, it becomes",
            "Because the pear is the better fruit example than the apple, after the team handles it, it becomes",
        )
    if difficulty == "hard":
        return (
            "Although the pear looks familiar, the apple is the better fruit example, so after the team handles it, it becomes",
            "Although the pear looks familiar, the apple is still treated as the worse example, so after the team handles it, it becomes",
        )
    return (
        "If both the apple and the pear appear but the apple fits the fruit category better, after the team handles it, it becomes",
        "If both the apple and the pear appear but the apple fits the fruit category worse, after the team handles it, it becomes",
    )


def single_token_candidates(tokenizer, words: List[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for word in words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) == 1:
            out.append((word, int(token_ids[0])))
    return out


def margin_for_prompt(model, tokenizer, prompt: str, positive_ids: List[int], negative_ids: List[int]) -> float:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1].float().cpu()
    positive_mean = float(logits[positive_ids].mean().item())
    negative_mean = float(logits[negative_ids].mean().item())
    return positive_mean - negative_mean


def build_summary() -> Dict[str, object]:
    model, tokenizer = load_model()
    positive_candidates = single_token_candidates(tokenizer, POSITIVE_WORDS)
    negative_candidates = single_token_candidates(tokenizer, NEGATIVE_WORDS)
    if not positive_candidates or not negative_candidates:
        raise RuntimeError("苹果上下文偏置探针未找到足够的单 token 候选结果词")
    positive_ids = [token_id for _, token_id in positive_candidates]
    negative_ids = [token_id for _, token_id in negative_candidates]

    pair_rows: List[Dict[str, object]] = []
    for family_name in FAMILIES:
        for difficulty in DIFFICULTIES:
            positive_prompt, negative_prompt = build_prompt_pair(family_name, difficulty)
            positive_margin = margin_for_prompt(model, tokenizer, positive_prompt, positive_ids, negative_ids)
            negative_margin = margin_for_prompt(model, tokenizer, negative_prompt, positive_ids, negative_ids)
            shift = positive_margin - negative_margin
            pair_rows.append(
                {
                    "family_name": family_name,
                    "difficulty": difficulty,
                    "positive_margin": positive_margin,
                    "negative_margin": negative_margin,
                    "shift": shift,
                }
            )

    mean_shift = mean(float(row["shift"]) for row in pair_rows)
    positive_shift_rate = sum(1 for row in pair_rows if float(row["shift"]) > 0.0) / len(pair_rows)
    context_bias_shift_score = clamp01(0.70 * (0.5 + 0.5 * math.tanh(mean_shift / 2.5)) + 0.30 * positive_shift_rate)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage156_apple_context_bias_shift",
        "title": "苹果上下文偏置扭转探针",
        "status_short": "apple_context_bias_shift_ready",
        "prompt_pair_count": len(pair_rows),
        "positive_candidate_words": [word for word, _ in positive_candidates],
        "negative_candidate_words": [word for word, _ in negative_candidates],
        "mean_shift": mean_shift,
        "positive_shift_rate": positive_shift_rate,
        "context_bias_shift_score": context_bias_shift_score,
        "pair_rows": pair_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage156: 苹果上下文偏置扭转探针",
        "",
        "## 核心结果",
        f"- 提示对数量: {summary['prompt_pair_count']}",
        f"- 平均偏置位移: {summary['mean_shift']:.4f}",
        f"- 正向位移占比: {summary['positive_shift_rate']:.4f}",
        f"- 偏置扭转分数: {summary['context_bias_shift_score']:.4f}",
        f"- 正向候选词: {', '.join(summary['positive_candidate_words'])}",
        f"- 负向候选词: {', '.join(summary['negative_candidate_words'])}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果上下文偏置扭转探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
