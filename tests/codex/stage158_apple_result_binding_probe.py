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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE158_APPLE_RESULT_BINDING_PROBE_REPORT.md"

FAMILIES = ["simple_binding", "order_swap", "tool_interference", "repair_binding", "adversarial_binding"]
DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_prompt_pair(family_name: str, difficulty: str) -> Tuple[str, str]:
    if family_name == "simple_binding":
        if difficulty == "easy":
            return (
                "The team washed the apple, not the pear, so the item that became fresh was the",
                "The team washed the pear, not the apple, so the item that became fresh was the",
            )
        if difficulty == "medium":
            return (
                "After the team checked both fruits, it washed the apple, not the pear, so the item that became fresh was the",
                "After the team checked both fruits, it washed the pear, not the apple, so the item that became fresh was the",
            )
        if difficulty == "hard":
            return (
                "Although the report mentioned the pear, the team washed the apple, so the item that became fresh was the",
                "Although the report mentioned the apple, the team washed the pear, so the item that became fresh was the",
            )
        return (
            "If both the apple and the pear stayed on the table, the team washed the apple, so the item that became fresh was the",
            "If both the apple and the pear stayed on the table, the team washed the pear, so the item that became fresh was the",
        )
    if family_name == "order_swap":
        if difficulty == "easy":
            return (
                "The team moved the pear first but sliced the apple later, so the item that became ready was the",
                "The team moved the apple first but sliced the pear later, so the item that became ready was the",
            )
        if difficulty == "medium":
            return (
                "After moving the pear first, the team sliced the apple later, so the item that became ready was the",
                "After moving the apple first, the team sliced the pear later, so the item that became ready was the",
            )
        if difficulty == "hard":
            return (
                "Although the report emphasized the pear, the team sliced the apple later, so the item that became ready was the",
                "Although the report emphasized the apple, the team sliced the pear later, so the item that became ready was the",
            )
        return (
            "If the team handled both fruits but sliced the apple later, the item that became ready was the",
            "If the team handled both fruits but sliced the pear later, the item that became ready was the",
        )
    if family_name == "tool_interference":
        if difficulty == "easy":
            return (
                "The team used the knife on the apple, not the pear, so the item that became served was the",
                "The team used the knife on the pear, not the apple, so the item that became served was the",
            )
        if difficulty == "medium":
            return (
                "After moving the knife, the team used it on the apple, not the pear, so the item that became served was the",
                "After moving the knife, the team used it on the pear, not the apple, so the item that became served was the",
            )
        if difficulty == "hard":
            return (
                "Although the note mentioned the pear, the team used the knife on the apple, so the item that became served was the",
                "Although the note mentioned the apple, the team used the knife on the pear, so the item that became served was the",
            )
        return (
            "If the knife touched both fruits but was finally used on the apple, the item that became served was the",
            "If the knife touched both fruits but was finally used on the pear, the item that became served was the",
        )
    if family_name == "repair_binding":
        if difficulty == "easy":
            return (
                "The team first chose the pear, then corrected itself and stored the apple, so the item that was saved was the",
                "The team first chose the apple, then corrected itself and stored the pear, so the item that was saved was the",
            )
        if difficulty == "medium":
            return (
                "After first choosing the pear, the team corrected itself and stored the apple, so the item that was saved was the",
                "After first choosing the apple, the team corrected itself and stored the pear, so the item that was saved was the",
            )
        if difficulty == "hard":
            return (
                "Although the archive pushed the pear, the team corrected itself and stored the apple, so the item that was saved was the",
                "Although the archive pushed the apple, the team corrected itself and stored the pear, so the item that was saved was the",
            )
        return (
            "If the archive first pushed the pear but the team later stored the apple, the item that was saved was the",
            "If the archive first pushed the apple but the team later stored the pear, the item that was saved was the",
        )
    if difficulty == "easy":
        return (
            "The team discussed the apple and the pear, but only packed the apple, so the item that was chosen was the",
            "The team discussed the apple and the pear, but only packed the pear, so the item that was chosen was the",
        )
    if difficulty == "medium":
        return (
            "Because the team discussed both fruits but only packed the apple, the item that was chosen was the",
            "Because the team discussed both fruits but only packed the pear, the item that was chosen was the",
        )
    if difficulty == "hard":
        return (
            "Although the report highlighted the pear, the team only packed the apple, so the item that was chosen was the",
            "Although the report highlighted the apple, the team only packed the pear, so the item that was chosen was the",
        )
    return (
        "If the report mentioned the pear, the knife, and the basket, but the team only packed the apple, the item that was chosen was the",
        "If the report mentioned the apple, the knife, and the basket, but the team only packed the pear, the item that was chosen was the",
    )


def single_token_id(tokenizer, word: str) -> int:
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"结果绑定探针候选词不是单 token: {word}")
    return int(token_ids[0])


def margin_for_prompt(model, tokenizer, prompt: str, apple_id: int, pear_id: int) -> float:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1].float().cpu()
    return float(logits[apple_id].item() - logits[pear_id].item())


def build_summary() -> Dict[str, object]:
    model, tokenizer = load_model()
    apple_id = single_token_id(tokenizer, "apple")
    pear_id = single_token_id(tokenizer, "pear")
    pair_rows: List[Dict[str, object]] = []
    for family_name in FAMILIES:
        for difficulty in DIFFICULTIES:
            apple_prompt, pear_prompt = build_prompt_pair(family_name, difficulty)
            apple_margin = margin_for_prompt(model, tokenizer, apple_prompt, apple_id, pear_id)
            pear_margin = margin_for_prompt(model, tokenizer, pear_prompt, apple_id, pear_id)
            binding_shift = apple_margin - pear_margin
            pair_rows.append(
                {
                    "family_name": family_name,
                    "difficulty": difficulty,
                    "apple_target_margin": apple_margin,
                    "pear_target_margin": pear_margin,
                    "binding_shift": binding_shift,
                }
            )
    mean_binding_shift = mean(float(row["binding_shift"]) for row in pair_rows)
    positive_binding_rate = sum(1 for row in pair_rows if float(row["binding_shift"]) > 0.0) / len(pair_rows)
    result_binding_score = clamp01(0.70 * (0.5 + 0.5 * math.tanh(mean_binding_shift / 2.5)) + 0.30 * positive_binding_rate)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage158_apple_result_binding_probe",
        "title": "苹果结果绑定探针",
        "status_short": "apple_result_binding_ready",
        "prompt_pair_count": len(pair_rows),
        "candidate_words": ["apple", "pear"],
        "mean_binding_shift": mean_binding_shift,
        "positive_binding_rate": positive_binding_rate,
        "apple_result_binding_score": result_binding_score,
        "pair_rows": pair_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage158: 苹果结果绑定探针",
        "",
        "## 核心结果",
        f"- 提示对数量: {summary['prompt_pair_count']}",
        f"- 平均结果绑定位移: {summary['mean_binding_shift']:.4f}",
        f"- 正向绑定占比: {summary['positive_binding_rate']:.4f}",
        f"- 结果绑定分数: {summary['apple_result_binding_score']:.4f}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果结果绑定探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
