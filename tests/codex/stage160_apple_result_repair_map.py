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
from stage158_apple_result_binding_probe import FAMILIES, DIFFICULTIES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE160_APPLE_RESULT_REPAIR_MAP_REPORT.md"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def base_prompt(family_name: str, difficulty: str) -> str:
    if family_name == "simple_binding":
        return {
            "easy": "The team washed the apple, not the pear, so the item that became fresh was the",
            "medium": "After the team checked both fruits, it washed the apple, not the pear, so the item that became fresh was the",
            "hard": "Although the report mentioned the pear, the team washed the apple, so the item that became fresh was the",
            "adversarial": "If both the apple and the pear stayed on the table, the team washed the apple, so the item that became fresh was the",
        }[difficulty]
    if family_name == "order_swap":
        return {
            "easy": "The team moved the pear first but sliced the apple later, so the item that became ready was the",
            "medium": "After moving the pear first, the team sliced the apple later, so the item that became ready was the",
            "hard": "Although the report emphasized the pear, the team sliced the apple later, so the item that became ready was the",
            "adversarial": "If the team handled both fruits but sliced the apple later, the item that became ready was the",
        }[difficulty]
    if family_name == "tool_interference":
        return {
            "easy": "The team used the knife on the apple, not the pear, so the item that became served was the",
            "medium": "After moving the knife, the team used it on the apple, not the pear, so the item that became served was the",
            "hard": "Although the note mentioned the pear, the team used the knife on the apple, so the item that became served was the",
            "adversarial": "If the knife touched both fruits but was finally used on the apple, the item that became served was the",
        }[difficulty]
    if family_name == "repair_binding":
        return {
            "easy": "The team first chose the pear, then corrected itself and stored the apple, so the item that was saved was the",
            "medium": "After first choosing the pear, the team corrected itself and stored the apple, so the item that was saved was the",
            "hard": "Although the archive pushed the pear, the team corrected itself and stored the apple, so the item that was saved was the",
            "adversarial": "If the archive first pushed the pear but the team later stored the apple, the item that was saved was the",
        }[difficulty]
    return {
        "easy": "The team discussed the apple and the pear, but only packed the apple, so the item that was chosen was the",
        "medium": "Because the team discussed both fruits but only packed the apple, the item that was chosen was the",
        "hard": "Although the report highlighted the pear, the team only packed the apple, so the item that was chosen was the",
        "adversarial": "If the report mentioned the pear, the knife, and the basket, but the team only packed the apple, the item that was chosen was the",
    }[difficulty]


def repair_prompt(prompt: str, family_name: str, difficulty: str) -> str:
    if family_name == "simple_binding":
        repair = " The apple was the one that changed."
    elif family_name == "order_swap":
        repair = " The apple, not the earlier pear, was the later changed fruit."
    elif family_name == "tool_interference":
        repair = " The tool acted on the apple itself."
    elif family_name == "repair_binding":
        repair = " The final corrected choice was the apple."
    else:
        repair = " The chosen object was the apple."
    if difficulty == "adversarial":
        repair += " Ignore the distractors."
    return prompt + repair + " Therefore the item was the"


def single_token_id(tokenizer, word: str) -> int:
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"结果修复图候选词不是单 token: {word}")
    return int(token_ids[0])


def apple_margin(model, tokenizer, prompt: str, apple_id: int, pear_id: int) -> float:
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
            raw_prompt = base_prompt(family_name, difficulty)
            fixed_prompt = repair_prompt(raw_prompt, family_name, difficulty)
            raw_margin = apple_margin(model, tokenizer, raw_prompt, apple_id, pear_id)
            fixed_margin = apple_margin(model, tokenizer, fixed_prompt, apple_id, pear_id)
            repair_gain = fixed_margin - raw_margin
            pair_rows.append(
                {
                    "family_name": family_name,
                    "difficulty": difficulty,
                    "raw_margin": raw_margin,
                    "fixed_margin": fixed_margin,
                    "repair_gain": repair_gain,
                }
            )

    family_rows: List[Dict[str, object]] = []
    for family_name in FAMILIES:
        family_subset = [row for row in pair_rows if row["family_name"] == family_name]
        family_rows.append(
            {
                "family_name": family_name,
                "mean_raw_margin": mean(float(row["raw_margin"]) for row in family_subset),
                "mean_fixed_margin": mean(float(row["fixed_margin"]) for row in family_subset),
                "mean_repair_gain": mean(float(row["repair_gain"]) for row in family_subset),
                "positive_repair_rate": sum(1 for row in family_subset if float(row["repair_gain"]) > 0.0) / len(family_subset),
            }
        )
    family_rows.sort(key=lambda row: float(row["mean_repair_gain"]), reverse=True)

    mean_repair_gain = mean(float(row["repair_gain"]) for row in pair_rows)
    positive_repair_rate = sum(1 for row in pair_rows if float(row["repair_gain"]) > 0.0) / len(pair_rows)
    repair_map_score = clamp01(0.70 * (0.5 + 0.5 * math.tanh(mean_repair_gain / 2.5)) + 0.30 * positive_repair_rate)

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage160_apple_result_repair_map",
        "title": "苹果结果修复图",
        "status_short": "apple_result_repair_map_ready",
        "prompt_pair_count": len(pair_rows),
        "mean_repair_gain": mean_repair_gain,
        "positive_repair_rate": positive_repair_rate,
        "apple_result_repair_score": repair_map_score,
        "best_repair_family_name": str(family_rows[0]["family_name"]),
        "worst_repair_family_name": str(family_rows[-1]["family_name"]),
        "family_rows": family_rows,
        "pair_rows": pair_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage160: 苹果结果修复图",
        "",
        "## 核心结果",
        f"- 提示对数量: {summary['prompt_pair_count']}",
        f"- 平均修复增益: {summary['mean_repair_gain']:.4f}",
        f"- 正向修复占比: {summary['positive_repair_rate']:.4f}",
        f"- 结果修复分数: {summary['apple_result_repair_score']:.4f}",
        f"- 最强修复家族: {summary['best_repair_family_name']}",
        f"- 最弱修复家族: {summary['worst_repair_family_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果结果修复图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
