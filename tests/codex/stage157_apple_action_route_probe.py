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
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE157_APPLE_ACTION_ROUTE_PROBE_REPORT.md"

FAMILIES = ["harvest_scene", "kitchen_scene", "store_scene", "repair_scene", "adversarial_scene"]
DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]
FRUIT_ACTIONS = ["eat", "wash", "slice", "peel"]
BRAND_ACTIONS = ["buy", "charge", "update", "install"]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_prompt_pair(family_name: str, difficulty: str) -> Tuple[str, str]:
    if family_name == "harvest_scene":
        if difficulty == "easy":
            return (
                "The basket held the apple from the orchard, so the cook will",
                "The desk held the Apple device from the store, so the engineer will",
            )
        if difficulty == "medium":
            return (
                "Because the basket held the apple from the orchard, the cook will",
                "Because the desk held the Apple device from the store, the engineer will",
            )
        if difficulty == "hard":
            return (
                "Although the report mentioned the pear, the basket still held the apple from the orchard, so the cook will",
                "Although the report mentioned the phone, the desk still held the Apple device from the store, so the engineer will",
            )
        return (
            "If the basket held both the apple and the pear from the orchard, the cook will",
            "If the desk held both the Apple device and the phone from the store, the engineer will",
        )
    if family_name == "kitchen_scene":
        if difficulty == "easy":
            return (
                "The recipe focused on the apple, so the chef will",
                "The support note focused on the Apple device, so the technician will",
            )
        if difficulty == "medium":
            return (
                "Because the recipe focused on the apple beside the pear, the chef will",
                "Because the support note focused on the Apple device beside the phone, the technician will",
            )
        if difficulty == "hard":
            return (
                "Although the menu mentioned the pear, the recipe focused on the apple, so the chef will",
                "Although the support guide mentioned the phone, it focused on the Apple device, so the technician will",
            )
        return (
            "If both the apple and the pear are on the table, the chef will",
            "If both the Apple device and the phone are on the desk, the technician will",
        )
    if family_name == "store_scene":
        if difficulty == "easy":
            return (
                "The market displayed the apple, so the shopper will",
                "The shop displayed the Apple device, so the customer will",
            )
        if difficulty == "medium":
            return (
                "Because the market displayed the apple near the pear, the shopper will",
                "Because the shop displayed the Apple device near the phone, the customer will",
            )
        if difficulty == "hard":
            return (
                "Although the sign highlighted the pear, the market still displayed the apple, so the shopper will",
                "Although the sign highlighted the phone, the shop still displayed the Apple device, so the customer will",
            )
        return (
            "If the market displays both the apple and the pear, the shopper will",
            "If the shop displays both the Apple device and the phone, the customer will",
        )
    if family_name == "repair_scene":
        if difficulty == "easy":
            return (
                "The cook checked the apple after lunch, so the cook will",
                "The engineer checked the Apple device after lunch, so the engineer will",
            )
        if difficulty == "medium":
            return (
                "Because the cook checked the apple after lunch, the cook will",
                "Because the engineer checked the Apple device after lunch, the engineer will",
            )
        if difficulty == "hard":
            return (
                "Although the note mentioned the pear, the cook checked the apple after lunch, so the cook will",
                "Although the note mentioned the phone, the engineer checked the Apple device after lunch, so the engineer will",
            )
        return (
            "If the cook checked both the apple and the pear after lunch, the cook will",
            "If the engineer checked both the Apple device and the phone after lunch, the engineer will",
        )
    if difficulty == "easy":
        return (
            "The team discussed the apple for dinner, so it will",
            "The team discussed the Apple device for testing, so it will",
        )
    if difficulty == "medium":
        return (
            "Because the team discussed the apple for dinner, it will",
            "Because the team discussed the Apple device for testing, it will",
        )
    if difficulty == "hard":
        return (
            "Although the team mentioned the pear, it discussed the apple for dinner, so it will",
            "Although the team mentioned the phone, it discussed the Apple device for testing, so it will",
        )
    return (
        "If the team discussed both the apple and the pear for dinner, it will",
        "If the team discussed both the Apple device and the phone for testing, it will",
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
    return float(logits[positive_ids].mean().item() - logits[negative_ids].mean().item())


def build_summary() -> Dict[str, object]:
    model, tokenizer = load_model()
    fruit_candidates = single_token_candidates(tokenizer, FRUIT_ACTIONS)
    brand_candidates = single_token_candidates(tokenizer, BRAND_ACTIONS)
    if not fruit_candidates or not brand_candidates:
        raise RuntimeError("苹果动作选路探针缺少足够的单 token 动作候选词")
    fruit_ids = [token_id for _, token_id in fruit_candidates]
    brand_ids = [token_id for _, token_id in brand_candidates]

    pair_rows: List[Dict[str, object]] = []
    for family_name in FAMILIES:
        for difficulty in DIFFICULTIES:
            fruit_prompt, brand_prompt = build_prompt_pair(family_name, difficulty)
            fruit_margin = margin_for_prompt(model, tokenizer, fruit_prompt, fruit_ids, brand_ids)
            brand_margin = margin_for_prompt(model, tokenizer, brand_prompt, fruit_ids, brand_ids)
            route_shift = fruit_margin - brand_margin
            pair_rows.append(
                {
                    "family_name": family_name,
                    "difficulty": difficulty,
                    "fruit_margin": fruit_margin,
                    "brand_margin": brand_margin,
                    "route_shift": route_shift,
                }
            )

    mean_route_shift = mean(float(row["route_shift"]) for row in pair_rows)
    positive_route_rate = sum(1 for row in pair_rows if float(row["route_shift"]) > 0.0) / len(pair_rows)
    action_route_score = clamp01(0.70 * (0.5 + 0.5 * math.tanh(mean_route_shift / 2.5)) + 0.30 * positive_route_rate)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage157_apple_action_route_probe",
        "title": "苹果动作选路探针",
        "status_short": "apple_action_route_ready",
        "prompt_pair_count": len(pair_rows),
        "fruit_action_words": [word for word, _ in fruit_candidates],
        "brand_action_words": [word for word, _ in brand_candidates],
        "mean_route_shift": mean_route_shift,
        "positive_route_rate": positive_route_rate,
        "apple_action_route_score": action_route_score,
        "pair_rows": pair_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage157: 苹果动作选路探针",
        "",
        "## 核心结果",
        f"- 提示对数量: {summary['prompt_pair_count']}",
        f"- 平均动作选路位移: {summary['mean_route_shift']:.4f}",
        f"- 正向位移占比: {summary['positive_route_rate']:.4f}",
        f"- 动作选路分数: {summary['apple_action_route_score']:.4f}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果动作选路探针")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
