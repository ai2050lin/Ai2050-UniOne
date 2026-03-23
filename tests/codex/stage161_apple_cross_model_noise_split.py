#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage161_apple_cross_model_noise_split_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE161_APPLE_CROSS_MODEL_NOISE_SPLIT_REPORT.md"
STAGE159_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323" / "summary.json"

BRAND_WORDS = {"iphone", "icloud", "appstore", "mac", "ipad", "device", "software"}
FOOD_WORDS = {"cider", "potato", "tomato", "onion", "vegetable", "wheat", "carrot"}
FRUIT_WORDS = {"fruit", "banana", "peach", "strawberry", "lemon", "avocado", "apple", "apples", "pineapple", "pear", "grape", "melon", "berry", "cherry"}


def classify_neighbor(word: str, group: str, gpt2_shared_words: set[str]) -> str:
    lowered = word.lower()
    if lowered in gpt2_shared_words:
        return "shared_core"
    if lowered in BRAND_WORDS:
        return "brand_noise"
    if lowered in FOOD_WORDS:
        return "food_bleed"
    if lowered in FRUIT_WORDS:
        return "fruit_expansion"
    if not re.fullmatch(r"[a-z]+", lowered):
        return "token_noise"
    if lowered.startswith("app") or lowered.endswith("module") or lowered.endswith("id"):
        return "string_noise"
    return "open_noise"


def build_summary() -> Dict[str, object]:
    summary = json.loads(STAGE159_SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    model_rows = summary["model_rows"]
    gpt2_neighbors = next(row["top_fruit_neighbors"] for row in model_rows if row["model_name"] == "GPT-2")
    gpt2_shared_words = {str(item["word"]).lower() for item in gpt2_neighbors}

    split_rows: List[Dict[str, object]] = []
    for row in model_rows:
        model_name = str(row["model_name"])
        classified = []
        for item in row["top_fruit_neighbors"]:
            label = classify_neighbor(str(item["word"]), str(item["group"]), gpt2_shared_words)
            classified.append(
                {
                    "word": str(item["word"]),
                    "group": str(item["group"]),
                    "similarity": float(item["similarity"]),
                    "noise_label": label,
                }
            )
        counts: Dict[str, int] = {}
        for item in classified:
            counts[item["noise_label"]] = counts.get(item["noise_label"], 0) + 1
        clean_ratio = counts.get("shared_core", 0) + counts.get("fruit_expansion", 0)
        clean_ratio = clean_ratio / max(1, len(classified))
        noise_ratio = 1.0 - clean_ratio
        split_rows.append(
            {
                "model_name": model_name,
                "clean_ratio": clean_ratio,
                "noise_ratio": noise_ratio,
                "label_counts": counts,
                "classified_neighbors": classified,
            }
        )

    strongest_clean = max(split_rows, key=lambda row: float(row["clean_ratio"]))
    weakest_clean = min(split_rows, key=lambda row: float(row["clean_ratio"]))
    noise_split_score = mean(float(row["clean_ratio"]) for row in split_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage161_apple_cross_model_noise_split",
        "title": "苹果跨模型噪声拆分",
        "status_short": "apple_cross_model_noise_split_ready",
        "model_count": len(split_rows),
        "mean_clean_ratio": noise_split_score,
        "mean_noise_ratio": 1.0 - noise_split_score,
        "strongest_clean_model_name": str(strongest_clean["model_name"]),
        "weakest_clean_model_name": str(weakest_clean["model_name"]),
        "split_rows": split_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage161: 苹果跨模型噪声拆分",
        "",
        "## 核心结果",
        f"- 模型数: {summary['model_count']}",
        f"- 平均干净比率: {summary['mean_clean_ratio']:.4f}",
        f"- 平均噪声比率: {summary['mean_noise_ratio']:.4f}",
        f"- 最干净模型: {summary['strongest_clean_model_name']}",
        f"- 最脏模型: {summary['weakest_clean_model_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果跨模型噪声拆分")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
