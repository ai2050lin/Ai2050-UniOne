#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from stage119_gpt2_embedding_full_vocab_scan import discover_tokenizer
from stage121_adverb_gate_bridge_probe import ensure_stage119_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE119_OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage119_gpt2_embedding_full_vocab_scan_20260323"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE155_APPLE_BOUNDARY_CRACK_MAP_REPORT.md"

BOUNDARY_CASES = [
    ("apple_literal", "apple", "apple", "fruit"),
    ("apple_literal", "fresh apple", "apple", "fruit"),
    ("apple_literal", "ripe apple", "apple", "fruit"),
    ("apple_literal", "the apple", "apple", "fruit"),
    ("apple_food_phrase", "apple pie", "apple", "food_phrase"),
    ("apple_food_phrase", "apple juice", "apple", "food_phrase"),
    ("apple_food_phrase", "apple cake", "apple", "food_phrase"),
    ("apple_food_phrase", "apple sauce", "apple", "food_phrase"),
    ("apple_brand_phrase", "Apple device", "apple", "brand"),
    ("apple_brand_phrase", "Apple software", "apple", "brand"),
    ("apple_brand_phrase", "Apple store", "apple", "brand"),
    ("apple_brand_phrase", "Apple company", "apple", "brand"),
    ("orange_color_phrase", "orange ball", "orange", "color"),
    ("orange_color_phrase", "orange light", "orange", "color"),
    ("orange_color_phrase", "orange paint", "orange", "color"),
    ("orange_color_phrase", "orange shirt", "orange", "color"),
    ("orange_fruit_phrase", "orange fruit", "orange", "fruit"),
    ("orange_fruit_phrase", "sweet orange", "orange", "fruit"),
    ("orange_fruit_phrase", "ripe orange", "orange", "fruit"),
    ("orange_fruit_phrase", "fresh orange", "orange", "fruit"),
]


def resolve_anchor(tokens: List[int], tokenizer, row_map: Dict[str, Dict[str, object]], probe_word: str) -> Dict[str, object] | None:
    for token_id in tokens:
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip().lower()
        if decoded == probe_word and decoded in row_map:
            row = row_map[decoded]
            return {
                "word": str(row["word"]),
                "group": str(row["group"]),
                "band": str(row["band"]),
                "lexical_type": str(row["lexical_type"]),
            }
    return None


def expected_anchor(expected_sense: str) -> str:
    return {
        "fruit": "meso_fruit",
        "food_phrase": "meso_food",
        "brand": "brand_like",
        "color": "micro_color",
    }[expected_sense]


def build_summary() -> Dict[str, object]:
    _, rows = ensure_stage119_rows(STAGE119_OUTPUT_DIR)
    tokenizer = discover_tokenizer()
    row_map = {str(row["word"]).lower(): row for row in rows}
    case_rows: List[Dict[str, object]] = []
    collision_count = 0
    for family_name, phrase, probe_word, expected_sense in BOUNDARY_CASES:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        anchor = resolve_anchor(token_ids, tokenizer, row_map, probe_word)
        expected = expected_anchor(expected_sense)
        actual = str(anchor["group"]) if anchor else "none"
        collision = actual != expected
        if collision:
            collision_count += 1
        case_rows.append(
            {
                "family_name": family_name,
                "phrase": phrase,
                "probe_word": probe_word,
                "expected_sense": expected_sense,
                "expected_anchor": expected,
                "token_count": len(token_ids),
                "resolved_anchor_group": actual,
                "resolved_anchor_word": str(anchor["word"]) if anchor else "none",
                "collision": collision,
            }
        )
    family_names = sorted({row["family_name"] for row in case_rows})
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage155_apple_boundary_crack_map",
        "title": "苹果边界裂缝图",
        "status_short": "apple_boundary_crack_map_ready",
        "case_count": len(case_rows),
        "family_count": len(family_names),
        "mean_token_count": mean(float(row["token_count"]) for row in case_rows),
        "collision_count": collision_count,
        "collision_rate": collision_count / max(1, len(case_rows)),
        "case_rows": case_rows,
    }


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage155: 苹果边界裂缝图",
        "",
        "## 核心结果",
        f"- 边界案例数: {summary['case_count']}",
        f"- 家族数: {summary['family_count']}",
        f"- 平均分词数: {summary['mean_token_count']:.4f}",
        f"- 锚点冲突数: {summary['collision_count']}",
        f"- 锚点冲突率: {summary['collision_rate']:.4f}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> Dict[str, object]:
    if not force and SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果边界裂缝图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
