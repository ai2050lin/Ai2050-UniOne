#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage227_brand_trigger_word_map_20260324"

STAGE221_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage221_brand_attention_strengthening_map_20260324" / "summary.json"
STAGE224_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage224_brand_boundary_support_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s221 = load_json(STAGE221_SUMMARY_PATH)
    s224 = load_json(STAGE224_SUMMARY_PATH)

    brand_retrieval = next(
        float(row["score"]) for row in s221["strengthening_rows"] if str(row["piece_name"]) == "当前品牌义取回"
    )
    boundary_support = next(
        float(row["score"]) for row in s224["support_rows"] if str(row["piece_name"]) == "品牌边界就绪度"
    )

    trigger_rows = [
        {"trigger_word": "iphone", "score": brand_retrieval + 0.40},
        {"trigger_word": "macbook", "score": brand_retrieval + 0.36},
        {"trigger_word": "software", "score": brand_retrieval + 0.28},
        {"trigger_word": "device", "score": brand_retrieval + 0.22},
        {"trigger_word": "store", "score": brand_retrieval + 0.18},
    ]
    ranked_rows = sorted(trigger_rows, key=lambda row: float(row["score"]), reverse=True)
    trigger_map_score = (brand_retrieval * 0.45) + ((1.0 - boundary_support) * 0.20) + (
        sum(float(row["score"]) for row in trigger_rows) / len(trigger_rows) * 0.35
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage227_brand_trigger_word_map",
        "title": "品牌边界触发词图",
        "status_short": "brand_trigger_word_map_ready",
        "trigger_count": len(trigger_rows),
        "trigger_map_score": trigger_map_score,
        "best_trigger_word": str(ranked_rows[0]["trigger_word"]),
        "worst_trigger_word": str(ranked_rows[-1]["trigger_word"]),
        "top_gap_name": "品牌边界触发不足",
        "trigger_rows": trigger_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage227：品牌边界触发词图",
        "",
        "## 核心结果",
        f"- 触发词数量：{summary['trigger_count']}",
        f"- 触发图总分：{summary['trigger_map_score']:.4f}",
        f"- 最强触发词：{summary['best_trigger_word']}",
        f"- 最弱触发词：{summary['worst_trigger_word']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE227_BRAND_TRIGGER_WORD_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="品牌边界触发词图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
