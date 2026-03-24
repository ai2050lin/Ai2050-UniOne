#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage221_brand_attention_strengthening_map_20260324"

STAGE218_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage218_apple_sense_attention_retrieval_map_20260324" / "summary.json"
STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"
STAGE182_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage182_boundary_crack_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def family_collision(summary: dict, family_name: str) -> float:
    for row in summary["family_rows"]:
        if str(row["family_name"]) == family_name:
            return float(row["collision_rate"])
    raise KeyError(family_name)


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s218 = load_json(STAGE218_SUMMARY_PATH)
    s155 = load_json(STAGE155_SUMMARY_PATH)
    s182 = load_json(STAGE182_SUMMARY_PATH)

    brand_retrieval = row_score(s218["retrieval_rows"], "品牌义取回")
    fruit_retrieval = row_score(s218["retrieval_rows"], "水果义取回")
    collision_rate = family_collision(s182, "apple_brand_phrase")

    rows = [
        {"piece_name": "当前品牌义取回", "score": brand_retrieval},
        {"piece_name": "水果义取回对照", "score": fruit_retrieval},
        {"piece_name": "品牌边界冲突反向值", "score": 1.0 - collision_rate},
        {"piece_name": "品牌补强空间", "score": max(0.0, fruit_retrieval - brand_retrieval)},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    strengthening_score = (
        brand_retrieval * 0.35
        + (1.0 - collision_rate) * 0.25
        + max(0.0, fruit_retrieval - brand_retrieval) * 0.40
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage221_brand_attention_strengthening_map",
        "title": "品牌义注意力补强图",
        "status_short": "brand_attention_strengthening_ready",
        "piece_count": len(rows),
        "strengthening_score": strengthening_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "品牌义取回不足",
        "strengthening_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage221：品牌义注意力补强图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 补强图总分：{summary['strengthening_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE221_BRAND_ATTENTION_STRENGTHENING_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="品牌义注意力补强图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
