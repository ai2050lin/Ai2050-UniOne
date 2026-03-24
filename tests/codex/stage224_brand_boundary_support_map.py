#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage224_brand_boundary_support_map_20260324"

STAGE221_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage221_brand_attention_strengthening_map_20260324" / "summary.json"
STAGE182_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage182_boundary_crack_puzzle_expansion_20260323" / "summary.json"
STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def family_row(summary: dict, family_name: str) -> dict:
    for row in summary["family_rows"]:
        if str(row["family_name"]) == family_name:
            return row
    raise KeyError(family_name)


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s221 = load_json(STAGE221_SUMMARY_PATH)
    s182 = load_json(STAGE182_SUMMARY_PATH)
    s155 = load_json(STAGE155_SUMMARY_PATH)

    brand_row = family_row(s182, "apple_brand_phrase")
    literal_row = family_row(s182, "apple_literal")
    brand_attention = row_score(s221["strengthening_rows"], "当前品牌义取回")
    brand_space = row_score(s221["strengthening_rows"], "品牌补强空间")

    rows = [
        {"piece_name": "品牌边界就绪度", "score": float(brand_row["boundary_readiness"])},
        {"piece_name": "品牌义注意力取回", "score": brand_attention},
        {"piece_name": "品牌补强空间", "score": brand_space},
        {"piece_name": "水果义边界对照", "score": float(literal_row["boundary_readiness"])},
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    support_score = (
        float(rows[0]["score"]) * 0.30
        + float(rows[1]["score"]) * 0.25
        + float(rows[2]["score"]) * 0.20
        + float(rows[3]["score"]) * 0.25
    )
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage224_brand_boundary_support_map",
        "title": "品牌义边界支撑图",
        "status_short": "brand_boundary_support_ready",
        "piece_count": len(rows),
        "collision_rate": float(s155["collision_rate"]),
        "support_score": support_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "品牌边界未站住",
        "support_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage224：品牌义边界支撑图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 冲突率：{summary['collision_rate']:.4f}",
        f"- 支撑图总分：{summary['support_score']:.4f}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE224_BRAND_BOUNDARY_SUPPORT_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="品牌义边界支撑图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
