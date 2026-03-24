#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage218_apple_sense_attention_retrieval_map_20260324"

STAGE155_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json"
STAGE157_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json"
STAGE182_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage182_boundary_crack_puzzle_expansion_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def family_readiness(summary: dict, family_name: str) -> float:
    for row in summary["family_rows"]:
        if str(row["family_name"]) == family_name:
            return float(row["boundary_readiness"])
    raise KeyError(family_name)


def build_summary() -> dict:
    s155 = load_json(STAGE155_SUMMARY_PATH)
    s157 = load_json(STAGE157_SUMMARY_PATH)
    s182 = load_json(STAGE182_SUMMARY_PATH)

    route_support = float(s157["apple_action_route_score"])
    literal_readiness = family_readiness(s182, "apple_literal")
    brand_readiness = family_readiness(s182, "apple_brand_phrase")
    food_readiness = family_readiness(s182, "apple_food_phrase")

    rows = [
        {
            "piece_name": "水果义取回",
            "score": literal_readiness * 0.60 + route_support * 0.40,
        },
        {
            "piece_name": "品牌义取回",
            "score": brand_readiness * 0.60 + route_support * 0.40,
        },
        {
            "piece_name": "食物短语取回",
            "score": food_readiness * 0.60 + route_support * 0.40,
        },
        {
            "piece_name": "上下文选路支撑",
            "score": route_support,
        },
    ]
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]))
    retrieval_score = sum(float(row["score"]) for row in rows) / float(len(rows))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage218_apple_sense_attention_retrieval_map",
        "title": "苹果义项注意力取回图",
        "status_short": "apple_sense_attention_retrieval_ready",
        "piece_count": len(rows),
        "collision_rate": float(s155["collision_rate"]),
        "retrieval_score": retrieval_score,
        "weakest_piece_name": str(ranked_rows[0]["piece_name"]),
        "strongest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "retrieval_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage218：苹果义项注意力取回图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 边界冲突率：{summary['collision_rate']:.4f}",
        f"- 取回图总分：{summary['retrieval_score']:.4f}",
        f"- 最弱义项取回：{summary['weakest_piece_name']}",
        f"- 最强义项取回：{summary['strongest_piece_name']}",
    ]
    (output_dir / "STAGE218_APPLE_SENSE_ATTENTION_RETRIEVAL_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="苹果义项注意力取回图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
