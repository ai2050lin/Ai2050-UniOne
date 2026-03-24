#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage230_parameter_boundary_trigger_pack_20260324"

STAGE221_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage221_brand_attention_strengthening_map_20260324" / "summary.json"
STAGE224_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage224_brand_boundary_support_map_20260324" / "summary.json"
STAGE227_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage227_brand_trigger_word_map_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pick_score(rows: list[dict], name: str) -> float:
    for row in rows:
        if str(row["piece_name"]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s221 = load_json(STAGE221_SUMMARY_PATH)
    s224 = load_json(STAGE224_SUMMARY_PATH)
    s227 = load_json(STAGE227_SUMMARY_PATH)

    fruit_retrieval = pick_score(s221["strengthening_rows"], "水果义取回对照")
    brand_conflict_reverse = pick_score(s221["strengthening_rows"], "品牌边界冲突反向值")
    brand_boundary_ready = pick_score(s224["support_rows"], "品牌边界就绪度")
    boundary_support = pick_score(s224["support_rows"], "品牌补强空间")

    trigger_scores = [float(row["score"]) for row in s227["trigger_rows"]]
    trigger_mean = sum(trigger_scores) / len(trigger_scores)

    piece_rows = [
        {"piece_name": "水果义参数占优", "score": fruit_retrieval},
        {"piece_name": "品牌义参数触发支持", "score": trigger_mean},
        {"piece_name": "参数边界分裂就绪度", "score": brand_boundary_ready},
        {"piece_name": "参数级品牌冲突缺口", "score": 1.0 - max(brand_conflict_reverse, boundary_support)},
    ]
    ranked_rows = sorted(piece_rows, key=lambda row: float(row["score"]), reverse=True)
    score = sum(float(row["score"]) for row in piece_rows) / len(piece_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage230_parameter_boundary_trigger_pack",
        "title": "参数级品牌边界触发包",
        "status_short": "parameter_boundary_trigger_pack_ready",
        "piece_count": len(piece_rows),
        "parameter_boundary_score": score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "品牌义参数边界仍未站住",
        "piece_rows": piece_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage230：参数级品牌边界触发包",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 参数边界总分：{summary['parameter_boundary_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE230_PARAMETER_BOUNDARY_TRIGGER_PACK_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="参数级品牌边界触发包")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
