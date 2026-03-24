#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage233_brand_parameter_trigger_lattice_20260324"

STAGE227_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage227_brand_trigger_word_map_20260324" / "summary.json"
STAGE230_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage230_parameter_boundary_trigger_pack_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s227 = load_json(STAGE227_SUMMARY_PATH)
    s230 = load_json(STAGE230_SUMMARY_PATH)

    piece_map = {str(row["piece_name"]): float(row["score"]) for row in s230["piece_rows"]}
    trigger_mean = sum(float(row["score"]) for row in s227["trigger_rows"]) / len(s227["trigger_rows"])
    best_trigger_score = max(float(row["score"]) for row in s227["trigger_rows"])

    lattice_rows = [
        {"piece_name": "品牌义强触发晶格", "score": best_trigger_score},
        {"piece_name": "品牌义平均触发地板", "score": trigger_mean},
        {"piece_name": "品牌义参数边界支撑", "score": piece_map["品牌义参数触发支持"]},
        {"piece_name": "品牌义边界分裂就绪度", "score": piece_map["参数边界分裂就绪度"]},
        {"piece_name": "水果义压制背景", "score": piece_map["水果义参数占优"]},
    ]
    ranked_rows = sorted(lattice_rows, key=lambda row: float(row["score"]), reverse=True)
    lattice_score = sum(float(row["score"]) for row in lattice_rows) / len(lattice_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage233_brand_parameter_trigger_lattice",
        "title": "品牌义参数触发晶格",
        "status_short": "brand_parameter_trigger_lattice_ready",
        "piece_count": len(lattice_rows),
        "lattice_score": lattice_score,
        "strongest_piece_name": str(ranked_rows[0]["piece_name"]),
        "weakest_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "品牌义参数晶格仍然稀薄",
        "best_trigger_word": str(s227["best_trigger_word"]),
        "lattice_rows": lattice_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage233：品牌义参数触发晶格",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 晶格总分：{summary['lattice_score']:.4f}",
        f"- 最强部件：{summary['strongest_piece_name']}",
        f"- 最弱部件：{summary['weakest_piece_name']}",
        f"- 最强触发词：{summary['best_trigger_word']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE233_BRAND_PARAMETER_TRIGGER_LATTICE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="品牌义参数触发晶格")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
