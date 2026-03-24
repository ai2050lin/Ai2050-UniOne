#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage228_natural_fidelity_gain_map_20260324"

STAGE222_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage222_attention_source_fidelity_bridge_20260324" / "summary.json"
STAGE223_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage223_natural_vs_repair_closure_split_map_20260324" / "summary.json"
STAGE225_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage225_attention_to_natural_fidelity_chain_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def row_score(rows: list[dict], name: str) -> float:
    for row in rows:
        key = "piece_name" if "piece_name" in row else "trigger_word"
        if str(row[key]) == name:
            return float(row["score"])
    raise KeyError(name)


def build_summary() -> dict:
    s222 = load_json(STAGE222_SUMMARY_PATH)
    s223 = load_json(STAGE223_SUMMARY_PATH)
    s225 = load_json(STAGE225_SUMMARY_PATH)

    natural_fidelity = row_score(s222["bridge_rows"], "天然来源保真")
    repair_closure = row_score(s223["split_rows"], "修复闭合")
    natural_constraint = row_score(s225["chain_rows"], "天然断裂约束")

    gain_rows = [
        {"piece_name": "当前天然保真", "score": natural_fidelity},
        {"piece_name": "修复增益对照", "score": repair_closure - natural_fidelity},
        {"piece_name": "断裂解除空间", "score": 1.0 - natural_constraint},
        {"piece_name": "理论提升空间", "score": max(0.0, repair_closure - natural_fidelity * 0.5)},
    ]
    ranked_rows = sorted(gain_rows, key=lambda row: float(row["score"]), reverse=True)
    gain_map_score = sum(float(row["score"]) for row in gain_rows) / len(gain_rows)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage228_natural_fidelity_gain_map",
        "title": "天然保真增强图",
        "status_short": "natural_fidelity_gain_map_ready",
        "piece_count": len(gain_rows),
        "gain_map_score": gain_map_score,
        "best_gain_piece_name": str(ranked_rows[0]["piece_name"]),
        "worst_gain_piece_name": str(ranked_rows[-1]["piece_name"]),
        "top_gap_name": "天然来源保真不足",
        "gain_rows": gain_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage228：天然保真增强图",
        "",
        "## 核心结果",
        f"- 部件数量：{summary['piece_count']}",
        f"- 增强图总分：{summary['gain_map_score']:.4f}",
        f"- 最强增强块：{summary['best_gain_piece_name']}",
        f"- 最弱增强块：{summary['worst_gain_piece_name']}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE228_NATURAL_FIDELITY_GAIN_MAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="天然保真增强图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
