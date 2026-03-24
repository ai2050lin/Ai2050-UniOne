#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage212_natural_trace_decay_map_20260323"

STAGE209_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage209_natural_trace_retention_map_20260323" / "summary.json"
STAGE206_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage206_retained_trace_transfer_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_decay(score: float) -> str:
    if score >= 0.75:
        return "稳定段"
    if score >= 0.45:
        return "过渡段"
    return "衰减段"


def build_summary() -> dict:
    s209 = load_json(STAGE209_SUMMARY_PATH)
    s206 = load_json(STAGE206_SUMMARY_PATH)

    rows = []
    for row in s209["segment_rows"]:
        score = float(row["score"])
        decay_strength = 1.0 - score
        rows.append(
            {
                "segment_name": str(row["segment_name"]),
                "retention_score": score,
                "decay_strength": decay_strength,
                "decay_status": classify_decay(score),
            }
        )

    ranked_rows = sorted(rows, key=lambda item: float(item["retention_score"]))
    retention_score = sum(float(item["retention_score"]) for item in rows) / float(len(rows))
    decay_map_score = 1.0 - retention_score
    continuity_gap = float(s206["continuity_gap"])
    strongest_decay_segment_name = str(ranked_rows[0]["segment_name"])
    weakest_decay_segment_name = str(ranked_rows[-1]["segment_name"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage212_natural_trace_decay_map",
        "title": "天然痕迹衰减图",
        "status_short": "natural_trace_decay_map_ready",
        "segment_count": len(rows),
        "continuity_gap": continuity_gap,
        "decay_map_score": decay_map_score,
        "strongest_decay_segment_name": strongest_decay_segment_name,
        "weakest_decay_segment_name": weakest_decay_segment_name,
        "decay_rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage212：天然痕迹衰减图",
        "",
        "## 核心结果",
        f"- 分段数量：{summary['segment_count']}",
        f"- 连续性缺口：{summary['continuity_gap']:.4f}",
        f"- 衰减图总分：{summary['decay_map_score']:.4f}",
        f"- 最强衰减段：{summary['strongest_decay_segment_name']}",
        f"- 最弱衰减段：{summary['weakest_decay_segment_name']}",
    ]
    (output_dir / "STAGE212_NATURAL_TRACE_DECAY_MAP_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="天然痕迹衰减图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
