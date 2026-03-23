#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage194_bottom_block_intervention_priority_20260323"

STAGE188_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage188_apple_neuron_role_card_20260323" / "summary.json"
STAGE192_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage192_time_unfolded_role_slicing_20260323" / "summary.json"
STAGE193_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage193_cross_model_invariant_3d_blocks_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_priority(score: float) -> str:
    if score < 0.45:
        return "一级干预"
    if score < 0.60:
        return "二级干预"
    return "持续观察"


def build_summary() -> dict:
    s188 = load_json(STAGE188_SUMMARY_PATH)
    s192 = load_json(STAGE192_SUMMARY_PATH)
    s193 = load_json(STAGE193_SUMMARY_PATH)

    target_rows = []
    target_rows.extend(
        {
            "target_name": str(row["role_name"]),
            "score": float(row["score"]),
            "source_type": "角色束",
            "priority": classify_priority(float(row["score"])),
        }
        for row in s188["role_rows"]
    )
    target_rows.extend(
        {
            "target_name": str(row["slice_name"]),
            "score": float(row["score"]),
            "source_type": "时间切片",
            "priority": classify_priority(float(row["score"])),
        }
        for row in s192["slice_rows"]
    )
    target_rows.extend(
        {
            "target_name": str(row["block_name"]),
            "score": float(row["score"]),
            "source_type": "跨模型拼块",
            "priority": classify_priority(float(row["score"])),
        }
        for row in s193["block_rows"]
    )
    ranked_rows = sorted(target_rows, key=lambda row: float(row["score"]))
    level1_count = sum(1 for row in target_rows if str(row["priority"]) == "一级干预")
    level2_count = sum(1 for row in target_rows if str(row["priority"]) == "二级干预")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage194_bottom_block_intervention_priority",
        "title": "底层拼块因果干预优先级",
        "status_short": "bottom_block_intervention_priority_ready",
        "target_count": len(target_rows),
        "level1_count": level1_count,
        "level2_count": level2_count,
        "top_priority_name": str(ranked_rows[0]["target_name"]),
        "second_priority_name": str(ranked_rows[1]["target_name"]),
        "target_rows": target_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage194：底层拼块因果干预优先级",
        "",
        "## 核心结果",
        f"- 目标数量：{summary['target_count']}",
        f"- 一级干预数量：{summary['level1_count']}",
        f"- 二级干预数量：{summary['level2_count']}",
        f"- 头号干预目标：{summary['top_priority_name']}",
        f"- 次级干预目标：{summary['second_priority_name']}",
    ]
    (output_dir / "STAGE194_BOTTOM_BLOCK_INTERVENTION_PRIORITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="底层拼块因果干预优先级")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
