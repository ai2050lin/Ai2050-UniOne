#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage184_cross_model_puzzle_expansion_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE184_CROSS_MODEL_PUZZLE_EXPANSION_REPORT.md"

STAGE181_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage181_cross_model_shared_puzzle_board_20260323" / "summary.json"
STAGE139_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s181 = load_json(STAGE181_SUMMARY_PATH)
    s139 = load_json(STAGE139_SUMMARY_PATH)
    s140 = load_json(STAGE140_SUMMARY_PATH)

    qwen = s139["transfer_summary"]["qwen_core_metrics"]
    deepseek = s140["transfer_summary"]["qwen_core_metrics"]

    spread_rows = [
        {
            "piece_name": "副词动态选路",
            "qwen_score": float(qwen["route_shift_layer_localization_score"]),
            "deepseek_score": float(deepseek["route_shift_layer_localization_score"]),
        },
        {
            "piece_name": "复杂语篇重提",
            "qwen_score": float(qwen["complex_discourse_noun_propagation_score"]),
            "deepseek_score": float(deepseek["complex_discourse_noun_propagation_score"]),
        },
        {
            "piece_name": "结果链",
            "qwen_score": float(qwen["noun_verb_result_chain_score"]),
            "deepseek_score": float(deepseek["noun_verb_result_chain_score"]),
        },
        {
            "piece_name": "条件场",
            "qwen_score": float(qwen["conditional_gating_field_score"]),
            "deepseek_score": float(deepseek["conditional_gating_field_score"]),
        },
    ]
    for row in spread_rows:
        row["spread"] = abs(float(row["qwen_score"]) - float(row["deepseek_score"]))
        row["shared_floor"] = min(float(row["qwen_score"]), float(row["deepseek_score"]))

    ranked_shared = sorted(s181["block_rows"], key=lambda row: float(row["score"]))
    strongest_consensus = str(ranked_shared[-1]["block_name"])
    weakest_consensus = str(ranked_shared[0]["block_name"])
    highest_spread_row = max(spread_rows, key=lambda row: float(row["spread"]))
    lowest_spread_row = min(spread_rows, key=lambda row: float(row["spread"]))
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage184_cross_model_puzzle_expansion",
        "title": "跨模型共同拼图扩张",
        "status_short": "cross_model_puzzle_expansion_ready",
        "shared_block_count": len(s181["block_rows"]),
        "spread_piece_count": len(spread_rows),
        "strongest_consensus_block_name": strongest_consensus,
        "weakest_consensus_block_name": weakest_consensus,
        "highest_spread_piece_name": str(highest_spread_row["piece_name"]),
        "lowest_spread_piece_name": str(lowest_spread_row["piece_name"]),
        "shared_rows": s181["block_rows"],
        "spread_rows": spread_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage184: 跨模型共同拼图扩张",
        "",
        "## 核心结果",
        f"- 共同块数量: {summary['shared_block_count']}",
        f"- 扩张块数量: {summary['spread_piece_count']}",
        f"- 最强共同块: {summary['strongest_consensus_block_name']}",
        f"- 最弱共同块: {summary['weakest_consensus_block_name']}",
        f"- 最大分歧块: {summary['highest_spread_piece_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型共同拼图扩张")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
