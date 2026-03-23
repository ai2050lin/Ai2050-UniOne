#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage178_puzzle_board_overview_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE178_PUZZLE_BOARD_OVERVIEW_REPORT.md"

SUMMARY_PATHS = {
    "共享核": PROJECT_ROOT / "tests" / "codex_temp" / "stage154_apple_fruit_shared_core_20260323" / "summary.json",
    "边界裂缝": PROJECT_ROOT / "tests" / "codex_temp" / "stage155_apple_boundary_crack_map_20260323" / "summary.json",
    "上下文偏置": PROJECT_ROOT / "tests" / "codex_temp" / "stage156_apple_context_bias_shift_20260323" / "summary.json",
    "动作选路": PROJECT_ROOT / "tests" / "codex_temp" / "stage157_apple_action_route_probe_20260323" / "summary.json",
    "结果绑定": PROJECT_ROOT / "tests" / "codex_temp" / "stage158_apple_result_binding_probe_20260323" / "summary.json",
    "结果修复": PROJECT_ROOT / "tests" / "codex_temp" / "stage160_apple_result_repair_map_20260323" / "summary.json",
    "类别纤维": PROJECT_ROOT / "tests" / "codex_temp" / "stage166_category_fiber_map_20260323" / "summary.json",
    "来源痕迹": PROJECT_ROOT / "tests" / "codex_temp" / "stage172_provenance_trace_probe_20260323" / "summary.json",
    "多实体压力": PROJECT_ROOT / "tests" / "codex_temp" / "stage173_multi_entity_recovery_stress_test_20260323" / "summary.json",
    "跨模型共同核": PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323" / "summary.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_block_score(block_name: str, summary: dict) -> float:
    if block_name == "共享核":
        return float(summary["shared_core_score"])
    if block_name == "边界裂缝":
        return 1.0 - float(summary["collision_rate"])
    if block_name == "上下文偏置":
        return float(summary["context_bias_shift_score"])
    if block_name == "动作选路":
        return float(summary["apple_action_route_score"])
    if block_name == "结果绑定":
        return float(summary["apple_result_binding_score"])
    if block_name == "结果修复":
        return float(summary["apple_result_repair_score"])
    if block_name == "类别纤维":
        return float(summary["category_fiber_score"])
    if block_name == "来源痕迹":
        return float(summary["provenance_trace_score"])
    if block_name == "多实体压力":
        return float(summary["stress_survival_score"])
    if block_name == "跨模型共同核":
        return float(summary["shared_core_consensus_score"])
    raise KeyError(block_name)


def classify_block(score: float) -> str:
    if score >= 0.75:
        return "稳定"
    if score >= 0.6:
        return "可用"
    if score >= 0.45:
        return "过渡"
    return "缺口"


def build_summary() -> dict:
    block_rows = []
    for block_name, path in SUMMARY_PATHS.items():
        summary = load_json(path)
        score = get_block_score(block_name, summary)
        block_rows.append(
            {
                "block_name": block_name,
                "score": score,
                "status": classify_block(score),
                "source_experiment": str(summary["experiment_id"]),
            }
        )
    ranked_rows = sorted(block_rows, key=lambda row: float(row["score"]))
    stable_block_count = sum(1 for row in block_rows if str(row["status"]) == "稳定")
    gap_block_count = sum(1 for row in block_rows if str(row["status"]) == "缺口")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage178_puzzle_board_overview",
        "title": "拼图板块总图",
        "status_short": "puzzle_board_overview_ready",
        "puzzle_block_count": len(block_rows),
        "stable_block_count": stable_block_count,
        "gap_block_count": gap_block_count,
        "strongest_block_name": str(ranked_rows[-1]["block_name"]),
        "weakest_block_name": str(ranked_rows[0]["block_name"]),
        "block_rows": block_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage178: 拼图板块总图",
        "",
        "## 核心结果",
        f"- 拼图块数量: {summary['puzzle_block_count']}",
        f"- 稳定拼图块数量: {summary['stable_block_count']}",
        f"- 缺口拼图块数量: {summary['gap_block_count']}",
        f"- 最强拼图块: {summary['strongest_block_name']}",
        f"- 最弱拼图块: {summary['weakest_block_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="拼图板块总图")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
