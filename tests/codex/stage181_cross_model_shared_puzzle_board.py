#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage181_cross_model_shared_puzzle_board_20260323"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "STAGE181_CROSS_MODEL_SHARED_PUZZLE_BOARD_REPORT.md"

STAGE139_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
STAGE140_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"
STAGE159_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage159_triple_model_apple_kernel_20260323" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_block(score: float) -> str:
    if score >= 0.65:
        return "共同稳定"
    if score >= 0.45:
        return "共同过渡"
    return "共同薄弱"


def build_summary() -> dict:
    s139 = load_json(STAGE139_SUMMARY_PATH)
    s140 = load_json(STAGE140_SUMMARY_PATH)
    s159 = load_json(STAGE159_SUMMARY_PATH)

    qwen = s139["transfer_summary"]["qwen_core_metrics"]
    deepseek = s140["transfer_summary"]["qwen_core_metrics"]
    gpt2_ref = s139["transfer_summary"]["gpt2_reference_snapshot"]

    block_rows = [
        {
            "block_name": "苹果共同核",
            "score": float(s159["shared_core_consensus_score"]),
        },
        {
            "block_name": "副词动态选路",
            "score": min(
                float(gpt2_ref["stage123_route_shift_layer_localization_score"]),
                float(qwen["route_shift_layer_localization_score"]),
                float(deepseek["route_shift_layer_localization_score"]),
            ),
        },
        {
            "block_name": "复杂语篇重提",
            "score": min(
                float(gpt2_ref["stage133_complex_discourse_noun_propagation_score"]),
                float(qwen["complex_discourse_noun_propagation_score"]),
                float(deepseek["complex_discourse_noun_propagation_score"]),
            ),
        },
        {
            "block_name": "结果链",
            "score": min(
                float(gpt2_ref["stage137_noun_verb_result_chain_score"]),
                float(qwen["noun_verb_result_chain_score"]),
                float(deepseek["noun_verb_result_chain_score"]),
            ),
        },
        {
            "block_name": "条件场",
            "score": min(
                float(qwen["conditional_gating_field_score"]),
                float(deepseek["conditional_gating_field_score"]),
            ),
        },
    ]
    for row in block_rows:
        row["status"] = classify_block(float(row["score"]))
    ranked_rows = sorted(block_rows, key=lambda row: float(row["score"]))
    stable_shared_count = sum(1 for row in block_rows if str(row["status"]) == "共同稳定")
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage181_cross_model_shared_puzzle_board",
        "title": "跨模型共同拼图板",
        "status_short": "cross_model_shared_puzzle_board_ready",
        "shared_block_count": len(block_rows),
        "stable_shared_count": stable_shared_count,
        "strongest_shared_block_name": str(ranked_rows[-1]["block_name"]),
        "weakest_shared_block_name": str(ranked_rows[0]["block_name"]),
        "block_rows": block_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage181: 跨模型共同拼图板",
        "",
        "## 核心结果",
        f"- 共同拼图块数量: {summary['shared_block_count']}",
        f"- 共同稳定块数量: {summary['stable_shared_count']}",
        f"- 最强共同拼图块: {summary['strongest_shared_block_name']}",
        f"- 最弱共同拼图块: {summary['weakest_shared_block_name']}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    if not force and SUMMARY_PATH.exists():
        return load_json(SUMMARY_PATH)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨模型共同拼图板")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重建")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
