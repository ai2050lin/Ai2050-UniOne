#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage263_qwen_deepseek_complete_behavior_suite import OUTPUT_DIR as STAGE263_OUTPUT_DIR, run_analysis as run_stage263


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage264_qwen_deepseek_complete_structural_aggregate_20260324"
QWEN_STAGE139 = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
DEEPSEEK_STAGE140 = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"
DEEPSEEK_STAGE235 = PROJECT_ROOT / "tests" / "codex_temp" / "stage235_deepseek_direct_fidelity_recheck_20260324" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def qwen_structure_score(payload: dict) -> dict:
    score_parts = {
        "theory_check_pass_rate": float(payload["transfer_summary"]["theory_check_pass_rate"]),
        "adverb_context_route_shift_score": float(payload["dynamic_summary"]["adverb_context_route_shift_score"]),
        "syntax_stability_rate": float(payload["noun_context_summary"]["syntax_stability_rate"]),
        "noun_verb_result_chain_score": float(payload["result_summary"]["noun_verb_result_chain_score"]),
    }
    return {"parts": score_parts, "score": sum(score_parts.values()) / len(score_parts)}


def deepseek_structure_score(payload140: dict, payload235: dict) -> dict:
    score_parts = {
        "theory_check_pass_rate": float(payload140["transfer_summary"]["theory_check_pass_rate"]),
        "adverb_context_route_shift_score": float(payload140["dynamic_summary"]["adverb_context_route_shift_score"]),
        "syntax_stability_rate": float(payload140["noun_context_summary"]["syntax_stability_rate"]),
        "noun_verb_result_chain_score": float(payload140["result_summary"]["noun_verb_result_chain_score"]),
        "direct_fidelity_recheck_score": float(payload235["recheck_score"]),
    }
    return {"parts": score_parts, "score": sum(score_parts.values()) / len(score_parts)}


def build_summary() -> dict:
    behavior = run_stage263(output_dir=STAGE263_OUTPUT_DIR, force=False)
    qwen139 = load_json(QWEN_STAGE139)
    deepseek140 = load_json(DEEPSEEK_STAGE140)
    deepseek235 = load_json(DEEPSEEK_STAGE235)
    qwen_behavior = next(row for row in behavior["model_rows"] if row["model_tag"] == "qwen4b")
    deepseek_behavior = next(row for row in behavior["model_rows"] if row["model_tag"] == "deepseek14b")
    qwen_struct = qwen_structure_score(qwen139)
    deepseek_struct = deepseek_structure_score(deepseek140, deepseek235)
    model_rows = [
        {
            "model_tag": "qwen4b",
            "display_name": "Qwen3-4B",
            "direct_behavior_score": qwen_behavior["direct_score"],
            "historical_structure_score": qwen_struct["score"],
            "complete_score": (qwen_behavior["direct_score"] + qwen_struct["score"]) / 2.0,
            "structure_parts": qwen_struct["parts"],
        },
        {
            "model_tag": "deepseek14b",
            "display_name": "DeepSeek-R1-14B",
            "direct_behavior_score": deepseek_behavior["direct_score"],
            "historical_structure_score": deepseek_struct["score"],
            "complete_score": (deepseek_behavior["direct_score"] + deepseek_struct["score"]) / 2.0,
            "structure_parts": deepseek_struct["parts"],
        },
    ]
    strongest = max(model_rows, key=lambda row: row["complete_score"])
    weakest = min(model_rows, key=lambda row: row["complete_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage264_qwen_deepseek_complete_structural_aggregate",
        "title": "Qwen 与 DeepSeek 完整测试结构汇总层",
        "status_short": "qwen_deepseek_complete_structural_aggregate_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": model_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage264：Qwen 与 DeepSeek 完整测试结构汇总层",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 行为直测分：{row['direct_behavior_score']:.4f}",
                f"- 历史结构分：{row['historical_structure_score']:.4f}",
                f"- 完整总分：{row['complete_score']:.4f}",
            ]
        )
    (output_dir / "STAGE264_QWEN_DEEPSEEK_COMPLETE_STRUCTURAL_AGGREGATE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 与 DeepSeek 完整测试结构汇总层")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
