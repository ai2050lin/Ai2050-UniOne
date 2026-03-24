#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage263_qwen_deepseek_complete_behavior_suite import OUTPUT_DIR as STAGE263_OUTPUT_DIR, run_analysis as run_stage263
from stage264_qwen_deepseek_complete_structural_aggregate import OUTPUT_DIR as STAGE264_OUTPUT_DIR, run_analysis as run_stage264


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage265_qwen_deepseek_complete_final_review_20260324"


def build_summary() -> dict:
    behavior = run_stage263(output_dir=STAGE263_OUTPUT_DIR, force=False)
    aggregate = run_stage264(output_dir=STAGE264_OUTPUT_DIR, force=False)
    behavior_rows = {row["model_tag"]: row for row in behavior["model_rows"]}
    aggregate_rows = {row["model_tag"]: row for row in aggregate["model_rows"]}
    model_rows = []
    for model_tag in ["qwen4b", "deepseek14b"]:
        direct = behavior_rows[model_tag]
        whole = aggregate_rows[model_tag]
        model_rows.append(
            {
                "model_tag": model_tag,
                "display_name": whole["display_name"],
                "direct_behavior_score": direct["direct_score"],
                "historical_structure_score": whole["historical_structure_score"],
                "complete_score": whole["complete_score"],
                "strongest_category": direct["strongest_category"],
                "weakest_category": direct["weakest_category"],
            }
        )
    strongest = max(model_rows, key=lambda row: row["complete_score"])
    weakest = min(model_rows, key=lambda row: row["complete_score"])
    gap = strongest["complete_score"] - weakest["complete_score"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage265_qwen_deepseek_complete_final_review",
        "title": "Qwen 与 DeepSeek 完整测试总评",
        "status_short": "qwen_deepseek_complete_final_review_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "score_gap": gap,
        "model_rows": model_rows,
        "top_gap_name": "完整测试已经能分出强弱，但两边都还没有进入参数级硬主核闭合区",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage265：Qwen 与 DeepSeek 完整测试总评",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 分差：{summary['score_gap']:.4f}",
        f"- 头号缺口：{summary['top_gap_name']}",
    ]
    for row in summary["model_rows"]:
        lines.extend(
            [
                "",
                f"## {row['display_name']}",
                f"- 完整总分：{row['complete_score']:.4f}",
                f"- 行为直测分：{row['direct_behavior_score']:.4f}",
                f"- 历史结构分：{row['historical_structure_score']:.4f}",
                f"- 最强类别：{row['strongest_category']}",
                f"- 最弱类别：{row['weakest_category']}",
            ]
        )
    (output_dir / "STAGE265_QWEN_DEEPSEEK_COMPLETE_FINAL_REVIEW_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="Qwen 与 DeepSeek 完整测试总评")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

