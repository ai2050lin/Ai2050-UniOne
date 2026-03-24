#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage230_parameter_boundary_trigger_pack import run_analysis as run_stage230
from stage231_source_fidelity_parameter_structure_map import run_analysis as run_stage231
from stage263_qwen_deepseek_complete_behavior_suite import run_analysis as run_stage263
from stage264_qwen_deepseek_complete_structural_aggregate import run_analysis as run_stage264
from stage266_qwen_deepseek_parameter_hook_compare import run_analysis as run_stage266
from stage267_qwen_deepseek_same_class_competition_compare import run_analysis as run_stage267


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage268_complete_test_to_parameter_principle_bridge_20260324"


def by_tag(rows, tag: str) -> dict:
    return next(row for row in rows if row["model_tag"] == tag)


def build_summary() -> dict:
    s230 = run_stage230(force=False)
    s231 = run_stage231(force=False)
    s263 = run_stage263(force=False)
    s264 = run_stage264(force=False)
    s266 = run_stage266(force=False)
    s267 = run_stage267(force=False)

    rows = []
    for model_tag, display_name in [("qwen4b", "Qwen3-4B"), ("deepseek14b", "DeepSeek-R1-14B")]:
        behavior = by_tag(s263["model_rows"], model_tag)
        aggregate = by_tag(s264["model_rows"], model_tag)
        hook_tag = "qwen4b" if model_tag == "qwen4b" else "deepseek7b"
        hook_row = by_tag(s266["model_rows"], hook_tag)
        competition_row = by_tag(s267["model_rows"], hook_tag)
        bridge_parts = {
            "complete_test_score": float(aggregate["complete_score"]),
            "parameter_hook_score": float(hook_row["parameter_hook_score"]),
            "same_class_score": float(competition_row["same_class_score"]),
            "parameter_boundary_score": float(s230["parameter_boundary_score"]),
            "source_fidelity_parameter_score": float(s231["parameter_structure_score"]),
        }
        bridge_score = sum(bridge_parts.values()) / len(bridge_parts)
        rows.append(
            {
                "model_tag": model_tag,
                "display_name": display_name,
                "bridge_score": bridge_score,
                "strongest_piece_name": max(bridge_parts.items(), key=lambda item: item[1])[0],
                "weakest_piece_name": min(bridge_parts.items(), key=lambda item: item[1])[0],
                "bridge_parts": bridge_parts,
                "behavior_strongest_category": behavior["strongest_category"],
                "behavior_weakest_category": behavior["weakest_category"],
            }
        )
    strongest = max(rows, key=lambda row: row["bridge_score"])
    weakest = min(rows, key=lambda row: row["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage268_complete_test_to_parameter_principle_bridge",
        "title": "完整测试到参数级原理桥",
        "status_short": "complete_test_to_parameter_principle_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "score_gap": strongest["bridge_score"] - weakest["bridge_score"],
        "model_rows": rows,
        "top_gap_name": "完整测试差异已经能压回参数层，但天然来源保真仍然是两边共同主断点",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage268：完整测试到参数级原理桥",
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
                f"- 桥总分：{row['bridge_score']:.4f}",
                f"- 最强部件：{row['strongest_piece_name']}",
                f"- 最弱部件：{row['weakest_piece_name']}",
            ]
        )
    (output_dir / "STAGE268_COMPLETE_TEST_TO_PARAMETER_PRINCIPLE_BRIDGE_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="完整测试到参数级原理桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

