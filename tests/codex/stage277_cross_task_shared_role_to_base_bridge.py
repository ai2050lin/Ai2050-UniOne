#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage272_translation_refactor_parameter_role_card import run_analysis as run_stage272
from stage273_translation_content_to_output_bridge import run_analysis as run_stage273
from stage274_refactor_object_constraint_structure_bridge import run_analysis as run_stage274


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage277_cross_task_shared_role_to_base_bridge_20260324"


def build_summary() -> dict:
    s272 = run_stage272(force=False)
    s273 = run_stage273(force=False)
    s274 = run_stage274(force=False)

    rows = []
    for tag, display in [("qwen4b", "Qwen3-4B"), ("deepseek7b", "DeepSeek-R1-Distill-Qwen-7B")]:
        r272 = next(row for row in s272["model_rows"] if row["model_tag"] == tag)
        r273 = next(row for row in s273["model_rows"] if row["model_tag"] == tag)
        r274 = next(row for row in s274["model_rows"] if row["model_tag"] == tag)
        parts = {
            "task_role_entry": float(r272["role_score"]),
            "translation_content_preservation": float(r273["bridge_score"]),
            "refactor_structure_bridge": float(r274["bridge_score"]),
        }
        bridge_score = sum(parts.values()) / len(parts)
        rows.append(
            {
                "model_tag": tag,
                "display_name": display,
                "bridge_score": bridge_score,
                "strongest_part_name": max(parts.items(), key=lambda item: item[1])[0],
                "weakest_part_name": min(parts.items(), key=lambda item: item[1])[0],
                "bridge_parts": parts,
            }
        )
    strongest = max(rows, key=lambda row: row["bridge_score"])
    weakest = min(rows, key=lambda row: row["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage277_cross_task_shared_role_to_base_bridge",
        "title": "跨任务共享角色到底盘桥",
        "status_short": "cross_task_shared_role_to_base_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "翻译和重构虽然后段功能不同，但前段共享的不是词面，而是操作、对象、约束这些可复用角色，它们共同落在同一底盘参数体系上",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage277 跨任务共享角色到底盘桥",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE277_CROSS_TASK_SHARED_ROLE_TO_BASE_BRIDGE_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="跨任务共享角色到底盘桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
