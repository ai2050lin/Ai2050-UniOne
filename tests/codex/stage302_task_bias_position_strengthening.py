#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_STAGE272 = PROJECT_ROOT / "tests" / "codex_temp" / "stage272_translation_refactor_parameter_role_card_20260324" / "summary.json"
INPUT_STAGE283 = PROJECT_ROOT / "tests" / "codex_temp" / "stage283_shared_task_role_base_to_bottom_param_bridge_20260324" / "summary.json"
INPUT_STAGE299 = PROJECT_ROOT / "tests" / "codex_temp" / "stage299_bias_position_role_card_20260324" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage302_task_bias_position_strengthening_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary() -> dict:
    s272 = load_json(INPUT_STAGE272)
    s283 = load_json(INPUT_STAGE283)
    s299 = load_json(INPUT_STAGE299)

    task_rows = []
    for role_model, bridge_model in zip(s272["model_rows"], s283["model_rows"]):
        operation_strength = 0.0
        constraint_strength = 0.0
        for task in role_model["task_rows"]:
            for role in task["role_rows"]:
                if role["role_name"] == "operation_role":
                    operation_strength += role["role_score"]
                if role["role_name"] == "constraint_role":
                    constraint_strength += role["role_score"]
        task_rows.append(
            {
                "model_tag": role_model["model_tag"],
                "display_name": role_model["display_name"],
                "operation_strength": operation_strength / max(1, len(role_model["task_rows"])),
                "constraint_strength": constraint_strength / max(1, len(role_model["task_rows"])),
                "shared_task_bridge": bridge_model["bridge_score"],
            }
        )

    strengthening_score = (
        sum(row["operation_strength"] + row["constraint_strength"] for row in task_rows) / max(1, len(task_rows)) * 0.20
        + sum(row["shared_task_bridge"] for row in task_rows) / max(1, len(task_rows)) * 0.40
        + float(s299["role_score"]) * 0.40
    )

    strongest_model = max(task_rows, key=lambda row: row["operation_strength"] + row["constraint_strength"])["display_name"]
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage302_task_bias_position_strengthening",
        "title": "任务偏转位补强",
        "status_short": "task_bias_position_strengthening_ready",
        "strengthening_score": float(strengthening_score),
        "strongest_model": strongest_model,
        "task_rows": task_rows,
        "top_gap_name": "任务偏转位正在通过操作角色和约束角色显影，但它们在偏置位总图里仍然偏少",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="任务偏转位补强")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
