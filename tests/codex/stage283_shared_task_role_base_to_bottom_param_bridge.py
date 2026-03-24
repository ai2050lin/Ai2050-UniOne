#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage272_translation_refactor_parameter_role_card import run_analysis as run_stage272
from stage277_cross_task_shared_role_to_base_bridge import run_analysis as run_stage277
from stage280_translation_refactor_shared_base_compression import run_analysis as run_stage280


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage283_shared_task_role_base_to_bottom_param_bridge_20260324"


def build_summary() -> dict:
    s272 = run_stage272(force=False)
    s277 = run_stage277(force=False)
    s280 = run_stage280(force=False)

    rows = []
    for tag, display in [("qwen4b", "Qwen3-4B"), ("deepseek7b", "DeepSeek-R1-Distill-Qwen-7B")]:
        r272 = next(row for row in s272["model_rows"] if row["model_tag"] == tag)
        r277 = next(row for row in s277["model_rows"] if row["model_tag"] == tag)
        r280 = next(row for row in s280["model_rows"] if row["model_tag"] == tag)
        parts = {
            "role_entry": float(r272["role_score"]),
            "shared_role_bridge": float(r277["bridge_score"]),
            "shared_base_compression": float(r280["compression_score"]),
        }
        bridge_score = sum(parts.values()) / len(parts)
        rows.append(
            {
                "model_tag": tag,
                "display_name": display,
                "bridge_score": float(bridge_score),
                "strongest_part_name": max(parts.items(), key=lambda item: item[1])[0],
                "weakest_part_name": min(parts.items(), key=lambda item: item[1])[0],
                "parts": parts,
            }
        )
    strongest = max(rows, key=lambda item: item["bridge_score"])
    weakest = min(rows, key=lambda item: item["bridge_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage283_shared_task_role_base_to_bottom_param_bridge",
        "title": "共享任务角色底盘到底层参数桥",
        "status_short": "shared_task_role_base_to_bottom_param_bridge_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "翻译与重构前段共享的不只是表面角色，而是已经可以被继续压到更底层的参数底盘桥上",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary = build_summary()
    write_outputs(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="共享任务角色底盘到底层参数桥")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
