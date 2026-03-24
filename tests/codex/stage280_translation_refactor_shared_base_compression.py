#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage272_translation_refactor_parameter_role_card import run_analysis as run_stage272
from stage278_translation_target_language_readout_position_map import run_analysis as run_stage278
from stage279_refactor_structure_rewrite_causal_map import run_analysis as run_stage279


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage280_translation_refactor_shared_base_compression_20260324"


def build_summary() -> dict:
    s272 = run_stage272(force=False)
    s278 = run_stage278(force=False)
    s279 = run_stage279(force=False)

    rows = []
    for tag, display in [("qwen4b", "Qwen3-4B"), ("deepseek7b", "DeepSeek-R1-Distill-Qwen-7B")]:
        r272 = next(row for row in s272["model_rows"] if row["model_tag"] == tag)
        r278 = next(row for row in s278["model_rows"] if row["model_tag"] == tag)
        r279 = next(row for row in s279["model_rows"] if row["model_tag"] == tag)
        parts = {
            "shared_task_role_base": float(r272["role_score"]),
            "translation_readout_extension": float(r278["readout_score"]),
            "refactor_structure_extension": float(r279["causal_score"]),
        }
        score = sum(parts.values()) / len(parts)
        rows.append(
            {
                "model_tag": tag,
                "display_name": display,
                "compression_score": score,
                "strongest_part_name": max(parts.items(), key=lambda item: item[1])[0],
                "weakest_part_name": min(parts.items(), key=lambda item: item[1])[0],
                "parts": parts,
            }
        )

    strongest = max(rows, key=lambda row: row["compression_score"])
    weakest = min(rows, key=lambda row: row["compression_score"])
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage280_translation_refactor_shared_base_compression",
        "title": "翻译与重构共享底盘压缩",
        "status_short": "translation_refactor_shared_base_compression_ready",
        "strongest_model": strongest["display_name"],
        "weakest_model": weakest["display_name"],
        "model_rows": rows,
        "top_gap_name": "翻译和重构前段共享的是同一套任务角色底盘，后段只是沿不同扩展位继续展开成目标语言读出或结构改写",
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    lines = [
        "# Stage280 翻译与重构共享底盘压缩",
        "",
        f"- 最强模型：{summary['strongest_model']}",
        f"- 最弱模型：{summary['weakest_model']}",
        f"- 关键结论：{summary['top_gap_name']}",
    ]
    (output_dir / "STAGE280_TRANSLATION_REFACTOR_SHARED_BASE_COMPRESSION_REPORT.md").write_text(
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
    parser = argparse.ArgumentParser(description="翻译与重构共享底盘压缩")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(output_dir=Path(args.output_dir), force=bool(args.force))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
