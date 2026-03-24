#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage310_shared_carrier_operator_core_compression import run_analysis as run_stage310
from stage311_bias_deflection_operator_core_compression import run_analysis as run_stage311
from stage309_operator_to_architecture_bridge import run_analysis as run_stage309


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / "stage312_operator_special_format_export_20260324"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_summary() -> tuple[dict, list[dict]]:
    s310 = run_stage310(force=False)
    s311 = run_stage311(force=False)
    s309 = run_stage309(force=False)

    records: list[dict] = []
    for row in s310["core_rows"]:
        records.append(
            {
                "schema": "agi_operator_trace_v1",
                "record_type": "shared_carrier_core",
                "dim_index": row["dim_index"],
                "role": row["core_role"],
                "score": row["core_strength"],
                "base_load": row["base_load"],
                "delta_load": row["mean_delta_load"],
            }
        )
    for row in s311["core_rows"]:
        records.append(
            {
                "schema": "agi_operator_trace_v1",
                "record_type": "bias_deflection_core",
                "dim_index": row["dim_index"],
                "role": row["core_role"],
                "score": row["core_strength"],
                "selectivity": row["selectivity"],
                "delta_load": row["mean_delta_load"],
            }
        )
    records.append(
        {
            "schema": "agi_operator_trace_v1",
            "record_type": "operator_architecture_template",
            "template": s309["architecture_template"],
            "score": s309["bridge_score"],
        }
    )

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage312_operator_special_format_export",
        "title": "三算子原始参数特殊格式导出",
        "status_short": "operator_special_format_export_ready",
        "record_count": len(records),
        "export_schema_name": "agi_operator_trace_v1",
        "jsonl_path": str(OUTPUT_DIR / "agi_operator_trace_v1.jsonl"),
        "top_gap_name": "当前已能把共享承载、偏置偏转和联合放大导出成统一分析格式，但记录仍然来自工作性主核，不是完整底层全量参数转储",
    }
    return summary, records


def write_outputs(summary: dict, records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    write_jsonl(output_dir / "agi_operator_trace_v1.jsonl", records)


def run_analysis(*, output_dir: Path = OUTPUT_DIR, force: bool = False) -> dict:
    summary_path = output_dir / "summary.json"
    if not force and summary_path.exists():
        return load_json(summary_path)
    summary, records = build_summary()
    write_outputs(summary, records, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="三算子原始参数特殊格式导出")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(output_dir=Path(args.output_dir), force=bool(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
