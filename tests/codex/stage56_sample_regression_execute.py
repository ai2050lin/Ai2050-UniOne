from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_control_axis_regression import build_control_regression
from stage56_family_patch_offset_raw_chain import build_rows as build_family_rows, build_summary as build_family_summary
from stage56_fullsample_regression_runner import (
    build_design_rows,
    build_summary as build_runner_summary,
    read_json,
    read_jsonl,
)
from stage56_sign_stability_runner import build_summary as build_sign_summary
from stage56_static_raw_chain import build_rows as build_static_rows, build_summary as build_static_summary

ROOT = Path(__file__).resolve().parents[2]


def build_execute_summary(
    pair_density_rows: List[Dict[str, object]],
    complete_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    design_rows = build_design_rows(pair_density_rows, complete_rows)
    static_rows = build_static_rows(design_rows)
    family_rows = build_family_rows(static_rows)
    runner_summary = build_runner_summary(family_rows)
    static_summary = build_static_summary(static_rows)
    family_summary = build_family_summary(family_rows)
    control_summary = build_control_regression(family_rows)
    sign_summary = build_sign_summary(family_rows)
    return {
        "record_type": "stage56_sample_regression_execute_summary",
        "row_count": len(family_rows),
        "runner_summary": runner_summary,
        "static_summary": static_summary,
        "family_summary": family_summary,
        "control_summary": control_summary,
        "sign_summary": sign_summary,
        "main_judgment": (
            "当前样本集回归执行入口已经把设计矩阵、静态原始链、family patch / concept offset 原始链、"
            "控制轴回归和符号稳定分析收进同一执行面板。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return (
        "# Stage56 样本集回归执行摘要\n\n"
        f"- row_count: {summary.get('row_count', 0)}\n"
        f"- main_judgment: {summary.get('main_judgment', '')}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute the unified sample-regression suite once the runtime environment is available")
    ap.add_argument(
        "--pair-density-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_pair_density_tensor_field_20260319_1512" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--complete-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1645" / "joined_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_sample_regression_execute_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_density_rows = read_jsonl(Path(args.pair_density_jsonl))
    complete_rows = list(read_json(Path(args.complete_joined_json)).get("rows", []))
    summary = build_execute_summary(pair_density_rows, complete_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": summary["row_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
