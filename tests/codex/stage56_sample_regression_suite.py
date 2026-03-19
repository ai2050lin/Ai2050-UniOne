from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_control_axis_regression import build_control_regression
from stage56_fullsample_regression_runner import (
    build_design_rows,
    build_summary as build_runner_summary,
    read_json,
    read_jsonl,
)
from stage56_static_raw_chain import build_static_raw_rows, build_summary as build_static_summary

ROOT = Path(__file__).resolve().parents[2]


def build_suite(
    pair_density_rows: List[Dict[str, object]],
    complete_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    design_rows = build_design_rows(pair_density_rows, complete_rows)
    static_rows = build_static_raw_rows(design_rows)
    runner_summary = build_runner_summary(static_rows)
    control_summary = build_control_regression(static_rows)
    static_summary = build_static_summary(static_rows)
    return {
        "record_type": "stage56_sample_regression_suite_summary",
        "design_row_count": len(static_rows),
        "runner_summary": runner_summary,
        "control_summary": control_summary,
        "static_summary": static_summary,
        "main_judgment": (
            "当前样本集回归三大任务块已经被统一收束到一个套件入口："
            "样本级设计矩阵、静态原始估计链、控制轴回归。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    return (
        "# Stage56 样本集回归套件摘要\n\n"
        f"- design_row_count: {summary.get('design_row_count', 0)}\n"
        f"- main_judgment: {summary.get('main_judgment', '')}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the unified sample regression suite over current sample-level design data")
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
        default=str(ROOT / "tests" / "codex_temp" / "stage56_sample_regression_suite_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_density_rows = read_jsonl(Path(args.pair_density_jsonl))
    complete_rows = list(read_json(Path(args.complete_joined_json)).get("rows", []))
    summary = build_suite(pair_density_rows, complete_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "design_row_count": summary["design_row_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
