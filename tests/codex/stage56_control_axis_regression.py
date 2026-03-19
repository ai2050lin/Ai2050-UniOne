from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_fullsample_regression_runner import fit_linear_regression, read_json

ROOT = Path(__file__).resolve().parents[2]


def build_control_regression(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "style_control_proxy",
        "logic_control_proxy",
        "syntax_control_proxy",
    ]
    fits = [
        fit_linear_regression(rows, feature_names, "union_joint_adv"),
        fit_linear_regression(rows, feature_names, "union_synergy_joint"),
        fit_linear_regression(rows, feature_names, "strict_positive_synergy"),
    ]
    sign_consistency = {
        feature: [
            dict(fit.get("weights", {})).get(feature, 0.0)
            for fit in fits
        ]
        for feature in feature_names
    }
    return {
        "record_type": "stage56_control_axis_regression_summary",
        "row_count": len(rows),
        "fits": fits,
        "sign_consistency": sign_consistency,
        "main_judgment": (
            "控制轴样本级回归已经具备独立入口，可以单独检验 style / logic / syntax 的样本级符号是否稳定。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 控制轴样本级回归摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Sign Consistency",
    ]
    for key, values in dict(summary.get("sign_consistency", {})).items():
        lines.append(f"- {key}: {', '.join(f'{float(v):+.6f}' for v in values)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the first sample-level regression for control-axis terms")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_control_axis_regression_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    summary = build_control_regression(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
