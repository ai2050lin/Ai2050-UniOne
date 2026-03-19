from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_summary(kernel_summary: Dict[str, object], strict_summary: Dict[str, object], gap_summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "record_type": "stage56_dual_equation_layered_summary",
        "general_layer": {
            "equation": "U_general = kernel_v4",
            "stable_signs": dict(kernel_summary.get("signs", {})),
        },
        "strict_layer": {
            "equation": "U_strict = strict_module_base",
            "final_choice": dict(strict_summary.get("final_choice", {})),
        },
        "discriminator_layer": {
            "equation": "D_strict = dual_gap_final",
            "stable_signs": dict(gap_summary.get("signs", {})),
        },
        "main_judgment": (
            "双主式更适合写成分层结构：一般闭包核负责主核层，严格闭包模块负责严格层，"
            "dual_gap 负责判别层，而不是与前两层同层并场。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 双主式层级化摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## General Layer",
        f"- equation: {dict(summary.get('general_layer', {})).get('equation', '')}",
        f"- stable_signs: {json.dumps(dict(summary.get('general_layer', {})).get('stable_signs', {}), ensure_ascii=False)}",
        "",
        "## Strict Layer",
        f"- equation: {dict(summary.get('strict_layer', {})).get('equation', '')}",
        f"- final_choice: {json.dumps(dict(summary.get('strict_layer', {})).get('final_choice', {}), ensure_ascii=False)}",
        "",
        "## Discriminator Layer",
        f"- equation: {dict(summary.get('discriminator_layer', {})).get('equation', '')}",
        f"- stable_signs: {json.dumps(dict(summary.get('discriminator_layer', {})).get('stable_signs', {}), ensure_ascii=False)}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize a layered dual-equation structure")
    ap.add_argument(
        "--kernel-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_validation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--strict-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_finalizer_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--gap-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_gap_classifier_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_layered_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kernel_summary = read_json(Path(args.kernel_summary_json))
    strict_summary = read_json(Path(args.strict_summary_json))
    gap_summary = read_json(Path(args.gap_summary_json))
    summary = build_summary(kernel_summary, strict_summary, gap_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
