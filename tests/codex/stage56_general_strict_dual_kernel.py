from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_fullsample_regression_runner import read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    kernel_summary: Dict[str, object],
    strict_summary: Dict[str, object],
) -> Dict[str, object]:
    kernel_sign = dict(kernel_summary.get("sign_matrix", {})).get("kernel_v4_term", {})
    strict_combined_sign = dict(strict_summary.get("sign_matrix", {})).get("strict_module_combined_term", {})
    strict_residual_sign = dict(strict_summary.get("sign_matrix", {})).get("strict_module_residual_term", {})
    return {
        "record_type": "stage56_general_strict_dual_kernel_summary",
        "general_kernel_sign": kernel_sign,
        "strict_module_combined_sign": strict_combined_sign,
        "strict_module_residual_sign": strict_residual_sign,
        "equation_text": (
            "general_kernel = kernel_v4; "
            "strict_kernel_module = strict_module_combined; "
            "strict_residual_penalty = strict_module_residual"
        ),
        "main_judgment": (
            "一般闭包核和严格闭包模块已经开始明显分离："
            "一般核由短闭式核承担，严格闭包更像额外附着在一般核上的正向组合模块与负向残差模块。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines: List[str] = [
        "# Stage56 一般核 / 严格核双结构摘要",
        "",
        f"- equation_text: {summary.get('equation_text', '')}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Signs",
        f"- general_kernel_sign: {json.dumps(summary.get('general_kernel_sign', {}), ensure_ascii=False)}",
        f"- strict_module_combined_sign: {json.dumps(summary.get('strict_module_combined_sign', {}), ensure_ascii=False)}",
        f"- strict_module_residual_sign: {json.dumps(summary.get('strict_module_residual_sign', {}), ensure_ascii=False)}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize a dual-branch general/strict kernel view")
    ap.add_argument(
        "--kernel-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_closed_form_v4_refit_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--strict-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_load_module_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_general_strict_dual_kernel_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kernel_summary = read_json(Path(args.kernel_summary_json))
    strict_summary = read_json(Path(args.strict_summary_json))
    summary = build_summary(kernel_summary, strict_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
