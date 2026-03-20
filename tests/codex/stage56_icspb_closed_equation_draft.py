from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    short_form_summary: Dict[str, object],
    corpus_summary: Dict[str, object],
    kernel_summary: Dict[str, object],
    strict_summary: Dict[str, object],
) -> Dict[str, object]:
    return {
        "record_type": "stage56_icspb_closed_equation_draft_summary",
        "closed_equations": {
            "general_equation": "U_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)",
            "strict_equation": "U_strict(pair) ~= U_general(pair) + b * S(pair) + s * L_select(pair)",
            "discriminator_equation": "D_strict(pair) = dual_gap_final(pair)",
            "state_dictionary": {
                "G": "kernel_v4",
                "S": "strict_module_base",
                "D": "dual_gap_final",
                "gd": "主驱动通道",
                "L_base": "(gs + sd) / 2",
                "L_select": "(sd - gs) / 2",
            },
        },
        "supporting_signs": {
            "kernel_signs": dict(kernel_summary.get("signs", {})),
            "strict_choice": dict(strict_summary.get("final_choice", {})),
            "corpus_signs": dict(corpus_summary.get("sign_matrix", {})),
        },
        "main_judgment": (
            "当前 ICSPB 闭式方程草案已经可以写成“主核层 + 严格层 + 判别层”的三层结构，"
            "并且在自然语料口径下已经找到与 G / L_base / L_select 对应的自然代理。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 ICSPB 闭式方程草案摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Closed Equations",
    ]
    for key, value in dict(summary.get("closed_equations", {})).items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Draft a more explicit ICSPB closed equation system")
    ap.add_argument(
        "--short-form-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layered_equation_short_form_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--corpus-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_shortform_validation_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--kernel-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_kernel_v4_validation_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--strict-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_module_finalizer_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_icspb_closed_equation_draft_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(Path(args.short_form_json)),
        read_json(Path(args.corpus_json)),
        read_json(Path(args.kernel_json)),
        read_json(Path(args.strict_json)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
