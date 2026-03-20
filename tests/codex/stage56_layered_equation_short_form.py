from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from stage56_fullsample_regression_runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def build_summary(
    generalized_refinement: Dict[str, object],
    strict_select_summary: Dict[str, object],
    formal_summary: Dict[str, object],
) -> Dict[str, object]:
    refined = dict(generalized_refinement.get("refined_formulas", {}))
    strict_signs = dict(strict_select_summary.get("sign_matrix", {}))
    strict_choice = dict(formal_summary.get("layer_stability", {}).get("strict_choice", {}))

    short_form = {
        "general_short_form": "U_general(pair) ~= a * G(pair) + d * D(pair) + p * gd(pair) - l * L_base(pair)",
        "strict_short_form": (
            "U_strict(pair) ~= U_general(pair) + b * S(pair) + s * L_select(pair)"
        ),
        "discriminator_short_form": "D_strict(pair) = dual_gap_final(pair)",
        "state_dictionary": {
            "G": "kernel_v4",
            "S": "strict_module_base",
            "D": "dual_gap_final",
            "L_base": "(gs + sd) / 2",
            "L_select": "(sd - gs) / 2",
        },
    }

    return {
        "record_type": "stage56_layered_equation_short_form_summary",
        "strict_select_signs": strict_signs,
        "strict_choice": strict_choice,
        "refined_formulas": refined,
        "short_form": short_form,
        "main_judgment": (
            "当前分层双主式已经可以进一步压成更短的正式写法："
            "一般层由 G、D、gd 与 L_base 决定，严格层则在其基础上再叠加 S 与 L_select。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 分层方程短式摘要",
        "",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Short Form",
    ]
    for key, value in dict(summary.get("short_form", {})).items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compress layered equations into a shorter formal system")
    ap.add_argument(
        "--generalized-refinement-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_generalized_formula_refinement_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--strict-select-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_strict_select_expansion_20260320" / "summary.json"),
    )
    ap.add_argument(
        "--formal-summary-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_dual_equation_formal_system_20260319" / "summary.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_layered_equation_short_form_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    generalized_refinement = read_json(Path(args.generalized_refinement_json))
    strict_select_summary = read_json(Path(args.strict_select_json))
    formal_summary = read_json(Path(args.formal_summary_json))
    summary = build_summary(generalized_refinement, strict_select_summary, formal_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
